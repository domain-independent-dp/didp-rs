use crate::hashable_state;
use crate::solver;
use crate::successor_generator;
use didp_parser::variable;
use didp_parser::Transition;
use rustc_hash::FxHashMap;
use std::error::Error;
use std::fmt;
use std::rc::Rc;
use std::str;

#[derive(Default)]
pub struct ForwardRecursion<T> {
    memo_capacity: Option<usize>,
    parameters: solver::SolverParameters<T>,
}

impl<T: variable::Numeric> solver::Solver<T> for ForwardRecursion<T>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    fn solve(
        &mut self,
        model: &didp_parser::Model<T>,
    ) -> Result<solver::Solution<T>, Box<dyn Error>> {
        let time_keeper = self.parameters.time_limit.map(solver::TimeKeeper::new);
        let generator = successor_generator::SuccessorGenerator::<Transition<T>>::new(model, false);
        let mut memo = FxHashMap::default();
        if let Some(capacity) = self.memo_capacity {
            memo.reserve(capacity);
        }
        let mut expanded = 0;
        let state = hashable_state::HashableState::new(&model.target);
        let cost = forward_recursion(
            state,
            model,
            &generator,
            &mut memo,
            &time_keeper,
            &mut expanded,
        );
        let mut transitions = Vec::new();
        match model.reduce_function {
            didp_parser::ReduceFunction::Max | didp_parser::ReduceFunction::Min
                if cost.is_some() =>
            {
                let mut state = hashable_state::HashableState::new(&model.target);
                while let Some((_, Some(transition))) = memo.get(&state) {
                    let transition = transition.clone();
                    state = transition.apply(&state, &model.table_registry);
                    transitions.push(transition);
                }
            }
            _ => {}
        }
        println!("Expanded: {}", expanded);
        Ok(solver::Solution {
            cost,
            transitions,
            is_optimal: cost.is_some()
                && (model.reduce_function == didp_parser::ReduceFunction::Max
                    || model.reduce_function == didp_parser::ReduceFunction::Min),
            is_infeasible: cost.is_none(),
            ..Default::default()
        })
    }

    #[inline]
    fn set_primal_bound(&mut self, primal_bound: T) {
        self.parameters.primal_bound = Some(primal_bound)
    }

    #[inline]
    fn set_time_limit(&mut self, time_limit: u64) {
        self.parameters.time_limit = Some(time_limit)
    }

    #[inline]
    fn get_primal_bound(&self) -> Option<T> {
        self.parameters.primal_bound
    }

    #[inline]
    fn get_time_limit(&self) -> Option<u64> {
        self.parameters.time_limit
    }
}

impl<T: variable::Numeric> ForwardRecursion<T> {
    pub fn new(config: &yaml_rust::Yaml) -> Result<ForwardRecursion<T>, solver::ConfigErr>
    where
        <T as str::FromStr>::Err: fmt::Debug,
    {
        let map = match config {
            yaml_rust::Yaml::Hash(map) => map,
            yaml_rust::Yaml::Null => return Ok(ForwardRecursion::default()),
            _ => {
                return Err(solver::ConfigErr::new(format!(
                    "expected Hash, but found `{:?}`",
                    config
                )))
            }
        };
        let memo_capacity = match map.get(&yaml_rust::Yaml::from_str("capacity")) {
            Some(yaml_rust::Yaml::Integer(value)) => Some(*value as usize),
            None => Some(1000000),
            value => {
                return Err(solver::ConfigErr::new(format!(
                    "expected Integer, but found `{:?}`",
                    value
                )))
            }
        };
        let parameters = solver::SolverParameters::parse_from_map(map)?;
        Ok(ForwardRecursion {
            memo_capacity,
            parameters,
        })
    }
}

type StateMemo<T> =
    FxHashMap<hashable_state::HashableState, (Option<T>, Option<Rc<didp_parser::Transition<T>>>)>;

pub fn forward_recursion<T: variable::Numeric>(
    state: hashable_state::HashableState,
    model: &didp_parser::Model<T>,
    generator: &successor_generator::SuccessorGenerator<Transition<T>>,
    memo: &mut StateMemo<T>,
    time_keeper: &Option<solver::TimeKeeper>,
    expanded: &mut i32,
) -> Option<T> {
    *expanded += 1;
    if model.is_goal(&state) {
        return Some(T::zero());
    }
    if let Some((cost, _)) = memo.get(&state) {
        return *cost;
    }
    if time_keeper
        .as_ref()
        .map_or(false, |time_keeper| time_keeper.check_time_limit())
    {
        return None;
    }
    let mut cost = None;
    let mut best_transition = None;
    for transition in generator.applicable_transitions(&state) {
        let successor = transition.apply(&state, &model.table_registry);
        if model.check_constraints(&successor) {
            let successor_cost =
                forward_recursion(successor, model, generator, memo, time_keeper, expanded);
            if let Some(successor_cost) = successor_cost {
                let current_cost =
                    transition.eval_cost(successor_cost, &state, &model.table_registry);
                if cost.is_none() {
                    cost = Some(current_cost);
                    match model.reduce_function {
                        didp_parser::ReduceFunction::Min | didp_parser::ReduceFunction::Max => {
                            best_transition = Some(transition);
                        }
                        _ => {}
                    }
                } else {
                    match model.reduce_function {
                        didp_parser::ReduceFunction::Min => {
                            if current_cost < cost.unwrap() {
                                cost = Some(current_cost);
                                best_transition = Some(transition);
                            }
                        }
                        didp_parser::ReduceFunction::Max => {
                            if current_cost > cost.unwrap() {
                                cost = Some(current_cost);
                                best_transition = Some(transition);
                            }
                        }
                        didp_parser::ReduceFunction::Sum => {
                            cost = Some(cost.unwrap() + current_cost);
                        }
                        didp_parser::ReduceFunction::Product => {
                            cost = Some(cost.unwrap() * current_cost);
                        }
                    }
                }
            }
        }
    }
    memo.insert(state, (cost, best_transition));
    cost
}
