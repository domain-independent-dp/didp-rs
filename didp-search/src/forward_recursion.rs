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
pub struct ForwardRecursion {
    memo_capacity: Option<usize>,
}

impl<T: variable::Numeric> solver::Solver<T> for ForwardRecursion
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    fn solve(
        &mut self,
        model: &didp_parser::Model<T>,
    ) -> Result<solver::Solution<T>, Box<dyn Error>> {
        let generator = successor_generator::SuccessorGenerator::<Transition<T>>::new(model, false);
        let mut memo = FxHashMap::default();
        if let Some(capacity) = self.memo_capacity {
            memo.reserve(capacity);
        }
        let mut expanded = 0;
        let state = hashable_state::HashableState::new(&model.target);
        let cost = forward_recursion(state, model, &generator, &mut memo, &mut expanded);
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
        Ok(cost.map(|cost| (cost, transitions)))
    }
}

impl ForwardRecursion {
    pub fn new(config: &yaml_rust::Yaml) -> Result<ForwardRecursion, solver::ConfigErr> {
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
        Ok(ForwardRecursion { memo_capacity })
    }
}

type StateMemo<T> =
    FxHashMap<hashable_state::HashableState, (Option<T>, Option<Rc<didp_parser::Transition<T>>>)>;

pub fn forward_recursion<T: variable::Numeric>(
    state: hashable_state::HashableState,
    model: &didp_parser::Model<T>,
    generator: &successor_generator::SuccessorGenerator<Transition<T>>,
    memo: &mut StateMemo<T>,
    expanded: &mut i32,
) -> Option<T> {
    *expanded += 1;
    if model.is_goal(&state) {
        return Some(T::zero());
    }
    if let Some((cost, _)) = memo.get(&state) {
        return *cost;
    }
    let mut cost = None;
    let mut best_transition = None;
    for transition in generator.applicable_transitions(&state) {
        let successor = transition.apply(&state, &model.table_registry);
        if model.check_constraints(&successor) {
            let successor_cost = forward_recursion(successor, model, generator, memo, expanded);
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
