use crate::hashable_state;
use crate::solver;
use crate::successor_generator;
use dypdl::variable_type;
use dypdl::Transition;
use rustc_hash::FxHashMap;
use std::error::Error;
use std::fmt;
use std::rc::Rc;
use std::str;

/// Forward recursion solver.
///
/// It performs a naive recursion while memoizing all encoutered states.
/// It works only if the state space is acyclic.
pub struct ForwardRecursion<T> {
    /// The initial capacity of the data structure storing all generated states.
    pub initial_registry_capacity: Option<usize>,
    /// Common parameters for heuristic search solvers.
    pub parameters: solver::SolverParameters<T>,
}

impl<T: variable_type::Numeric> solver::Solver<T> for ForwardRecursion<T>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    fn solve(&mut self, model: &dypdl::Model) -> Result<solver::Solution<T>, Box<dyn Error>> {
        let time_keeper = self.parameters.time_limit.map_or_else(
            solver::TimeKeeper::default,
            solver::TimeKeeper::with_time_limit,
        );
        let generator = successor_generator::SuccessorGenerator::<Transition>::new(model, false);
        let mut memo = FxHashMap::default();
        if let Some(capacity) = self.initial_registry_capacity {
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
            dypdl::ReduceFunction::Max | dypdl::ReduceFunction::Min if cost.is_some() => {
                let mut state = hashable_state::HashableState::new(&model.target);
                while let Some((_, Some(transition))) = memo.get(&state) {
                    let transition = transition.as_ref().clone();
                    state = transition.apply(&state, &model.table_registry);
                    transitions.push(transition);
                }
            }
            _ => {}
        }
        Ok(solver::Solution {
            cost,
            transitions,
            is_optimal: cost.is_some()
                && (model.reduce_function == dypdl::ReduceFunction::Max
                    || model.reduce_function == dypdl::ReduceFunction::Min),
            expanded,
            generated: expanded,
            is_infeasible: cost.is_none(),
            time: time_keeper.elapsed_time(),
            ..Default::default()
        })
    }

    #[inline]
    fn set_primal_bound(&mut self, primal_bound: T) {
        self.parameters.primal_bound = Some(primal_bound)
    }

    #[inline]
    fn get_primal_bound(&self) -> Option<T> {
        self.parameters.primal_bound
    }

    #[inline]
    fn set_time_limit(&mut self, time_limit: f64) {
        self.parameters.time_limit = Some(time_limit)
    }

    #[inline]
    fn get_time_limit(&self) -> Option<f64> {
        self.parameters.time_limit
    }

    #[inline]
    fn set_quiet(&mut self, quiet: bool) {
        self.parameters.quiet = quiet
    }

    #[inline]
    fn get_quiet(&self) -> bool {
        self.parameters.quiet
    }
}

type StateMemo<T> =
    FxHashMap<hashable_state::HashableState, (Option<T>, Option<Rc<dypdl::Transition>>)>;

/// Performs a naive recursion while memoizing all encountered states.
pub fn forward_recursion<T: variable_type::Numeric>(
    state: hashable_state::HashableState,
    model: &dypdl::Model,
    generator: &successor_generator::SuccessorGenerator<Transition>,
    memo: &mut StateMemo<T>,
    time_keeper: &solver::TimeKeeper,
    expanded: &mut usize,
) -> Option<T> {
    *expanded += 1;
    if model.is_base(&state) {
        return Some(T::zero());
    }
    if let Some((cost, _)) = memo.get(&state) {
        return *cost;
    }
    if time_keeper.check_time_limit() {
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
                        dypdl::ReduceFunction::Min | dypdl::ReduceFunction::Max => {
                            best_transition = Some(transition);
                        }
                        _ => {}
                    }
                } else {
                    match model.reduce_function {
                        dypdl::ReduceFunction::Min => {
                            if current_cost < cost.unwrap() {
                                cost = Some(current_cost);
                                best_transition = Some(transition);
                            }
                        }
                        dypdl::ReduceFunction::Max => {
                            if current_cost > cost.unwrap() {
                                cost = Some(current_cost);
                                best_transition = Some(transition);
                            }
                        }
                        dypdl::ReduceFunction::Sum => {
                            cost = Some(cost.unwrap() + current_cost);
                        }
                        dypdl::ReduceFunction::Product => {
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
