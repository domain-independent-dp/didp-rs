use super::data_structure::{HashableState, SuccessorGenerator};
use super::search::{Search, Solution};
use super::util;
use dypdl::{variable_type, TransitionInterface};
use rustc_hash::FxHashMap;
use std::error::Error;
use std::fmt;
use std::rc::Rc;
use std::str;

/// Forward recursion solver.
///
/// It performs a naive recursion while memoizing all encountered states.
/// It works only if the state space is acyclic.
pub struct ForwardRecursion<T>
where
    T: variable_type::Numeric + fmt::Display + Ord + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    model: Rc<dypdl::Model>,
    parameters: util::Parameters<T>,
    initial_registry_capacity: Option<usize>,
    terminated: bool,
    solution: Solution<T>,
}

impl<T> ForwardRecursion<T>
where
    T: variable_type::Numeric + fmt::Display + Ord + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    /// Create a new forward recursion solver.
    pub fn new(
        model: Rc<dypdl::Model>,
        parameters: util::Parameters<T>,
        initial_registry_capacity: Option<usize>,
    ) -> ForwardRecursion<T> {
        ForwardRecursion {
            model,
            parameters,
            initial_registry_capacity,
            terminated: false,
            solution: Solution::default(),
        }
    }
}

impl<T> Search<T> for ForwardRecursion<T>
where
    T: variable_type::Numeric + fmt::Display + Ord + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    fn search_next(&mut self) -> Result<(Solution<T>, bool), Box<dyn Error>> {
        if self.terminated {
            return Ok((self.solution.clone(), true));
        }

        let time_keeper = self
            .parameters
            .time_limit
            .map_or_else(util::TimeKeeper::default, util::TimeKeeper::with_time_limit);
        let generator = SuccessorGenerator::from_model(self.model.clone(), false);
        let mut memo = FxHashMap::default();

        if let Some(capacity) = self.initial_registry_capacity {
            memo.reserve(capacity);
        }

        let state = HashableState::from(self.model.target.clone());
        self.solution.cost = forward_recursion(
            state,
            self.model.as_ref(),
            &generator,
            &mut memo,
            &time_keeper,
            &mut self.solution.expanded,
        );

        match self.model.reduce_function {
            dypdl::ReduceFunction::Max | dypdl::ReduceFunction::Min
                if self.solution.cost.is_some() =>
            {
                let mut state = HashableState::from(self.model.target.clone());
                while let Some((_, Some(transition))) = memo.get(&state) {
                    let transition = transition.as_ref().clone();
                    state = transition.apply(&state, &self.model.table_registry);
                    self.solution.transitions.push(transition);
                }
            }
            _ => {}
        }

        self.solution.is_optimal = self.solution.cost.is_some()
            && (self.model.reduce_function == dypdl::ReduceFunction::Max
                || self.model.reduce_function == dypdl::ReduceFunction::Min);
        self.solution.generated = self.solution.expanded;
        self.solution.is_infeasible = self.solution.cost.is_none();
        self.solution.time = time_keeper.elapsed_time();

        Ok((self.solution.clone(), true))
    }
}

type StateMemo<T> = FxHashMap<HashableState, (Option<T>, Option<Rc<dypdl::Transition>>)>;

/// Performs a naive recursion while memoizing all encountered states.
pub fn forward_recursion<T: variable_type::Numeric>(
    state: HashableState,
    model: &dypdl::Model,
    generator: &SuccessorGenerator,
    memo: &mut StateMemo<T>,
    time_keeper: &util::TimeKeeper,
    expanded: &mut usize,
) -> Option<T> {
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
    *expanded += 1;

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
