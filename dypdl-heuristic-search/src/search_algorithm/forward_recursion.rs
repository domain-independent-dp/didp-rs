use super::data_structure::{HashableState, SuccessorGenerator};
use super::search::{Parameters, Search, Solution};
use super::util;
use dypdl::{variable_type, StateFunctionCache, Transition, TransitionInterface};
use rustc_hash::FxHashMap;
use std::error::Error;
use std::fmt;
use std::rc::Rc;
use std::str;

/// Forward recursion solver.
///
/// It performs a naive recursion while memoizing all encountered states.
/// It works only if the state space is acyclic.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::{Search, Parameters};
/// use dypdl_heuristic_search::search_algorithm::{ForwardRecursion};
/// use std::rc::Rc;
///
/// let mut model = Model::default();
/// let variable = model.add_integer_variable("variable", 0).unwrap();
/// model.add_base_case(
///     vec![Condition::comparison_i(ComparisonOperator::Ge, variable, 1)]
/// ).unwrap();
/// let mut increment = Transition::new("increment");
/// increment.set_cost(IntegerExpression::Cost + 1);
/// increment.add_effect(variable, variable + 1).unwrap();
/// model.add_forward_transition(increment.clone()).unwrap();
///
/// let model = Rc::new(model);
/// let parameters = Parameters::default();
/// let mut solver = ForwardRecursion::new(model, parameters);
/// let solution = solver.search().unwrap();
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions, vec![increment]);
/// assert!(!solution.is_infeasible);
/// ```
pub struct ForwardRecursion<T>
where
    T: variable_type::Numeric + fmt::Display + Ord + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    model: Rc<dypdl::Model>,
    parameters: Parameters<T>,
    terminated: bool,
    solution: Solution<T>,
}

impl<T> ForwardRecursion<T>
where
    T: variable_type::Numeric + fmt::Display + Ord + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    /// Create a new forward recursion solver.
    pub fn new(model: Rc<dypdl::Model>, parameters: Parameters<T>) -> ForwardRecursion<T> {
        ForwardRecursion {
            model,
            parameters,
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
        let generator = SuccessorGenerator::<Transition>::from_model(self.model.clone(), false);
        let mut memo = FxHashMap::default();

        if let Some(capacity) = self.parameters.initial_registry_capacity {
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
                let mut function_cache = StateFunctionCache::new(&self.model.state_functions);

                while let Some((_, Some(transition))) = memo.get(&state) {
                    function_cache.clear();
                    let transition = transition.as_ref().clone();
                    state = transition.apply(
                        &state,
                        &mut function_cache,
                        &self.model.state_functions,
                        &self.model.table_registry,
                    );
                    self.solution.transitions.push(transition);
                }
            }
            _ => {}
        }

        if self.model.reduce_function == dypdl::ReduceFunction::Max
            || self.model.reduce_function == dypdl::ReduceFunction::Min
        {
            self.solution.is_optimal = self.solution.cost.is_some();
            self.solution.best_bound = self.solution.cost;
        }

        self.solution.generated = self.solution.expanded;
        self.solution.is_infeasible = self.solution.cost.is_none();
        self.solution.time = time_keeper.elapsed_time();

        Ok((self.solution.clone(), true))
    }
}

type StateMemo<T> = FxHashMap<HashableState, (Option<T>, Option<Rc<dypdl::Transition>>)>;

/// Performs a naive recursion while memoizing all encountered states.
pub fn forward_recursion<T: variable_type::Numeric + Ord>(
    state: HashableState,
    model: &dypdl::Model,
    generator: &SuccessorGenerator,
    memo: &mut StateMemo<T>,
    time_keeper: &util::TimeKeeper,
    expanded: &mut usize,
) -> Option<T> {
    let mut function_cache = StateFunctionCache::new(&model.state_functions);

    if let Some(cost) = model.eval_base_cost(&state, &mut function_cache) {
        return Some(cost);
    }

    if let Some((cost, _)) = memo.get(&state) {
        return *cost;
    }

    if time_keeper.check_time_limit(true) {
        return None;
    }

    let mut cost = None;
    let mut best_transition = None;
    *expanded += 1;

    let mut applicable_transitions = Vec::new();

    generator.generate_applicable_transitions(
        &state,
        &mut function_cache,
        &mut applicable_transitions,
    );

    for transition in applicable_transitions {
        let successor = transition.apply(
            &state,
            &mut function_cache,
            &model.state_functions,
            &model.table_registry,
        );

        if model.check_constraints(&successor, &mut function_cache) {
            let successor_cost =
                forward_recursion(successor, model, generator, memo, time_keeper, expanded);

            if let Some(successor_cost) = successor_cost {
                let current_cost = transition.eval_cost(
                    successor_cost,
                    &state,
                    &mut function_cache,
                    &model.state_functions,
                    &model.table_registry,
                );
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
