use super::data_structure::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use super::data_structure::{
    LazySearchNode, SearchNode, SuccessorGenerator, TransitionChain, TransitionChainInterface,
};
use super::search::{Search, Solution};
use super::util;
use dypdl::{variable_type, TransitionInterface};
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::error::Error;
use std::fmt;
use std::rc::Rc;
use std::str;

/// Dijkstra's algorithm Solver.
///
/// This solver uses forward search based on the shortest path problem.
/// It only works with problems where the cost expressions are in the form of `cost + w`, `cost * w`, or `min(cost, w)
/// where `cost` is `IntegerExpression::Cost`or `ContinuousExpression::Cost` and `w` is a numeric expression independent of `cost`.
/// In addition `w` must be non-negative, and the model must be minimization.
///
/// `lazy` indicates whether the solver uses lazy evaluation.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::search_algorithm::{Dijkstra, Search};
/// use dypdl_heuristic_search::search_algorithm::util::{
///     ForwardSearchParameters, Parameters,
/// };
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
/// let mut solver = Dijkstra::new(model, parameters, false, None);
/// let solution = solver.search().unwrap();
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions, vec![increment]);
/// assert!(!solution.is_infeasible);
/// ```
pub struct Dijkstra<T>
where
    T: variable_type::Numeric + fmt::Display + Ord + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    model: Rc<dypdl::Model>,
    parameters: util::Parameters<T>,
    lazy: bool,
    initial_registry_capacity: Option<usize>,
    terminated: bool,
    solution: Solution<T>,
}

impl<T> Dijkstra<T>
where
    T: variable_type::Numeric + fmt::Display + Ord + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    /// Create a new Dijkstra's algorithm solver.
    pub fn new(
        model: Rc<dypdl::Model>,
        parameters: util::Parameters<T>,
        lazy: bool,
        initial_registry_capacity: Option<usize>,
    ) -> Dijkstra<T> {
        Dijkstra {
            model,
            parameters,
            lazy,
            initial_registry_capacity,
            terminated: false,
            solution: Solution::default(),
        }
    }
}

impl<T> Search<T> for Dijkstra<T>
where
    T: variable_type::Numeric + fmt::Display + Ord + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    fn search_next(&mut self) -> Result<(Solution<T>, bool), Box<dyn Error>> {
        if self.terminated {
            return Ok((self.solution.clone(), true));
        }

        let generator = SuccessorGenerator::from_model(self.model.clone(), false);

        self.solution = if self.lazy {
            lazy_dijkstra(
                &self.model,
                generator,
                self.parameters,
                self.initial_registry_capacity,
            )
        } else {
            dijkstra(
                &self.model,
                generator,
                self.parameters,
                self.initial_registry_capacity,
            )
        };
        self.terminated = true;
        Ok((self.solution.clone(), true))
    }
}

/// Performs Dijkstra's algorithm.
///
/// `registry_capacity` is the initial capacity of the state registry.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::search_algorithm::{dijkstra, Search};
/// use dypdl_heuristic_search::search_algorithm::data_structure::successor_generator::{
///     SuccessorGenerator
/// };
/// use dypdl_heuristic_search::search_algorithm::util::{
///     ForwardSearchParameters, Parameters,
/// };
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
/// let generator = SuccessorGenerator::from_model(model.clone(), false);
/// let parameters = Parameters::default();
/// let solution = dijkstra(&model, generator, parameters, None);
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions, vec![increment]);
/// assert!(!solution.is_infeasible);
/// ```
pub fn dijkstra<T>(
    model: &Rc<dypdl::Model>,
    generator: SuccessorGenerator,
    parameters: util::Parameters<T>,
    registry_capacity: Option<usize>,
) -> Solution<T>
where
    T: variable_type::Numeric + Ord + fmt::Display,
{
    let time_keeper = parameters
        .time_limit
        .map_or_else(util::TimeKeeper::default, util::TimeKeeper::with_time_limit);
    let primal_bound = parameters.primal_bound;
    let mut open = BinaryHeap::<Reverse<Rc<_>>>::default();
    let mut registry = StateRegistry::new(model.clone());

    if let Some(capacity) = registry_capacity {
        registry.reserve(capacity);
    }

    let cost = T::zero();
    let initial_state = StateInRegistry::from(model.target.clone());
    let constructor = |state: StateInRegistry, cost: T, _: Option<&SearchNode<T>>| {
        Some(SearchNode {
            state,
            cost,
            ..Default::default()
        })
    };
    let (initial_node, _) = registry.insert(initial_state, cost, constructor).unwrap();
    open.push(Reverse(initial_node));
    let mut expanded = 0;
    let mut generated = 1;
    let mut cost_max = T::zero();

    while let Some(Reverse(node)) = open.pop() {
        if *node.closed.borrow() {
            continue;
        }

        *node.closed.borrow_mut() = true;

        if node.cost() > cost_max {
            cost_max = node.cost();

            if !parameters.quiet {
                println!("cost = {}, expanded: {}", cost_max, expanded);
            }
        }

        if model.is_base(node.state()) {
            return Solution {
                cost: Some(node.cost()),
                is_optimal: true,
                transitions: node
                    .transitions
                    .as_ref()
                    .map_or_else(Vec::new, |transition| transition.transitions()),
                expanded,
                generated,
                time: time_keeper.elapsed_time(),
                ..Default::default()
            };
        }

        if time_keeper.check_time_limit() {
            return Solution {
                best_bound: Some(cost_max),
                expanded,
                generated,
                time: time_keeper.elapsed_time(),
                time_out: true,
                ..Default::default()
            };
        }

        expanded += 1;

        for transition in generator.applicable_transitions(node.state()) {
            if let Some((successor, new_generated)) =
                node.generate_successor(transition, &mut registry, primal_bound)
            {
                open.push(Reverse(successor));

                if new_generated {
                    generated += 1;
                }
            }
        }
    }

    Solution {
        is_infeasible: true,
        expanded,
        generated,
        time: time_keeper.elapsed_time(),
        ..Default::default()
    }
}

/// Search node stored in the open list of lazy Dijkstra.
#[derive(Debug, Clone)]
pub struct DijkstraEdge<T: variable_type::Numeric + Ord> {
    pub cost: T,
    pub parent: Rc<LazySearchNode<T>>,
    pub transition: Rc<dypdl::Transition>,
}

impl<T: variable_type::Numeric + Ord> PartialEq for DijkstraEdge<T> {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl<T: variable_type::Numeric + Ord> Eq for DijkstraEdge<T> {}

impl<T: variable_type::Numeric + Ord> Ord for DijkstraEdge<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost.cmp(&other.cost)
    }
}

impl<T: variable_type::Numeric + Ord> PartialOrd for DijkstraEdge<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Performs lazy Dijkstra's algorithm.
///
/// Pointers to parent nodes and transitions are stored in the open list, and a state is generated when it is expanded.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::search_algorithm::{lazy_dijkstra, Search};
/// use dypdl_heuristic_search::search_algorithm::data_structure::successor_generator::{
///     SuccessorGenerator
/// };
/// use dypdl_heuristic_search::search_algorithm::util::{
///     ForwardSearchParameters, Parameters,
/// };
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
/// let generator = SuccessorGenerator::from_model(model.clone(), false);
/// let parameters = Parameters::default();
/// let solution = lazy_dijkstra(&model, generator, parameters, None);
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions, vec![increment]);
/// assert!(!solution.is_infeasible);
/// ```
pub fn lazy_dijkstra<T>(
    model: &Rc<dypdl::Model>,
    generator: SuccessorGenerator,
    parameters: util::Parameters<T>,
    registry_capacity: Option<usize>,
) -> Solution<T>
where
    T: variable_type::Numeric + Ord + fmt::Display,
{
    let time_keeper = parameters
        .time_limit
        .map_or_else(util::TimeKeeper::default, util::TimeKeeper::with_time_limit);
    let primal_bound = parameters.primal_bound;
    let mut open = BinaryHeap::default();
    let mut registry = StateRegistry::<T, LazySearchNode<_>>::new(model.clone());

    if let Some(capacity) = registry_capacity {
        registry.reserve(capacity);
    }

    let cost = T::zero();
    let initial_state = StateInRegistry::from(model.target.clone());

    let constructor = |state: StateInRegistry, cost: T, _: Option<&LazySearchNode<T>>| {
        Some(LazySearchNode {
            state,
            cost,
            ..Default::default()
        })
    };
    let (initial_node, _) = registry.insert(initial_state, cost, constructor).unwrap();

    for transition in generator.applicable_transitions(&initial_node.state) {
        let cost = transition.eval_cost(
            initial_node.cost,
            &initial_node.state,
            &model.table_registry,
        );
        open.push(Reverse(DijkstraEdge {
            cost,
            parent: initial_node.clone(),
            transition,
        }));
    }

    let mut expanded = 0;
    let mut cost_max = T::zero();

    while let Some(Reverse(edge)) = open.pop() {
        let state = edge
            .transition
            .apply(&edge.parent.state, &model.table_registry);
        if !model.check_constraints(&state) {
            continue;
        }
        let constructor = |state: StateInRegistry, cost: T, _: Option<&LazySearchNode<T>>| {
            Some(LazySearchNode {
                state,
                cost,
                transitions: Some(Rc::new(TransitionChain::new(
                    edge.parent.transitions.clone(),
                    edge.transition,
                ))),
            })
        };
        if let Some((node, _)) = registry.insert(state, edge.cost, constructor) {
            expanded += 1;

            if node.cost > cost_max {
                cost_max = node.cost;
                if !parameters.quiet {
                    println!("cost = {}, expanded: {}", cost_max, expanded);
                }
            }

            if model.is_base(&node.state) {
                return Solution {
                    cost: Some(node.cost),
                    is_optimal: true,
                    transitions: node
                        .transitions
                        .as_ref()
                        .map_or_else(Vec::new, |transition| transition.transitions()),
                    expanded,
                    generated: expanded,
                    time: time_keeper.elapsed_time(),
                    ..Default::default()
                };
            }

            if time_keeper.check_time_limit() {
                return Solution {
                    best_bound: Some(cost_max),
                    expanded,
                    generated: expanded,
                    time: time_keeper.elapsed_time(),
                    time_out: true,
                    ..Default::default()
                };
            }

            for transition in generator.applicable_transitions(&node.state) {
                let cost = transition.eval_cost(node.cost, &node.state, &model.table_registry);

                if primal_bound.is_some() && cost >= primal_bound.unwrap() {
                    continue;
                }

                open.push(Reverse(DijkstraEdge {
                    cost,
                    parent: node.clone(),
                    transition,
                }));
            }
        }
    }

    Solution {
        is_infeasible: true,
        expanded,
        generated: expanded,
        time: time_keeper.elapsed_time(),
        ..Default::default()
    }
}
