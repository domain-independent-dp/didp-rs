use crate::search_node::{trace_transitions, SearchNode};
use crate::solver;
use crate::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use crate::successor_generator::SuccessorGenerator;
use didp_parser::variable;
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections;
use std::error::Error;
use std::fmt;
use std::rc::Rc;
use std::str;

#[derive(Default)]
pub struct Dijkstra<T> {
    parameters: solver::SolverParameters<T>,
    registry_capacity: Option<usize>,
}

impl<T> solver::Solver<T> for Dijkstra<T>
where
    T: variable::Numeric + Ord + fmt::Display,
{
    fn solve(
        &mut self,
        model: &didp_parser::Model<T>,
    ) -> Result<solver::Solution<T>, Box<dyn Error>> {
        let generator = SuccessorGenerator::<didp_parser::Transition<T>>::new(model, false);
        Ok(dijkstra(
            model,
            generator,
            self.parameters,
            self.registry_capacity,
        ))
    }

    #[inline]
    fn set_primal_bound(&mut self, primal_bound: T) {
        self.parameters.primal_bound = Some(primal_bound)
    }

    #[inline]
    fn set_time_limit(&mut self, time_limit: u64) {
        self.parameters.time_limit = Some(time_limit)
    }
}

impl<T: variable::Numeric> Dijkstra<T>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    pub fn new(config: &yaml_rust::Yaml) -> Result<Dijkstra<T>, Box<dyn Error>> {
        let map = match config {
            yaml_rust::Yaml::Hash(map) => map,
            yaml_rust::Yaml::Null => return Ok(Dijkstra::default()),
            _ => {
                return Err(solver::ConfigErr::new(format!(
                    "expected Hash, but found `{:?}`",
                    config
                ))
                .into())
            }
        };
        let parameters = solver::SolverParameters::parse_from_map(map)?;
        let registry_capacity = match map.get(&yaml_rust::Yaml::from_str("registry_capacity")) {
            Some(yaml_rust::Yaml::Integer(value)) => Some(*value as usize),
            None => Some(1000000),
            value => {
                return Err(solver::ConfigErr::new(format!(
                    "expected Integer, but found `{:?}`",
                    value
                ))
                .into())
            }
        };
        Ok(Dijkstra {
            parameters,
            registry_capacity,
        })
    }
}

pub fn dijkstra<T>(
    model: &didp_parser::Model<T>,
    generator: SuccessorGenerator<didp_parser::Transition<T>>,
    parameters: solver::SolverParameters<T>,
    registry_capacity: Option<usize>,
) -> solver::Solution<T>
where
    T: variable::Numeric + Ord + fmt::Display,
{
    let time_keeper = parameters.time_limit.map(solver::TimeKeeper::new);
    let primal_bound = parameters.primal_bound;
    let mut open = collections::BinaryHeap::new();
    let mut registry = StateRegistry::new(model);
    if let Some(capacity) = registry_capacity {
        registry.reserve(capacity);
    }

    let cost = T::zero();
    let initial_state = StateInRegistry::new(&model.target);
    let constructor = |state: StateInRegistry, cost: T, _: Option<&Rc<SearchNode<T>>>| {
        Some(Rc::new(SearchNode {
            state,
            cost,
            ..Default::default()
        }))
    };
    let (initial_node, _) = registry.insert(initial_state, cost, constructor).unwrap();
    open.push(Reverse(initial_node));
    let mut expanded = 0;
    let mut cost_max = T::zero();

    while let Some(Reverse(node)) = open.pop() {
        if *node.closed.borrow() {
            continue;
        }
        *node.closed.borrow_mut() = true;
        expanded += 1;
        if node.cost() > cost_max {
            cost_max = node.cost();
            println!("cost = {}, expanded: {}", cost_max, expanded);
        }
        if model.is_goal(node.state()) {
            println!("Expanded: {}", expanded);
            return solver::Solution {
                cost: Some(node.cost()),
                is_optimal: true,
                transitions: trace_transitions(node),
                ..Default::default()
            };
        }
        if time_keeper
            .as_ref()
            .map_or(false, |time_keeper| time_keeper.check_time_limit())
        {
            println!("Expanded: {}", expanded);
            return solver::Solution {
                best_bound: Some(cost_max),
                ..Default::default()
            };
        }
        for transition in generator.applicable_transitions(node.state()) {
            let cost = transition.eval_cost(node.cost(), node.state(), &model.table_registry);
            if primal_bound.is_some() && cost >= primal_bound.unwrap() {
                continue;
            }
            let state = transition.apply(node.state(), &model.table_registry);
            if model.check_constraints(&state) {
                let constructor =
                    |state: StateInRegistry, cost: T, _: Option<&Rc<SearchNode<T>>>| {
                        Some(Rc::new(SearchNode {
                            state,
                            cost,
                            parent: Some(node.clone()),
                            operator: Some(transition.clone()),
                            closed: RefCell::new(false),
                        }))
                    };
                if let Some((successor, dominated)) = registry.insert(state, cost, constructor) {
                    if let Some(dominated) = dominated {
                        if !*dominated.closed.borrow() {
                            *dominated.closed.borrow_mut() = true;
                        }
                    }
                    open.push(Reverse(successor));
                }
            }
        }
    }
    println!("Expanded: {}", expanded);
    solver::Solution {
        is_infeasible: true,
        ..Default::default()
    }
}
