use crate::solver;
use crate::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use crate::successor_generator::SuccessorGenerator;
use didp_parser::variable;
use didp_parser::Transition;
use std::fmt;
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct RecursiveNode<T: variable::Numeric> {
    pub state: StateInRegistry,
    pub cost: T,
}

impl<T: variable::Numeric> StateInformation<T> for RecursiveNode<T> {
    fn cost(&self) -> T {
        self.cost
    }

    fn state(&self) -> &StateInRegistry {
        &self.state
    }
}

pub fn forward_iterative_exist_dfs<T>(
    model: &didp_parser::Model<T>,
    generator: &SuccessorGenerator<Transition<T>>,
    parameters: solver::SolverParameters<T>,
    maximize: bool,
    capacity: Option<usize>,
) -> solver::Solution<T>
where
    T: variable::Numeric + fmt::Display,
{
    let time_keeper = parameters.time_limit.map(solver::TimeKeeper::new);
    let mut primal_bound = parameters.primal_bound;
    let mut expanded = 0;
    let mut prob = StateRegistry::new(model);
    if let Some(capacity) = capacity {
        prob.reserve(capacity);
    };
    let mut incumbent = None;
    let node = RecursiveNode {
        state: StateInRegistry::new(&model.target),
        cost: T::zero(),
    };
    loop {
        let parameters = ExistDFSParameters {
            primal_bound,
            maximize,
        };
        match exist_dfs(
            node.clone(),
            model,
            generator,
            &mut prob,
            parameters,
            &time_keeper,
            &mut expanded,
        ) {
            (Some((new_cost, transitions)), _) => {
                if let Some(current_cost) = primal_bound {
                    match model.reduce_function {
                        didp_parser::ReduceFunction::Max if new_cost > current_cost => {
                            primal_bound = Some(new_cost);
                            incumbent = Some(transitions);
                            println!("New primal bound: {}, expanded: {}", new_cost, expanded);
                        }
                        didp_parser::ReduceFunction::Min if new_cost < current_cost => {
                            primal_bound = Some(new_cost);
                            incumbent = Some(transitions);
                            println!("New primal bound: {}, expanded: {}", new_cost, expanded);
                        }
                        _ => {}
                    }
                } else {
                    primal_bound = Some(new_cost);
                    incumbent = Some(transitions);
                    println!("New primal bound: {}, expanded: {}", new_cost, expanded);
                }
            }
            (_, time_out) => {
                println!("Expanded: {}", expanded);
                return incumbent.map_or_else(
                    || solver::Solution {
                        is_infeasible: true,
                        ..Default::default()
                    },
                    |mut incumbent| solver::Solution {
                        cost: primal_bound,
                        is_optimal: !time_out,
                        transitions: {
                            incumbent.reverse();
                            incumbent
                        },
                        ..Default::default()
                    },
                );
            }
        }
    }
}

pub struct ExistDFSParameters<T> {
    pub primal_bound: Option<T>,
    pub maximize: bool,
}

type ExistDFSSolution<T> = (Option<(T, Vec<Rc<Transition<T>>>)>, bool);

pub fn exist_dfs<T: variable::Numeric>(
    node: RecursiveNode<T>,
    model: &didp_parser::Model<T>,
    generator: &SuccessorGenerator<Transition<T>>,
    prob: &mut StateRegistry<T, RecursiveNode<T>>,
    parameters: ExistDFSParameters<T>,
    time_keeper: &Option<solver::TimeKeeper>,
    expanded: &mut u32,
) -> ExistDFSSolution<T> {
    let state = node.state;
    let cost = node.cost;
    let primal_bound = parameters.primal_bound;
    let maximize = parameters.maximize;
    *expanded += 1;
    if model.is_goal(&state) {
        if maximize && primal_bound.is_some() && cost <= primal_bound.unwrap() {
            return (None, false);
        }
        return (Some((cost, Vec::new())), false);
    }
    if prob.get(&state, cost).is_some() {
        return (None, false);
    }
    if time_keeper
        .as_ref()
        .map_or(false, |time_keeper| time_keeper.check_time_limit())
    {
        return (None, true);
    }
    for transition in generator.applicable_transitions(&state) {
        let cost = transition.eval_cost(cost, &state, &model.table_registry);
        if !maximize && primal_bound.is_some() && cost >= primal_bound.unwrap() {
            continue;
        }
        let successor = transition.apply(&state, &model.table_registry);
        if model.check_constraints(&successor) {
            let node = RecursiveNode {
                state: successor,
                cost,
            };
            let result = exist_dfs(
                node,
                model,
                generator,
                prob,
                ExistDFSParameters {
                    maximize,
                    primal_bound,
                },
                time_keeper,
                expanded,
            );
            match result {
                (Some((cost, mut transitions)), _) => {
                    transitions.push(transition);
                    return (Some((cost, transitions)), false);
                }
                (_, true) => return (None, true),
                _ => {}
            }
        }
    }
    let constructor = |state: StateInRegistry, cost: T, _: Option<&RecursiveNode<T>>| {
        Some(RecursiveNode { state, cost })
    };
    prob.insert(state, cost, constructor);
    (None, false)
}
