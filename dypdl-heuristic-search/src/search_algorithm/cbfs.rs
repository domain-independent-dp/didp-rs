use super::data_structure::state_registry::{StateInRegistry, StateRegistry};
use super::data_structure::{exceed_bound, BfsNodeInterface, SuccessorGenerator};
use super::search::{Search, Solution};
use super::util;
use dypdl::variable_type;
use std::collections;
use std::error::Error;
use std::fmt;
use std::rc::Rc;

/// Cyclic Best-First Search (CBFS).
pub struct Cbfs<T, N, H, F>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    N: BfsNodeInterface<T>,
    H: Fn(&StateInRegistry, &dypdl::Model) -> Option<T>,
    F: Fn(T, T, &StateInRegistry, &dypdl::Model) -> T,
{
    generator: SuccessorGenerator,
    h_evaluator: H,
    f_evaluator: F,
    f_pruning: bool,
    ordered_by_f: bool,
    primal_bound: Option<T>,
    get_all_solutions: bool,
    quiet: bool,
    open: Vec<collections::BinaryHeap<Rc<N>>>,
    registry: StateRegistry<T, N>,
    time_keeper: util::TimeKeeper,
    solution: Solution<T>,
}

impl<T, N, H, F> Cbfs<T, N, H, F>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    N: BfsNodeInterface<T>,
    H: Fn(&StateInRegistry, &dypdl::Model) -> Option<T>,
    F: Fn(T, T, &StateInRegistry, &dypdl::Model) -> T,
{
    /// Create a new CBFS solver.
    pub fn new(
        model: Rc<dypdl::Model>,
        h_evaluator: H,
        f_evaluator: F,
        f_pruning: bool,
        ordered_by_f: bool,
        parameters: util::ForwardSearchParameters<T>,
    ) -> Cbfs<T, N, H, F> {
        let time_keeper = parameters
            .parameters
            .time_limit
            .map_or_else(util::TimeKeeper::default, util::TimeKeeper::with_time_limit);
        let primal_bound = parameters.parameters.primal_bound;
        let get_all_solutions = parameters.parameters.get_all_solutions;
        let quiet = parameters.parameters.quiet;

        let mut open = vec![collections::BinaryHeap::new()];
        let mut registry = StateRegistry::new(model);

        if let Some(capacity) = parameters.initial_registry_capacity {
            registry.reserve(capacity);
        }

        let mut solution = Solution::default();

        if let Some((node, h, f)) =
            N::generate_initial_node(&mut registry, &h_evaluator, &f_evaluator)
        {
            open[0].push(node);
            solution.generated += 1;

            if !quiet {
                println!("Initial h = {}", h);
            }

            if f_pruning {
                solution.best_bound = Some(f);
            }
        } else {
            solution.is_infeasible = true;
        }

        Cbfs {
            generator: parameters.generator,
            h_evaluator,
            f_evaluator,
            f_pruning,
            ordered_by_f,
            primal_bound,
            get_all_solutions,
            quiet,
            open,
            registry,
            time_keeper,
            solution,
        }
    }
}

impl<T, N, H, F> Search<T> for Cbfs<T, N, H, F>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    N: BfsNodeInterface<T>,
    H: Fn(&StateInRegistry, &dypdl::Model) -> Option<T>,
    F: Fn(T, T, &StateInRegistry, &dypdl::Model) -> T,
{
    fn search_next(&mut self) -> Result<(Solution<T>, bool), Box<dyn Error>> {
        let mut i = 0;
        let mut no_node = true;

        loop {
            if let Some(node) = self.open[i].pop() {
                if node.closed() {
                    continue;
                }

                node.close();

                let f = node.get_bound(self.registry.model());

                if self.f_pruning
                    && self.ordered_by_f
                    && exceed_bound(self.registry.model(), f, self.primal_bound)
                {
                    self.open[i].clear();
                } else {
                    if no_node {
                        no_node = false;
                    }

                    if self.registry.model().is_base(node.state()) {
                        if exceed_bound(self.registry.model(), node.cost(), self.primal_bound) {
                            if self.get_all_solutions {
                                let mut solution = self.solution.clone();
                                solution.cost = Some(node.cost());
                                solution.transitions = node.transitions();
                                solution.time = self.time_keeper.elapsed_time();

                                return Ok((solution, false));
                            }

                            continue;
                        } else {
                            if !self.quiet {
                                println!(
                                    "New primal bound: {}, expanded: {}",
                                    node.cost(),
                                    self.solution.expanded
                                );
                            }

                            let cost = node.cost();
                            self.solution.cost = Some(cost);
                            self.solution.time = self.time_keeper.elapsed_time();
                            self.solution.transitions = node.transitions();
                            self.primal_bound = Some(cost);

                            if let Some(bound) = self.solution.best_bound {
                                self.solution.is_optimal = cost == bound;
                            }

                            self.solution.time = self.time_keeper.elapsed_time();

                            return Ok((self.solution.clone(), self.solution.is_optimal));
                        }
                    }

                    if self.time_keeper.check_time_limit() {
                        if !self.quiet {
                            println!("Reached time limit.");
                            println!("Expanded: {}", self.solution.expanded);
                        }

                        self.solution.time_out = true;
                        self.solution.time = self.time_keeper.elapsed_time();

                        return Ok((self.solution.clone(), true));
                    }

                    self.solution.expanded += 1;

                    let primal_bound = if self.f_pruning {
                        self.primal_bound
                    } else {
                        None
                    };

                    for transition in self.generator.applicable_transitions(node.state()) {
                        let successor = node.generate_successor(
                            transition,
                            &mut self.registry,
                            &self.h_evaluator,
                            &self.f_evaluator,
                            primal_bound,
                        );

                        if let Some((successor, _, _, new_generated)) = successor {
                            while i + 1 >= self.open.len() {
                                self.open.push(collections::BinaryHeap::new());
                            }

                            self.open[i + 1].push(successor);

                            if new_generated {
                                self.solution.generated += 1;
                            }
                        }
                    }
                }
            }

            if no_node && i + 1 == self.open.len() {
                break;
            } else if i + 1 == self.open.len() {
                i = 0;
                no_node = true;
            } else {
                i += 1;
            }
        }

        self.solution.is_infeasible = self.solution.cost.is_none();
        self.solution.is_optimal = self.solution.cost.is_some();
        self.solution.time = self.time_keeper.elapsed_time();
        Ok((self.solution.clone(), true))
    }
}
