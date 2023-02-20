use super::data_structure::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use super::data_structure::{
    exceed_bound, SearchNode, SuccessorGenerator, TransitionChainInterface,
};
use super::search::{Search, Solution};
use super::util;
use dypdl::variable_type;
use std::error::Error;
use std::fmt;
use std::rc::Rc;

/// Depth-First Branch-and-Bound (DFBB).
pub struct Dfbb<T, H, F>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    H: Fn(&StateInRegistry, &dypdl::Model) -> Option<T>,
    F: Fn(T, T, &StateInRegistry, &dypdl::Model) -> T,
{
    generator: SuccessorGenerator,
    h_evaluator: H,
    f_evaluator: F,
    f_pruning: bool,
    primal_bound: Option<T>,
    get_all_solutions: bool,
    quiet: bool,
    open: Vec<Rc<SearchNode<T>>>,
    registry: StateRegistry<T, SearchNode<T>>,
    time_keeper: util::TimeKeeper,
    solution: Solution<T>,
}

impl<T, H, F> Dfbb<T, H, F>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    H: Fn(&StateInRegistry, &dypdl::Model) -> Option<T>,
    F: Fn(T, T, &StateInRegistry, &dypdl::Model) -> T,
{
    /// Create a new DFBB solver.
    pub fn new(
        model: Rc<dypdl::Model>,
        h_evaluator: H,
        f_evaluator: F,
        f_pruning: bool,
        parameters: util::ForwardSearchParameters<T>,
    ) -> Dfbb<T, H, F> {
        let time_keeper = parameters
            .parameters
            .time_limit
            .map_or_else(util::TimeKeeper::default, util::TimeKeeper::with_time_limit);
        let primal_bound = parameters.parameters.primal_bound;
        let get_all_solutions = parameters.parameters.get_all_solutions;
        let quiet = parameters.parameters.quiet;

        let mut open = Vec::new();
        let mut registry = StateRegistry::new(model);

        if let Some(capacity) = parameters.initial_registry_capacity {
            registry.reserve(capacity);
        }

        let mut solution = Solution::default();
        let node = SearchNode::generate_initial_node(&mut registry).unwrap();
        open.push(node);
        solution.generated += 1;

        Dfbb {
            generator: parameters.generator,
            h_evaluator,
            f_evaluator,
            f_pruning,
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

impl<T, H, F> Search<T> for Dfbb<T, H, F>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    H: Fn(&StateInRegistry, &dypdl::Model) -> Option<T>,
    F: Fn(T, T, &StateInRegistry, &dypdl::Model) -> T,
{
    fn search_next(&mut self) -> Result<(Solution<T>, bool), Box<dyn Error>> {
        while let Some(node) = self.open.pop() {
            if *node.closed.borrow() {
                continue;
            }

            *node.closed.borrow_mut() = true;

            if self.registry.model().is_base(node.state()) {
                if exceed_bound(self.registry.model(), node.cost(), self.primal_bound) {
                    if self.get_all_solutions {
                        let mut solution = self.solution.clone();
                        solution.cost = Some(node.cost());
                        solution.transitions = node
                            .transitions
                            .as_ref()
                            .map_or_else(Vec::new, |transitions| transitions.transitions());
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
                    self.solution.transitions = node
                        .transitions
                        .as_ref()
                        .map_or_else(Vec::new, |transition| transition.transitions());
                    self.primal_bound = Some(cost);

                    return Ok((self.solution.clone(), false));
                }
            }

            if let (true, Some(bound)) = (self.f_pruning, self.primal_bound) {
                if let Some(h) = (self.h_evaluator)(node.state(), self.registry.model()) {
                    let f = (self.f_evaluator)(node.cost(), h, node.state(), self.registry.model());

                    if self.f_pruning && exceed_bound(self.registry.model(), f, Some(bound)) {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            if self.time_keeper.check_time_limit() {
                if !self.quiet {
                    println!("Expanded: {}", self.solution.expanded);
                }

                self.solution.time_out = true;
                self.solution.time = self.time_keeper.elapsed_time();

                return Ok((self.solution.clone(), true));
            }

            self.solution.expanded += 1;

            for transition in self.generator.applicable_transitions(node.state()) {
                if let Some((successor, new_generated)) =
                    node.generate_successor(transition, &mut self.registry, None)
                {
                    self.open.push(successor);

                    if new_generated {
                        self.solution.generated += 1;
                    }
                }
            }
        }

        self.solution.is_infeasible = self.solution.cost.is_none();
        self.solution.is_optimal = self.solution.cost.is_some();
        self.solution.time = self.time_keeper.elapsed_time();
        Ok((self.solution.clone(), true))
    }
}
