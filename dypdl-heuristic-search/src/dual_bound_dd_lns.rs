use super::f_evaluator_type::FEvaluatorType;
use super::search_algorithm::{
    beam_search, rollout, Cabs, CabsParameters, CostNode, DdLns, DdLnsParameters, FNode,
    NeighborhoodSearchInput, Search, SearchInput, Solution, StateInRegistry, SuccessorGenerator,
    TransitionMutex, TransitionWithId,
};
use dypdl::{variable_type, Transition};
use std::fmt;
use std::rc::Rc;
use std::str;

/// Create a Large Neighborhood Search with Decision Diagrams (DD-LNS) solver using the dual bound as a heuristic function.
///
/// It performs LNS by constructing restricted multi-valued decision diagrams (MDD).
/// It starts from a initial solution (`transitions`) and improves it.
/// If `transitions` is `None`, it first performs CABS to find an initial feasible solution and then performs DD-LNS to improve the solution.
///
/// This solver uses forward search based on the shortest path problem.
/// It only works with problems where the cost expressions are in the form of `cost + w`, `cost * w`, `max(cost, w)`, or `min(cost, w)`
/// where `cost` is `IntegerExpression::Cost`or `ContinuousExpression::Cost` and `w` is a numeric expression independent of `cost`.
/// `f_evaluator_type` must be specified appropriately according to the cost expressions.
///
/// # Panics
///
/// If CABS to find an initial feasible solution panics.
///
/// # References
///
/// Xavier Gillard and Pierre Schaus. "Large Neighborhood Search with Decision Diagrams,"
/// Proceedings of the 31st International Joint Conference on Artificial Intelligence (IJCAI), pp. 4754-4760, 2022.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::{
///     CabsParameters, create_dual_bound_dd_lns, DdLnsParameters, FEvaluatorType,
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
/// model.add_dual_bound(IntegerExpression::from(0)).unwrap();
///
/// let model = Rc::new(model);
/// let parameters = DdLnsParameters::default();
/// let cabs_parameters = CabsParameters::default();
/// let f_evaluator_type = FEvaluatorType::Plus;
///
/// let mut solver = create_dual_bound_dd_lns(
///     model, None, parameters, cabs_parameters, f_evaluator_type,
/// );
/// let solution = solver.search().unwrap();
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions, vec![increment]);
/// assert!(!solution.is_infeasible);
/// ```
pub fn create_dual_bound_dd_lns<T>(
    model: Rc<dypdl::Model>,
    transitions: Option<Vec<Transition>>,
    mut parameters: DdLnsParameters<T>,
    cabs_parameters: CabsParameters<T>,
    f_evaluator_type: FEvaluatorType,
) -> Box<dyn Search<T>>
where
    T: variable_type::Numeric + fmt::Display + Ord + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    let generator = SuccessorGenerator::<TransitionWithId>::from_model(model.clone(), false);
    let base_cost_evaluator = move |cost, base_cost| f_evaluator_type.eval(cost, base_cost);
    let root_cost = match f_evaluator_type {
        FEvaluatorType::Plus => T::zero(),
        FEvaluatorType::Product => T::one(),
        FEvaluatorType::Max => T::min_value(),
        FEvaluatorType::Min => T::max_value(),
        FEvaluatorType::Overwrite => T::zero(),
    };

    let generator_transitions = generator
        .transitions
        .iter()
        .chain(generator.forced_transitions.iter())
        .map(|t| t.as_ref().clone());

    let (solution, transition_mutex) = if let Some(transitions) = transitions {
        // Validate the initial solution.
        let solution_cost = if let Some(result) = rollout(
            &generator.model.target,
            root_cost,
            &transitions,
            base_cost_evaluator,
            &model,
        ) {
            if result.is_base {
                Some(result.cost)
            } else {
                if !parameters.beam_search_parameters.parameters.quiet {
                    println!("Initial solution does not result in a base state");
                }
                None
            }
        } else {
            if !parameters.beam_search_parameters.parameters.quiet {
                println!("Initial solution is infeasible");
            }
            None
        };
        let id_max = generator_transitions
            .clone()
            .map(|t| t.id)
            .max()
            .unwrap_or(0);
        let transitions = transitions
            .into_iter()
            .enumerate()
            .map(|(i, transition)| TransitionWithId {
                id: id_max + 1 + i,
                forced: false,
                transition,
            })
            .collect::<Vec<_>>();
        let solution = Solution {
            cost: solution_cost,
            transitions: transitions.clone(),
            ..Default::default()
        };

        // Consider the given transitions in TransitionMutex
        let transitions = generator_transitions
            .chain(transitions.into_iter())
            .collect();

        (Some(solution), TransitionMutex::new(transitions))
    } else {
        (None, TransitionMutex::new(generator_transitions.collect()))
    };

    if model.has_dual_bounds() {
        let h_model = model.clone();
        let h_evaluator = move |state: &_| h_model.eval_dual_bound(state);
        let f_evaluator = move |g, h, _: &_| f_evaluator_type.eval(g, h);
        let g_model = model.clone();
        let node_generator = move |state, cost| {
            FNode::generate_root_node(
                state,
                cost,
                &g_model,
                &h_evaluator,
                &f_evaluator,
                cabs_parameters
                    .beam_search_parameters
                    .parameters
                    .primal_bound,
            )
        };

        // Perform CABS if no initial solution is given.
        let solution = solution.unwrap_or_else(|| {
            let h_model = model.clone();
            let h_evaluator = move |state: &_| h_model.eval_dual_bound(state);
            let f_evaluator = move |g, h, _: &_| f_evaluator_type.eval(g, h);
            let t_model = model.clone();
            let transition_evaluator = move |node: &FNode<_, _>, transition, primal_bound| {
                node.generate_successor_node(
                    transition,
                    &t_model,
                    &h_evaluator,
                    &f_evaluator,
                    primal_bound,
                )
            };
            let input = SearchInput {
                node: node_generator(StateInRegistry::from(model.target.clone()), root_cost),
                generator: generator.clone(),
                solution_suffix: &[],
            };
            let beam_search = move |input: &SearchInput<_, _>, parameters| {
                beam_search(
                    input,
                    &transition_evaluator,
                    base_cost_evaluator,
                    parameters,
                )
            };
            let mut cabs = Cabs::new(input, beam_search.clone(), cabs_parameters);
            let (solution, _) = cabs.search_inner();
            solution
        });

        let h_evaluator = move |state: &_| model.eval_dual_bound(state);
        let f_evaluator = move |g, h, _: &_| f_evaluator_type.eval(g, h);
        let transition_evaluator =
            move |node: &FNode<_, _>, transition, registry: &mut _, primal_bound| {
                node.insert_successor_node(
                    transition,
                    registry,
                    &h_evaluator,
                    &f_evaluator,
                    primal_bound,
                )
            };
        parameters.beam_search_parameters.parameters.time_limit = parameters
            .beam_search_parameters
            .parameters
            .time_limit
            .map(|time_limit| {
                let remaining = time_limit - solution.time;
                if remaining < 0.0 {
                    0.0
                } else {
                    remaining
                }
            });

        let input = NeighborhoodSearchInput {
            root_cost,
            node_generator,
            successor_generator: generator,
            solution,
            phantom: Default::default(),
        };

        Box::new(DdLns::new(
            input,
            transition_evaluator,
            base_cost_evaluator,
            transition_mutex,
            parameters,
        ))
    } else {
        let g_model = model.clone();
        let node_generator =
            move |state, cost| Some(CostNode::generate_root_node(state, cost, &g_model));

        // Perform CABS if no initial solution is given.
        let solution = solution.unwrap_or_else(|| {
            let t_model = model.clone();
            let transition_evaluator = move |node: &CostNode<_, _>, transition, _| {
                node.generate_successor_node(transition, &t_model)
            };
            let input = SearchInput {
                node: node_generator(StateInRegistry::from(model.target.clone()), root_cost),
                generator: generator.clone(),
                solution_suffix: &[],
            };
            let beam_search = move |input: &SearchInput<_, _>, parameters| {
                beam_search(
                    input,
                    &transition_evaluator,
                    base_cost_evaluator,
                    parameters,
                )
            };
            let mut cabs = Cabs::new(input, beam_search.clone(), cabs_parameters);
            let (solution, _) = cabs.search_inner();
            solution
        });

        let transition_evaluator = move |node: &CostNode<_, _>, transition, registry: &mut _, _| {
            node.insert_successor_node(transition, registry)
        };
        parameters.beam_search_parameters.parameters.time_limit = parameters
            .beam_search_parameters
            .parameters
            .time_limit
            .map(|time_limit| {
                let remaining = time_limit - solution.time;
                if remaining < 0.0 {
                    0.0
                } else {
                    remaining
                }
            });

        let input = NeighborhoodSearchInput {
            root_cost,
            node_generator,
            successor_generator: generator,
            solution,
            phantom: Default::default(),
        };

        Box::new(DdLns::new(
            input,
            transition_evaluator,
            base_cost_evaluator,
            transition_mutex,
            parameters,
        ))
    }
}
