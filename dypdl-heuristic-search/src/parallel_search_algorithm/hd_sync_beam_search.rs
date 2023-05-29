use super::data_structure::SearchNodeMessage;
use super::hd_search_statistics::HdSearchStatistics;
use crate::search_algorithm::data_structure::{exceed_bound, Beam};
use crate::search_algorithm::util::TimeKeeper;
use crate::search_algorithm::{
    get_solution_cost_and_suffix, BeamSearchParameters, BfsNode, SearchInput, Solution,
    StateRegistry,
};
use bus::{Bus, BusReader};
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use dypdl::{variable_type, Model, TransitionInterface};
use std::error::Error;
use std::fmt::Display;
use std::rc::Rc;
use std::sync::Arc;
use std::{cmp, iter, mem, thread};

/// Performs hash distributed beam search with layer synchronization.
///
/// This function uses forward search based on the shortest path problem.
/// It only works with problems where the cost expressions are in the form of `cost + w`, `cost * w`, `max(cost, w)`, or `min(cost, w)`
/// where `cost` is `IntegerExpression::Cost`or `ContinuousExpression::Cost` and `w` is a numeric expression independent of `cost`.
///
/// It keeps the best `beam_size` nodes at each layer.
///
/// Type parameter `N` is a node type that implements `BfsNode`.
/// Type parameter `E` is a type of a function that evaluates a transition and generate a successor node.
/// The last argument of the function is the primal bound of the solution cost.
/// Type parameter `B` is a type of a function that combines the g-value (the cost to a state) and the base cost.
/// It should be the same function as the cost expression, e.g., `cost + base_cost` for `cost + w`.
///
/// # Panics
///
/// If it fails to create a thread pool or reserve memory for the state registry.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::Search;
/// use dypdl_heuristic_search::parallel_search_algorithm::{
///     DistributedFNode, FNodeMessage, hd_sync_beam_search,
/// };
/// use dypdl_heuristic_search::search_algorithm::{
///     BeamSearchParameters, SearchInput, SuccessorGenerator,
/// };
/// use std::sync::Arc;
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
/// let model = Arc::new(model);
///
/// let state = model.target.clone();
/// let cost = 0;
/// let h_evaluator = |_: &_| Some(0);
/// let f_evaluator = |g, h, _: &_| g + h;
/// let primal_bound = None;
/// let node = FNodeMessage::generate_root_node(
///     state,
///     cost,
///     &model,
///     &h_evaluator,
///     &f_evaluator,
///     primal_bound,
/// );
/// let generator = SuccessorGenerator::<Transition, Arc<Transition>, Arc<_>>::from_model(
///     model.clone(), false,
/// );
/// let input = SearchInput {
///     node,
///     generator,
///     solution_suffix: &[],
/// };
/// let transition_evaluator =
///     move |node: &DistributedFNode<_>, transition, primal_bound| {
///         node.generate_sendable_successor_node(
///             transition,
///             &model,
///             &h_evaluator,
///             &f_evaluator,
///             primal_bound,
///         )
/// };
/// let base_cost_evaluator = |cost, base_cost| cost + base_cost;
/// let parameters = BeamSearchParameters::default();
/// let threads = 1;
/// let (solution, _) = hd_sync_beam_search(
///     &input, transition_evaluator, base_cost_evaluator, parameters, threads,
/// ).unwrap();
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions.len(), 1);
/// assert_eq!(Transition::from(solution.transitions[0].clone()), increment);
/// assert!(!solution.is_infeasible);
/// ```
pub fn hd_sync_beam_search<'a, T, N, M, E, B, V>(
    input: &'a SearchInput<'a, M, V, Arc<V>, Arc<Model>>,
    transition_evaluator: E,
    base_cost_evaluator: B,
    parameters: BeamSearchParameters<T>,
    threads: usize,
) -> Result<(Solution<T, V>, HdSearchStatistics), Box<dyn Error>>
where
    T: variable_type::Numeric + Ord + Display + Send + Sync,
    N: BfsNode<T, V>,
    N: From<M>,
    M: Clone + SearchNodeMessage,
    E: Fn(&N, Arc<V>, Option<T>) -> Option<M> + Send + Sync,
    B: Fn(T, T) -> T + Send + Sync,
    V: TransitionInterface + Clone + Default + Send + Sync,
{
    let threads = cmp::min(threads, parameters.beam_size);
    let base_beam_size = parameters.beam_size / threads;
    let modulo = parameters.beam_size % threads;

    let (node_txs, node_rxs): (Vec<_>, Vec<_>) = (0..threads).map(|_| unbounded()).unzip();
    let (solution_tx, solution_rx) = bounded(1);
    let (optimality_tx, optimality_rx) = bounded(1);
    let (statistics_tx, statistics_rx) = bounded(threads);

    let (local_layer_tx, local_layer_rx) = bounded(threads - 1);
    let mut global_layer_tx = Bus::new(1);
    let follower_channels = (0..threads - 1)
        .map(|_| LayerChannel::Follower(local_layer_tx.clone(), global_layer_tx.add_rx()))
        .collect::<Vec<_>>();
    let leader_channel = LayerChannel::Leader(local_layer_rx, global_layer_tx);
    let layer_channels = iter::once(leader_channel).chain(follower_channels);

    thread::scope(|s| {
        for (id, (node_rx, layer_channel)) in node_rxs.into_iter().zip(layer_channels).enumerate() {
            let node_txs = node_txs.clone();

            let solution_tx = solution_tx.clone();
            let optimality_tx = optimality_tx.clone();
            let statistics_tx = statistics_tx.clone();
            let channels = Channels {
                id,
                node_txs,
                node_rx,
                layer_channel,
                solution_tx,
                optimality_tx,
                statistics_tx,
            };

            let mut parameters = parameters;
            parameters.beam_size = base_beam_size + if id < modulo { 1 } else { 0 };
            let transition_evaluator = &transition_evaluator;
            let base_cost_evaluator = &base_cost_evaluator;

            s.spawn(move || {
                single_sync_beam_search(
                    input,
                    transition_evaluator,
                    base_cost_evaluator,
                    parameters,
                    channels,
                )
            });
        }
    });

    let mut solution = Solution::default();

    if let Some((cost, transitions)) = solution_rx.recv()? {
        solution.cost = Some(cost);
        solution.transitions = transitions;
    }

    let optimality_message = optimality_rx.recv()?;

    if optimality_message.proved {
        solution.is_optimal = solution.cost.is_some();
        solution.is_infeasible = solution.cost.is_none();
    }

    solution.best_bound = optimality_message.bound;
    solution.time_out = optimality_message.time_out;

    let mut statistics = HdSearchStatistics {
        expanded: Vec::with_capacity(threads),
        generated: Vec::with_capacity(threads),
        kept: Vec::with_capacity(threads),
        sent: Vec::with_capacity(threads),
    };

    for _ in 0..threads {
        let information = statistics_rx.recv()?;
        solution.expanded += information.expanded;
        solution.generated += information.generated;
        statistics.expanded.push(information.expanded);
        statistics.generated.push(information.generated);
        statistics.kept.push(information.kept);
        statistics.sent.push(information.sent);
    }

    Ok((solution, statistics))
}

#[derive(Default, Clone)]
struct LocalLayerMessage<T> {
    id: usize,
    pruned: bool,
    is_empty: bool,
    bound: Option<T>,
    cost: Option<T>,
}

#[derive(Clone)]
enum GlobalLayerMessage<T> {
    Terminate(Option<usize>),
    Bound(Option<T>),
}

#[derive(Default)]
struct OptimalityMessage<T> {
    bound: Option<T>,
    proved: bool,
    time_out: bool,
}

#[derive(Default)]
struct Statistics {
    expanded: usize,
    generated: usize,
    kept: usize,
    sent: usize,
}

enum LayerChannel<T> {
    Leader(Receiver<LocalLayerMessage<T>>, Bus<GlobalLayerMessage<T>>),
    Follower(
        Sender<LocalLayerMessage<T>>,
        BusReader<GlobalLayerMessage<T>>,
    ),
}

struct Channels<T, M, V> {
    id: usize,
    node_txs: Vec<Sender<Option<M>>>,
    node_rx: Receiver<Option<M>>,
    layer_channel: LayerChannel<T>,
    solution_tx: Sender<Option<(T, Vec<V>)>>,
    optimality_tx: Sender<OptimalityMessage<T>>,
    statistics_tx: Sender<Statistics>,
}

fn single_sync_beam_search<'a, T, N, M, E, B, V>(
    input: &'a SearchInput<'a, M, V, Arc<V>, Arc<Model>>,
    transition_evaluator: E,
    base_cost_evaluator: B,
    parameters: BeamSearchParameters<T>,
    mut channels: Channels<T, M, V>,
) where
    T: variable_type::Numeric + Ord + Display + Sync,
    N: BfsNode<T, V>,
    N: From<M>,
    M: Clone + SearchNodeMessage,
    E: Fn(&N, Arc<V>, Option<T>) -> Option<M>,
    B: Fn(T, T) -> T,
    V: TransitionInterface + Clone + Default,
{
    let id = channels.id;
    let time_keeper = if let LayerChannel::Leader(..) = &channels.layer_channel {
        Some(
            parameters
                .parameters
                .time_limit
                .map_or_else(TimeKeeper::default, TimeKeeper::with_time_limit),
        )
    } else {
        None
    };
    let quiet = id != 0 || parameters.parameters.quiet;
    let mut primal_bound = parameters.parameters.primal_bound;
    let threads = channels.node_txs.len();

    let model = &input.generator.model;
    let generator = &input.generator;
    let suffix = input.solution_suffix;
    let mut current_beam = Beam::<_, N, Rc<_>, _>::new(parameters.beam_size);
    let mut next_beam = Beam::<_, _, Rc<_>, _>::new(parameters.beam_size);
    let mut registry = StateRegistry::<_, N, _, _, _>::new(model.clone());
    let capacity = parameters
        .parameters
        .initial_registry_capacity
        .unwrap_or_else(|| current_beam.capacity());
    registry.reserve(capacity);

    let mut sent = 0;
    let mut kept = 0;
    let mut generated = 0;

    if let Some(node) = input.node.clone() {
        if id == node.assign_thread(threads) {
            let node = N::from(node);
            current_beam.insert(&mut registry, node);
            generated += 1;

            if !parameters.keep_all_layers {
                registry.clear();
            }
        }
    }

    let mut expanded = 0;
    let mut pruned = false;
    let mut best_dual_bound = None;
    let mut layer_index = 0;

    loop {
        let mut incumbent = None;
        let previously_pruned = pruned;
        let mut layer_dual_bound = None;

        {
            let mut expanded_all = false;
            let mut sent_all = false;
            let mut received_all = 0;
            let mut iter = current_beam.drain();

            while !sent_all || received_all < threads - 1 {
                if !expanded_all {
                    // Expands a node.
                    if let Some(node) = iter.next() {
                        if let Some(bound) = node.bound(model) {
                            if exceed_bound(model, bound, primal_bound) {
                                continue;
                            }
                        }

                        if let Some((cost, suffix)) = get_solution_cost_and_suffix(
                            model,
                            &*node,
                            suffix,
                            &base_cost_evaluator,
                        ) {
                            if !exceed_bound(model, cost, primal_bound) {
                                primal_bound = Some(cost);
                                incumbent = Some((node, cost, suffix));

                                // Optimal solution, ignore remaining open nodes.
                                if Some(cost) == best_dual_bound {
                                    expanded_all = true;
                                }
                            }
                            continue;
                        }

                        if pruned && incumbent.is_some() {
                            continue;
                        }

                        expanded += 1;

                        for transition in generator.applicable_transitions(node.state()) {
                            if let Some(successor) =
                                transition_evaluator(&node, transition, primal_bound)
                            {
                                let sent_to = successor.assign_thread(threads);

                                if sent_to == id {
                                    kept += 1;
                                    let successor = N::from(successor);

                                    if let Some(bound) = successor.bound(model) {
                                        if !exceed_bound(model, bound, layer_dual_bound) {
                                            layer_dual_bound = Some(bound);
                                        }
                                    }

                                    let (new_generated, beam_pruning) =
                                        next_beam.insert(&mut registry, successor);

                                    if !pruned && beam_pruning {
                                        pruned = true;
                                    }

                                    if new_generated {
                                        generated += 1;
                                    }
                                } else {
                                    channels.node_txs[sent_to].send(Some(successor)).unwrap();
                                    sent += 1;
                                }
                            }
                        }
                    } else {
                        expanded_all = true;
                    }
                }

                // Notifies the other threads that the current thread sent all nodes
                if expanded_all && !sent_all {
                    sent_all = true;
                    channels.node_txs.iter().enumerate().for_each(|(i, tx)| {
                        if i != id {
                            tx.send(None).unwrap()
                        }
                    });
                }

                if received_all < threads - 1 {
                    // Receives a node.
                    while let Ok(node) = channels.node_rx.try_recv() {
                        if let Some(node) = node {
                            let node = N::from(node);

                            if let Some(bound) = node.bound(model) {
                                if !exceed_bound(model, bound, layer_dual_bound) {
                                    layer_dual_bound = Some(bound);
                                }
                            }

                            let (new_generated, beam_pruning) =
                                next_beam.insert(&mut registry, node);

                            if !pruned && beam_pruning {
                                pruned = true;
                            }

                            if new_generated {
                                generated += 1;
                            }
                        } else {
                            received_all += 1;
                        }
                    }
                }
            }
        }

        // Aggregates the information of the current layer.
        match &mut channels.layer_channel {
            LayerChannel::Follower(tx, rx) => {
                // Sends the information to the leader.
                let information = LocalLayerMessage {
                    id,
                    pruned,
                    is_empty: next_beam.is_empty(),
                    bound: layer_dual_bound,
                    cost: incumbent.as_ref().map(|(_, cost, _)| *cost),
                };
                tx.send(information).unwrap();

                // Receives the aggregated information from the leader.
                match rx.recv().unwrap() {
                    // Termination.
                    GlobalLayerMessage::Terminate(goal_id) => {
                        if Some(id) == goal_id {
                            let (node, cost, suffix) = incumbent.unwrap();
                            let mut transitions = node.transitions();
                            transitions.extend_from_slice(suffix);

                            channels
                                .solution_tx
                                .send(Some((cost, transitions)))
                                .unwrap()
                        }

                        // Sends the statistics to the original thread.
                        channels
                            .statistics_tx
                            .send(Statistics {
                                expanded,
                                generated,
                                kept,
                                sent,
                            })
                            .unwrap();

                        return;
                    }
                    // Updates the dual bound.
                    GlobalLayerMessage::Bound(bound) => {
                        best_dual_bound = bound;
                    }
                }
            }
            LayerChannel::Leader(rx, tx) => {
                let mut is_empty = next_beam.is_empty();
                let mut cost = incumbent.as_ref().map(|(_, cost, _)| *cost);
                let mut goal_id = if cost.is_some() { Some(id) } else { None };

                // Receives and aggregates the information from each follower.
                for _ in 0..threads - 1 {
                    let information = rx.recv().unwrap();
                    pruned |= information.pruned;
                    is_empty &= information.is_empty;

                    if let Some(bound) = information.bound {
                        if !exceed_bound(model, bound, layer_dual_bound) {
                            layer_dual_bound = Some(bound);
                        }
                    }

                    if let Some(other) = information.cost {
                        if !exceed_bound(model, other, cost)
                            || (Some(other) == cost && information.id < goal_id.unwrap())
                        {
                            cost = Some(other);
                            goal_id = Some(information.id);
                        }
                    }
                }

                if let (false, Some(value)) = (previously_pruned, layer_dual_bound) {
                    if best_dual_bound
                        .map_or(true, |bound| !exceed_bound(model, bound, Some(value)))
                    {
                        best_dual_bound = Some(value);
                    }
                }

                let time_out = time_keeper.as_ref().unwrap().check_time_limit(quiet);

                if !quiet {
                    println!(
                        "Searched layer: {}, elapsed time: {}",
                        layer_index,
                        time_keeper.as_ref().unwrap().elapsed_time()
                    );
                }

                if is_empty || time_out || goal_id.is_some() {
                    let mut proved = !pruned && is_empty;

                    if let Some(goal_id) = goal_id {
                        if threads > 1 {
                            // Sends the termination signal to all followers
                            // with the id of the thread that finds the best solution.
                            tx.broadcast(GlobalLayerMessage::Terminate(Some(goal_id)));
                        }

                        if cost == best_dual_bound {
                            proved = true;
                        }

                        if goal_id == id {
                            let (node, cost, suffix) = incumbent.unwrap();
                            let mut transitions = node.transitions();
                            transitions.extend_from_slice(suffix);

                            channels
                                .solution_tx
                                .send(Some((cost, transitions)))
                                .unwrap()
                        }
                    } else {
                        if threads > 1 {
                            // Sends the termination signal to all followers without a solution.
                            tx.broadcast(GlobalLayerMessage::Terminate(None));
                        }

                        // Sends no solution to the original thread.
                        channels.solution_tx.send(None).unwrap();

                        if proved {
                            best_dual_bound = None;
                        }
                    }

                    // Sends the statistics to the original thread.
                    channels
                        .statistics_tx
                        .send(Statistics {
                            expanded,
                            generated,
                            kept,
                            sent,
                        })
                        .unwrap();
                    // Sends the optimality information to the original thread.
                    channels
                        .optimality_tx
                        .send(OptimalityMessage {
                            bound: best_dual_bound,
                            proved,
                            time_out,
                        })
                        .unwrap();

                    return;
                } else if threads > 1 {
                    // Sends the dual bound to all followers.
                    tx.broadcast(GlobalLayerMessage::Bound(best_dual_bound));
                }
            }
        }

        mem::swap(&mut current_beam, &mut next_beam);

        if !parameters.keep_all_layers {
            registry.clear();
        }

        layer_index += 1;
    }
}
