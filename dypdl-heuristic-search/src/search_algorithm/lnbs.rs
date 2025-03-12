use super::beam_search::BeamSearchParameters;
use super::data_structure::{
    exceed_bound, HashableSignatureVariables, StateInRegistry, TransitionMutex, TransitionWithId,
};
use super::neighborhood_search::NeighborhoodSearchInput;
use super::rollout::get_trace;
use super::search::{Parameters, Search, SearchInput, Solution};
use super::util::{print_primal_bound, update_bound_if_better, TimeKeeper};
use dypdl::{variable_type, Model, ReduceFunction, Transition, TransitionInterface};
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use rustc_hash::FxHashMap;
use std::cmp;
use std::error::Error;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::Deref;
use std::rc::Rc;
use std::str;

/// Parameters for LNBS
#[derive(Debug, PartialEq, Clone, Copy, Default)]
pub struct LnbsParameters<T> {
    /// Maximum beam size.
    pub max_beam_size: Option<usize>,
    /// Random seed.
    pub seed: u64,
    /// Cost can be negative.
    pub has_negative_cost: bool,
    /// Bias the weight by the cost difference.
    pub use_cost_weight: bool,
    /// Select an arm uniformly at random.
    pub no_bandit: bool,
    /// Do not use transition mutex.
    pub no_transition_mutex: bool,
    /// Parameters for beam search.
    pub beam_search_parameters: BeamSearchParameters<T>,
}

/// Large Neighborhood Beam Search (LNBS).
///
/// It performs Large Neighborhood Beam Search (LNBS), which improves a solution by finding a partial path using beam search.
///
/// This solver uses forward search based on the shortest path problem.
/// It only works with problems where the cost expressions are in the form of `cost + w`, `cost * w`, `max(cost, w)`, or `min(cost, w)`
/// where `cost` is `IntegerExpression::Cost`or `ContinuousExpression::Cost` and `w` is a numeric expression independent of `cost`.
///
/// Type parameter `B` is a type of a function that performs beam search.
/// The function takes a `SearchInput` and `BeamSearchParameters` and returns a `Solution`.
///
/// `parameters.parameters.time_limit` is required in this solver.
///
/// Note that a solution found by this solver may not apply a forced transition when it is applicable.
///
/// # References
///
/// Ryo Kuroiwa and J. Christopher Beck. "Large Neighborhood Beam Search for Domain-Independent Dynamic Programming,"
/// Proceedings of the 29th International Conference on Principles and Practice of Constraint Programming (CP), 2023.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::{Parameters, Search, Solution};
/// use dypdl_heuristic_search::search_algorithm::{
///     beam_search, BeamSearchParameters, FNode, Lnbs, LnbsParameters, NeighborhoodSearchInput,
///     SearchInput, SuccessorGenerator, TransitionMutex, TransitionWithId,
/// };
/// use std::rc::Rc;
/// use std::marker::PhantomData;
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
/// let model = Rc::new(model);
///
/// let h_evaluator = |_: &_, _: &mut _| Some(0);
/// let f_evaluator = |g, h, _: &_| g + h;
/// let primal_bound = None;
/// let successor_generator = SuccessorGenerator::<TransitionWithId>::from_model(
///     model.clone(), false,
/// );
/// let t_model = model.clone();
/// let transition_evaluator = move |node: &FNode<_, _>, transition, cache: &mut _, primal_bound| {
///     node.generate_successor_node(
///         transition,
///         cache,
///         &t_model,
///         &h_evaluator,
///         &f_evaluator,
///         primal_bound,
///     )
/// };
/// let base_cost_evaluator = |cost, base_cost| cost + base_cost;
/// let beam_search = move |input: &SearchInput<_, _>, parameters| {
///     beam_search(input, &transition_evaluator, base_cost_evaluator, parameters)
/// };
/// let node_generator = |state, cost| {
///     let mut function_cache = StateFunctionCache::new(&model.state_functions);
///
///     FNode::generate_root_node(
///         state,
///         &mut function_cache,
///         cost,
///         &model,
///         &h_evaluator,
///         &f_evaluator,
///         primal_bound,
///     )
/// };
///
/// let solution = Solution::<_, _> {
///     cost: Some(1),
///     transitions: vec![TransitionWithId {
///         transition: increment.clone(),
///         id: 0,
///         forced: false,
///     }],
///     ..Default::default()
/// };
///
/// let parameters = LnbsParameters {
///     beam_search_parameters: BeamSearchParameters {
///         parameters: Parameters {
///             time_limit: Some(10.0),
///             ..Default::default()
///         },
///         ..Default::default()
///     },    
///     ..Default::default()
/// };
///
/// let transition_mutex = TransitionMutex::new(
///     successor_generator.transitions
///     .iter()
///     .chain(successor_generator.forced_transitions.iter())
///     .map(|t| t.as_ref().clone())
///     .collect()
/// );
///
/// let input = NeighborhoodSearchInput {
///     root_cost: 0,
///     node_generator,
///     successor_generator,
///     solution,
///     phantom: PhantomData::default(),
/// };
///
/// let mut solver = Lnbs::new(input, beam_search, transition_mutex, parameters);
/// let solution = solver.search().unwrap();
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions, vec![increment]);
/// assert!(!solution.is_infeasible);
/// ```
pub struct Lnbs<
    T,
    N,
    B,
    G,
    V = Transition,
    D = Rc<V>,
    R = Rc<Model>,
    K = Rc<HashableSignatureVariables>,
> where
    T: variable_type::Numeric + Display,
    <T as str::FromStr>::Err: Debug,
    B: FnMut(
        &SearchInput<N, TransitionWithId<V>, D, R>,
        BeamSearchParameters<T>,
    ) -> Solution<T, TransitionWithId<V>>,
    G: FnMut(StateInRegistry<K>, T) -> Option<N>,
    V: TransitionInterface + Clone + Default,
    D: Deref<Target = TransitionWithId<V>> + Clone,
    R: Deref<Target = Model> + Clone,
    K: Hash
        + Eq
        + Clone
        + Debug
        + Deref<Target = HashableSignatureVariables>
        + From<HashableSignatureVariables>,
{
    input: NeighborhoodSearchInput<T, N, G, StateInRegistry<K>, TransitionWithId<V>, D, R>,
    beam_search: B,
    max_beam_size: Option<usize>,
    has_negative_cost: bool,
    use_cost_weight: bool,
    no_bandit: bool,
    no_transition_mutex: bool,
    initial_beam_size: usize,
    keep_all_layers: bool,
    quiet: bool,
    transition_mutex: TransitionMutex,
    neighborhood_beam_size: FxHashMap<(usize, usize), (usize, bool)>,
    depth_arms: Vec<usize>,
    reward_mean: Vec<f64>,
    time_mean: Vec<f64>,
    trials: Vec<f64>,
    total_trials: f64,
    lambda: Option<f64>,
    time_limit: f64,
    depth_exhausted: Vec<bool>,
    rng: Pcg64Mcg,
    time_keeper: TimeKeeper,
    first_call: bool,
}

impl<T, N, B, G, V, D, R, K> Lnbs<T, N, B, G, V, D, R, K>
where
    T: variable_type::Numeric + Ord + Display,
    <T as str::FromStr>::Err: Debug,
    B: FnMut(
        &SearchInput<N, TransitionWithId<V>, D, R>,
        BeamSearchParameters<T>,
    ) -> Solution<T, TransitionWithId<V>>,
    G: FnMut(StateInRegistry<K>, T) -> Option<N>,
    V: TransitionInterface + Clone + Default,
    Transition: From<V>,
    D: Deref<Target = TransitionWithId<V>> + From<TransitionWithId<V>> + Clone,
    R: Deref<Target = Model> + Clone,
    K: Hash
        + Eq
        + Clone
        + Debug
        + Deref<Target = HashableSignatureVariables>
        + From<HashableSignatureVariables>,
{
    /// Create a new LNBS solver.
    pub fn new(
        input: NeighborhoodSearchInput<T, N, G, StateInRegistry<K>, TransitionWithId<V>, D, R>,
        beam_search: B,
        transition_mutex: TransitionMutex,
        parameters: LnbsParameters<T>,
    ) -> Lnbs<T, N, B, G, V, D, R, K> {
        let time_limit = parameters
            .beam_search_parameters
            .parameters
            .time_limit
            .unwrap();
        let mut time_keeper = TimeKeeper::with_time_limit(time_limit);

        let max_depth = input.solution.transitions.len();

        let (depth_arms, reward_mean, time_mean, trials, depth_exhausted) = if max_depth > 0 {
            let log = (usize::BITS - 1) - max_depth.leading_zeros();
            let mut depth_arms: Vec<usize> = (1..=log).map(|i| 2usize.pow(i)).collect();

            if let Some(last) = depth_arms.last() {
                if *last < max_depth {
                    depth_arms.push(max_depth);
                }
            } else {
                depth_arms.push(max_depth);
            }

            let reward_mean = vec![0.0; depth_arms.len()];
            let time_mean = vec![0.0; depth_arms.len()];
            let trials = vec![0.0; depth_arms.len()];
            let depth_exhausted = vec![false; depth_arms.len()];

            (depth_arms, reward_mean, time_mean, trials, depth_exhausted)
        } else {
            (
                Vec::default(),
                Vec::default(),
                Vec::default(),
                Vec::default(),
                Vec::default(),
            )
        };

        time_keeper.stop();

        Lnbs {
            input,
            beam_search,
            max_beam_size: parameters.max_beam_size,
            has_negative_cost: parameters.has_negative_cost,
            use_cost_weight: parameters.use_cost_weight,
            no_bandit: parameters.no_bandit,
            no_transition_mutex: parameters.no_transition_mutex,
            initial_beam_size: parameters.beam_search_parameters.beam_size,
            keep_all_layers: parameters.beam_search_parameters.keep_all_layers,
            quiet: parameters.beam_search_parameters.parameters.quiet,
            transition_mutex,
            neighborhood_beam_size: FxHashMap::default(),
            depth_arms,
            reward_mean,
            time_mean,
            trials,
            total_trials: 0.0,
            lambda: None,
            time_limit,
            depth_exhausted,
            rng: Pcg64Mcg::seed_from_u64(parameters.seed),
            time_keeper,
            first_call: true,
        }
    }

    /// Search for the next solution, returning the solution using `TransitionWithId`.
    pub fn search_inner(&mut self) -> (Solution<T, TransitionWithId<V>>, bool) {
        if self.input.solution.is_terminated() || self.input.solution.cost.is_none() {
            return (self.input.solution.clone(), true);
        }

        self.time_keeper.start();

        if self.first_call {
            self.first_call = false;

            self.time_keeper.stop();

            return (
                self.input.solution.clone(),
                self.input.solution.cost.is_none(),
            );
        }

        let (states, costs): (Vec<_>, Vec<_>) = get_trace(
            &self.input.successor_generator.model.target,
            self.input.root_cost,
            &self.input.solution.transitions,
            &self.input.successor_generator.model,
        )
        .unzip();

        let last = self.depth_arms.last_mut().unwrap();

        if *last > self.input.solution.transitions.len() {
            *last = self.input.solution.transitions.len();
        };

        loop {
            let result = self.select_depth();

            if result.is_none() {
                self.time_keeper.stop();

                return (self.input.solution.clone(), true);
            }

            let (arm, depth) = result.unwrap();
            let time_start = self.time_keeper.elapsed_time();
            // println!("{:?}", self.reward_mean);
            // println!("{:?}", self.time_mean);
            // println!("{:?}", self.trials);
            // println!("Depth: {}", depth);

            let result = self.select_start(&costs[..], depth);

            if result.is_none() {
                self.depth_exhausted[arm] = true;
                continue;
            }

            let model = &self.input.successor_generator.model;

            let (start, beam_size) = result.unwrap();

            let prefix = &self.input.solution.transitions[..start];
            let prefix_cost = if start == 0 {
                self.input.root_cost
            } else {
                costs[start - 1]
            };
            let target = StateInRegistry::from(if start == 0 {
                model.target.clone()
            } else {
                states[start - 1].clone()
            });
            let node = (self.input.node_generator)(target, prefix_cost);
            let suffix = &self.input.solution.transitions[start + depth..];
            let generator = if self.no_transition_mutex {
                self.input.successor_generator.clone()
            } else {
                self.transition_mutex.filter_successor_generator(
                    &self.input.successor_generator,
                    prefix,
                    suffix,
                )
            };
            let input = SearchInput::<N, TransitionWithId<V>, D, R> {
                node,
                generator,
                solution_suffix: suffix,
            };

            let parameters = BeamSearchParameters {
                beam_size,
                keep_all_layers: self.keep_all_layers,
                parameters: Parameters {
                    primal_bound: self.input.solution.cost,
                    get_all_solutions: false,
                    quiet: true,
                    time_limit: self.time_keeper.remaining_time_limit(),
                    ..Default::default()
                },
            };

            let solution = (self.beam_search)(&input, parameters);

            self.input.solution.expanded += solution.expanded;
            self.input.solution.generated += solution.generated;

            let len = self.input.solution.transitions.len();

            let exhausted = solution.is_optimal || solution.is_infeasible;
            let (next_beam_size, done) = if let Some(max_beam_size) = self.max_beam_size {
                if beam_size >= max_beam_size {
                    (max_beam_size, true)
                } else {
                    (cmp::min(2 * beam_size, max_beam_size), exhausted)
                }
            } else {
                (2 * beam_size, exhausted)
            };
            self.neighborhood_beam_size
                .insert((start, depth), (next_beam_size, done));

            if let Some(cost) = solution.cost {
                if !exceed_bound(model, cost, self.input.solution.cost) {
                    self.neighborhood_beam_size = FxHashMap::from_iter(
                        self.neighborhood_beam_size
                            .drain()
                            .filter(|(k, _)| k.0 <= start && k.0 + k.1 >= start + depth),
                    );

                    self.depth_exhausted.iter_mut().for_each(|x| *x = false);

                    if Some(cost) == self.input.solution.best_bound {
                        self.input.solution.is_optimal = true;
                    }

                    if depth == len && solution.is_optimal {
                        self.input.solution.is_optimal = true;
                        self.input.solution.best_bound = solution.cost;
                    }

                    let mut transitions = prefix.to_vec();
                    transitions.extend(solution.transitions);
                    self.input.solution.transitions = transitions;

                    let current_cost = self.input.solution.cost.unwrap();
                    let reward = (cost - current_cost).to_continuous().abs()
                        / cmp::max(cost.abs(), current_cost.abs()).to_continuous();
                    let reward = if reward > 1.0 { 1.0 } else { reward };
                    let time = (self.time_keeper.elapsed_time() - time_start) / self.time_limit;
                    self.update_bandit(arm, reward, time);

                    self.input.solution.cost = Some(cost);
                    self.input.solution.time = self.time_keeper.elapsed_time();

                    if !self.quiet {
                        println!(
                            "A new primal bound is found with depth: {}, #beam: {}",
                            depth, beam_size
                        );
                        print_primal_bound(&self.input.solution);
                    }

                    self.time_keeper.stop();

                    return (self.input.solution.clone(), self.input.solution.is_optimal);
                }
            }

            if start == 0 && depth == len {
                if let Some(bound) = solution.best_bound {
                    update_bound_if_better(&mut self.input.solution, bound, model, self.quiet);
                }
            }

            if exhausted && depth == len {
                self.input.solution.is_optimal = self.input.solution.cost.is_some();
                self.input.solution.is_infeasible = self.input.solution.cost.is_none();
                self.input.solution.time = self.time_keeper.elapsed_time();

                if self.input.solution.is_optimal {
                    self.input.solution.best_bound = self.input.solution.cost;
                }

                self.time_keeper.stop();

                return (self.input.solution.clone(), true);
            }

            if solution.time_out {
                self.input.solution.time = self.time_keeper.elapsed_time();

                self.time_keeper.stop();

                return (self.input.solution.clone(), true);
            }

            let time = (self.time_keeper.elapsed_time() - time_start) / self.time_limit;
            self.update_bandit(arm, 0.0, time);
        }
    }

    fn update_bandit(&mut self, arm: usize, reward: f64, time: f64) {
        if self.lambda.is_none() {
            self.lambda = Some(time / 10.0);
        }

        self.total_trials += 1.0;
        self.trials[arm] += 1.0;
        self.reward_mean[arm] =
            (self.reward_mean[arm] * (self.trials[arm] - 1.0) + reward) / self.trials[arm];
        self.time_mean[arm] =
            (self.time_mean[arm] * (self.trials[arm] - 1.0) + time) / self.trials[arm];
    }

    fn select_depth(&mut self) -> Option<(usize, usize)> {
        if self.no_bandit {
            self.select_random()
        } else {
            self.select_ucb()
        }
    }

    fn select_ucb(&mut self) -> Option<(usize, usize)> {
        let last = self.depth_arms.len() - 1;
        let last_depth = self.depth_arms[last];
        self.depth_arms
            .iter()
            .enumerate()
            .filter_map(|(i, depth)| {
                if self.depth_exhausted[i] || (i < last && *depth >= last_depth) {
                    None
                } else {
                    if self.trials[i] < 0.5 {
                        return Some((f64::INFINITY, (i, *depth)));
                    }

                    let r = self.reward_mean[i];
                    let c = self.time_mean[i];
                    let lambda = self.lambda.unwrap();
                    let epsilon = (2.0 * self.total_trials.ln() / self.trials[i]).sqrt();
                    let numerator = if r + epsilon <= 1.0 { r + epsilon } else { 1.0 };
                    let denominator = if c - epsilon >= lambda {
                        c - epsilon
                    } else {
                        lambda
                    };

                    let score = r / c + epsilon / c + epsilon / c * numerator / denominator;

                    Some((score, (i, *depth)))
                }
            })
            .rev()
            .max_by(|(a, _), (b, _)| a.total_cmp(b))
            .map(|(_, arm)| arm)
    }

    fn select_random(&mut self) -> Option<(usize, usize)> {
        let last = self.depth_arms.len() - 1;
        let last_depth = self.depth_arms[last];
        self.depth_arms
            .iter()
            .enumerate()
            .filter_map(|(i, depth)| {
                if self.depth_exhausted[i] || (i < last && *depth >= last_depth) {
                    None
                } else {
                    let score = self.rng.random::<f64>();

                    Some((score, (i, *depth)))
                }
            })
            .max_by(|(a, _), (b, _)| a.total_cmp(b))
            .map(|(_, arm)| arm)
    }

    fn select_start(&mut self, costs: &[T], depth: usize) -> Option<(usize, usize)> {
        let not_cost_algebraic_minimization = self.has_negative_cost
            || self.input.successor_generator.model.reduce_function == ReduceFunction::Max;

        let (weights, starts): (Vec<_>, Vec<_>) = std::iter::once(T::zero())
            .chain(costs.iter().copied())
            .zip(costs[depth - 1..].iter().copied())
            .enumerate()
            .filter_map(|(start, (before, after))| {
                let entry = self
                    .neighborhood_beam_size
                    .entry((start, depth))
                    .or_insert((self.initial_beam_size, false));

                if entry.1 || (!not_cost_algebraic_minimization && after <= before) {
                    None
                } else if self.use_cost_weight {
                    Some((after - before, (start, entry.0)))
                } else {
                    Some((T::one(), (start, entry.0)))
                }
            })
            .unzip();

        if starts.is_empty() {
            return None;
        }

        let weights = if not_cost_algebraic_minimization && self.use_cost_weight {
            if self.input.successor_generator.model.reduce_function == ReduceFunction::Max {
                let max_weight = weights.iter().copied().max().unwrap();

                if let Some(second_max) = weights.iter().copied().filter(|v| *v < max_weight).max()
                {
                    weights
                        .into_iter()
                        .zip(starts.iter())
                        .map(|(v, (_, beam_size))| {
                            (max_weight - cmp::min(v, second_max)).to_continuous()
                                / *beam_size as f64
                        })
                        .collect()
                } else {
                    weights
                        .into_iter()
                        .zip(starts.iter())
                        .map(|(_, (_, beam_size))| 1.0 / *beam_size as f64)
                        .collect()
                }
            } else {
                let min_weight = weights.iter().copied().min().unwrap();

                if let Some(second_min) = weights.iter().copied().filter(|v| *v > min_weight).min()
                {
                    weights
                        .into_iter()
                        .zip(starts.iter())
                        .map(|(v, (_, beam_size))| {
                            (cmp::max(v, second_min) - min_weight).to_continuous()
                                / *beam_size as f64
                        })
                        .collect()
                } else {
                    weights
                        .into_iter()
                        .zip(starts.iter())
                        .map(|(_, (_, beam_size))| 1.0 / *beam_size as f64)
                        .collect()
                }
            }
        } else {
            let mut weights = weights
                .iter()
                .map(|v| v.to_continuous())
                .collect::<Vec<_>>();

            if self.use_cost_weight {
                weights
                    .iter_mut()
                    .zip(starts.iter())
                    .for_each(|(v, (_, beam_size))| *v /= *beam_size as f64);
            }

            weights
        };

        let dist = WeightedIndex::new(weights).unwrap();

        starts[dist.sample(&mut self.rng)].into()
    }
}

impl<T, N, B, G, V, D, R, K> Search<T> for Lnbs<T, N, B, G, V, D, R, K>
where
    T: variable_type::Numeric + Ord + Display,
    <T as str::FromStr>::Err: Debug,
    B: FnMut(
        &SearchInput<N, TransitionWithId<V>, D, R>,
        BeamSearchParameters<T>,
    ) -> Solution<T, TransitionWithId<V>>,
    G: FnMut(StateInRegistry<K>, T) -> Option<N>,
    V: TransitionInterface + Clone + Default,
    Transition: From<V>,
    D: Deref<Target = TransitionWithId<V>> + From<TransitionWithId<V>> + Clone,
    R: Deref<Target = Model> + Clone,
    K: Hash
        + Eq
        + Clone
        + Debug
        + Deref<Target = HashableSignatureVariables>
        + From<HashableSignatureVariables>,
{
    fn search_next(&mut self) -> Result<(Solution<T>, bool), Box<dyn Error>> {
        let (solution, is_terminated) = self.search_inner();
        let solution = Solution {
            cost: solution.cost,
            best_bound: solution.best_bound,
            is_optimal: solution.is_optimal,
            is_infeasible: solution.is_infeasible,
            transitions: solution
                .transitions
                .into_iter()
                .map(|t| dypdl::Transition::from(t.transition))
                .collect(),
            expanded: solution.expanded,
            generated: solution.generated,
            time: solution.time,
            time_out: solution.time_out,
        };

        Ok((solution, is_terminated))
    }
}
