use super::beam_search::BeamSearchParameters;
use super::data_structure::{
    exceed_bound, BfsNode, HashableSignatureVariables, StateInRegistry, SuccessorGenerator,
    TransitionMutex, TransitionWithId,
};
use super::rollout::get_trace;
use super::search::{Parameters, Search, SearchInput, Solution};
use super::util::{print_primal_bound, update_bound_if_better, TimeKeeper};
use dypdl::{variable_type, Model, ReduceFunction, Transition, TransitionInterface};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use rustc_hash::FxHashMap;
use std::cmp;
use std::error::Error;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::marker::PhantomData;
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
/// Type parameter `G` is a type of a function that generates a root node given a state and its cost.
///
/// `parameters.parameters.time_limit` is required in this solver.
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
///     beam_search, BeamSearchParameters, FNode, Lnbs, LnbsParameters, SearchInput,
///     SuccessorGenerator, TransitionWithId,
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
/// let model = Rc::new(model);
///
/// let state = model.target.clone();
/// let cost = 0;
/// let h_evaluator = |_: &_| Some(0);
/// let f_evaluator = |g, h, _: &_| g + h;
/// let primal_bound = None;
/// let generator = SuccessorGenerator::<TransitionWithId>::from_model(model.clone(), false);
/// let t_model = model.clone();
/// let transition_evaluator = move |node: &FNode<_, _>, transition, primal_bound| {
///     node.generate_successor_node(
///         transition,
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
/// let root_generator = |state, cost| {
///     FNode::generate_root_node(
///         state,
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
/// let mut solver = Lnbs::new(generator, beam_search, root_generator, solution, parameters);
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
    N: BfsNode<T, TransitionWithId<V>, K>,
    B: Fn(
        &SearchInput<N, TransitionWithId<V>, D, R>,
        BeamSearchParameters<T>,
    ) -> Solution<T, TransitionWithId<V>>,
    G: Fn(StateInRegistry<K>, T) -> Option<N>,
    V: TransitionInterface + Clone + Default,
    D: Deref<Target = TransitionWithId<V>> + Clone,
    R: Deref<Target = Model> + Clone,
    K: Hash + Eq + Clone + Debug,
{
    generator: SuccessorGenerator<TransitionWithId<V>, D, R>,
    beam_search: B,
    root_generator: G,
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
    solution: Solution<T, TransitionWithId<V>>,
    phantom: PhantomData<(D, R, K)>,
}

impl<T, N, B, G, V, D, R, K> Lnbs<T, N, B, G, V, D, R, K>
where
    T: variable_type::Numeric + Ord + Display,
    <T as str::FromStr>::Err: Debug,
    N: BfsNode<T, TransitionWithId<V>, K> + Clone,
    B: Fn(
        &SearchInput<N, TransitionWithId<V>, D, R>,
        BeamSearchParameters<T>,
    ) -> Solution<T, TransitionWithId<V>>,
    G: Fn(StateInRegistry<K>, T) -> Option<N>,
    V: TransitionInterface + Clone + Default,
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
        generator: SuccessorGenerator<TransitionWithId<V>, D, R>,
        beam_search: B,
        root_generator: G,
        solution: Solution<T, TransitionWithId<V>>,
        parameters: LnbsParameters<T>,
    ) -> Lnbs<T, N, B, G, V, D, R, K> {
        let time_limit = parameters
            .beam_search_parameters
            .parameters
            .time_limit
            .unwrap();
        let mut time_keeper = TimeKeeper::with_time_limit(time_limit);
        let transition_mutex = TransitionMutex::new(&generator.model, generator.backward);

        let max_depth = solution.transitions.len();

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
            generator,
            beam_search,
            root_generator,
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
            solution,
            phantom: PhantomData::default(),
        }
    }

    //// Search for the next solution, returning the solution using `TransitionWithId`.
    pub fn search_inner(&mut self) -> (Solution<T, TransitionWithId<V>>, bool) {
        if self.solution.is_terminated() || self.solution.cost.is_none() {
            return (self.solution.clone(), true);
        }

        self.time_keeper.start();

        let (states, costs): (Vec<_>, Vec<_>) = get_trace(
            &self.generator.model.target,
            T::zero(),
            &self.solution.transitions,
            &self.generator.model,
        )
        .unzip();

        let last = self.depth_arms.last_mut().unwrap();

        if *last > self.solution.transitions.len() {
            *last = self.solution.transitions.len();
        };

        loop {
            let result = self.select_depth();

            if result.is_none() {
                self.time_keeper.stop();

                return (self.solution.clone(), true);
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

            let (start, beam_size) = result.unwrap();

            let prefix = &self.solution.transitions[..start];
            let prefix_cost = if start == 0 {
                T::zero()
            } else {
                costs[start - 1]
            };
            let target = StateInRegistry::from(if start == 0 {
                self.generator.model.target.clone()
            } else {
                states[start - 1].clone()
            });
            let node = (self.root_generator)(target, prefix_cost);
            let suffix = &self.solution.transitions[start + depth..];
            let generator = if self.no_transition_mutex {
                self.generator.clone()
            } else {
                self.transition_mutex
                    .filter_successor_generator(&self.generator, prefix, suffix)
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
                    primal_bound: self.solution.cost,
                    get_all_solutions: false,
                    quiet: true,
                    time_limit: self.time_keeper.remaining_time_limit(),
                    ..Default::default()
                },
            };

            let solution = (self.beam_search)(&input, parameters);

            self.solution.expanded += solution.expanded;
            self.solution.generated += solution.generated;

            let len = self.solution.transitions.len();

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
                if !exceed_bound(&self.generator.model, cost, self.solution.cost) {
                    self.neighborhood_beam_size = FxHashMap::from_iter(
                        self.neighborhood_beam_size
                            .drain()
                            .filter(|(k, _)| k.0 <= start && k.0 + k.1 >= start + depth),
                    );

                    self.depth_exhausted.iter_mut().for_each(|x| *x = false);

                    if Some(cost) == self.solution.best_bound {
                        self.solution.is_optimal = true;
                    }

                    if depth == len && solution.is_optimal {
                        self.solution.is_optimal = true;
                        self.solution.best_bound = solution.cost;
                    }

                    let mut transitions = prefix.to_vec();
                    transitions.extend(solution.transitions.into_iter());
                    self.solution.transitions = transitions;

                    let current_cost = self.solution.cost.unwrap();
                    let reward = (cost - current_cost).to_continuous().abs()
                        / current_cost.to_continuous().abs();
                    let time = (self.time_keeper.elapsed_time() - time_start) / self.time_limit;
                    self.update_bandit(arm, reward, time);

                    self.solution.cost = Some(cost);
                    self.solution.time = self.time_keeper.elapsed_time();

                    if !self.quiet {
                        println!("Depth: {}, #beam: {}", depth, beam_size);
                        print_primal_bound(&self.solution);
                    }

                    self.time_keeper.stop();

                    return (self.solution.clone(), self.solution.is_optimal);
                }
            }

            if start == 0 && depth == len {
                if let Some(bound) = solution.best_bound {
                    update_bound_if_better(
                        &mut self.solution,
                        bound,
                        &self.generator.model,
                        self.quiet,
                    );
                }
            }

            if exhausted && depth == len {
                self.solution.is_optimal = self.solution.cost.is_some();
                self.solution.is_infeasible = self.solution.cost.is_none();
                self.solution.time = self.time_keeper.elapsed_time();

                if self.solution.is_optimal {
                    self.solution.best_bound = self.solution.cost;
                }

                self.time_keeper.stop();

                return (self.solution.clone(), true);
            }

            if solution.time_out {
                self.solution.time = self.time_keeper.elapsed_time();

                self.time_keeper.stop();

                return (self.solution.clone(), true);
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
                    let score = self.rng.gen::<f64>();

                    Some((score, (i, *depth)))
                }
            })
            .max_by(|(a, _), (b, _)| a.total_cmp(b))
            .map(|(_, arm)| arm)
    }

    fn select_start(&mut self, costs: &[T], depth: usize) -> Option<(usize, usize)> {
        let search_non_positive =
            self.has_negative_cost || self.generator.model.reduce_function == ReduceFunction::Max;

        let (weights, starts): (Vec<_>, Vec<_>) = std::iter::once(T::zero())
            .chain(costs.iter().copied())
            .zip(costs[depth - 1..].iter().copied())
            .enumerate()
            .filter_map(|(start, (before, after))| {
                let entry = self
                    .neighborhood_beam_size
                    .entry((start, depth))
                    .or_insert((self.initial_beam_size, false));

                if entry.1 || (!search_non_positive && after <= before) {
                    None
                } else if !search_non_positive && self.use_cost_weight {
                    Some((after - before, (start, entry.0)))
                } else {
                    Some((T::one(), (start, entry.0)))
                }
            })
            .unzip();

        if starts.is_empty() {
            return None;
        }

        let mut weights = weights
            .iter()
            .map(|v| v.to_continuous())
            .collect::<Vec<_>>();

        if self.generator.model.reduce_function == ReduceFunction::Max {
            weights.iter_mut().for_each(|v| *v = 1.0 / *v);
        };

        if self.use_cost_weight {
            weights
                .iter_mut()
                .zip(starts.iter())
                .for_each(|(v, (_, beam_size))| *v /= *beam_size as f64);
        }

        let dist = WeightedIndex::new(weights).unwrap();

        starts[dist.sample(&mut self.rng)].into()
    }
}

impl<T, N, B, G, V, D, R, K> Search<T> for Lnbs<T, N, B, G, V, D, R, K>
where
    T: variable_type::Numeric + Ord + Display,
    <T as str::FromStr>::Err: Debug,
    N: BfsNode<T, TransitionWithId<V>, K> + Clone,
    B: Fn(
        &SearchInput<N, TransitionWithId<V>, D, R>,
        BeamSearchParameters<T>,
    ) -> Solution<T, TransitionWithId<V>>,
    G: Fn(StateInRegistry<K>, T) -> Option<N>,
    V: TransitionInterface + Clone + Default,
    Transition: From<TransitionWithId<V>>,
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
                .map(dypdl::Transition::from)
                .collect(),
            expanded: solution.expanded,
            generated: solution.generated,
            time: solution.time,
            time_out: solution.time_out,
        };

        Ok((solution, is_terminated))
    }
}
