use crate::Parameters;

use super::beam_search::{beam_search, BeamSearchParameters};
use super::cabs::Cabs;
use super::data_structure::beam::{BeamInterface, InformationInBeam};
use super::data_structure::state_registry::StateInRegistry;
use super::data_structure::{
    exceed_bound, BeamNeighborhood, BeamSearchProblemInstance, CustomCostNodeInterface,
    SuccessorGenerator, TransitionConstraints, TransitionWithCustomCost,
};
use super::rollout::get_trace;
use super::search::{Search, Solution};
use super::util;
use dypdl::{variable_type, ReduceFunction};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use rustc_hash::FxHashMap;
use std::error::Error;
use std::fmt;
use std::marker::PhantomData;
use std::rc::Rc;
use std::str;

/// Parameters for LNBS
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LnbsParameters {
    /// Initial beam size for reconstruction beam search.
    pub initial_beam_size: usize,
    /// Cost can be negative.
    pub has_negative_cost: bool,
    /// Do not bias the weight by the cost difference.
    pub no_cost_weight: bool,
    /// Randomly select an arm.
    pub no_bandit: bool,
    /// Do not use transition constraints.
    pub no_transition_constraints: bool,
    /// Random seed.
    pub seed: u64,
}

/// Large Neighborhood Beam Search (LNBS).
///
/// It iterates beam search with exponentially increasing beam width.
///
/// This solver uses forward search based on the shortest path problem.
/// It only works with problems where the cost expressions are in the form of `cost + w`, `cost * w`, `max(cost, w)`, or `min(cost, w)`
/// where `cost` is `IntegerExpression::Cost`or `ContinuousExpression::Cost` and `w` is a numeric expression independent of `cost`.
///
/// It uses `h_evaluator` and `f_evaluator` for pruning.
/// If `h_evaluator` returns `None`, the state is pruned.
/// If `parameters.f_pruning` and `f_evaluator` returns a value that exceeds the f bound, the state is pruned.
///
/// Beam search searches layer by layer, where the i th layer contains states that can be reached with i transitions.
/// By default, this solver only keeps states in the current layer to check for duplicates.
/// If `parameters.keep_all_layers` is `true`, this solver keeps states in all layers to check for duplicates.
///
/// `parameters.parameters.time_limit` is required in this solver.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::search_algorithm::data_structure::beam::Beam;
/// use dypdl_heuristic_search::search_algorithm::{
///     BeamSearchParameters, Lnbs, LnbsParameters, Search,
/// };
/// use dypdl_heuristic_search::search_algorithm::data_structure::BeamSearchNode;
/// use dypdl_heuristic_search::search_algorithm::data_structure::successor_generator::{
///     SuccessorGenerator
/// };
/// use dypdl_heuristic_search::Parameters;
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
/// let h_evaluator = |_: &_, _: &_| Some(0);
/// let f_evaluator = |g, h, _: &_, _: &_| g + h;
///
/// let model = Rc::new(model);
/// let generator = SuccessorGenerator::from_model_without_custom_cost(model.clone(), false);
/// let beam_constructor = |beam_size| Beam::<_, _, BeamSearchNode<_, _>>::new(beam_size);
/// let lnbs_parameters = LnbsParameters {
///     initial_beam_size: 1,
///     has_negative_cost: false,
///     no_cost_weight: false,
///     no_bandit: false,
///     no_transition_constraints: false,
///     seed: 0,
/// };
/// let parameters = BeamSearchParameters {
///     beam_size: 1,
///     parameters: Parameters {
///         time_limit: Some(1800.0),
///         ..Default::default()
///     },
///     ..Default::default()
/// };
///
/// let mut solver = Lnbs::new(
///     generator, h_evaluator, f_evaluator, beam_constructor, lnbs_parameters, parameters
/// );
/// let solution = solver.search().unwrap();
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions, vec![increment]);
/// assert!(!solution.is_infeasible);
/// ```
pub struct Lnbs<T, I, B, C, H, F>
where
    T: variable_type::Numeric + fmt::Display + Ord,
    <T as str::FromStr>::Err: fmt::Debug,
    I: InformationInBeam<T, T> + CustomCostNodeInterface<T, T>,
    B: BeamInterface<T, T, I>,
    C: Fn(usize) -> B,
    H: Fn(&StateInRegistry, &dypdl::Model) -> Option<T>,
    F: Fn(T, T, &StateInRegistry, &dypdl::Model) -> T,
{
    generator: SuccessorGenerator<TransitionWithCustomCost, Rc<TransitionWithCustomCost>>,
    transition_constraints: TransitionConstraints,
    h_evaluator: H,
    f_evaluator: F,
    beam_constructor: C,
    bandit_parameters: LnbsParameters,
    maximize: bool,
    f_pruning: bool,
    keep_all_layers: bool,
    primal_bound: Option<T>,
    quiet: bool,
    beam_size: usize,
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
    time_keeper: util::TimeKeeper,
    solution: Solution<T, TransitionWithCustomCost>,
    phantom: PhantomData<I>,
}

impl<T, I, B, C, H, F> Lnbs<T, I, B, C, H, F>
where
    T: variable_type::Numeric + fmt::Display + Ord,
    <T as str::FromStr>::Err: fmt::Debug,
    I: InformationInBeam<T, T> + CustomCostNodeInterface<T, T>,
    B: BeamInterface<T, T, I>,
    C: Fn(usize) -> B,
    H: Fn(&StateInRegistry, &dypdl::Model) -> Option<T>,
    F: Fn(T, T, &StateInRegistry, &dypdl::Model) -> T,
{
    /// Create a new LNBS solver.
    pub fn new(
        generator: SuccessorGenerator<TransitionWithCustomCost, Rc<TransitionWithCustomCost>>,
        h_evaluator: H,
        f_evaluator: F,
        beam_constructor: C,
        lnbs_parameters: LnbsParameters,
        parameters: BeamSearchParameters<T, T>,
    ) -> Lnbs<T, I, B, C, H, F> {
        let time_limit = parameters.parameters.time_limit.unwrap();
        let time_keeper = util::TimeKeeper::with_time_limit(time_limit);
        let transition_constraints =
            TransitionConstraints::new(&generator.model, generator.backward);
        let seed = lnbs_parameters.seed;
        Lnbs {
            generator,
            transition_constraints,
            h_evaluator,
            f_evaluator,
            beam_constructor,
            bandit_parameters: lnbs_parameters,
            maximize: parameters.maximize,
            f_pruning: parameters.f_pruning,
            keep_all_layers: parameters.keep_all_layers,
            primal_bound: parameters.parameters.primal_bound,
            quiet: parameters.parameters.quiet,
            beam_size: parameters.beam_size,
            neighborhood_beam_size: FxHashMap::default(),
            depth_arms: Vec::default(),
            reward_mean: Vec::default(),
            time_mean: Vec::default(),
            trials: Vec::default(),
            total_trials: 0.0,
            lambda: None,
            time_limit,
            depth_exhausted: Vec::default(),
            rng: Pcg64Mcg::seed_from_u64(seed),
            time_keeper,
            solution: Solution::default(),
            phantom: PhantomData::default(),
        }
    }

    //// Search for the next solution, returning the solution using `TransitionWithCustomCost`.
    pub fn search_inner(&mut self) -> (Solution<T, TransitionWithCustomCost>, bool) {
        if self.solution.is_terminated() {
            return (self.solution.clone(), true);
        }

        if self.solution.cost.is_none() {
            return self.search_initial_solution();
        }

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
            let (arm, depth) = self.select_depth();
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
            let suffix = &self.solution.transitions[start + depth..];
            let generator = if self.bandit_parameters.no_transition_constraints {
                self.generator.clone()
            } else {
                self.transition_constraints.filter_successor_generator(
                    &self.generator,
                    prefix,
                    suffix,
                )
            };
            let problem = BeamSearchProblemInstance {
                target,
                generator,
                cost: prefix_cost,
                g: prefix_cost,
                solution_suffix: suffix,
            };
            let neighborhood = BeamNeighborhood {
                problem,
                prefix,
                start,
                depth,
                beam_size,
            };

            let parameters = BeamSearchParameters {
                beam_size: neighborhood.beam_size,
                maximize: self.maximize,
                f_pruning: self.f_pruning,
                f_bound: self.solution.cost,
                keep_all_layers: self.keep_all_layers,
                parameters: Parameters {
                    primal_bound: self.solution.cost,
                    get_all_solutions: false,
                    quiet: true,
                    time_limit: self.time_keeper.remaining_time_limit(),
                },
            };

            let solution = beam_search(
                &neighborhood.problem,
                &self.beam_constructor,
                &self.h_evaluator,
                &self.f_evaluator,
                parameters,
            );

            self.solution.expanded += solution.expanded;
            self.solution.generated += solution.generated;

            let len = self.solution.transitions.len();

            let exhausted = solution.is_optimal || solution.is_infeasible;
            self.neighborhood_beam_size
                .insert((start, depth), (2 * neighborhood.beam_size, exhausted));

            if let Some(cost) = solution.cost {
                if !exceed_bound(&self.generator.model, cost, self.solution.cost) {
                    self.neighborhood_beam_size = FxHashMap::from_iter(
                        self.neighborhood_beam_size
                            .drain()
                            .filter(|(k, _)| k.0 <= start && k.0 + k.1 >= start + depth),
                    );

                    self.depth_exhausted.iter_mut().for_each(|x| *x = false);

                    if !self.quiet {
                        println!(
                            "New primal bound: {}, depth: {}, #beam: {}",
                            cost, neighborhood.depth, neighborhood.beam_size
                        );
                    }

                    if depth == len {
                        self.solution.is_optimal = solution.is_optimal;
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

                    return (self.solution.clone(), self.solution.is_optimal);
                }
            }

            if exhausted && depth == len {
                self.solution.is_optimal = self.solution.cost.is_some();
                self.solution.is_infeasible = self.solution.cost.is_none();
                self.solution.time = self.time_keeper.elapsed_time();

                return (self.solution.clone(), true);
            }

            if solution.time_out {
                self.solution.time = self.time_keeper.elapsed_time();

                return (self.solution.clone(), true);
            }

            let time = (self.time_keeper.elapsed_time() - time_start) / self.time_limit;
            self.update_bandit(arm, 0.0, time);
        }
    }

    fn search_initial_solution(&mut self) -> (Solution<T, TransitionWithCustomCost>, bool) {
        let parameters = BeamSearchParameters {
            maximize: self.maximize,
            f_pruning: self.f_pruning,
            f_bound: self.primal_bound,
            keep_all_layers: self.keep_all_layers,
            beam_size: self.beam_size,
            parameters: Parameters {
                primal_bound: self.primal_bound,
                get_all_solutions: false,
                quiet: true,
                time_limit: self.time_keeper.remaining_time_limit(),
            },
        };

        let mut cabs = Cabs::new(
            self.generator.clone(),
            &self.h_evaluator,
            &self.f_evaluator,
            &self.beam_constructor,
            parameters,
        );

        let (solution, is_terminated) = cabs.search_inner();
        self.solution = solution;

        if let Some(cost) = self.solution.cost {
            if !self.quiet {
                println!("Initial primal bound: {}", cost);
            }

            let max_depth = self.solution.transitions.len();

            if max_depth > 0 {
                let log = (usize::BITS - 1) - max_depth.leading_zeros();
                self.depth_arms = (1..=log).map(|i| 2usize.pow(i)).collect();

                if let Some(last) = self.depth_arms.last() {
                    if *last < max_depth {
                        self.depth_arms.push(max_depth);
                    }
                } else {
                    self.depth_arms.push(max_depth);
                }

                self.reward_mean = vec![0.0; self.depth_arms.len()];
                self.time_mean = vec![0.0; self.depth_arms.len()];
                self.trials = vec![0.0; self.depth_arms.len()];
                self.depth_exhausted = vec![false; self.depth_arms.len()];
            }
        }

        (self.solution.clone(), is_terminated)
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

    fn select_depth(&mut self) -> (usize, usize) {
        if self.bandit_parameters.no_bandit {
            self.select_random()
        } else {
            self.select_ucb()
        }
    }

    fn select_ucb(&mut self) -> (usize, usize) {
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
            .unwrap()
            .1
    }

    fn select_random(&mut self) -> (usize, usize) {
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
            .unwrap()
            .1
    }

    fn select_start(&mut self, costs: &[T], depth: usize) -> Option<(usize, usize)> {
        let search_non_positive = self.bandit_parameters.has_negative_cost
            || self.generator.model.reduce_function == ReduceFunction::Max;

        let (mut weights, starts): (Vec<_>, Vec<_>) = std::iter::once(T::zero())
            .chain(costs.iter().copied())
            .zip(costs[depth - 1..].iter().copied())
            .enumerate()
            .filter_map(|(start, (before, after))| {
                let entry = self
                    .neighborhood_beam_size
                    .entry((start, depth))
                    .or_insert((self.bandit_parameters.initial_beam_size, false));

                if entry.1 || (!search_non_positive && after <= before) {
                    None
                } else if self.bandit_parameters.no_cost_weight {
                    Some((T::one(), (start, entry.0)))
                } else {
                    Some((after - before, (start, entry.0)))
                }
            })
            .unzip();

        if starts.is_empty() {
            return None;
        }

        if search_non_positive {
            let min = *weights.iter().min().unwrap();

            if min <= T::zero() {
                weights.iter_mut().for_each(|v| *v = *v - min + T::one());
            }
        }

        let mut weights = weights
            .iter()
            .map(|v| v.to_continuous())
            .collect::<Vec<_>>();

        if self.generator.model.reduce_function == ReduceFunction::Max {
            weights.iter_mut().for_each(|v| *v = 1.0 / *v);
        };

        if !self.bandit_parameters.no_cost_weight {
            weights
                .iter_mut()
                .zip(starts.iter())
                .for_each(|(v, (_, beam_size))| *v /= *beam_size as f64);
        }

        let dist = WeightedIndex::new(weights).unwrap();

        starts[dist.sample(&mut self.rng)].into()
    }
}

impl<T, I, B, C, H, F> Search<T> for Lnbs<T, I, B, C, H, F>
where
    T: variable_type::Numeric + fmt::Display + Ord,
    <T as str::FromStr>::Err: fmt::Debug,
    I: InformationInBeam<T, T> + CustomCostNodeInterface<T, T>,
    B: BeamInterface<T, T, I>,
    C: Fn(usize) -> B,
    H: Fn(&StateInRegistry, &dypdl::Model) -> Option<T>,
    F: Fn(T, T, &StateInRegistry, &dypdl::Model) -> T,
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
