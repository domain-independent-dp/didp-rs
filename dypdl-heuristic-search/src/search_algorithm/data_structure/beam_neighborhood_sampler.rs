use super::beam_neighborhood::BeamNeighborhood;
use super::transition_with_custom_cost::TransitionWithCustomCost;
use dypdl::variable_type::Numeric;
use dypdl::Model;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rustc_hash::FxHashMap;
use std::mem;
use std::ops::Deref;
use std::rc::Rc;

type Bucket<'a, T, U, D, R> = (Vec<Box<BeamNeighborhood<'a, T, U, D, R>>>, Vec<f64>);
type BucketMap<'a, T, U, D, R> = FxHashMap<(usize, usize), Bucket<'a, T, U, D, R>>;

#[derive(Debug, Default, Clone)]
pub struct BeamNeighborhoodSampler<'a, T, U, D = Rc<TransitionWithCustomCost>, R = Rc<Model>>
where
    T: Numeric,
    U: Numeric,
    D: Deref<Target = TransitionWithCustomCost> + Clone,
    R: Deref<Target = Model>,
{
    sizes: Vec<(usize, usize)>,
    size_weights: Vec<f64>,
    buckets: BucketMap<'a, T, U, D, R>,
    start_depth_to_beam_size: FxHashMap<(usize, usize), usize>,
    start_beam_size_to_depth: FxHashMap<(usize, usize), usize>,
}

impl<'a, T, U, D, R> BeamNeighborhoodSampler<'a, T, U, D, R>
where
    T: Numeric,
    U: Numeric,
    D: Deref<Target = TransitionWithCustomCost> + Clone + From<TransitionWithCustomCost>,
    R: Deref<Target = Model> + Clone,
{
    pub fn contains(&self, start: usize, depth: usize, beam_size: usize) -> bool {
        if let Some(other) = self.start_depth_to_beam_size.get(&(start, depth)) {
            return beam_size <= *other;
        }

        if let Some(other) = self.start_beam_size_to_depth.get(&(start, beam_size)) {
            return depth <= *other;
        }

        false
    }

    pub fn insert(&mut self, neighborhood: BeamNeighborhood<'a, T, U, D, R>, priority: f64) {
        self.start_depth_to_beam_size
            .entry((neighborhood.start, neighborhood.depth))
            .and_modify(|beam_size| {
                if *beam_size < neighborhood.beam_size {
                    *beam_size = neighborhood.beam_size;
                }
            })
            .or_insert(neighborhood.beam_size);
        self.start_beam_size_to_depth
            .entry((neighborhood.start, neighborhood.beam_size))
            .and_modify(|depth| {
                if *depth < neighborhood.depth {
                    *depth = neighborhood.depth;
                }
            })
            .or_insert(neighborhood.depth);

        let entry = self
            .buckets
            .entry((neighborhood.depth, neighborhood.beam_size))
            .or_insert_with(|| (Vec::with_capacity(1), Vec::with_capacity(1)));
        entry.0.push(Box::new(neighborhood));
        entry.1.push(priority);
    }

    pub fn contains_size(&self, depth: usize, beam_size: usize) -> bool {
        if let Some(bucket) = self.buckets.get(&(depth, beam_size)) {
            !bucket.0.is_empty()
        } else {
            false
        }
    }

    pub fn set_priority(&mut self, depth: usize, beam_size: usize, priority: f64) {
        self.sizes.push((depth, beam_size));
        self.size_weights.push(priority);
    }

    pub fn pop<V: Rng>(&mut self, rng: &mut V) -> Option<BeamNeighborhood<'a, T, U, D, R>> {
        loop {
            if self.sizes.is_empty() {
                return None;
            }

            let dist = WeightedIndex::new(self.size_weights.iter()).unwrap();
            let index = dist.sample(rng);
            let size = self.sizes[index];

            let last = self.sizes.pop().unwrap();
            let last_weight = self.size_weights.pop().unwrap();

            if index < self.sizes.len() {
                self.sizes[index] = last;
                self.size_weights[index] = last_weight;
            }

            let bucket = self.buckets.get_mut(&(size.0, size.1));

            if bucket.is_none() {
                continue;
            }

            let (bucket, weights) = bucket.unwrap();

            if bucket.is_empty() {
                continue;
            }

            let dist = WeightedIndex::new(weights.iter()).unwrap();
            let index = dist.sample(rng);
            let mut neighborhood = bucket.pop().unwrap();
            let last_weight = weights.pop().unwrap();

            if index < bucket.len() {
                mem::swap(&mut bucket[index], &mut neighborhood);
                weights[index] = last_weight;
            }

            return Some(*neighborhood);
        }
    }

    pub fn get_subsuming_neighborhoods(
        &self,
        start: usize,
        depth: usize,
    ) -> Vec<(usize, usize, usize)> {
        let mut result = Vec::default();

        for size in self.sizes.iter() {
            if size.0 <= depth {
                continue;
            }

            if let Some((bucket, _)) = self.buckets.get(&(size.0, size.1)) {
                for neighborhood in bucket {
                    if neighborhood.includes(start, depth) {
                        result.push((
                            neighborhood.start,
                            neighborhood.depth,
                            neighborhood.beam_size,
                        ));
                    }
                }
            }
        }

        result.sort();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::super::successor_generator::SuccessorGenerator;
    use super::super::BeamSearchProblemInstance;
    use dypdl::Transition;

    use super::*;

    #[test]
    fn beam_neighborhood_sampler_insert() {
        let mut sampler = BeamNeighborhoodSampler::<i32, _, _, _>::default();
        let model = Rc::new(Model {
            forward_transitions: vec![Transition::default(), Transition::default()],
            ..Default::default()
        });
        let generator =
            SuccessorGenerator::<TransitionWithCustomCost>::from_model_without_custom_cost(
                model, false,
            );
        let problem = BeamSearchProblemInstance {
            target: generator.model.target.clone().into(),
            generator,
            cost: 0,
            g: 0,
            solution_suffix: &[],
        };

        let neighborhood = BeamNeighborhood {
            problem: problem.clone(),
            prefix: &[],
            start: 0,
            depth: 2,
            beam_size: 8,
        };
        sampler.insert(neighborhood, 10.0);
        assert!(sampler.contains(0, 1, 8));
        assert!(!sampler.contains(0, 1, 16));
        assert!(sampler.contains(0, 2, 1));
        assert!(sampler.contains(0, 2, 2));
        assert!(sampler.contains(0, 2, 4));
        assert!(sampler.contains(0, 2, 8));
        assert!(!sampler.contains(0, 3, 8));
        assert!(!sampler.contains(1, 2, 8));

        let neighborhood = BeamNeighborhood {
            problem: problem.clone(),
            prefix: &[],
            start: 0,
            depth: 3,
            beam_size: 8,
        };
        sampler.insert(neighborhood, 10.0);
        assert!(sampler.contains(0, 1, 8));
        assert!(!sampler.contains(0, 1, 16));
        assert!(sampler.contains(0, 2, 1));
        assert!(sampler.contains(0, 2, 2));
        assert!(sampler.contains(0, 2, 4));
        assert!(sampler.contains(0, 2, 8));
        assert!(sampler.contains(0, 3, 8));
        assert!(!sampler.contains(0, 4, 8));
        assert!(!sampler.contains(1, 2, 8));

        let neighborhood = BeamNeighborhood {
            problem: problem.clone(),
            prefix: &[],
            start: 0,
            depth: 2,
            beam_size: 16,
        };
        sampler.insert(neighborhood, 10.0);
        assert!(sampler.contains(0, 1, 8));
        assert!(sampler.contains(0, 1, 16));
        assert!(!sampler.contains(0, 2, 32));
        assert!(sampler.contains(0, 2, 1));
        assert!(sampler.contains(0, 2, 2));
        assert!(sampler.contains(0, 2, 4));
        assert!(sampler.contains(0, 2, 8));
        assert!(sampler.contains(0, 2, 16));
        assert!(!sampler.contains(0, 2, 32));
        assert!(sampler.contains(0, 3, 8));
        assert!(!sampler.contains(0, 4, 8));
        assert!(!sampler.contains(1, 2, 8));
    }

    #[test]
    fn beam_neighborhood_sampler_get_subsuming_neighborhoods_no_priority() {
        let mut sampler = BeamNeighborhoodSampler::<i32, _, _, _>::default();
        let model = Rc::new(Model {
            forward_transitions: vec![Transition::default(), Transition::default()],
            ..Default::default()
        });
        let generator =
            SuccessorGenerator::<TransitionWithCustomCost>::from_model_without_custom_cost(
                model, false,
            );
        let problem = BeamSearchProblemInstance {
            target: generator.model.target.clone().into(),
            generator,
            cost: 0,
            g: 0,
            solution_suffix: &[],
        };

        let neighborhood = BeamNeighborhood {
            problem: problem.clone(),
            prefix: &[],
            start: 0,
            depth: 4,
            beam_size: 16,
        };
        sampler.insert(neighborhood, 10.0);
        let neighborhood = BeamNeighborhood {
            problem: problem.clone(),
            prefix: &[],
            start: 1,
            depth: 3,
            beam_size: 8,
        };
        sampler.insert(neighborhood, 12.0);
        let neighborhood = BeamNeighborhood {
            problem: problem.clone(),
            prefix: &[],
            start: 0,
            depth: 2,
            beam_size: 16,
        };
        sampler.insert(neighborhood, 10.0);

        assert_eq!(sampler.get_subsuming_neighborhoods(1, 2), vec![]);
    }

    #[test]
    fn beam_neighborhood_sampler_get_subsuming_neighborhoods_with_priority() {
        let mut sampler = BeamNeighborhoodSampler::<i32, _, _, _>::default();
        let model = Rc::new(Model {
            forward_transitions: vec![Transition::default(), Transition::default()],
            ..Default::default()
        });
        let generator =
            SuccessorGenerator::<TransitionWithCustomCost>::from_model_without_custom_cost(
                model, false,
            );
        let problem = BeamSearchProblemInstance {
            target: generator.model.target.clone().into(),
            generator,
            cost: 0,
            g: 0,
            solution_suffix: &[],
        };

        let neighborhood = BeamNeighborhood {
            problem: problem.clone(),
            prefix: &[],
            start: 0,
            depth: 4,
            beam_size: 16,
        };
        sampler.insert(neighborhood, 10.0);
        sampler.set_priority(4, 16, 0.0);
        let neighborhood = BeamNeighborhood {
            problem: problem.clone(),
            prefix: &[],
            start: 1,
            depth: 3,
            beam_size: 8,
        };
        sampler.insert(neighborhood, 12.0);
        sampler.set_priority(3, 8, 0.0);
        let neighborhood = BeamNeighborhood {
            problem: problem.clone(),
            prefix: &[],
            start: 0,
            depth: 2,
            beam_size: 16,
        };
        sampler.insert(neighborhood, 10.0);
        sampler.set_priority(2, 16, 0.0);

        assert_eq!(
            sampler.get_subsuming_neighborhoods(1, 2),
            vec![(0, 4, 16), (1, 3, 8)]
        );
    }
}
