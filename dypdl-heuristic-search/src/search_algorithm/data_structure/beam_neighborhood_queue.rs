use super::beam_neighborhood::BeamNeighborhood;
use super::transition_with_custom_cost::TransitionWithCustomCost;
use dypdl::variable_type::Numeric;
use dypdl::Model;
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::ops::Deref;
use std::rc::Rc;

#[derive(Debug, Default, Clone, Copy)]
pub struct BeamNeighborhoodSize<P>(usize, usize, P);

impl<P: PartialEq> PartialEq for BeamNeighborhoodSize<P> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.2 == other.2
    }
}

impl<P: Eq> Eq for BeamNeighborhoodSize<P> {}

impl<P: Ord> Ord for BeamNeighborhoodSize<P> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.2.cmp(&other.2)
    }
}

impl<P: PartialOrd> PartialOrd for BeamNeighborhoodSize<P> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.2.partial_cmp(&other.2)
    }
}

#[derive(Debug, Clone)]
pub struct PrioritizedBeamNeighborhood<'a, P, T, U, D, R>(BeamNeighborhood<'a, T, U, D, R>, P)
where
    T: Numeric,
    U: Numeric,
    D: Deref<Target = TransitionWithCustomCost> + Clone,
    R: Deref<Target = Model>;

impl<'a, P, T, U, D, R> PartialEq for PrioritizedBeamNeighborhood<'a, P, T, U, D, R>
where
    P: PartialEq,
    T: Numeric,
    U: Numeric,
    D: Deref<Target = TransitionWithCustomCost> + Clone,
    R: Deref<Target = Model>,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

impl<'a, P, T, U, D, R> Eq for PrioritizedBeamNeighborhood<'a, P, T, U, D, R>
where
    P: Eq,
    T: Numeric,
    U: Numeric,
    D: Deref<Target = TransitionWithCustomCost> + Clone,
    R: Deref<Target = Model>,
{
}

impl<'a, P, T, U, D, R> Ord for PrioritizedBeamNeighborhood<'a, P, T, U, D, R>
where
    P: Ord,
    T: Numeric,
    U: Numeric,
    D: Deref<Target = TransitionWithCustomCost> + Clone,
    R: Deref<Target = Model>,
{
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.1.cmp(&other.1)
    }
}

impl<'a, P, T, U, D, R> PartialOrd for PrioritizedBeamNeighborhood<'a, P, T, U, D, R>
where
    P: PartialOrd,
    T: Numeric,
    U: Numeric,
    D: Deref<Target = TransitionWithCustomCost> + Clone,
    R: Deref<Target = Model>,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.1.partial_cmp(&other.1)
    }
}

type Bucket<'a, P, T, U, D, R> = BinaryHeap<Box<PrioritizedBeamNeighborhood<'a, P, T, U, D, R>>>;
type BucketMap<'a, P, T, U, D, R> = FxHashMap<(usize, usize), Bucket<'a, P, T, U, D, R>>;

#[derive(Debug, Default, Clone)]
pub struct BeamNeighborhoodQueue<'a, P, Q, T, U, D = Rc<TransitionWithCustomCost>, R = Rc<Model>>
where
    P: Ord,
    Q: Ord,
    T: Numeric,
    U: Numeric,
    D: Deref<Target = TransitionWithCustomCost> + Clone,
    R: Deref<Target = Model>,
{
    queue: BinaryHeap<BeamNeighborhoodSize<P>>,
    buckets: BucketMap<'a, Q, T, U, D, R>,
    start_depth_to_beam_size: FxHashMap<(usize, usize), usize>,
    start_beam_size_to_depth: FxHashMap<(usize, usize), usize>,
}

impl<'a, P, Q, T, U, D, R> BeamNeighborhoodQueue<'a, P, Q, T, U, D, R>
where
    P: Ord,
    Q: Ord,
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

    pub fn insert(&mut self, neighborhood: BeamNeighborhood<'a, T, U, D, R>, priority: Q) {
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
            .or_insert_with(|| BinaryHeap::with_capacity(1));
        let neighborhood = Box::new(PrioritizedBeamNeighborhood(neighborhood, priority));
        entry.push(neighborhood);
    }

    pub fn contains_size(&self, depth: usize, beam_size: usize) -> bool {
        if let Some(bucket) = self.buckets.get(&(depth, beam_size)) {
            !bucket.is_empty()
        } else {
            false
        }
    }

    pub fn set_priority(&mut self, depth: usize, beam_size: usize, priority: P) {
        let size = BeamNeighborhoodSize(depth, beam_size, priority);
        self.queue.push(size);
    }

    pub fn pop(&mut self) -> Option<BeamNeighborhood<'a, T, U, D, R>> {
        loop {
            let size = self.queue.pop()?;
            let bucket = self.buckets.get_mut(&(size.0, size.1));

            if bucket.is_none() {
                continue;
            }

            let bucket = bucket.unwrap();
            let top = bucket.pop();

            if top.is_none() {
                continue;
            }

            return Some(top.unwrap().0);
        }
    }

    pub fn get_subsuming_neighborhoods(
        &self,
        start: usize,
        depth: usize,
    ) -> Vec<(usize, usize, usize)> {
        let mut result = Vec::default();

        for size in self.queue.iter() {
            if size.0 <= depth {
                continue;
            }

            if let Some(bucket) = self.buckets.get(&(size.0, size.1)) {
                for neighborhood in bucket {
                    if neighborhood.0.includes(start, depth) {
                        result.push((
                            neighborhood.0.start,
                            neighborhood.0.depth,
                            neighborhood.0.beam_size,
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
    fn beam_neighborhood_size_eq() {
        let a = BeamNeighborhoodSize(2, 1, 10);
        let b = BeamNeighborhoodSize(3, 16, 10);
        assert_eq!(a, b);
        assert!(a <= b);
        assert!(b >= a);
    }

    #[test]
    fn beam_neighborhood_size_neq() {
        let a = BeamNeighborhoodSize(2, 1, 10);
        let b = BeamNeighborhoodSize(2, 1, 11);
        assert_ne!(a, b);
        assert!(a < b);
        assert!(a <= b);
        assert!(b > a);
        assert!(b >= a);
    }

    #[test]
    fn prioritized_beam_neighborhood_eq() {
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
        let a = BeamNeighborhood {
            problem: problem.clone(),
            prefix: &[],
            start: 0,
            depth: 2,
            beam_size: 1,
        };
        let a = PrioritizedBeamNeighborhood(a, 2);
        let b = BeamNeighborhood {
            problem,
            prefix: &[],
            start: 1,
            depth: 2,
            beam_size: 1,
        };
        let b = PrioritizedBeamNeighborhood(b, 2);
        assert_eq!(a, b);
        assert!(a <= b);
        assert!(b >= a);
    }

    #[test]
    fn prioritized_beam_neighborhood_neq() {
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
        let a = BeamNeighborhood {
            problem: problem.clone(),
            prefix: &[],
            start: 0,
            depth: 2,
            beam_size: 1,
        };
        let a = PrioritizedBeamNeighborhood(a, 2);
        let b = BeamNeighborhood {
            problem,
            prefix: &[],
            start: 0,
            depth: 2,
            beam_size: 1,
        };
        let b = PrioritizedBeamNeighborhood(b, 3);
        assert_ne!(a, b);
        assert!(a <= b);
        assert!(a < b);
        assert!(b >= a);
        assert!(b > a);
    }

    #[test]
    fn beam_neighborhood_queue_insert() {
        let mut queue = BeamNeighborhoodQueue::<i32, _, _, _>::default();
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
        queue.insert(neighborhood, 10);
        assert!(queue.contains(0, 1, 8));
        assert!(!queue.contains(0, 1, 16));
        assert!(queue.contains(0, 2, 1));
        assert!(queue.contains(0, 2, 2));
        assert!(queue.contains(0, 2, 4));
        assert!(queue.contains(0, 2, 8));
        assert!(!queue.contains(0, 3, 8));
        assert!(!queue.contains(1, 2, 8));

        let neighborhood = BeamNeighborhood {
            problem: problem.clone(),
            prefix: &[],
            start: 0,
            depth: 3,
            beam_size: 8,
        };
        queue.insert(neighborhood, 10);
        assert!(queue.contains(0, 1, 8));
        assert!(!queue.contains(0, 1, 16));
        assert!(queue.contains(0, 2, 1));
        assert!(queue.contains(0, 2, 2));
        assert!(queue.contains(0, 2, 4));
        assert!(queue.contains(0, 2, 8));
        assert!(queue.contains(0, 3, 8));
        assert!(!queue.contains(0, 4, 8));
        assert!(!queue.contains(1, 2, 8));

        let neighborhood = BeamNeighborhood {
            problem: problem.clone(),
            prefix: &[],
            start: 0,
            depth: 2,
            beam_size: 16,
        };
        queue.insert(neighborhood, 10);
        assert!(queue.contains(0, 1, 8));
        assert!(queue.contains(0, 1, 16));
        assert!(!queue.contains(0, 2, 32));
        assert!(queue.contains(0, 2, 1));
        assert!(queue.contains(0, 2, 2));
        assert!(queue.contains(0, 2, 4));
        assert!(queue.contains(0, 2, 8));
        assert!(queue.contains(0, 2, 16));
        assert!(!queue.contains(0, 2, 32));
        assert!(queue.contains(0, 3, 8));
        assert!(!queue.contains(0, 4, 8));
        assert!(!queue.contains(1, 2, 8));
    }

    #[test]
    fn beam_neighborhood_queue_pop() {
        let mut queue = BeamNeighborhoodQueue::<i32, _, _, _>::default();
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

        let neighborhood1 = BeamNeighborhood {
            problem: problem.clone(),
            prefix: &[],
            start: 0,
            depth: 2,
            beam_size: 8,
        };
        queue.insert(neighborhood1.clone(), 10);

        queue.set_priority(2, 8, 3);

        let neighborhood2 = BeamNeighborhood {
            problem: problem.clone(),
            prefix: &[],
            start: 1,
            depth: 2,
            beam_size: 8,
        };
        queue.insert(neighborhood2.clone(), 12);

        let neighborhood3 = BeamNeighborhood {
            problem: problem.clone(),
            prefix: &[],
            start: 0,
            depth: 2,
            beam_size: 16,
        };
        queue.insert(neighborhood3.clone(), 10);

        queue.set_priority(2, 16, 2);

        let neighborhood = queue.pop();
        assert_eq!(neighborhood, Some(neighborhood2));

        queue.set_priority(2, 8, 1);

        let neighborhood = queue.pop();
        assert_eq!(neighborhood, Some(neighborhood3));

        queue.set_priority(2, 16, 0);

        let neighborhood = queue.pop();
        assert_eq!(neighborhood, Some(neighborhood1));

        queue.set_priority(2, 8, 0);

        let neighborhood = queue.pop();
        assert_eq!(neighborhood, None);
    }

    #[test]
    fn beam_neighborhood_queue_get_subsuming_neighborhoods_no_priority() {
        let mut queue = BeamNeighborhoodQueue::<i32, _, _, _>::default();
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
        queue.insert(neighborhood, 10);
        let neighborhood = BeamNeighborhood {
            problem: problem.clone(),
            prefix: &[],
            start: 1,
            depth: 3,
            beam_size: 8,
        };
        queue.insert(neighborhood, 12);
        let neighborhood = BeamNeighborhood {
            problem: problem.clone(),
            prefix: &[],
            start: 0,
            depth: 2,
            beam_size: 16,
        };
        queue.insert(neighborhood, 10);

        assert_eq!(queue.get_subsuming_neighborhoods(1, 2), vec![]);
    }

    #[test]
    fn beam_neighborhood_queue_get_subsuming_neighborhoods_with_priority() {
        let mut queue = BeamNeighborhoodQueue::<i32, _, _, _>::default();
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
        queue.insert(neighborhood, 10);
        queue.set_priority(4, 16, 0);
        let neighborhood = BeamNeighborhood {
            problem: problem.clone(),
            prefix: &[],
            start: 1,
            depth: 3,
            beam_size: 8,
        };
        queue.insert(neighborhood, 12);
        queue.set_priority(3, 8, 0);
        let neighborhood = BeamNeighborhood {
            problem: problem.clone(),
            prefix: &[],
            start: 0,
            depth: 2,
            beam_size: 16,
        };
        queue.insert(neighborhood, 10);
        queue.set_priority(2, 16, 0);

        assert_eq!(
            queue.get_subsuming_neighborhoods(1, 2),
            vec![(0, 4, 16), (1, 3, 8)]
        );
    }
}
