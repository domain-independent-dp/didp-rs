use super::beam_search_problem_instance::BeamSearchProblemInstance;
use super::state_registry::StateInRegistry;
use super::successor_generator::SuccessorGenerator;
use super::transition_constraint::TransitionConstraints;
use super::transition_with_custom_cost::TransitionWithCustomCost;
use dypdl::variable_type::Numeric;
use dypdl::{Model, TransitionInterface};
use std::ops::Deref;
use std::rc::Rc;

/// Neighborhood for beam search.
#[derive(Debug, PartialEq, Clone)]
pub struct BeamNeighborhood<'a, T, U, D = Rc<TransitionWithCustomCost>, R = Rc<Model>>
where
    T: Numeric,
    U: Numeric,
    D: Deref<Target = TransitionWithCustomCost> + Clone,
    R: Deref<Target = Model>,
{
    /// Problem.
    pub problem:
        BeamSearchProblemInstance<'a, T, U, StateInRegistry, TransitionWithCustomCost, D, R>,
    /// Prefix.
    pub prefix: &'a [TransitionWithCustomCost],
    /// Start point.
    pub start: usize,
    /// Depth:
    pub depth: usize,
    /// Beam size;
    pub beam_size: usize,
}

/// Iterator for BeamNeighborhood.
pub struct BeamNeighborhoodIter<'a, T, U, D = Rc<TransitionWithCustomCost>, R = Rc<Model>>
where
    T: Numeric,
    U: Numeric,
    D: Deref<Target = TransitionWithCustomCost> + Clone + From<TransitionWithCustomCost>,
    R: Deref<Target = Model>,
{
    transitions: &'a [TransitionWithCustomCost],
    generator: &'a SuccessorGenerator<TransitionWithCustomCost, D, R>,
    transition_constraints: &'a TransitionConstraints,
    item: Option<BeamNeighborhood<'a, T, U, D, R>>,
}

impl<'a, T, U, D, R> Iterator for BeamNeighborhoodIter<'a, T, U, D, R>
where
    T: Numeric,
    U: Numeric,
    D: Deref<Target = TransitionWithCustomCost> + Clone + From<TransitionWithCustomCost>,
    R: Deref<Target = Model> + Clone,
{
    type Item = BeamNeighborhood<'a, T, U, D, R>;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.item.clone();

        if let Some(item) = self.item.as_ref() {
            if item.start + item.depth < self.transitions.len() {
                let transition = &self.transitions[item.start];
                let prefix = &self.transitions[..item.start + 1];
                let state = &item.problem.target;
                let cost = item.problem.cost;
                let g = item.problem.g;
                let registry = &item.problem.generator.model.table_registry;

                let cost = transition.eval_cost(cost, state, registry);
                let g = transition.custom_cost.eval_cost(g, state, registry);
                let target = transition.apply(state, registry);
                let solution_suffix = &item.problem.solution_suffix[1..];
                let generator = self.transition_constraints.filter_successor_generator(
                    self.generator,
                    prefix,
                    solution_suffix,
                );
                let problem = BeamSearchProblemInstance {
                    target,
                    generator,
                    cost,
                    g,
                    solution_suffix,
                };
                self.item = Some(BeamNeighborhood {
                    problem,
                    prefix,
                    start: item.start + 1,
                    depth: item.depth,
                    beam_size: item.beam_size,
                });
            } else {
                self.item = None;
            }
        }

        result
    }
}

impl<'a, T, U, D, R> BeamNeighborhood<'a, T, U, D, R>
where
    T: Numeric,
    U: Numeric,
    D: Deref<Target = TransitionWithCustomCost> + Clone + From<TransitionWithCustomCost>,
    R: Deref<Target = Model> + Clone,
{
    /// Returns an iterator of neighborhoods of a sequence of transitions given depth, beam size, successor generator, and transition constraints.
    pub fn generate_neighborhoods(
        transitions: &'a [TransitionWithCustomCost],
        depth: usize,
        beam_size: usize,
        generator: &'a SuccessorGenerator<TransitionWithCustomCost, D, R>,
        transition_constraints: &'a TransitionConstraints,
    ) -> BeamNeighborhoodIter<'a, T, U, D, R> {
        let solution_suffix = &transitions[depth..];
        let problem = BeamSearchProblemInstance {
            target: StateInRegistry::from(generator.model.target.clone()),
            generator: transition_constraints.filter_successor_generator(
                generator,
                &[],
                solution_suffix,
            ),
            cost: T::zero(),
            g: U::zero(),
            solution_suffix,
        };
        let item = Some(BeamNeighborhood {
            problem,
            prefix: &[],
            start: 0,
            depth,
            beam_size,
        });

        BeamNeighborhoodIter {
            generator,
            transition_constraints,
            transitions,
            item,
        }
    }

    /// Generate a neighborhood with a different beam size.
    pub fn generate_with_different_width(self, beam_size: usize) -> Self {
        Self {
            problem: self.problem,
            prefix: self.prefix,
            start: self.start,
            depth: self.depth,
            beam_size,
        }
    }

    /// Generate a neighborhood with a increased depth.
    pub fn generate_deeper(
        self,
        increment: usize,
        beam_size: usize,
        generator: &SuccessorGenerator<TransitionWithCustomCost, D, R>,
        transition_constraints: &TransitionConstraints,
    ) -> Self {
        let solution_suffix = &self.problem.solution_suffix[increment..];
        let problem = BeamSearchProblemInstance {
            target: self.problem.target,
            cost: self.problem.cost,
            g: self.problem.g,
            generator: transition_constraints.filter_successor_generator(
                generator,
                self.prefix,
                solution_suffix,
            ),
            solution_suffix,
        };

        Self {
            problem,
            prefix: self.prefix,
            start: self.start,
            depth: self.depth + increment,
            beam_size,
        }
    }

    /// Returns if given interval is included.
    pub fn includes(&self, start: usize, depth: usize) -> bool {
        self.start <= start && self.start + self.depth >= start + depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::expression::*;
    use dypdl::prelude::*;
    use dypdl::{BaseCase, Effect, GroundedCondition};
    use rustc_hash::FxHashMap;

    #[test]
    fn generate_neighborhoods_with_two_one() {
        let model = Rc::new(Model {
            base_cases: vec![BaseCase::new(vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Ge,
                    Box::new(IntegerExpression::Variable(0)),
                    Box::new(IntegerExpression::Constant(2)),
                ),
                ..Default::default()
            }])],
            target: State {
                signature_variables: SignatureVariables {
                    set_variables: vec![{
                        let mut set = Set::with_capacity(3);
                        set.insert(0);
                        set.insert(1);
                        set.insert(2);
                        set
                    }],
                    ..Default::default()
                },
                ..Default::default()
            },
            state_metadata: StateMetadata {
                object_type_names: vec![String::from("object")],
                name_to_object_type: {
                    let mut map = FxHashMap::default();
                    map.insert(String::from("object"), 0);
                    map
                },
                object_numbers: vec![3],
                set_variable_names: vec![String::from("v")],
                name_to_set_variable: {
                    let mut map = FxHashMap::default();
                    map.insert(String::from("v"), 0);
                    map
                },
                set_variable_to_object: vec![0],
                ..Default::default()
            },
            forward_transitions: vec![
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Set(Box::new(SetCondition::IsIn(
                            ElementExpression::Constant(0),
                            SetExpression::Reference(ReferenceExpression::Variable(0)),
                        ))),
                        ..Default::default()
                    }],
                    effect: Effect {
                        set_effects: vec![(
                            0,
                            SetExpression::SetElementOperation(
                                SetElementOperator::Remove,
                                ElementExpression::Constant(0),
                                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                                    0,
                                ))),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::Constant(1)),
                    ..Default::default()
                },
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Set(Box::new(SetCondition::IsIn(
                            ElementExpression::Constant(1),
                            SetExpression::Reference(ReferenceExpression::Variable(0)),
                        ))),
                        ..Default::default()
                    }],
                    effect: Effect {
                        set_effects: vec![(
                            0,
                            SetExpression::SetElementOperation(
                                SetElementOperator::Remove,
                                ElementExpression::Constant(1),
                                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                                    0,
                                ))),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::Constant(2)),
                    ..Default::default()
                },
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Set(Box::new(SetCondition::IsIn(
                            ElementExpression::Constant(2),
                            SetExpression::Reference(ReferenceExpression::Variable(0)),
                        ))),
                        ..Default::default()
                    }],
                    effect: Effect {
                        set_effects: vec![(
                            0,
                            SetExpression::SetElementOperation(
                                SetElementOperator::Remove,
                                ElementExpression::Constant(2),
                                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                                    0,
                                ))),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::Constant(3)),
                    ..Default::default()
                },
            ],
            ..Default::default()
        });
        let generator =
            SuccessorGenerator::<TransitionWithCustomCost>::from_model_without_custom_cost(
                model.clone(),
                false,
            );
        let transition_constraints = &TransitionConstraints::new(&model, false);
        let transitions: Vec<TransitionWithCustomCost> = generator
            .transitions
            .iter()
            .map(|t: &_| (**t).clone())
            .collect();
        let mut iter = BeamNeighborhood::generate_neighborhoods(
            &transitions[..],
            2,
            1,
            &generator,
            transition_constraints,
        );
        assert_eq!(
            iter.next(),
            Some(BeamNeighborhood {
                problem: BeamSearchProblemInstance {
                    target: model.target.clone().into(),
                    generator: SuccessorGenerator::new(
                        vec![],
                        vec![
                            Rc::from(transitions[0].clone()),
                            Rc::from(transitions[1].clone())
                        ],
                        false,
                        model.clone()
                    ),
                    cost: 0,
                    g: 0,
                    solution_suffix: &transitions[2..]
                },
                prefix: &[],
                start: 0,
                depth: 2,
                beam_size: 1,
            })
        );
        assert_eq!(
            iter.next(),
            Some(BeamNeighborhood {
                problem: BeamSearchProblemInstance {
                    target: State {
                        signature_variables: SignatureVariables {
                            set_variables: vec![{
                                let mut set = Set::with_capacity(3);
                                set.insert(1);
                                set.insert(2);
                                set
                            }],
                            ..Default::default()
                        },
                        ..Default::default()
                    }
                    .into(),
                    generator: SuccessorGenerator::new(
                        vec![],
                        vec![
                            Rc::from(transitions[1].clone()),
                            Rc::from(transitions[2].clone())
                        ],
                        false,
                        model
                    ),
                    cost: 1,
                    g: 1,
                    solution_suffix: &[]
                },
                prefix: &transitions[..1],
                start: 1,
                depth: 2,
                beam_size: 1,
            })
        );
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn generate_neighborhood_with_different_width() {
        let model = Rc::new(Model {
            base_cases: vec![BaseCase::new(vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Ge,
                    Box::new(IntegerExpression::Variable(0)),
                    Box::new(IntegerExpression::Constant(2)),
                ),
                ..Default::default()
            }])],
            target: State {
                signature_variables: SignatureVariables {
                    set_variables: vec![{
                        let mut set = Set::with_capacity(3);
                        set.insert(0);
                        set.insert(1);
                        set.insert(2);
                        set
                    }],
                    ..Default::default()
                },
                ..Default::default()
            },
            state_metadata: StateMetadata {
                object_type_names: vec![String::from("object")],
                name_to_object_type: {
                    let mut map = FxHashMap::default();
                    map.insert(String::from("object"), 0);
                    map
                },
                object_numbers: vec![3],
                set_variable_names: vec![String::from("v")],
                name_to_set_variable: {
                    let mut map = FxHashMap::default();
                    map.insert(String::from("v"), 0);
                    map
                },
                set_variable_to_object: vec![0],
                ..Default::default()
            },
            forward_transitions: vec![
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Set(Box::new(SetCondition::IsIn(
                            ElementExpression::Constant(0),
                            SetExpression::Reference(ReferenceExpression::Variable(0)),
                        ))),
                        ..Default::default()
                    }],
                    effect: Effect {
                        set_effects: vec![(
                            0,
                            SetExpression::SetElementOperation(
                                SetElementOperator::Remove,
                                ElementExpression::Constant(0),
                                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                                    0,
                                ))),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::Constant(1)),
                    ..Default::default()
                },
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Set(Box::new(SetCondition::IsIn(
                            ElementExpression::Constant(1),
                            SetExpression::Reference(ReferenceExpression::Variable(0)),
                        ))),
                        ..Default::default()
                    }],
                    effect: Effect {
                        set_effects: vec![(
                            0,
                            SetExpression::SetElementOperation(
                                SetElementOperator::Remove,
                                ElementExpression::Constant(1),
                                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                                    0,
                                ))),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::Constant(2)),
                    ..Default::default()
                },
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Set(Box::new(SetCondition::IsIn(
                            ElementExpression::Constant(2),
                            SetExpression::Reference(ReferenceExpression::Variable(0)),
                        ))),
                        ..Default::default()
                    }],
                    effect: Effect {
                        set_effects: vec![(
                            0,
                            SetExpression::SetElementOperation(
                                SetElementOperator::Remove,
                                ElementExpression::Constant(2),
                                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                                    0,
                                ))),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::Constant(3)),
                    ..Default::default()
                },
            ],
            ..Default::default()
        });
        let generator =
            SuccessorGenerator::<TransitionWithCustomCost>::from_model_without_custom_cost(
                model.clone(),
                false,
            );
        let transition_constraints = &TransitionConstraints::new(&model, false);
        let transitions: Vec<TransitionWithCustomCost> = generator
            .transitions
            .iter()
            .map(|t: &_| (**t).clone())
            .collect();
        let solution_suffix = &transitions[2..];
        let problem = BeamSearchProblemInstance {
            target: StateInRegistry::from(generator.model.target.clone()),
            generator: transition_constraints.filter_successor_generator(
                &generator,
                &[],
                solution_suffix,
            ),
            cost: 0,
            g: 0,
            solution_suffix,
        };
        let neighborhood = BeamNeighborhood {
            problem: problem.clone(),
            prefix: &[],
            start: 0,
            depth: 2,
            beam_size: 1,
        };
        let new_neighborhood = neighborhood.generate_with_different_width(2);
        assert_eq!(
            new_neighborhood,
            BeamNeighborhood {
                problem,
                prefix: &[],
                start: 0,
                depth: 2,
                beam_size: 2,
            }
        );
    }

    #[test]
    fn generate_neighborhood_with_different_depth() {
        let model = Rc::new(Model {
            base_cases: vec![BaseCase::new(vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Ge,
                    Box::new(IntegerExpression::Variable(0)),
                    Box::new(IntegerExpression::Constant(2)),
                ),
                ..Default::default()
            }])],
            target: State {
                signature_variables: SignatureVariables {
                    set_variables: vec![{
                        let mut set = Set::with_capacity(3);
                        set.insert(0);
                        set.insert(1);
                        set.insert(2);
                        set
                    }],
                    ..Default::default()
                },
                ..Default::default()
            },
            state_metadata: StateMetadata {
                object_type_names: vec![String::from("object")],
                name_to_object_type: {
                    let mut map = FxHashMap::default();
                    map.insert(String::from("object"), 0);
                    map
                },
                object_numbers: vec![3],
                set_variable_names: vec![String::from("v")],
                name_to_set_variable: {
                    let mut map = FxHashMap::default();
                    map.insert(String::from("v"), 0);
                    map
                },
                set_variable_to_object: vec![0],
                ..Default::default()
            },
            forward_transitions: vec![
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Set(Box::new(SetCondition::IsIn(
                            ElementExpression::Constant(0),
                            SetExpression::Reference(ReferenceExpression::Variable(0)),
                        ))),
                        ..Default::default()
                    }],
                    effect: Effect {
                        set_effects: vec![(
                            0,
                            SetExpression::SetElementOperation(
                                SetElementOperator::Remove,
                                ElementExpression::Constant(0),
                                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                                    0,
                                ))),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::Constant(1)),
                    ..Default::default()
                },
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Set(Box::new(SetCondition::IsIn(
                            ElementExpression::Constant(1),
                            SetExpression::Reference(ReferenceExpression::Variable(0)),
                        ))),
                        ..Default::default()
                    }],
                    effect: Effect {
                        set_effects: vec![(
                            0,
                            SetExpression::SetElementOperation(
                                SetElementOperator::Remove,
                                ElementExpression::Constant(1),
                                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                                    0,
                                ))),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::Constant(2)),
                    ..Default::default()
                },
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Set(Box::new(SetCondition::IsIn(
                            ElementExpression::Constant(2),
                            SetExpression::Reference(ReferenceExpression::Variable(0)),
                        ))),
                        ..Default::default()
                    }],
                    effect: Effect {
                        set_effects: vec![(
                            0,
                            SetExpression::SetElementOperation(
                                SetElementOperator::Remove,
                                ElementExpression::Constant(2),
                                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                                    0,
                                ))),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::Constant(3)),
                    ..Default::default()
                },
            ],
            ..Default::default()
        });
        let generator =
            SuccessorGenerator::<TransitionWithCustomCost>::from_model_without_custom_cost(
                model.clone(),
                false,
            );
        let transition_constraints = &TransitionConstraints::new(&model, false);
        let transitions: Vec<TransitionWithCustomCost> = generator
            .transitions
            .iter()
            .map(|t: &_| (**t).clone())
            .collect();
        let solution_suffix = &transitions[2..];
        let problem = BeamSearchProblemInstance {
            target: StateInRegistry::from(generator.model.target.clone()),
            generator: transition_constraints.filter_successor_generator(
                &generator,
                &[],
                solution_suffix,
            ),
            cost: 0,
            g: 0,
            solution_suffix,
        };
        let neighborhood = BeamNeighborhood {
            problem,
            prefix: &[],
            start: 0,
            depth: 2,
            beam_size: 1,
        };
        let new_neighborhood =
            neighborhood.generate_deeper(1, 2, &generator, transition_constraints);
        let problem = BeamSearchProblemInstance {
            target: StateInRegistry::from(generator.model.target.clone()),
            generator,
            cost: 0,
            g: 0,
            solution_suffix: &[],
        };
        assert_eq!(
            new_neighborhood,
            BeamNeighborhood {
                problem,
                prefix: &[],
                start: 0,
                depth: 3,
                beam_size: 2,
            }
        );
    }

    #[test]
    fn interval_included_same() {
        let model = Rc::new(Model {
            base_cases: vec![BaseCase::new(vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Ge,
                    Box::new(IntegerExpression::Variable(0)),
                    Box::new(IntegerExpression::Constant(2)),
                ),
                ..Default::default()
            }])],
            target: State {
                signature_variables: SignatureVariables {
                    set_variables: vec![{
                        let mut set = Set::with_capacity(3);
                        set.insert(0);
                        set.insert(1);
                        set.insert(2);
                        set
                    }],
                    ..Default::default()
                },
                ..Default::default()
            },
            state_metadata: StateMetadata {
                object_type_names: vec![String::from("object")],
                name_to_object_type: {
                    let mut map = FxHashMap::default();
                    map.insert(String::from("object"), 0);
                    map
                },
                object_numbers: vec![3],
                set_variable_names: vec![String::from("v")],
                name_to_set_variable: {
                    let mut map = FxHashMap::default();
                    map.insert(String::from("v"), 0);
                    map
                },
                set_variable_to_object: vec![0],
                ..Default::default()
            },
            forward_transitions: vec![
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Set(Box::new(SetCondition::IsIn(
                            ElementExpression::Constant(0),
                            SetExpression::Reference(ReferenceExpression::Variable(0)),
                        ))),
                        ..Default::default()
                    }],
                    effect: Effect {
                        set_effects: vec![(
                            0,
                            SetExpression::SetElementOperation(
                                SetElementOperator::Remove,
                                ElementExpression::Constant(0),
                                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                                    0,
                                ))),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::Constant(1)),
                    ..Default::default()
                },
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Set(Box::new(SetCondition::IsIn(
                            ElementExpression::Constant(1),
                            SetExpression::Reference(ReferenceExpression::Variable(0)),
                        ))),
                        ..Default::default()
                    }],
                    effect: Effect {
                        set_effects: vec![(
                            0,
                            SetExpression::SetElementOperation(
                                SetElementOperator::Remove,
                                ElementExpression::Constant(1),
                                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                                    0,
                                ))),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::Constant(2)),
                    ..Default::default()
                },
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Set(Box::new(SetCondition::IsIn(
                            ElementExpression::Constant(2),
                            SetExpression::Reference(ReferenceExpression::Variable(0)),
                        ))),
                        ..Default::default()
                    }],
                    effect: Effect {
                        set_effects: vec![(
                            0,
                            SetExpression::SetElementOperation(
                                SetElementOperator::Remove,
                                ElementExpression::Constant(2),
                                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                                    0,
                                ))),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::Constant(3)),
                    ..Default::default()
                },
            ],
            ..Default::default()
        });
        let generator =
            SuccessorGenerator::<TransitionWithCustomCost>::from_model_without_custom_cost(
                model.clone(),
                false,
            );
        let transition_constraints = &TransitionConstraints::new(&model, false);
        let transitions: Vec<TransitionWithCustomCost> = generator
            .transitions
            .iter()
            .map(|t: &_| (**t).clone())
            .collect();
        let solution_suffix = &transitions[2..];
        let problem = BeamSearchProblemInstance {
            target: StateInRegistry::from(generator.model.target.clone()),
            generator: transition_constraints.filter_successor_generator(
                &generator,
                &[],
                solution_suffix,
            ),
            cost: 0,
            g: 0,
            solution_suffix,
        };
        let neighborhood = BeamNeighborhood {
            problem,
            prefix: &[],
            start: 0,
            depth: 2,
            beam_size: 1,
        };
        assert!(neighborhood.includes(0, 2));
    }

    #[test]
    fn interval_included_smaller() {
        let model = Rc::new(Model {
            base_cases: vec![BaseCase::new(vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Ge,
                    Box::new(IntegerExpression::Variable(0)),
                    Box::new(IntegerExpression::Constant(2)),
                ),
                ..Default::default()
            }])],
            target: State {
                signature_variables: SignatureVariables {
                    set_variables: vec![{
                        let mut set = Set::with_capacity(3);
                        set.insert(0);
                        set.insert(1);
                        set.insert(2);
                        set
                    }],
                    ..Default::default()
                },
                ..Default::default()
            },
            state_metadata: StateMetadata {
                object_type_names: vec![String::from("object")],
                name_to_object_type: {
                    let mut map = FxHashMap::default();
                    map.insert(String::from("object"), 0);
                    map
                },
                object_numbers: vec![3],
                set_variable_names: vec![String::from("v")],
                name_to_set_variable: {
                    let mut map = FxHashMap::default();
                    map.insert(String::from("v"), 0);
                    map
                },
                set_variable_to_object: vec![0],
                ..Default::default()
            },
            forward_transitions: vec![
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Set(Box::new(SetCondition::IsIn(
                            ElementExpression::Constant(0),
                            SetExpression::Reference(ReferenceExpression::Variable(0)),
                        ))),
                        ..Default::default()
                    }],
                    effect: Effect {
                        set_effects: vec![(
                            0,
                            SetExpression::SetElementOperation(
                                SetElementOperator::Remove,
                                ElementExpression::Constant(0),
                                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                                    0,
                                ))),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::Constant(1)),
                    ..Default::default()
                },
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Set(Box::new(SetCondition::IsIn(
                            ElementExpression::Constant(1),
                            SetExpression::Reference(ReferenceExpression::Variable(0)),
                        ))),
                        ..Default::default()
                    }],
                    effect: Effect {
                        set_effects: vec![(
                            0,
                            SetExpression::SetElementOperation(
                                SetElementOperator::Remove,
                                ElementExpression::Constant(1),
                                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                                    0,
                                ))),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::Constant(2)),
                    ..Default::default()
                },
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Set(Box::new(SetCondition::IsIn(
                            ElementExpression::Constant(2),
                            SetExpression::Reference(ReferenceExpression::Variable(0)),
                        ))),
                        ..Default::default()
                    }],
                    effect: Effect {
                        set_effects: vec![(
                            0,
                            SetExpression::SetElementOperation(
                                SetElementOperator::Remove,
                                ElementExpression::Constant(2),
                                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                                    0,
                                ))),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::Constant(3)),
                    ..Default::default()
                },
            ],
            ..Default::default()
        });
        let generator =
            SuccessorGenerator::<TransitionWithCustomCost>::from_model_without_custom_cost(
                model.clone(),
                false,
            );
        let transition_constraints = &TransitionConstraints::new(&model, false);
        let solution_suffix = &[];
        let problem = BeamSearchProblemInstance {
            target: StateInRegistry::from(generator.model.target.clone()),
            generator: transition_constraints.filter_successor_generator(
                &generator,
                &[],
                solution_suffix,
            ),
            cost: 0,
            g: 0,
            solution_suffix,
        };
        let neighborhood = BeamNeighborhood {
            problem,
            prefix: &[],
            start: 0,
            depth: 3,
            beam_size: 1,
        };
        assert!(neighborhood.includes(0, 2));
    }

    #[test]
    fn interval_not_included_after() {
        let model = Rc::new(Model {
            base_cases: vec![BaseCase::new(vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Ge,
                    Box::new(IntegerExpression::Variable(0)),
                    Box::new(IntegerExpression::Constant(2)),
                ),
                ..Default::default()
            }])],
            target: State {
                signature_variables: SignatureVariables {
                    set_variables: vec![{
                        let mut set = Set::with_capacity(3);
                        set.insert(0);
                        set.insert(1);
                        set.insert(2);
                        set
                    }],
                    ..Default::default()
                },
                ..Default::default()
            },
            state_metadata: StateMetadata {
                object_type_names: vec![String::from("object")],
                name_to_object_type: {
                    let mut map = FxHashMap::default();
                    map.insert(String::from("object"), 0);
                    map
                },
                object_numbers: vec![3],
                set_variable_names: vec![String::from("v")],
                name_to_set_variable: {
                    let mut map = FxHashMap::default();
                    map.insert(String::from("v"), 0);
                    map
                },
                set_variable_to_object: vec![0],
                ..Default::default()
            },
            forward_transitions: vec![
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Set(Box::new(SetCondition::IsIn(
                            ElementExpression::Constant(0),
                            SetExpression::Reference(ReferenceExpression::Variable(0)),
                        ))),
                        ..Default::default()
                    }],
                    effect: Effect {
                        set_effects: vec![(
                            0,
                            SetExpression::SetElementOperation(
                                SetElementOperator::Remove,
                                ElementExpression::Constant(0),
                                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                                    0,
                                ))),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::Constant(1)),
                    ..Default::default()
                },
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Set(Box::new(SetCondition::IsIn(
                            ElementExpression::Constant(1),
                            SetExpression::Reference(ReferenceExpression::Variable(0)),
                        ))),
                        ..Default::default()
                    }],
                    effect: Effect {
                        set_effects: vec![(
                            0,
                            SetExpression::SetElementOperation(
                                SetElementOperator::Remove,
                                ElementExpression::Constant(1),
                                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                                    0,
                                ))),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::Constant(2)),
                    ..Default::default()
                },
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Set(Box::new(SetCondition::IsIn(
                            ElementExpression::Constant(2),
                            SetExpression::Reference(ReferenceExpression::Variable(0)),
                        ))),
                        ..Default::default()
                    }],
                    effect: Effect {
                        set_effects: vec![(
                            0,
                            SetExpression::SetElementOperation(
                                SetElementOperator::Remove,
                                ElementExpression::Constant(2),
                                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                                    0,
                                ))),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::Constant(3)),
                    ..Default::default()
                },
            ],
            ..Default::default()
        });
        let generator =
            SuccessorGenerator::<TransitionWithCustomCost>::from_model_without_custom_cost(
                model.clone(),
                false,
            );
        let transition_constraints = &TransitionConstraints::new(&model, false);
        let transitions: Vec<TransitionWithCustomCost> = generator
            .transitions
            .iter()
            .map(|t: &_| (**t).clone())
            .collect();
        let solution_suffix = &transitions[2..];
        let problem = BeamSearchProblemInstance {
            target: StateInRegistry::from(generator.model.target.clone()),
            generator: transition_constraints.filter_successor_generator(
                &generator,
                &[],
                solution_suffix,
            ),
            cost: 0,
            g: 0,
            solution_suffix,
        };
        let neighborhood = BeamNeighborhood {
            problem,
            prefix: &[],
            start: 0,
            depth: 2,
            beam_size: 1,
        };
        assert!(!neighborhood.includes(1, 2));
    }

    #[test]
    fn interval_not_included_before() {
        let model = Rc::new(Model {
            base_cases: vec![BaseCase::new(vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Ge,
                    Box::new(IntegerExpression::Variable(0)),
                    Box::new(IntegerExpression::Constant(2)),
                ),
                ..Default::default()
            }])],
            target: State {
                signature_variables: SignatureVariables {
                    set_variables: vec![{
                        let mut set = Set::with_capacity(3);
                        set.insert(0);
                        set.insert(1);
                        set.insert(2);
                        set
                    }],
                    ..Default::default()
                },
                ..Default::default()
            },
            state_metadata: StateMetadata {
                object_type_names: vec![String::from("object")],
                name_to_object_type: {
                    let mut map = FxHashMap::default();
                    map.insert(String::from("object"), 0);
                    map
                },
                object_numbers: vec![3],
                set_variable_names: vec![String::from("v")],
                name_to_set_variable: {
                    let mut map = FxHashMap::default();
                    map.insert(String::from("v"), 0);
                    map
                },
                set_variable_to_object: vec![0],
                ..Default::default()
            },
            forward_transitions: vec![
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Set(Box::new(SetCondition::IsIn(
                            ElementExpression::Constant(0),
                            SetExpression::Reference(ReferenceExpression::Variable(0)),
                        ))),
                        ..Default::default()
                    }],
                    effect: Effect {
                        set_effects: vec![(
                            0,
                            SetExpression::SetElementOperation(
                                SetElementOperator::Remove,
                                ElementExpression::Constant(0),
                                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                                    0,
                                ))),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::Constant(1)),
                    ..Default::default()
                },
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Set(Box::new(SetCondition::IsIn(
                            ElementExpression::Constant(1),
                            SetExpression::Reference(ReferenceExpression::Variable(0)),
                        ))),
                        ..Default::default()
                    }],
                    effect: Effect {
                        set_effects: vec![(
                            0,
                            SetExpression::SetElementOperation(
                                SetElementOperator::Remove,
                                ElementExpression::Constant(1),
                                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                                    0,
                                ))),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::Constant(2)),
                    ..Default::default()
                },
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Set(Box::new(SetCondition::IsIn(
                            ElementExpression::Constant(2),
                            SetExpression::Reference(ReferenceExpression::Variable(0)),
                        ))),
                        ..Default::default()
                    }],
                    effect: Effect {
                        set_effects: vec![(
                            0,
                            SetExpression::SetElementOperation(
                                SetElementOperator::Remove,
                                ElementExpression::Constant(2),
                                Box::new(SetExpression::Reference(ReferenceExpression::Variable(
                                    0,
                                ))),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::Constant(3)),
                    ..Default::default()
                },
            ],
            ..Default::default()
        });
        let generator =
            SuccessorGenerator::<TransitionWithCustomCost>::from_model_without_custom_cost(
                model.clone(),
                false,
            );
        let transition_constraints = &TransitionConstraints::new(&model, false);
        let transitions: Vec<TransitionWithCustomCost> = generator
            .transitions
            .iter()
            .map(|t: &_| (**t).clone())
            .collect();
        let solution_suffix = &[];
        let problem = BeamSearchProblemInstance {
            target: StateInRegistry::from(generator.model.target.clone()),
            generator: transition_constraints.filter_successor_generator(
                &generator,
                &transitions[..1],
                solution_suffix,
            ),
            cost: 0,
            g: 0,
            solution_suffix,
        };
        let neighborhood = BeamNeighborhood {
            problem,
            prefix: &transitions[..1],
            start: 1,
            depth: 2,
            beam_size: 1,
        };
        assert!(!neighborhood.includes(0, 2));
    }
}
