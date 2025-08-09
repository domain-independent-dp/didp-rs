use super::successor_generator::SuccessorGenerator;
use super::transition::TransitionWithId;
use dypdl::expression::*;
use dypdl::variable_type::Element;
use dypdl::{Model, Transition, TransitionInterface};
use rustc_hash::{FxHashMap, FxHashSet};
use std::ops::Deref;

#[derive(Default, PartialEq, Eq, Debug)]
struct AffectedElements {
    achieved: Vec<(usize, Element)>,
    removed: Vec<(usize, Element)>,
    arbitrary: Vec<usize>,
}

fn get_affected_elements(transition: &Transition) -> AffectedElements {
    let mut achieved = Vec::default();
    let mut removed = Vec::default();
    let mut arbitrary = Vec::default();

    for (var_id, expression) in &transition.effect.set_effects {
        match expression {
            SetExpression::SetElementOperation(
                SetElementOperator::Add,
                ElementExpression::Constant(element),
                _,
            ) => achieved.push((*var_id, *element)),
            SetExpression::SetElementOperation(
                SetElementOperator::Remove,
                ElementExpression::Constant(element),
                _,
            ) => removed.push((*var_id, *element)),
            _ => arbitrary.push(*var_id),
        }
    }

    AffectedElements {
        achieved,
        removed,
        arbitrary,
    }
}

#[derive(Default, PartialEq, Eq, Debug)]
struct RequiredElements {
    positively: Vec<(usize, Element)>,
    negatively: Vec<(usize, Element)>,
}

fn get_required_elements(transition: &Transition) -> RequiredElements {
    let mut positively = Vec::default();
    let mut negatively = Vec::default();

    for condition in transition.get_preconditions() {
        if let Condition::Set(condition) = condition {
            match condition.as_ref() {
                SetCondition::IsIn(
                    ElementExpression::Constant(element),
                    SetExpression::Reference(ReferenceExpression::Variable(var_id)),
                ) => positively.push((*var_id, *element)),
                SetCondition::IsIn(
                    ElementExpression::Constant(element),
                    SetExpression::Complement(expression),
                ) => {
                    if let SetExpression::Reference(ReferenceExpression::Variable(var_id)) =
                        expression.as_ref()
                    {
                        negatively.push((*var_id, *element))
                    }
                }
                _ => {}
            }
        } else if let Condition::Not(condition) = condition {
            if let Condition::Set(condition) = condition.as_ref() {
                match condition.as_ref() {
                    SetCondition::IsIn(
                        ElementExpression::Constant(element),
                        SetExpression::Reference(ReferenceExpression::Variable(var_id)),
                    ) => negatively.push((*var_id, *element)),
                    SetCondition::IsIn(
                        ElementExpression::Constant(element),
                        SetExpression::Complement(expression),
                    ) => {
                        if let SetExpression::Reference(ReferenceExpression::Variable(var_id)) =
                            expression.as_ref()
                        {
                            positively.push((*var_id, *element))
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    RequiredElements {
        positively,
        negatively,
    }
}

/// Data structure storing pairs of transitions that must not happen before or after each other.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::search_algorithm::data_structure::{
///     TransitionMutex,
/// };
/// use dypdl_heuristic_search::search_algorithm::SuccessorGenerator;
/// use std::rc::Rc;
///
/// let mut model = Model::default();
/// let object_type = model.add_object_type("object", 4).unwrap();
/// let set = model.create_set(object_type, &[0, 1, 2, 3]).unwrap();
/// let variable = model.add_set_variable("variable", object_type, set).unwrap();
///
/// let mut transition = Transition::new("remove 0");
/// transition.add_effect(variable, variable.remove(0)).unwrap();
/// model.add_forward_transition(transition.clone()).unwrap();
///
/// let mut transition = Transition::new("remove 1");
/// transition.add_effect(variable, variable.remove(1)).unwrap();
/// model.add_forward_forced_transition(transition.clone()).unwrap();
///
/// let mut transition = Transition::new("require 0");
/// transition.add_precondition(variable.contains(0));
/// model.add_forward_forced_transition(transition.clone()).unwrap();
///
/// let mut transition = Transition::new("require 1");
/// transition.add_precondition(variable.contains(1));
/// model.add_forward_transition(transition).unwrap();
///
/// let model = Rc::new(model);
/// let generator = SuccessorGenerator::<Transition>::from_model(model, false);
/// let transitions = generator
///     .transitions
///     .iter()
///     .chain(generator.forced_transitions.iter())
///     .map(|t| t.as_ref().clone())
///     .collect::<Vec<_>>();
/// let mutex = TransitionMutex::new(transitions);
///
/// assert_eq!(mutex.get_forbidden_before(false, 0), &[]);
/// assert_eq!(mutex.get_forbidden_before(true, 0), &[]);
/// assert_eq!(mutex.get_forbidden_before(true, 1), &[(false, 0)]);
/// assert_eq!(mutex.get_forbidden_before(false, 1), &[(true, 0)]);
///
/// assert_eq!(mutex.get_forbidden_after(false, 0), &[(true, 1)]);
/// assert_eq!(mutex.get_forbidden_after(true, 0), &[(false, 1)]);
/// assert_eq!(mutex.get_forbidden_after(true, 1), &[]);
/// assert_eq!(mutex.get_forbidden_after(false, 1), &[]);
///
/// let remove_0 = generator.transitions[0].clone();
/// let require_1 = generator.transitions[1].clone();
/// let prefix = &[(*remove_0).clone()];
/// let suffix = &[(*require_1).clone()];
/// let generator = mutex.filter_successor_generator(&generator, prefix, suffix);
/// assert_eq!(generator.transitions, vec![remove_0, require_1]);
/// assert_eq!(generator.forced_transitions, vec![]);
/// ```
#[derive(Default, PartialEq, Eq, Debug)]
pub struct TransitionMutex {
    forbidden_before: Vec<Vec<(bool, usize)>>,
    forced_forbidden_before: Vec<Vec<(bool, usize)>>,
    forbidden_after: Vec<Vec<(bool, usize)>>,
    forced_forbidden_after: Vec<Vec<(bool, usize)>>,
}

impl TransitionMutex {
    /// Get transitions that must not happen before the given transition.
    ///
    /// The first return value indicates whether it is a forced transition,
    /// and the second return value is the transition id.
    pub fn get_forbidden_before(&self, forced: bool, id: usize) -> &[(bool, usize)] {
        if forced {
            &self.forced_forbidden_before[id]
        } else {
            &self.forbidden_before[id]
        }
    }

    /// Get transitions that must not happen after the given transition.
    ///
    /// The first return value indicates whether it is a forced transition,
    /// and the second return value is the transition id.
    pub fn get_forbidden_after(&self, forced: bool, id: usize) -> &[(bool, usize)] {
        if forced {
            &self.forced_forbidden_after[id]
        } else {
            &self.forbidden_after[id]
        }
    }

    /// Create a successor generator filtering forbidden transitions by the given prefix ans suffix.
    pub fn filter_successor_generator<T, U, R>(
        &self,
        generator: &SuccessorGenerator<T, U, R>,
        prefix: &[TransitionWithId<T>],
        suffix: &[TransitionWithId<T>],
    ) -> SuccessorGenerator<T, U, R>
    where
        T: TransitionInterface,
        U: Deref<Target = TransitionWithId<T>> + Clone + From<TransitionWithId<T>>,
        R: Deref<Target = Model> + Clone,
    {
        let (forbidden_forced_ids, forbidden_ids): (Vec<_>, Vec<_>) = suffix
            .iter()
            .flat_map(|t| self.get_forbidden_before(t.forced, t.id).iter())
            .chain(
                prefix
                    .iter()
                    .flat_map(|t| self.get_forbidden_after(t.forced, t.id).iter()),
            )
            .copied()
            .partition(|(forced, _)| *forced);
        let forbidden_forced_ids =
            FxHashSet::<usize>::from_iter(forbidden_forced_ids.into_iter().map(|(_, id)| id));
        let forbidden_ids =
            FxHashSet::<usize>::from_iter(forbidden_ids.into_iter().map(|(_, id)| id));

        let forced_transitions = generator
            .forced_transitions
            .iter()
            .enumerate()
            .filter_map(|(id, t)| {
                if forbidden_forced_ids.contains(&id) {
                    None
                } else {
                    Some(t.clone())
                }
            })
            .collect();

        let transitions = generator
            .transitions
            .iter()
            .enumerate()
            .filter_map(|(id, t)| {
                if forbidden_ids.contains(&id) {
                    None
                } else {
                    Some(t.clone())
                }
            })
            .collect();

        SuccessorGenerator::new(
            forced_transitions,
            transitions,
            generator.backward,
            generator.model.clone(),
        )
    }

    /// Create a new transition mutex from the given transitions.
    pub fn new<T>(transitions: Vec<TransitionWithId<T>>) -> Self
    where
        T: TransitionInterface + Clone,
        Transition: From<T>,
    {
        let len = transitions
            .iter()
            .filter_map(|t| if !t.forced { Some(t.id) } else { None })
            .max()
            .map_or(0, |id_max| id_max + 1);
        let forced_len = transitions
            .iter()
            .filter_map(|t| if t.forced { Some(t.id) } else { None })
            .max()
            .map_or(0, |id_max| id_max + 1);

        let mut achievers = FxHashMap::default();
        let mut removers = FxHashMap::default();
        let mut arbitrary_affected = FxHashSet::default();
        let mut positively_conditioned = FxHashMap::default();
        let mut negatively_conditioned = FxHashMap::default();

        for t in transitions {
            let id = t.id;
            let forced = t.forced;
            let transition = Transition::from(t.transition);
            let affected = get_affected_elements(&transition);
            extend_element_transitions_map(&mut achievers, &affected.achieved, forced, id);
            extend_element_transitions_map(&mut removers, &affected.removed, forced, id);
            arbitrary_affected.extend(affected.arbitrary.into_iter());
            let required = get_required_elements(&transition);
            extend_element_transitions_map(
                &mut positively_conditioned,
                &required.positively,
                forced,
                id,
            );
            extend_element_transitions_map(
                &mut negatively_conditioned,
                &required.negatively,
                forced,
                id,
            );
        }

        let mut forbidden_before = vec![FxHashSet::default(); len];
        let mut forced_forbidden_before = vec![FxHashSet::default(); forced_len];
        let mut forbidden_after = vec![FxHashSet::default(); len];
        let mut forced_forbidden_after = vec![FxHashSet::default(); forced_len];

        // For each transition that positively requires an element.
        for ((var_id, element), operator_ids) in positively_conditioned {
            // If an element can be added, it does not matter.
            if arbitrary_affected.contains(&var_id) || achievers.contains_key(&(var_id, element)) {
                continue;
            }

            // A transition that removes the element must not happen before.
            for (forced, id) in operator_ids {
                if let Some(removers) = removers.get(&(var_id, element)) {
                    for remover in removers {
                        if forced {
                            forced_forbidden_before[id].insert(*remover);
                        } else {
                            forbidden_before[id].insert(*remover);
                        }

                        if remover.0 {
                            forced_forbidden_after[remover.1].insert((forced, id));
                        } else {
                            forbidden_after[remover.1].insert((forced, id));
                        }
                    }
                }
            }
        }

        // For each transition that negatively requires an element.
        for ((var_id, element), operator_ids) in negatively_conditioned {
            // If an element can be removed, it does not matter.
            if arbitrary_affected.contains(&var_id) || removers.contains_key(&(var_id, element)) {
                continue;
            }

            // A transition that adds the element must not happen before.
            for (forced, id) in operator_ids {
                if let Some(achievers) = achievers.get(&(var_id, element)) {
                    for achiever in achievers {
                        if forced {
                            forced_forbidden_before[id].insert(*achiever);
                        } else {
                            forbidden_before[id].insert(*achiever);
                        }

                        if achiever.0 {
                            forced_forbidden_after[achiever.1].insert((forced, id));
                        } else {
                            forbidden_after[achiever.1].insert((forced, id));
                        }
                    }
                }
            }
        }

        let forbidden_before = forbidden_before
            .into_iter()
            .map(|x| {
                let mut v = Vec::from_iter(x);
                v.sort();
                v
            })
            .collect();
        let forced_forbidden_before = forced_forbidden_before
            .into_iter()
            .map(|x| {
                let mut v = Vec::from_iter(x);
                v.sort();
                v
            })
            .collect();
        let forbidden_after = forbidden_after
            .into_iter()
            .map(|x| {
                let mut v = Vec::from_iter(x);
                v.sort();
                v
            })
            .collect();
        let forced_forbidden_after = forced_forbidden_after
            .into_iter()
            .map(|x| {
                let mut v = Vec::from_iter(x);
                v.sort();
                v
            })
            .collect();

        Self {
            forbidden_before,
            forced_forbidden_before,
            forbidden_after,
            forced_forbidden_after,
        }
    }
}

fn extend_element_transitions_map(
    map: &mut FxHashMap<(usize, Element), Vec<(bool, usize)>>,
    elements: &[(usize, Element)],
    forced: bool,
    id: usize,
) {
    for key in elements {
        map.entry(*key)
            .and_modify(|e| e.push((forced, id)))
            .or_insert_with(|| vec![(forced, id)]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::{Effect, GroundedCondition};
    use std::rc::Rc;

    #[test]
    fn get_affected_elements_without_set_effect() {
        let transition = Transition::default();
        let result = get_affected_elements(&transition);
        assert_eq!(result, AffectedElements::default());
    }

    #[test]
    fn get_affected_elements_with_set_effect() {
        let transition = Transition {
            effect: Effect {
                set_effects: vec![
                    (
                        0,
                        SetExpression::SetElementOperation(
                            SetElementOperator::Add,
                            ElementExpression::Constant(2),
                            Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
                        ),
                    ),
                    (
                        1,
                        SetExpression::SetElementOperation(
                            SetElementOperator::Remove,
                            ElementExpression::Constant(3),
                            Box::new(SetExpression::Reference(ReferenceExpression::Variable(2))),
                        ),
                    ),
                    (
                        2,
                        SetExpression::SetOperation(
                            SetOperator::Union,
                            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                            Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
                        ),
                    ),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        let result = get_affected_elements(&transition);
        assert_eq!(result.achieved, vec![(0, 2)]);
        assert_eq!(result.removed, vec![(1, 3)]);
        assert_eq!(result.arbitrary, vec![2]);
    }

    #[test]
    fn get_required_elements_without_set_effect() {
        let transition = Transition::default();
        let result = get_required_elements(&transition);
        assert_eq!(result, RequiredElements::default());
    }

    #[test]
    fn get_required_elements_with_set_effect() {
        let transition = Transition {
            elements_in_set_variable: vec![(0, 1)],
            preconditions: vec![
                GroundedCondition {
                    condition: Condition::Set(Box::new(SetCondition::IsIn(
                        ElementExpression::Constant(2),
                        SetExpression::Reference(ReferenceExpression::Variable(1)),
                    ))),
                    ..Default::default()
                },
                GroundedCondition {
                    condition: Condition::Set(Box::new(SetCondition::IsIn(
                        ElementExpression::Constant(3),
                        SetExpression::Complement(Box::new(SetExpression::Reference(
                            ReferenceExpression::Variable(2),
                        ))),
                    ))),
                    ..Default::default()
                },
                GroundedCondition {
                    condition: Condition::Set(Box::new(SetCondition::IsIn(
                        ElementExpression::Variable(0),
                        SetExpression::Complement(Box::new(SetExpression::Reference(
                            ReferenceExpression::Variable(3),
                        ))),
                    ))),
                    ..Default::default()
                },
                GroundedCondition {
                    condition: Condition::Not(Box::new(Condition::Set(Box::new(
                        SetCondition::IsIn(
                            ElementExpression::Constant(5),
                            SetExpression::Reference(ReferenceExpression::Variable(4)),
                        ),
                    )))),
                    ..Default::default()
                },
                GroundedCondition {
                    condition: Condition::Not(Box::new(Condition::Set(Box::new(
                        SetCondition::IsIn(
                            ElementExpression::Constant(6),
                            SetExpression::Complement(Box::new(SetExpression::Reference(
                                ReferenceExpression::Variable(5),
                            ))),
                        ),
                    )))),
                    ..Default::default()
                },
                GroundedCondition {
                    condition: Condition::Not(Box::new(Condition::Set(Box::new(
                        SetCondition::IsIn(
                            ElementExpression::Variable(0),
                            SetExpression::Reference(ReferenceExpression::Variable(6)),
                        ),
                    )))),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let result = get_required_elements(&transition);
        assert_eq!(result.positively, vec![(0, 1), (1, 2), (5, 6)]);
        assert_eq!(result.negatively, vec![(2, 3), (4, 5)]);
    }

    #[test]
    #[should_panic]
    fn get_forbidden_before_with_no_transitions() {
        let constraints = TransitionMutex::new(Vec::<TransitionWithId>::default());
        constraints.get_forbidden_before(false, 0);
    }

    #[test]
    #[should_panic]
    fn get_forbidden_after_with_no_transitions() {
        let constraints = TransitionMutex::new(Vec::<TransitionWithId>::default());
        constraints.get_forbidden_after(false, 0);
    }

    #[test]
    fn new() {
        let transitions = vec![
            TransitionWithId {
                id: 0,
                forced: true,
                transition: Transition {
                    effect: Effect {
                        set_effects: vec![
                            (
                                0,
                                SetExpression::SetElementOperation(
                                    SetElementOperator::Remove,
                                    ElementExpression::Constant(1),
                                    Box::new(SetExpression::Reference(
                                        ReferenceExpression::Variable(0),
                                    )),
                                ),
                            ),
                            (
                                1,
                                SetExpression::SetElementOperation(
                                    SetElementOperator::Add,
                                    ElementExpression::Constant(1),
                                    Box::new(SetExpression::Reference(
                                        ReferenceExpression::Variable(1),
                                    )),
                                ),
                            ),
                        ],
                        ..Default::default()
                    },
                    ..Default::default()
                },
            },
            TransitionWithId {
                id: 1,
                forced: true,
                transition: Transition {
                    preconditions: vec![
                        GroundedCondition {
                            condition: Condition::Set(Box::new(SetCondition::IsIn(
                                ElementExpression::Constant(1),
                                SetExpression::Reference(ReferenceExpression::Variable(0)),
                            ))),
                            ..Default::default()
                        },
                        GroundedCondition {
                            condition: Condition::Set(Box::new(SetCondition::IsIn(
                                ElementExpression::Constant(2),
                                SetExpression::Reference(ReferenceExpression::Variable(2)),
                            ))),
                            ..Default::default()
                        },
                    ],
                    effect: Effect {
                        set_effects: vec![
                            (
                                0,
                                SetExpression::SetElementOperation(
                                    SetElementOperator::Remove,
                                    ElementExpression::Constant(1),
                                    Box::new(SetExpression::Reference(
                                        ReferenceExpression::Variable(0),
                                    )),
                                ),
                            ),
                            (
                                2,
                                SetExpression::SetElementOperation(
                                    SetElementOperator::Remove,
                                    ElementExpression::Constant(2),
                                    Box::new(SetExpression::Reference(
                                        ReferenceExpression::Variable(2),
                                    )),
                                ),
                            ),
                        ],
                        ..Default::default()
                    },
                    ..Default::default()
                },
            },
            TransitionWithId {
                id: 0,
                forced: false,
                transition: Transition {
                    effect: Effect {
                        set_effects: vec![
                            (
                                1,
                                SetExpression::SetElementOperation(
                                    SetElementOperator::Add,
                                    ElementExpression::Constant(2),
                                    Box::new(SetExpression::Reference(
                                        ReferenceExpression::Variable(1),
                                    )),
                                ),
                            ),
                            (
                                0,
                                SetExpression::SetElementOperation(
                                    SetElementOperator::Remove,
                                    ElementExpression::Constant(2),
                                    Box::new(SetExpression::Reference(
                                        ReferenceExpression::Variable(0),
                                    )),
                                ),
                            ),
                            (
                                2,
                                SetExpression::SetElementOperation(
                                    SetElementOperator::Remove,
                                    ElementExpression::Constant(2),
                                    Box::new(SetExpression::Reference(
                                        ReferenceExpression::Variable(2),
                                    )),
                                ),
                            ),
                        ],
                        ..Default::default()
                    },
                    ..Default::default()
                },
            },
            TransitionWithId {
                id: 1,
                forced: false,
                transition: Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Not(Box::new(Condition::Set(Box::new(
                            SetCondition::IsIn(
                                ElementExpression::Constant(2),
                                SetExpression::Reference(ReferenceExpression::Variable(1)),
                            ),
                        )))),
                        ..Default::default()
                    }],
                    effect: Effect {
                        set_effects: vec![
                            (
                                1,
                                SetExpression::SetElementOperation(
                                    SetElementOperator::Add,
                                    ElementExpression::Constant(2),
                                    Box::new(SetExpression::Reference(
                                        ReferenceExpression::Variable(1),
                                    )),
                                ),
                            ),
                            (
                                2,
                                SetExpression::SetElementOperation(
                                    SetElementOperator::Remove,
                                    ElementExpression::Variable(0),
                                    Box::new(SetExpression::Reference(
                                        ReferenceExpression::Variable(2),
                                    )),
                                ),
                            ),
                        ],
                        ..Default::default()
                    },
                    ..Default::default()
                },
            },
        ];
        let mutex = TransitionMutex::new(transitions);
        assert_eq!(mutex.get_forbidden_before(true, 0), &[]);
        assert_eq!(mutex.get_forbidden_before(true, 1), &[(true, 0), (true, 1)]);
        assert_eq!(mutex.get_forbidden_before(false, 0), &[]);
        assert_eq!(
            mutex.get_forbidden_before(false, 1),
            &[(false, 0), (false, 1)]
        );
        assert_eq!(mutex.get_forbidden_after(true, 0), &[(true, 1)]);
        assert_eq!(mutex.get_forbidden_after(false, 0), &[(false, 1)]);
        assert_eq!(mutex.get_forbidden_after(true, 1), &[(true, 1)]);
        assert_eq!(mutex.get_forbidden_after(false, 1), &[(false, 1)]);
    }

    #[test]
    fn filter_successor_generator_without_any_constraints_and_suffix() {
        let model = Rc::new(Model {
            forward_transitions: vec![Transition::default()],
            forward_forced_transitions: vec![Transition::default()],
            ..Default::default()
        });
        let expected = SuccessorGenerator::<Transition>::from_model(model, false);
        let transitions = expected
            .transitions
            .iter()
            .chain(expected.forced_transitions.iter())
            .map(|t| t.as_ref().clone())
            .collect::<Vec<_>>();
        let mutex = TransitionMutex::new(transitions);
        let generator = mutex.filter_successor_generator(&expected, &[], &[]);
        assert_eq!(generator, expected);
    }

    #[test]
    fn filter_successor_generator_without_any_constraints() {
        let model = Rc::new(Model {
            forward_transitions: vec![Transition::default()],
            forward_forced_transitions: vec![Transition::default()],
            ..Default::default()
        });
        let expected = SuccessorGenerator::<Transition>::from_model(model, false);
        let transitions = expected
            .transitions
            .iter()
            .chain(expected.forced_transitions.iter())
            .map(|t| t.as_ref().clone())
            .collect::<Vec<_>>();
        let mutex = TransitionMutex::new(transitions);
        let generator = mutex.filter_successor_generator(
            &expected,
            &[],
            &[
                TransitionWithId {
                    id: 0,
                    forced: false,
                    ..Default::default()
                },
                TransitionWithId {
                    id: 0,
                    forced: true,
                    ..Default::default()
                },
            ],
        );
        assert_eq!(generator, expected);
    }

    #[test]
    fn filter_successor_generator_with_constraints_without_suffix() {
        let model = Rc::new(Model {
            forward_transitions: vec![Transition {
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
                            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                        ),
                    )],
                    ..Default::default()
                },
                ..Default::default()
            }],
            forward_forced_transitions: vec![Transition {
                preconditions: vec![GroundedCondition {
                    condition: Condition::Not(Box::new(Condition::Set(Box::new(
                        SetCondition::IsIn(
                            ElementExpression::Constant(2),
                            SetExpression::Reference(ReferenceExpression::Variable(1)),
                        ),
                    )))),
                    ..Default::default()
                }],
                effect: Effect {
                    set_effects: vec![(
                        1,
                        SetExpression::SetElementOperation(
                            SetElementOperator::Add,
                            ElementExpression::Constant(2),
                            Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
                        ),
                    )],
                    ..Default::default()
                },
                ..Default::default()
            }],
            ..Default::default()
        });
        let expected = SuccessorGenerator::<Transition>::from_model(model, false);
        let transitions = expected
            .transitions
            .iter()
            .chain(expected.forced_transitions.iter())
            .map(|t| t.as_ref().clone())
            .collect::<Vec<_>>();
        let mutex = TransitionMutex::new(transitions);
        let generator = mutex.filter_successor_generator(&expected, &[], &[]);
        assert_eq!(generator, expected);
    }

    #[test]
    fn filter_successor_generator_with_constraints_and_prefix() {
        let t1 = Transition {
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
                        Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                    ),
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        let t2 = Transition {
            preconditions: vec![GroundedCondition {
                condition: Condition::Not(Box::new(Condition::Set(Box::new(SetCondition::IsIn(
                    ElementExpression::Constant(2),
                    SetExpression::Reference(ReferenceExpression::Variable(1)),
                ))))),
                ..Default::default()
            }],
            effect: Effect {
                set_effects: vec![(
                    1,
                    SetExpression::SetElementOperation(
                        SetElementOperator::Add,
                        ElementExpression::Constant(2),
                        Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
                    ),
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        let model = Rc::new(Model {
            forward_transitions: vec![t1.clone(), Transition::default()],
            forward_forced_transitions: vec![t2.clone(), Transition::default()],
            ..Default::default()
        });
        let generator = SuccessorGenerator::<Transition>::from_model(model.clone(), false);
        let transitions = generator
            .transitions
            .iter()
            .chain(generator.forced_transitions.iter())
            .map(|t| t.as_ref().clone())
            .collect::<Vec<_>>();
        let mutex = TransitionMutex::new(transitions);
        let t1 = TransitionWithId {
            id: 0,
            forced: false,
            transition: t1,
        };
        let t2 = TransitionWithId {
            id: 0,
            forced: true,
            transition: t2,
        };
        let generator = mutex.filter_successor_generator(&generator, &[t1, t2], &[]);
        let forced_transitions = vec![Rc::new(TransitionWithId {
            id: 1,
            forced: true,
            ..Default::default()
        })];
        let transitions = vec![Rc::new(TransitionWithId {
            id: 1,
            forced: false,
            ..Default::default()
        })];
        let expected = SuccessorGenerator::new(forced_transitions, transitions, false, model);
        assert_eq!(generator, expected);
    }

    #[test]
    fn filter_successor_generator_with_constraints_and_suffix() {
        let t1 = Transition {
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
                        Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                    ),
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        let t2 = Transition {
            preconditions: vec![GroundedCondition {
                condition: Condition::Not(Box::new(Condition::Set(Box::new(SetCondition::IsIn(
                    ElementExpression::Constant(2),
                    SetExpression::Reference(ReferenceExpression::Variable(1)),
                ))))),
                ..Default::default()
            }],
            effect: Effect {
                set_effects: vec![(
                    1,
                    SetExpression::SetElementOperation(
                        SetElementOperator::Add,
                        ElementExpression::Constant(2),
                        Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
                    ),
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        let model = Rc::new(Model {
            forward_transitions: vec![t1.clone(), Transition::default()],
            forward_forced_transitions: vec![t2.clone(), Transition::default()],
            ..Default::default()
        });
        let generator = SuccessorGenerator::<Transition>::from_model(model.clone(), false);
        let transitions = generator
            .transitions
            .iter()
            .chain(generator.forced_transitions.iter())
            .map(|t| t.as_ref().clone())
            .collect::<Vec<_>>();
        let mutex = TransitionMutex::new(transitions);
        let t1 = TransitionWithId {
            id: 0,
            forced: false,
            transition: t1,
        };
        let t2 = TransitionWithId {
            id: 0,
            forced: true,
            transition: t2,
        };
        let generator = mutex.filter_successor_generator(&generator, &[], &[t1, t2]);
        let forced_transitions = vec![Rc::new(TransitionWithId {
            id: 1,
            forced: true,
            ..Default::default()
        })];
        let transitions = vec![Rc::new(TransitionWithId {
            id: 1,
            forced: false,
            ..Default::default()
        })];
        let expected =
            SuccessorGenerator::new(forced_transitions, transitions, false, model.clone());
        assert_eq!(generator, expected);
    }

    #[test]
    fn filter_successor_generator_with_constraints_and_prefix_and_suffix() {
        let t1 = Transition {
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
                        Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                    ),
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        let t2 = Transition {
            preconditions: vec![GroundedCondition {
                condition: Condition::Not(Box::new(Condition::Set(Box::new(SetCondition::IsIn(
                    ElementExpression::Constant(2),
                    SetExpression::Reference(ReferenceExpression::Variable(1)),
                ))))),
                ..Default::default()
            }],
            effect: Effect {
                set_effects: vec![(
                    1,
                    SetExpression::SetElementOperation(
                        SetElementOperator::Add,
                        ElementExpression::Constant(2),
                        Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
                    ),
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        let model = Rc::new(Model {
            forward_transitions: vec![t1.clone(), Transition::default()],
            forward_forced_transitions: vec![t2.clone(), Transition::default()],
            ..Default::default()
        });
        let generator = SuccessorGenerator::<Transition>::from_model(model.clone(), false);
        let transitions = generator
            .transitions
            .iter()
            .chain(generator.forced_transitions.iter())
            .map(|t| t.as_ref().clone())
            .collect::<Vec<_>>();
        let mutex = TransitionMutex::new(transitions);
        let t1 = TransitionWithId {
            id: 0,
            forced: false,
            transition: t1,
        };
        let t2 = TransitionWithId {
            id: 0,
            forced: true,
            transition: t2,
        };
        let generator = mutex.filter_successor_generator(&generator, &[t1], &[t2]);
        let forced_transitions = vec![Rc::new(TransitionWithId {
            id: 1,
            forced: true,
            ..Default::default()
        })];
        let transitions = vec![Rc::new(TransitionWithId {
            id: 1,
            forced: false,
            ..Default::default()
        })];
        let expected =
            SuccessorGenerator::new(forced_transitions, transitions, false, model.clone());
        assert_eq!(generator, expected);
    }
}
