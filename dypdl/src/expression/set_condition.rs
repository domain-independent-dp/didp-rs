use super::element_expression::ElementExpression;
use super::reference_expression::ReferenceExpression;
use super::set_expression::SetExpression;
use crate::state::StateInterface;
use crate::state_functions::{StateFunctionCache, StateFunctions};
use crate::table_registry::TableRegistry;
use crate::variable_type::Set;

/// Condition related to sets.
#[derive(Debug, PartialEq, Clone)]
pub enum SetCondition {
    /// Constant.
    Constant(bool),
    /// If a set is equal to another set.
    IsEqual(SetExpression, SetExpression),
    /// If a set is not equal to another set.
    IsNotEqual(SetExpression, SetExpression),
    /// If an element is included in a set.
    IsIn(ElementExpression, SetExpression),
    /// If a set is a subset of another set.
    IsSubset(SetExpression, SetExpression),
    /// If a set is empty.
    IsEmpty(SetExpression),
}

impl SetCondition {
    /// Returns the evaluation result.
    ///
    /// # Panics
    ///
    /// Panics if the cost of the transition state is used or a min/max reduce operation is performed on an empty set or vector.
    pub fn eval<T: StateInterface>(
        &self,
        state: &T,
        function_cache: &mut StateFunctionCache,
        state_functions: &StateFunctions,
        registry: &TableRegistry,
    ) -> bool {
        match self {
            Self::Constant(value) => *value,
            Self::IsIn(element, SetExpression::Reference(set)) => {
                let element = element.eval(state, function_cache, state_functions, registry);
                let set = set.eval(state, function_cache, state_functions, registry);
                set.contains(element)
            }
            Self::IsIn(element, SetExpression::StateFunction(i)) => {
                let element = element.eval(state, function_cache, state_functions, registry);
                let set = function_cache.get_set_value(*i, state, state_functions, registry);
                set.contains(element)
            }
            Self::IsIn(e, s) => s
                .eval(state, function_cache, state_functions, registry)
                .contains(e.eval(state, function_cache, state_functions, registry)),
            Self::IsEqual(x, y) => Self::evaluate_set_comparison(
                x,
                y,
                |x, y| x == y,
                state,
                function_cache,
                state_functions,
                registry,
            ),
            Self::IsNotEqual(x, y) => Self::evaluate_set_comparison(
                x,
                y,
                |x, y| x != y,
                state,
                function_cache,
                state_functions,
                registry,
            ),
            Self::IsSubset(x, y) => Self::evaluate_set_comparison(
                x,
                y,
                |x, y| x.is_subset(y),
                state,
                function_cache,
                state_functions,
                registry,
            ),
            Self::IsEmpty(SetExpression::Reference(set)) => {
                let set = set.eval(state, function_cache, state_functions, registry);
                set.count_ones(..) == 0
            }
            Self::IsEmpty(SetExpression::StateFunction(i)) => {
                let set = function_cache.get_set_value(*i, state, state_functions, registry);
                set.count_ones(..) == 0
            }
            Self::IsEmpty(s) => {
                s.eval(state, function_cache, state_functions, registry)
                    .count_ones(..)
                    == 0
            }
        }
    }

    /// Returns a simplified version by precomputation.
    ///
    /// # Panics
    ///
    /// Panics if a min/max reduce operation is performed on an empty set or vector.
    pub fn simplify(&self, registry: &TableRegistry) -> SetCondition {
        match self {
            Self::IsIn(element, set) => {
                match (element.simplify(registry), set.simplify(registry)) {
                    (
                        ElementExpression::Constant(element),
                        SetExpression::Reference(ReferenceExpression::Constant(set)),
                    ) => Self::Constant(set.contains(element)),
                    (element, set) => Self::IsIn(element, set),
                }
            }
            Self::IsEqual(x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (
                    SetExpression::Reference(ReferenceExpression::Constant(x)),
                    SetExpression::Reference(ReferenceExpression::Constant(y)),
                ) => Self::Constant(x == y),
                (
                    SetExpression::Reference(ReferenceExpression::Variable(x)),
                    SetExpression::Reference(ReferenceExpression::Variable(y)),
                )
                | (SetExpression::StateFunction(x), SetExpression::StateFunction(y))
                    if x == y =>
                {
                    Self::Constant(true)
                }
                (x, y) => Self::IsEqual(x, y),
            },
            Self::IsNotEqual(x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (
                    SetExpression::Reference(ReferenceExpression::Constant(x)),
                    SetExpression::Reference(ReferenceExpression::Constant(y)),
                ) => Self::Constant(x != y),
                (
                    SetExpression::Reference(ReferenceExpression::Variable(x)),
                    SetExpression::Reference(ReferenceExpression::Variable(y)),
                )
                | (SetExpression::StateFunction(x), SetExpression::StateFunction(y))
                    if x == y =>
                {
                    Self::Constant(false)
                }
                (x, y) => Self::IsNotEqual(x, y),
            },
            Self::IsSubset(x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (
                    SetExpression::Reference(ReferenceExpression::Constant(x)),
                    SetExpression::Reference(ReferenceExpression::Constant(y)),
                ) => Self::Constant(x.is_subset(&y)),
                (
                    SetExpression::Reference(ReferenceExpression::Variable(x)),
                    SetExpression::Reference(ReferenceExpression::Variable(y)),
                )
                | (SetExpression::StateFunction(x), SetExpression::StateFunction(y))
                    if x == y =>
                {
                    Self::Constant(true)
                }
                (x, y) => Self::IsSubset(x, y),
            },
            Self::IsEmpty(x) => match x.simplify(registry) {
                SetExpression::Reference(ReferenceExpression::Constant(x)) => {
                    Self::Constant(x.count_ones(..) == 0)
                }
                x => Self::IsEmpty(x),
            },
            _ => self.clone(),
        }
    }

    fn evaluate_set_comparison<T, F>(
        x: &SetExpression,
        y: &SetExpression,
        f: F,
        state: &T,
        function_cache: &mut StateFunctionCache,
        state_functions: &StateFunctions,
        registry: &TableRegistry,
    ) -> bool
    where
        T: StateInterface,
        F: Fn(&Set, &Set) -> bool,
    {
        match (x, y) {
            (SetExpression::StateFunction(i), SetExpression::StateFunction(j)) => {
                let (x, y) =
                    function_cache.get_set_value_pair(*i, *j, state, state_functions, registry);
                f(x, y)
            }
            (SetExpression::StateFunction(i), SetExpression::Reference(y)) => {
                let y = y.eval(state, function_cache, state_functions, registry);
                let x = function_cache.get_set_value(*i, state, state_functions, registry);
                f(x, y)
            }
            (SetExpression::Reference(x), SetExpression::StateFunction(i)) => {
                let x = x.eval(state, function_cache, state_functions, registry);
                let y = function_cache.get_set_value(*i, state, state_functions, registry);
                f(x, y)
            }
            (SetExpression::StateFunction(i), y) => {
                let y = y.eval(state, function_cache, state_functions, registry);
                let x = function_cache.get_set_value(*i, state, state_functions, registry);
                f(x, &y)
            }
            (x, SetExpression::StateFunction(i)) => {
                let x = x.eval(state, function_cache, state_functions, registry);
                let y = function_cache.get_set_value(*i, state, state_functions, registry);
                f(&x, y)
            }
            (SetExpression::Reference(x), SetExpression::Reference(y)) => {
                let x = x.eval(state, function_cache, state_functions, registry);
                let y = y.eval(state, function_cache, state_functions, registry);
                f(x, y)
            }
            (SetExpression::Reference(x), y) => {
                let x = x.eval(state, function_cache, state_functions, registry);
                let y = y.eval(state, function_cache, state_functions, registry);
                f(x, &y)
            }
            (x, SetExpression::Reference(y)) => {
                let x = x.eval(state, function_cache, state_functions, registry);
                let y = y.eval(state, function_cache, state_functions, registry);
                f(&x, y)
            }
            (x, y) => f(
                &x.eval(state, function_cache, state_functions, registry),
                &y.eval(state, function_cache, state_functions, registry),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::set_expression::{SetElementOperation, SetOperator};
    use super::*;
    use crate::state::*;

    fn generate_state() -> State {
        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(0);
        let mut set3 = Set::with_capacity(3);
        set3.insert(0);
        set3.insert(1);
        let set4 = Set::with_capacity(3);
        State {
            signature_variables: SignatureVariables {
                set_variables: vec![set1, set2, set3, set4],
                vector_variables: vec![vec![0, 2], vec![], vec![], vec![]],
                element_variables: vec![1, 2, 3, 4],
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn constant_eval() {
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let expression = SetCondition::Constant(true);
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::Constant(false);
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_in_eval() {
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let expression = SetCondition::IsIn(
            ElementExpression::Constant(0),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsIn(
            ElementExpression::Constant(1),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsIn(
            ElementExpression::Constant(2),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsIn(
            ElementExpression::Constant(0),
            SetExpression::Complement(Box::new(SetExpression::Reference(
                ReferenceExpression::Variable(0),
            ))),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsIn(
            ElementExpression::Constant(1),
            SetExpression::Complement(Box::new(SetExpression::Reference(
                ReferenceExpression::Variable(0),
            ))),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsIn(
            ElementExpression::Constant(2),
            SetExpression::Complement(Box::new(SetExpression::Reference(
                ReferenceExpression::Variable(0),
            ))),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_in_state_function_eval() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(1));
        assert!(f.is_ok());
        let f = f.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = SetCondition::IsIn(1.into(), f);

        assert!(condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_equal_eval() {
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(2)),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(3)),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(1)),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(2)),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(3)),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(2)),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(3)),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(2)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(3)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(2)),
            )))),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(3)),
            )))),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(1)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(2)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(3)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(2)),
            )))),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(3)),
            )))),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(2)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(3)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_equal_reference_state_function_eval_false() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(1));
        assert!(f.is_ok());
        let f = f.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = SetCondition::IsEqual(v.into(), f);

        assert!(!condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_equal_reference_state_function_eval_true() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.into());
        assert!(f.is_ok());
        let f = f.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = SetCondition::IsEqual(v.into(), f);

        assert!(condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_equal_state_function_reference_eval_false() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(1));
        assert!(f.is_ok());
        let f = f.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = SetCondition::IsEqual(f, v.into());

        assert!(!condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_equal_state_function_reference_eval_true() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.into());
        assert!(f.is_ok());
        let f = f.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = SetCondition::IsEqual(f, v.into());

        assert!(condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_equal_state_function_state_function_eval_false() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(1));
        assert!(f.is_ok());
        let f = f.unwrap();
        let g = state_functions.add_set_function("g", v.add(2));
        assert!(g.is_ok());
        let g = g.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = SetCondition::IsEqual(f, g);

        assert!(!condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_equal_state_function_state_function_eval_true() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(1));
        assert!(f.is_ok());
        let f = f.unwrap();
        let g = state_functions.add_set_function("g", v.add(1));
        assert!(g.is_ok());
        let g = g.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = SetCondition::IsEqual(f, g);

        assert!(condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_equal_state_function_expression_eval_false() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(1));
        assert!(f.is_ok());
        let f = f.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = SetCondition::IsEqual(f, v.add(2));

        assert!(!condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_equal_state_function_expression_eval_true() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(1));
        assert!(f.is_ok());
        let f = f.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = SetCondition::IsEqual(f, v.add(1));

        assert!(condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_equal_expression_state_function_eval_false() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(1));
        assert!(f.is_ok());
        let f = f.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = SetCondition::IsEqual(v.add(2), f);

        assert!(!condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_equal_expression_state_function_eval_true() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(1));
        assert!(f.is_ok());
        let f = f.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = SetCondition::IsEqual(v.add(1), f);

        assert!(condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_not_equal_eval() {
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(2)),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(3)),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(1)),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(2)),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(3)),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(2)),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(3)),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(2)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(3)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(2)),
            )))),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(3)),
            )))),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(1)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(2)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(3)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(2)),
            )))),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(3)),
            )))),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(2)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(3)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_not_equal_reference_state_function_eval_false() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.into());
        assert!(f.is_ok());
        let f = f.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = SetCondition::IsNotEqual(v.into(), f);

        assert!(!condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_not_equal_reference_state_function_eval_true() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(1));
        assert!(f.is_ok());
        let f = f.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = SetCondition::IsNotEqual(v.into(), f);

        assert!(condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_not_equal_state_function_reference_eval_false() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.into());
        assert!(f.is_ok());
        let f = f.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = SetCondition::IsNotEqual(f, v.into());

        assert!(!condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_not_equal_state_function_reference_eval_true() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(1));
        assert!(f.is_ok());
        let f = f.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = SetCondition::IsNotEqual(f, v.into());

        assert!(condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_not_equal_state_function_state_function_eval_false() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(1));
        assert!(f.is_ok());
        let f = f.unwrap();
        let g = state_functions.add_set_function("g", v.add(1));
        assert!(g.is_ok());
        let g = g.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = SetCondition::IsNotEqual(f, g);

        assert!(!condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_not_equal_state_function_state_function_eval_true() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(1));
        assert!(f.is_ok());
        let f = f.unwrap();
        let g = state_functions.add_set_function("g", v.add(2));
        assert!(g.is_ok());
        let g = g.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = SetCondition::IsNotEqual(f, g);

        assert!(condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_not_equal_state_function_expression_eval_false() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(1));
        assert!(f.is_ok());
        let f = f.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = SetCondition::IsNotEqual(f, v.add(1));

        assert!(!condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_not_equal_state_function_expression_eval_true() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(1));
        assert!(f.is_ok());
        let f = f.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = SetCondition::IsNotEqual(f, v.add(2));

        assert!(condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_not_equal_expression_state_function_eval_false() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(1));
        assert!(f.is_ok());
        let f = f.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = SetCondition::IsNotEqual(v.add(1), f);

        assert!(!condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_not_equal_expression_state_function_eval_true() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(1));
        assert!(f.is_ok());
        let f = f.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = SetCondition::IsNotEqual(v.add(1), f);

        assert!(!condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_subset_eval() {
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(2)),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(3)),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(1)),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(2)),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(3)),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(2)),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(3)),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(2)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(3)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(2)),
            )))),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(3)),
            )))),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(1)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(2)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(3)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(2)),
            )))),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(3)),
            )))),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(2)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(3)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
        );
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_subset_reference_state_function_eval_false() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.remove(0));
        assert!(f.is_ok());
        let f = f.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![{
                    let mut set = Set::with_capacity(3);
                    set.insert(0);
                    set
                }],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = SetCondition::IsSubset(v.into(), f);

        assert!(!condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_subset_reference_state_function_eval_true() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(1));
        assert!(f.is_ok());
        let f = f.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = SetCondition::IsSubset(v.into(), f);

        assert!(condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_subset_state_function_reference_eval_false() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(1));
        assert!(f.is_ok());
        let f = f.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![{
                    let mut set = Set::with_capacity(3);
                    set.insert(0);
                    set
                }],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = SetCondition::IsSubset(f, v.into());

        assert!(!condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_subset_state_function_reference_eval_true() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.remove(0));
        assert!(f.is_ok());
        let f = f.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![{
                    let mut set = Set::with_capacity(3);
                    set.insert(0);
                    set
                }],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = SetCondition::IsSubset(f, v.into());

        assert!(condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_subset_state_function_state_function_eval_false() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(1));
        assert!(f.is_ok());
        let f = f.unwrap();
        let g = state_functions.add_set_function("g", v.add(2));
        assert!(g.is_ok());
        let g = g.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = SetCondition::IsSubset(f, g);

        assert!(!condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_subset_state_function_state_function_eval_true() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(1));
        assert!(f.is_ok());
        let f = f.unwrap();
        let g = state_functions.add_set_function("g", v.add(1).add(2));
        assert!(g.is_ok());
        let g = g.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = SetCondition::IsSubset(f, g);

        assert!(condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_subset_state_function_expression_eval_false() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(1));
        assert!(f.is_ok());
        let f = f.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = SetCondition::IsSubset(f, v.add(2));

        assert!(!condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_subset_state_function_expression_eval_true() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(1));
        assert!(f.is_ok());
        let f = f.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = SetCondition::IsSubset(f, v.add(1).add(2));

        assert!(condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_subset_expression_state_function_eval_false() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(1));
        assert!(f.is_ok());
        let f = f.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = SetCondition::IsSubset(v.add(2), f);

        assert!(!condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_subset_expression_state_function_eval_true() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(1).add(2));
        assert!(f.is_ok());
        let f = f.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = SetCondition::IsSubset(v.add(1), f);

        assert!(condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_empty_eval() {
        let state = generate_state();
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let expression =
            SetCondition::IsEmpty(SetExpression::Reference(ReferenceExpression::Variable(0)));
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression =
            SetCondition::IsEmpty(SetExpression::Reference(ReferenceExpression::Variable(3)));
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsEmpty(SetExpression::Complement(Box::new(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        )));
        assert!(!expression.eval(&state, &mut function_cache, &state_functions, &registry));

        let expression = SetCondition::IsEmpty(SetExpression::Complement(Box::new(
            SetExpression::SetOperation(
                SetOperator::Union,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(2))),
            ),
        )));
        assert!(expression.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_empty_state_function_eval() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.remove(1));
        assert!(f.is_ok());
        let f = f.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(3)],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let condition = f.is_empty();

        assert!(condition.eval(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn constant_simplify() {
        let registry = TableRegistry::default();
        let expression = SetCondition::Constant(true);
        assert_eq!(expression.simplify(&registry), expression);

        let expression = SetCondition::Constant(false);
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn is_in_simplify() {
        let registry = TableRegistry::default();

        let mut set = Set::with_capacity(3);
        set.insert(0);
        let expression = SetCondition::IsIn(
            ElementExpression::Constant(0),
            SetExpression::Reference(ReferenceExpression::Constant(set)),
        );
        assert_eq!(expression.simplify(&registry), SetCondition::Constant(true));

        let mut set = Set::with_capacity(3);
        set.insert(0);
        let expression = SetCondition::IsIn(
            ElementExpression::Constant(1),
            SetExpression::Reference(ReferenceExpression::Constant(set)),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetCondition::Constant(false)
        );

        let expression = SetCondition::IsIn(
            ElementExpression::Constant(0),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn is_equal_simplify() {
        let registry = TableRegistry::default();

        let mut x = Set::with_capacity(3);
        x.insert(0);
        let mut y = Set::with_capacity(3);
        y.insert(0);
        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Constant(x)),
            SetExpression::Reference(ReferenceExpression::Constant(y)),
        );
        assert_eq!(expression.simplify(&registry), SetCondition::Constant(true));

        let mut x = Set::with_capacity(3);
        x.insert(2);
        let mut y = Set::with_capacity(3);
        y.insert(0);
        y.insert(1);
        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Constant(x)),
            SetExpression::Reference(ReferenceExpression::Constant(y)),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetCondition::Constant(false)
        );

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(expression.simplify(&registry), SetCondition::Constant(true));

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn is_equal_same_state_function_simplify() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.remove(1));
        assert!(f.is_ok());
        let f = f.unwrap();

        let condition = SetCondition::IsEqual(f.clone(), f);
        let registry = TableRegistry::default();

        assert_eq!(condition.simplify(&registry), SetCondition::Constant(true));
    }

    #[test]
    fn is_equal_different_state_function_simplify() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.remove(1));
        assert!(f.is_ok());
        let f = f.unwrap();
        let g = state_functions.add_set_function("g", v.add(1));
        assert!(g.is_ok());
        let g = g.unwrap();

        let condition = SetCondition::IsEqual(f.clone(), g);
        let registry = TableRegistry::default();

        assert_eq!(condition.simplify(&registry), condition);
    }

    #[test]
    fn is_not_equal_simplify() {
        let registry = TableRegistry::default();

        let mut x = Set::with_capacity(3);
        x.insert(0);
        let mut y = Set::with_capacity(3);
        y.insert(0);
        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Constant(x)),
            SetExpression::Reference(ReferenceExpression::Constant(y)),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetCondition::Constant(false)
        );

        let mut x = Set::with_capacity(3);
        x.insert(2);
        let mut y = Set::with_capacity(3);
        y.insert(0);
        y.insert(1);
        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Constant(x)),
            SetExpression::Reference(ReferenceExpression::Constant(y)),
        );
        assert_eq!(expression.simplify(&registry), SetCondition::Constant(true));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetCondition::Constant(false)
        );

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn is_not_equal_same_state_function_simplify() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.remove(1));
        assert!(f.is_ok());
        let f = f.unwrap();

        let condition = SetCondition::IsNotEqual(f.clone(), f);
        let registry = TableRegistry::default();

        assert_eq!(condition.simplify(&registry), SetCondition::Constant(false));
    }

    #[test]
    fn is_not_equal_different_state_function_simplify() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.remove(1));
        assert!(f.is_ok());
        let f = f.unwrap();
        let g = state_functions.add_set_function("g", v.add(1));
        assert!(g.is_ok());
        let g = g.unwrap();

        let condition = SetCondition::IsNotEqual(f.clone(), g);
        let registry = TableRegistry::default();

        assert_eq!(condition.simplify(&registry), condition);
    }

    #[test]
    fn is_subset_simplify() {
        let registry = TableRegistry::default();

        let mut x = Set::with_capacity(3);
        x.insert(0);
        let mut y = Set::with_capacity(3);
        y.insert(0);
        y.insert(1);
        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Constant(x)),
            SetExpression::Reference(ReferenceExpression::Constant(y)),
        );
        assert_eq!(expression.simplify(&registry), SetCondition::Constant(true));

        let mut x = Set::with_capacity(3);
        x.insert(2);
        let mut y = Set::with_capacity(3);
        y.insert(0);
        y.insert(1);
        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Constant(x)),
            SetExpression::Reference(ReferenceExpression::Constant(y)),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetCondition::Constant(false)
        );

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(expression.simplify(&registry), SetCondition::Constant(true));

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn is_subset_same_state_function_simplify() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.remove(1));
        assert!(f.is_ok());
        let f = f.unwrap();

        let condition = SetCondition::IsSubset(f.clone(), f);
        let registry = TableRegistry::default();

        assert_eq!(condition.simplify(&registry), SetCondition::Constant(true));
    }

    #[test]
    fn is_subset_different_state_function_simplify() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.remove(1));
        assert!(f.is_ok());
        let f = f.unwrap();
        let g = state_functions.add_set_function("g", v.add(1));
        assert!(g.is_ok());
        let g = g.unwrap();

        let condition = SetCondition::IsSubset(f.clone(), g);
        let registry = TableRegistry::default();

        assert_eq!(condition.simplify(&registry), condition);
    }

    #[test]
    fn is_empty_simplify() {
        let registry = TableRegistry::default();

        let expression = SetCondition::IsEmpty(SetExpression::Reference(
            ReferenceExpression::Constant(Set::with_capacity(3)),
        ));
        assert_eq!(expression.simplify(&registry), SetCondition::Constant(true));

        let expression =
            SetCondition::IsEmpty(SetExpression::Reference(ReferenceExpression::Variable(0)));
        assert_eq!(expression.simplify(&registry), expression);
    }
}
