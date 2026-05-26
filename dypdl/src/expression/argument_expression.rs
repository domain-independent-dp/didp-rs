use super::element_expression::ElementExpression;
use super::reference_expression::ReferenceExpression;
use super::set_expression::SetExpression;
use super::util;
use crate::state::{ElementResourceVariable, ElementVariable, SetVariable, StateInterface};
use crate::state_functions::{StateFunctionCache, StateFunctions};
use crate::table_registry::TableRegistry;
use crate::variable_type::{Element, Set};

/// An enum used to preform reduce operations over constants in a table.
#[derive(Debug, PartialEq, Clone)]
pub enum ArgumentExpression {
    Set(SetExpression),
    Element(ElementExpression),
}

impl From<SetExpression> for ArgumentExpression {
    #[inline]
    fn from(v: SetExpression) -> ArgumentExpression {
        Self::Set(v)
    }
}

impl From<ElementExpression> for ArgumentExpression {
    #[inline]
    fn from(v: ElementExpression) -> ArgumentExpression {
        Self::Element(v)
    }
}

macro_rules! impl_from {
    ($T:ty,$U:ty) => {
        impl From<$T> for ArgumentExpression {
            #[inline]
            fn from(v: $T) -> ArgumentExpression {
                Self::from(<$U>::from(v))
            }
        }
    };
}

impl_from!(Set, SetExpression);
impl_from!(SetVariable, SetExpression);
impl_from!(Element, ElementExpression);
impl_from!(ElementVariable, ElementExpression);
impl_from!(ElementResourceVariable, ElementExpression);

impl ArgumentExpression {
    /// Returns a simplified version by precomutation.
    ///
    /// # Panics
    ///
    /// Panics if a min/max reduce operation is performed on an empty set or vector.
    pub fn simplify(&self, registry: &TableRegistry) -> ArgumentExpression {
        match self {
            Self::Set(expression) => ArgumentExpression::Set(expression.simplify(registry)),
            Self::Element(expression) => ArgumentExpression::Element(expression.simplify(registry)),
        }
    }

    /// Returns a set of indices computed by cartesian products of args.
    ///
    /// # Panics
    ///
    /// Panics if the cost of the transitioned state is used or an empty set or vector is passed to a reduce operation or a min/max reduce operation is performed on an empty set or vector.
    pub fn eval_args<'a, I, U: StateInterface>(
        args: I,
        state: &U,
        function_cache: &mut StateFunctionCache,
        state_functions: &StateFunctions,
        registry: &TableRegistry,
    ) -> Vec<Vec<Element>>
    where
        I: Iterator<Item = &'a ArgumentExpression>,
    {
        let mut result = vec![vec![]];
        for expression in args {
            match expression {
                ArgumentExpression::Set(set) => {
                    result = match set {
                        SetExpression::Reference(set) => {
                            let set = set.eval(state, function_cache, state_functions, registry);
                            util::expand_vector_with_set(result, set)
                        }
                        SetExpression::StateFunction(i) => {
                            let set =
                                function_cache.get_set_value(*i, state, state_functions, registry);
                            util::expand_vector_with_set(result, set)
                        }
                        _ => util::expand_vector_with_set(
                            result,
                            &set.eval(state, function_cache, state_functions, registry),
                        ),
                    }
                }
                ArgumentExpression::Element(element) => {
                    let element = element.eval(state, function_cache, state_functions, registry);
                    result.iter_mut().for_each(|r| r.push(element));
                }
            }
        }
        result
    }

    /// Returns a set of indices if args only contain constants.
    ///
    /// # Panics
    ///
    /// Panics if a min/max reduce operation is performed on an empty set or vector.
    pub fn simplify_args<'a, I>(args: I) -> Option<Vec<Vec<Element>>>
    where
        I: Iterator<Item = &'a ArgumentExpression>,
    {
        let mut simplified_args = vec![vec![]];
        for expression in args {
            match expression {
                ArgumentExpression::Set(SetExpression::Reference(
                    ReferenceExpression::Constant(set),
                )) => simplified_args = util::expand_vector_with_set(simplified_args, set),
                ArgumentExpression::Element(ElementExpression::Constant(element)) => {
                    simplified_args.iter_mut().for_each(|r| r.push(*element));
                }
                _ => return None,
            }
        }
        Some(simplified_args)
    }
}

#[cfg(test)]
mod tests {
    use super::super::condition::Condition;
    use super::super::reference_expression::ReferenceExpression;
    use super::super::set_expression::SetElementOperation;
    use super::*;
    use crate::state::{SignatureVariables, State, StateMetadata};

    #[test]
    fn from_set_expression() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("Something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let s = metadata.create_set(ob, &[2, 4]);
        assert!(s.is_ok());
        let s = s.unwrap();
        assert_eq!(
            ArgumentExpression::from(SetExpression::Reference(ReferenceExpression::Constant(
                s.clone()
            ))),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(s)))
        );
    }

    #[test]
    fn from_set() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("Something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let s = metadata.create_set(ob, &[2, 4]);
        assert!(s.is_ok());
        let s = s.unwrap();

        assert_eq!(
            ArgumentExpression::from(s.clone()),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(s)))
        );
    }

    #[test]
    fn from_set_variable() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("Something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_set_variable(String::from("sv"), ob);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            ArgumentExpression::from(v),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Variable(
                v.id()
            )))
        );
    }

    #[test]
    fn from_element_expression() {
        assert_eq!(
            ArgumentExpression::from(ElementExpression::Constant(1)),
            ArgumentExpression::Element(ElementExpression::Constant(1)),
        );
    }

    #[test]
    fn from_element() {
        assert_eq!(
            ArgumentExpression::from(1),
            ArgumentExpression::Element(ElementExpression::Constant(1)),
        );
    }

    #[test]
    fn from_element_variable() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("Something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let v = metadata.add_element_variable(String::from("ev"), ob);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            ArgumentExpression::from(v),
            ArgumentExpression::Element(ElementExpression::Variable(v.id())),
        );
    }

    #[test]
    fn from_element_resource_variable() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("Something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let v = metadata.add_element_resource_variable(String::from("erv"), ob, true);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            ArgumentExpression::from(v),
            ArgumentExpression::Element(ElementExpression::ResourceVariable(v.id())),
        );
    }

    #[test]
    fn simplify_element() {
        let registry = TableRegistry::default();
        let expression = ArgumentExpression::Element(ElementExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(ElementExpression::Constant(0)),
            Box::new(ElementExpression::Constant(1)),
        ));
        assert_eq!(
            expression.simplify(&registry),
            ArgumentExpression::Element(ElementExpression::Constant(0))
        );
    }

    #[test]
    fn simplify_set() {
        let registry = TableRegistry::default();
        let expression = ArgumentExpression::Set(SetExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
        ));
        assert_eq!(
            expression.simplify(&registry),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Variable(0)))
        );
    }

    #[test]
    fn eval_args_with_state_function() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.remove(0).remove(1).add(4).add(5));
        assert!(f.is_ok());
        let f = f.unwrap();
        let g = state_functions.add_set_function("g", v.remove(0).remove(1).add(6).add(7));
        assert!(g.is_ok());
        let g = g.unwrap();

        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![{
                    let mut set = Set::with_capacity(10);
                    set.insert(0);
                    set.insert(1);
                    set
                }],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();
        let args = [
            ElementExpression::from(8).into(),
            v.into(),
            f.into(),
            v.remove(0).remove(1).add(2).add(3).into(),
            g.into(),
        ];
        assert_eq!(
            ArgumentExpression::eval_args(
                args.iter(),
                &state,
                &mut function_cache,
                &state_functions,
                &registry
            ),
            vec![
                vec![8, 0, 4, 2, 6],
                vec![8, 0, 4, 2, 7],
                vec![8, 0, 4, 3, 6],
                vec![8, 0, 4, 3, 7],
                vec![8, 0, 5, 2, 6],
                vec![8, 0, 5, 2, 7],
                vec![8, 0, 5, 3, 6],
                vec![8, 0, 5, 3, 7],
                vec![8, 1, 4, 2, 6],
                vec![8, 1, 4, 2, 7],
                vec![8, 1, 4, 3, 6],
                vec![8, 1, 4, 3, 7],
                vec![8, 1, 5, 2, 6],
                vec![8, 1, 5, 2, 7],
                vec![8, 1, 5, 3, 6],
                vec![8, 1, 5, 3, 7],
            ]
        );
    }
}
