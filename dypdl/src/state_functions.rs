use crate::{
    expression::*, state::StateInterface, table_registry::TableRegistry, util, util::ModelErr,
    variable_type::*,
};

use rustc_hash::FxHashMap;

/// Definition of state functions.
#[derive(Debug, PartialEq, Clone, Default)]
pub struct StateFunctions {
    /// Map from a set function id to the name.
    pub set_function_names: Vec<String>,
    /// Map from a name to a set function id.
    pub name_to_set_function: FxHashMap<String, usize>,
    /// Set functions.
    pub set_functions: Vec<SetExpression>,

    /// Map from a element function id to the name.
    pub element_function_names: Vec<String>,
    /// Map from a name to an element function id.
    pub name_to_element_function: FxHashMap<String, usize>,
    /// Element functions.
    pub element_functions: Vec<ElementExpression>,

    /// Map from an integer function id to the name.
    pub integer_function_names: Vec<String>,
    /// Map from a name to an integer function id.
    pub name_to_integer_function: FxHashMap<String, usize>,
    /// Integer functions.
    pub integer_functions: Vec<IntegerExpression>,

    /// Map from a continuous function id to the name.
    pub continuous_function_names: Vec<String>,
    /// Map from a name to a continuous function id.
    pub name_to_continuous_function: FxHashMap<String, usize>,
    /// Continuous functions.
    pub continuous_functions: Vec<ContinuousExpression>,

    /// Map from a boolean function id to the name.
    pub boolean_function_names: Vec<String>,
    /// Map from a name to a boolean function id.
    pub name_to_boolean_function: FxHashMap<String, usize>,
    /// Boolean functions.
    pub boolean_functions: Vec<Condition>,
}

macro_rules! define_handle {
    ($x:ident) => {
        /// A struct wrapping an id.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $x(usize);

        impl $x {
            /// Returns the id.
            #[inline]
            pub fn id(&self) -> usize {
                self.0
            }
        }
    };
}

define_handle!(SetStateFunction);
define_handle!(ElementStateFunction);
define_handle!(IntegerStateFunction);
define_handle!(ContinuousStateFunction);
define_handle!(BooleanStateFunction);

impl StateFunctions {
    /// Returns a set function given a name.
    ///
    /// # Errors
    ///
    /// If no such function.
    pub fn get_set_function(&self, name: &str) -> Result<SetExpression, ModelErr> {
        let id = util::get_id(name, &self.name_to_set_function)?;

        Ok(SetExpression::StateFunction(id))
    }

    /// Adds and returns a set function.
    ///
    /// # Errors
    ///
    /// If the name is already used.
    pub fn add_set_function<T>(
        &mut self,
        name: T,
        expression: SetExpression,
    ) -> Result<SetExpression, ModelErr>
    where
        String: From<T>,
    {
        let id = util::add_name(
            name,
            &mut self.set_function_names,
            &mut self.name_to_set_function,
        )?;
        self.set_functions.push(expression);

        Ok(SetExpression::StateFunction(id))
    }

    /// Returns an element function given a name.
    ///
    /// # Errors
    ///
    /// If no such function.
    pub fn get_element_function(&self, name: &str) -> Result<ElementExpression, ModelErr> {
        let id = util::get_id(name, &self.name_to_element_function)?;

        Ok(ElementExpression::StateFunction(id))
    }

    /// Adds and returns an element function.
    ///
    /// # Errors
    ///
    /// If the name is already used.
    pub fn add_element_function<T>(
        &mut self,
        name: T,
        expression: ElementExpression,
    ) -> Result<ElementExpression, ModelErr>
    where
        String: From<T>,
    {
        let id = util::add_name(
            name,
            &mut self.element_function_names,
            &mut self.name_to_element_function,
        )?;
        self.element_functions.push(expression);

        Ok(ElementExpression::StateFunction(id))
    }

    /// Returns an integer function given a name.
    ///
    /// # Errors
    ///
    /// If no such function.
    pub fn get_integer_function(&self, name: &str) -> Result<IntegerExpression, ModelErr> {
        let id = util::get_id(name, &self.name_to_integer_function)?;

        Ok(IntegerExpression::StateFunction(id))
    }

    /// Adds and returns an integer function.
    ///
    /// # Errors
    ///
    /// If the name is already used.
    pub fn add_integer_function<T>(
        &mut self,
        name: T,
        expression: IntegerExpression,
    ) -> Result<IntegerExpression, ModelErr>
    where
        String: From<T>,
    {
        let id = util::add_name(
            name,
            &mut self.integer_function_names,
            &mut self.name_to_integer_function,
        )?;
        self.integer_functions.push(expression);

        Ok(IntegerExpression::StateFunction(id))
    }

    /// Returns an continuous function given a name.
    ///
    /// # Errors
    ///
    /// If no such function.
    pub fn get_continuous_function(&self, name: &str) -> Result<ContinuousExpression, ModelErr> {
        let id = util::get_id(name, &self.name_to_continuous_function)?;

        Ok(ContinuousExpression::StateFunction(id))
    }

    /// Adds and returns a continuous function.
    ///
    /// # Errors
    ///
    /// If the name is already used.
    pub fn add_continuous_function<T>(
        &mut self,
        name: T,
        expression: ContinuousExpression,
    ) -> Result<ContinuousExpression, ModelErr>
    where
        String: From<T>,
    {
        let id = util::add_name(
            name,
            &mut self.continuous_function_names,
            &mut self.name_to_continuous_function,
        )?;
        self.continuous_functions.push(expression);

        Ok(ContinuousExpression::StateFunction(id))
    }

    /// Returns an boolean function given a name.
    ///
    /// # Errors
    ///
    /// If no such function.
    pub fn get_boolean_function(&self, name: &str) -> Result<Condition, ModelErr> {
        let id = util::get_id(name, &self.name_to_boolean_function)?;

        Ok(Condition::StateFunction(id))
    }

    /// Adds and returns a boolean function.
    ///
    /// # Errors
    ///
    /// If the name is already used.
    pub fn add_boolean_function<T>(
        &mut self,
        name: T,
        expression: Condition,
    ) -> Result<Condition, ModelErr>
    where
        String: From<T>,
    {
        let id = util::add_name(
            name,
            &mut self.boolean_function_names,
            &mut self.name_to_boolean_function,
        )?;
        self.boolean_functions.push(expression);

        Ok(Condition::StateFunction(id))
    }
}

/// Data structure to cache the values of state functions.
///
/// This can be used multiple times for evaluating expressions with the same state.
/// If the state is changed, the cache must be cleared.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StateFunctionCache {
    set_values: Vec<Option<Set>>,
    element_values: Vec<Option<Element>>,
    integer_values: Vec<Option<Integer>>,
    continuous_values: Vec<Option<Continuous>>,
    boolean_values: Vec<Option<bool>>,
}

macro_rules! define_getter {
    ($name:ident, $type:ty, $field:ident, $functions:ident) => {
        /// Get the value of a state function.
        ///
        /// # Panics
        ///
        /// If the function with id `i` does not exists.
        pub fn $name<S>(
            &mut self,
            i: usize,
            state: &S,
            functions: &StateFunctions,
            registry: &TableRegistry,
        ) -> $type
        where
            S: StateInterface,
        {
            if self.$field[i].is_none() {
                self.$field[i] =
                    Some(functions.$functions[i].eval(state, self, functions, &registry))
            }

            self.$field[i].unwrap()
        }
    };
}

macro_rules! define_setter {
    ($name:ident, $type:ty, $field:ident) => {
        /// Set the value of the state function.
        pub fn $name(&mut self, i: usize, value: $type) {
            self.$field[i] = Some(value);
        }
    };
}

impl StateFunctionCache {
    /// Create a new state function cache.
    pub fn new(state_functions: &StateFunctions) -> Self {
        let set_variables = vec![None; state_functions.set_functions.len()];
        let element_variables = vec![None; state_functions.element_functions.len()];
        let integer_variables = vec![None; state_functions.integer_functions.len()];
        let continuous_variables = vec![None; state_functions.continuous_functions.len()];
        let boolean_variables = vec![None; state_functions.boolean_functions.len()];

        Self {
            set_values: set_variables,
            element_values: element_variables,
            integer_values: integer_variables,
            continuous_values: continuous_variables,
            boolean_values: boolean_variables,
        }
    }

    /// Get the value of a state function.
    ///
    /// # Panics
    ///
    /// If the function with id `i` does not exists.
    pub fn get_set_value<S: StateInterface>(
        &mut self,
        i: usize,
        state: &S,
        functions: &StateFunctions,
        registry: &TableRegistry,
    ) -> &Set {
        if self.set_values[i].is_none() {
            self.set_values[i] =
                Some(functions.set_functions[i].eval(state, self, functions, registry))
        }

        self.set_values[i].as_ref().unwrap()
    }

    /// Get the values of two set state functions.
    ///
    /// # Panics
    ///
    /// If the function with id `i` or `j` does not exists.
    pub fn get_set_value_pair<S>(
        &mut self,
        i: usize,
        j: usize,
        state: &S,
        functions: &StateFunctions,
        registry: &TableRegistry,
    ) -> (&Set, &Set)
    where
        S: StateInterface,
    {
        // Evaluate the state functions.
        self.get_set_value(i, state, functions, registry);
        self.get_set_value(j, state, functions, registry);

        (
            self.set_values[i].as_ref().unwrap(),
            self.set_values[j].as_ref().unwrap(),
        )
    }

    define_setter!(set_set_value, Set, set_values);

    define_getter!(
        get_element_value,
        Element,
        element_values,
        element_functions
    );

    define_setter!(set_element_value, Element, element_values);

    define_getter!(
        get_integer_value,
        Integer,
        integer_values,
        integer_functions
    );

    define_setter!(set_integer_value, Integer, integer_values);

    define_getter!(
        get_continuous_value,
        Continuous,
        continuous_values,
        continuous_functions
    );

    define_setter!(set_continuous_value, Continuous, continuous_values);

    define_getter!(get_boolean_value, bool, boolean_values, boolean_functions);

    define_setter!(set_boolean_value, bool, boolean_values);

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.set_values.iter_mut().for_each(|v| *v = None);
        self.element_values.iter_mut().for_each(|v| *v = None);
        self.integer_values.iter_mut().for_each(|v| *v = None);
        self.continuous_values.iter_mut().for_each(|v| *v = None);
        self.boolean_values.iter_mut().for_each(|v| *v = None);
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::state::{SignatureVariables, State, StateMetadata};

    #[test]
    fn test_add_and_get_set_function() {
        let mut state_metadata = StateMetadata::default();

        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let v = state_metadata.add_set_variable("v", ob.unwrap());
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();

        let f = state_functions.add_set_function("f", v.add(1));
        assert!(f.is_ok());
        assert_eq!(f.unwrap(), SetExpression::StateFunction(0));

        let g = state_functions.add_set_function("g", v.remove(1));
        assert!(g.is_ok());
        assert_eq!(g.unwrap(), SetExpression::StateFunction(1));

        let f = state_functions.add_set_function("f", v.add(1));
        assert!(f.is_err());

        let f = state_functions.get_set_function("f");
        assert!(f.is_ok());
        assert_eq!(f.unwrap(), SetExpression::StateFunction(0));

        let g = state_functions.get_set_function("g");
        assert!(g.is_ok());
        assert_eq!(g.unwrap(), SetExpression::StateFunction(1));

        let h = state_functions.get_set_function("h");
        assert!(h.is_err());
    }

    #[test]
    fn tet_add_and_get_element_function() {
        let mut state_metadata = StateMetadata::default();

        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let v = state_metadata.add_element_variable("v", ob.unwrap());
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();

        let f = state_functions.add_element_function("f", v + 1);
        assert!(f.is_ok());
        assert_eq!(f.unwrap(), ElementExpression::StateFunction(0));

        let g = state_functions.add_element_function("g", v - 1);
        assert!(g.is_ok());
        assert_eq!(g.unwrap(), ElementExpression::StateFunction(1));

        let f = state_functions.add_element_function("f", v - 1);
        assert!(f.is_err());

        let f = state_functions.get_element_function("f");
        assert!(f.is_ok());
        assert_eq!(f.unwrap(), ElementExpression::StateFunction(0));

        let g = state_functions.get_element_function("g");
        assert!(g.is_ok());
        assert_eq!(g.unwrap(), ElementExpression::StateFunction(1));

        let h = state_functions.get_element_function("h");
        assert!(h.is_err());
    }

    #[test]
    fn test_add_and_get_integer_function_ok() {
        let mut state_metadata = StateMetadata::default();

        let v = state_metadata.add_integer_variable("v");
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();

        let f = state_functions.add_integer_function("f", v + 1);
        assert!(f.is_ok());
        assert_eq!(f.unwrap(), IntegerExpression::StateFunction(0));

        let g = state_functions.add_integer_function("g", v - 1);
        assert!(g.is_ok());
        assert_eq!(g.unwrap(), IntegerExpression::StateFunction(1));

        let f = state_functions.add_integer_function("f", v - 1);
        assert!(f.is_err());

        let f = state_functions.get_integer_function("f");
        assert!(f.is_ok());
        assert_eq!(f.unwrap(), IntegerExpression::StateFunction(0));

        let g = state_functions.get_integer_function("g");
        assert!(g.is_ok());
        assert_eq!(g.unwrap(), IntegerExpression::StateFunction(1));

        let h = state_functions.get_integer_function("h");
        assert!(h.is_err());
    }

    #[test]
    fn test_add_and_get_continuous_function() {
        let mut state_metadata = StateMetadata::default();

        let v = state_metadata.add_continuous_variable("v");
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();

        let f = state_functions.add_continuous_function("f", v + 1);
        assert!(f.is_ok());
        assert_eq!(f.unwrap(), ContinuousExpression::StateFunction(0));

        let g = state_functions.add_continuous_function("g", v - 1);
        assert!(g.is_ok());
        assert_eq!(g.unwrap(), ContinuousExpression::StateFunction(1));

        let f = state_functions.add_continuous_function("f", v - 1);
        assert!(f.is_err());

        let f = state_functions.get_continuous_function("f");
        assert!(f.is_ok());
        assert_eq!(f.unwrap(), ContinuousExpression::StateFunction(0));

        let g = state_functions.get_continuous_function("g");
        assert!(g.is_ok());
        assert_eq!(g.unwrap(), ContinuousExpression::StateFunction(1));

        let h = state_functions.get_continuous_function("h");
        assert!(h.is_err());
    }

    #[test]
    fn test_add_and_get_boolean_function() {
        let mut state_metadata = StateMetadata::default();

        let v = state_metadata.add_integer_variable("v");
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();

        let f = state_functions
            .add_boolean_function("f", Condition::comparison_i(ComparisonOperator::Eq, v, 1));
        assert!(f.is_ok());
        assert_eq!(f.unwrap(), Condition::StateFunction(0));

        let g = state_functions
            .add_boolean_function("g", Condition::comparison_i(ComparisonOperator::Ne, v, 1));
        assert!(g.is_ok());
        assert_eq!(g.unwrap(), Condition::StateFunction(1));

        let h = state_functions
            .add_boolean_function("h", Condition::comparison_i(ComparisonOperator::Eq, v, 1));
        assert!(h.is_ok());

        let f = state_functions.get_boolean_function("f");
        assert!(f.is_ok());
        assert_eq!(f.unwrap(), Condition::StateFunction(0));

        let g = state_functions.get_boolean_function("g");
        assert!(g.is_ok());
        assert_eq!(g.unwrap(), Condition::StateFunction(1));

        let h = state_functions.get_boolean_function("h");
        assert!(h.is_ok());
    }

    #[test]
    fn test_get_set_value() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(0));
        assert!(f.is_ok());
        let g = state_functions.add_set_function("g", v.add(2));
        assert!(g.is_ok());

        let set = state_metadata.create_set(ob, &[1]);
        assert!(set.is_ok());
        let set = set.unwrap();
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![set],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let expected1 = state_metadata.create_set(ob, &[0, 1]);
        assert!(expected1.is_ok());
        let expected1 = expected1.unwrap();
        assert_eq!(
            function_cache.get_set_value(0, &state, &state_functions, &registry),
            &expected1
        );

        let expected2 = state_metadata.create_set(ob, &[1, 2]);
        assert!(expected2.is_ok());
        let expected2 = expected2.unwrap();
        assert_eq!(
            function_cache.get_set_value(1, &state, &state_functions, &registry),
            &expected2
        );

        assert_eq!(
            function_cache.get_set_value(0, &state, &state_functions, &registry),
            &expected1
        );
        assert_eq!(
            function_cache.get_set_value(1, &state, &state_functions, &registry),
            &expected2
        );
    }

    #[should_panic]
    #[test]
    fn test_get_set_value_panic() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(0));
        assert!(f.is_ok());

        let set = state_metadata.create_set(ob, &[1]);
        assert!(set.is_ok());
        let set = set.unwrap();
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![set],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        function_cache.get_set_value(1, &state, &state_functions, &registry);
    }

    #[test]
    fn test_get_set_value_pairs() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(0));
        assert!(f.is_ok());
        let g = state_functions.add_set_function("g", v.add(2));
        assert!(g.is_ok());

        let set = state_metadata.create_set(ob, &[1]);
        assert!(set.is_ok());
        let set = set.unwrap();
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![set],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let expected1 = state_metadata.create_set(ob, &[0, 1]);
        assert!(expected1.is_ok());
        let expected1 = expected1.unwrap();
        let expected2 = state_metadata.create_set(ob, &[1, 2]);
        assert!(expected2.is_ok());
        let expected2 = expected2.unwrap();
        assert_eq!(
            function_cache.get_set_value_pair(0, 1, &state, &state_functions, &registry),
            (&expected1, &expected2)
        );
        assert_eq!(
            function_cache.get_set_value_pair(0, 1, &state, &state_functions, &registry),
            (&expected1, &expected2)
        );
    }

    #[should_panic]
    #[test]
    fn test_get_set_value_pairs_panic1() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(0));
        assert!(f.is_ok());

        let mut state_functions = StateFunctions::default();
        let g = state_functions.add_set_function("g", v.add(2));
        assert!(g.is_ok());

        let set = state_metadata.create_set(ob, &[1]);
        assert!(set.is_ok());
        let set = set.unwrap();
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![set],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        function_cache.get_set_value_pair(2, 1, &state, &state_functions, &registry);
    }

    #[should_panic]
    #[test]
    fn test_get_set_value_pairs_panic2() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(0));
        assert!(f.is_ok());

        let mut state_functions = StateFunctions::default();
        let g = state_functions.add_set_function("g", v.add(2));
        assert!(g.is_ok());

        let set = state_metadata.create_set(ob, &[1]);
        assert!(set.is_ok());
        let set = set.unwrap();
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![set],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        function_cache.get_set_value_pair(0, 2, &state, &state_functions, &registry);
    }

    #[test]
    fn test_set_set_value() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(0));
        assert!(f.is_ok());
        let g = state_functions.add_set_function("g", v.add(0));
        assert!(g.is_ok());

        let set = state_metadata.create_set(ob, &[1]);
        assert!(set.is_ok());
        let set = set.unwrap();
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![set],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        let expected1 = state_metadata.create_set(ob, &[0, 1]);
        assert!(expected1.is_ok());
        let expected1 = expected1.unwrap();

        function_cache.set_set_value(0, expected1.clone());

        assert_eq!(
            function_cache.get_set_value(0, &state, &state_functions, &registry),
            &expected1
        );

        let expected2 = state_metadata.create_set(ob, &[1, 2]);
        assert!(expected2.is_ok());
        let expected2 = expected2.unwrap();

        function_cache.set_set_value(1, expected2.clone());

        assert_eq!(
            function_cache.get_set_value(1, &state, &state_functions, &registry),
            &expected2
        );

        assert_eq!(
            function_cache.get_set_value(0, &state, &state_functions, &registry),
            &expected1
        );
        assert_eq!(
            function_cache.get_set_value(1, &state, &state_functions, &registry),
            &expected2
        );
    }

    #[should_panic]
    #[test]
    fn test_set_set_value_panic() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_set_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_set_function("f", v.add(0));
        assert!(f.is_ok());

        let mut function_cache = StateFunctionCache::new(&state_functions);

        let expected = state_metadata.create_set(ob, &[0, 1]);
        assert!(expected.is_ok());
        let expected = expected.unwrap();
        function_cache.set_set_value(1, expected);
    }

    #[test]
    fn test_get_element_value() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_element_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_element_function("f", v + 1);
        assert!(f.is_ok());
        let g = state_functions.add_element_function("g", v + 2);
        assert!(g.is_ok());

        let state = State {
            signature_variables: SignatureVariables {
                element_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        assert_eq!(
            function_cache.get_element_value(0, &state, &state_functions, &registry),
            1
        );
        assert_eq!(
            function_cache.get_element_value(1, &state, &state_functions, &registry),
            2
        );
        assert_eq!(
            function_cache.get_element_value(0, &state, &state_functions, &registry),
            1
        );
        assert_eq!(
            function_cache.get_element_value(1, &state, &state_functions, &registry),
            2
        );
    }

    #[should_panic]
    #[test]
    fn test_get_element_value_panic() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_element_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_element_function("f", v + 1);
        assert!(f.is_ok());

        let state = State {
            signature_variables: SignatureVariables {
                element_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        function_cache.get_element_value(1, &state, &state_functions, &registry);
    }

    #[test]
    fn test_set_element_value() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_element_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_element_function("f", v + 1);
        assert!(f.is_ok());
        let g = state_functions.add_element_function("g", v + 2);
        assert!(g.is_ok());

        let state = State {
            signature_variables: SignatureVariables {
                element_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        function_cache.set_element_value(0, 1);

        assert_eq!(
            function_cache.get_element_value(0, &state, &state_functions, &registry),
            1
        );

        function_cache.set_element_value(1, 2);

        assert_eq!(
            function_cache.get_element_value(1, &state, &state_functions, &registry),
            2
        );

        assert_eq!(
            function_cache.get_element_value(0, &state, &state_functions, &registry),
            1
        );

        assert_eq!(
            function_cache.get_element_value(1, &state, &state_functions, &registry),
            2
        );
    }

    #[should_panic]
    #[test]
    fn test_set_element_value_panic() {
        let mut state_metadata = StateMetadata::default();
        let ob = state_metadata.add_object_type("ob", 3);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = state_metadata.add_element_variable("v", ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_element_function("f", v + 1);
        assert!(f.is_ok());

        let mut function_cache = StateFunctionCache::new(&state_functions);

        function_cache.set_element_value(1, 1);
    }

    #[test]
    fn test_get_integer_value() {
        let mut state_metadata = StateMetadata::default();
        let v = state_metadata.add_integer_variable("v");
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_integer_function("f", v + 1);
        assert!(f.is_ok());
        let g = state_functions.add_integer_function("g", v + 2);
        assert!(g.is_ok());

        let state = State {
            signature_variables: SignatureVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        assert_eq!(
            function_cache.get_integer_value(0, &state, &state_functions, &registry),
            1
        );
        assert_eq!(
            function_cache.get_integer_value(1, &state, &state_functions, &registry),
            2
        );
        assert_eq!(
            function_cache.get_integer_value(0, &state, &state_functions, &registry),
            1
        );
        assert_eq!(
            function_cache.get_integer_value(1, &state, &state_functions, &registry),
            2
        );
    }

    #[should_panic]
    #[test]
    fn test_get_integer_value_panic() {
        let mut state_metadata = StateMetadata::default();
        let v = state_metadata.add_integer_variable("v");
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_integer_function("f", v + 1);
        assert!(f.is_ok());

        let state = State {
            signature_variables: SignatureVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        function_cache.get_integer_value(1, &state, &state_functions, &registry);
    }

    #[test]
    fn test_set_integer_value() {
        let mut state_metadata = StateMetadata::default();
        let v = state_metadata.add_integer_variable("v");
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_integer_function("f", v + 1);
        assert!(f.is_ok());
        let g = state_functions.add_integer_function("g", v + 2);
        assert!(g.is_ok());

        let state = State {
            signature_variables: SignatureVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        function_cache.set_integer_value(0, 1);

        assert_eq!(
            function_cache.get_integer_value(0, &state, &state_functions, &registry),
            1
        );

        function_cache.set_integer_value(1, 2);

        assert_eq!(
            function_cache.get_integer_value(1, &state, &state_functions, &registry),
            2
        );

        assert_eq!(
            function_cache.get_integer_value(0, &state, &state_functions, &registry),
            1
        );

        assert_eq!(
            function_cache.get_integer_value(1, &state, &state_functions, &registry),
            2
        );
    }

    #[should_panic]
    #[test]
    fn test_set_integer_value_panic() {
        let mut state_metadata = StateMetadata::default();
        let v = state_metadata.add_integer_variable("v");
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_integer_function("f", v + 1);
        assert!(f.is_ok());

        let mut function_cache = StateFunctionCache::new(&state_functions);

        function_cache.set_integer_value(1, 1);
    }

    #[test]
    fn test_get_continuous_value() {
        let mut state_metadata = StateMetadata::default();
        let v = state_metadata.add_continuous_variable("v");
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_continuous_function("f", v + 1);
        assert!(f.is_ok());
        let g = state_functions.add_continuous_function("g", v + 2);
        assert!(g.is_ok());

        let state = State {
            signature_variables: SignatureVariables {
                continuous_variables: vec![0.0],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        assert_relative_eq!(
            function_cache.get_continuous_value(0, &state, &state_functions, &registry),
            1.0
        );
        assert_relative_eq!(
            function_cache.get_continuous_value(1, &state, &state_functions, &registry),
            2.0
        );
        assert_relative_eq!(
            function_cache.get_continuous_value(0, &state, &state_functions, &registry),
            1.0
        );
        assert_relative_eq!(
            function_cache.get_continuous_value(1, &state, &state_functions, &registry),
            2.0
        );
    }

    #[should_panic]
    #[test]
    fn test_get_continuous_value_panic() {
        let mut state_metadata = StateMetadata::default();
        let v = state_metadata.add_continuous_variable("v");
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_continuous_function("f", v + 1);
        assert!(f.is_ok());

        let state = State {
            signature_variables: SignatureVariables {
                continuous_variables: vec![0.0],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        function_cache.get_continuous_value(1, &state, &state_functions, &registry);
    }

    #[test]
    fn test_set_continuous_value() {
        let mut state_metadata = StateMetadata::default();
        let v = state_metadata.add_continuous_variable("v");
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_continuous_function("f", v + 1);
        assert!(f.is_ok());
        let g = state_functions.add_continuous_function("g", v + 2);
        assert!(g.is_ok());

        let state = State {
            signature_variables: SignatureVariables {
                continuous_variables: vec![0.0],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        function_cache.set_continuous_value(0, 1.0);

        assert_relative_eq!(
            function_cache.get_continuous_value(0, &state, &state_functions, &registry),
            1.0
        );

        function_cache.set_continuous_value(1, 2.0);

        assert_relative_eq!(
            function_cache.get_continuous_value(1, &state, &state_functions, &registry),
            2.0
        );

        assert_relative_eq!(
            function_cache.get_continuous_value(0, &state, &state_functions, &registry),
            1.0
        );
        assert_relative_eq!(
            function_cache.get_continuous_value(1, &state, &state_functions, &registry),
            2.0
        );
    }

    #[should_panic]
    #[test]
    fn test_set_continuous_value_panic() {
        let mut state_metadata = StateMetadata::default();
        let v = state_metadata.add_continuous_variable("v");
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_continuous_function("f", v + 1);
        assert!(f.is_ok());

        let mut function_cache = StateFunctionCache::new(&state_functions);

        function_cache.set_continuous_value(1, 1.0);
    }

    #[test]
    fn test_get_boolean_value() {
        let mut state_metadata = StateMetadata::default();
        let v = state_metadata.add_integer_variable("v");
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions
            .add_boolean_function("f", Condition::comparison_i(ComparisonOperator::Eq, v, 0));
        assert!(f.is_ok());
        let g = state_functions
            .add_boolean_function("g", Condition::comparison_i(ComparisonOperator::Ne, v, 0));
        assert!(g.is_ok());

        let state = State {
            signature_variables: SignatureVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        assert!(function_cache.get_boolean_value(0, &state, &state_functions, &registry));
        assert!(!function_cache.get_boolean_value(1, &state, &state_functions, &registry));
        assert!(function_cache.get_boolean_value(0, &state, &state_functions, &registry));
        assert!(!function_cache.get_boolean_value(1, &state, &state_functions, &registry));
    }

    #[should_panic]
    #[test]
    fn test_get_boolean_value_panic() {
        let mut state_metadata = StateMetadata::default();
        let v = state_metadata.add_integer_variable("v");
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions
            .add_boolean_function("f", Condition::comparison_i(ComparisonOperator::Eq, v, 0));
        assert!(f.is_ok());

        let state = State {
            signature_variables: SignatureVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        function_cache.get_boolean_value(1, &state, &state_functions, &registry);
    }

    #[test]
    fn test_set_boolean_value() {
        let mut state_metadata = StateMetadata::default();
        let v = state_metadata.add_integer_variable("v");
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions
            .add_boolean_function("f", Condition::comparison_i(ComparisonOperator::Eq, v, 0));
        assert!(f.is_ok());
        let g = state_functions
            .add_boolean_function("g", Condition::comparison_i(ComparisonOperator::Ne, v, 0));
        assert!(g.is_ok());

        let state = State {
            signature_variables: SignatureVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        function_cache.set_boolean_value(0, true);

        assert!(function_cache.get_boolean_value(0, &state, &state_functions, &registry),);

        function_cache.set_boolean_value(1, false);

        assert!(!function_cache.get_boolean_value(1, &state, &state_functions, &registry),);

        assert!(function_cache.get_boolean_value(0, &state, &state_functions, &registry),);
        assert!(!function_cache.get_boolean_value(1, &state, &state_functions, &registry),);
    }

    #[should_panic]
    #[test]
    fn test_set_boolean_value_panic() {
        let mut state_metadata = StateMetadata::default();
        let v = state_metadata.add_integer_variable("v");
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions
            .add_boolean_function("f", Condition::comparison_i(ComparisonOperator::Eq, v, 0));
        assert!(f.is_ok());

        let mut function_cache = StateFunctionCache::new(&state_functions);

        function_cache.set_boolean_value(1, true);
    }

    #[test]
    fn test_clear() {
        let mut state_metadata = StateMetadata::default();
        let v = state_metadata.add_integer_variable("v");
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut state_functions = StateFunctions::default();
        let f = state_functions.add_integer_function("f", v + 1);
        assert!(f.is_ok());

        let state = State {
            signature_variables: SignatureVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut function_cache = StateFunctionCache::new(&state_functions);
        let registry = TableRegistry::default();

        assert_eq!(
            function_cache.get_integer_value(0, &state, &state_functions, &registry),
            1
        );

        let state = State {
            signature_variables: SignatureVariables {
                integer_variables: vec![1],
                ..Default::default()
            },
            ..Default::default()
        };

        function_cache.clear();

        assert_eq!(
            function_cache.get_integer_value(0, &state, &state_functions, &registry),
            2
        );
    }
}
