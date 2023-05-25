use crate::effect;
use crate::table_registry;
use crate::util::ModelErr;
use crate::variable_type::{Continuous, Element, Integer, Set, Vector};
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::panic;

/// Trait representing a state in DyPDL.
pub trait StateInterface: Sized {
    /// Returns the number of set variables;
    fn get_number_of_set_variables(&self) -> usize;

    /// Returns the value of a set variable.
    ///
    /// # Panics
    ///
    /// Panics if no variable has the id of `i`.
    fn get_set_variable(&self, i: usize) -> &Set;

    /// Returns the number of vector variables;
    fn get_number_of_vector_variables(&self) -> usize;

    /// Returns the value of a vector variable.
    ///
    /// # Panics
    ///
    /// Panics if no variable has the id of `i`.
    fn get_vector_variable(&self, i: usize) -> &Vector;

    /// Returns the number of element variables;
    fn get_number_of_element_variables(&self) -> usize;

    /// Returns the value of an element variable.
    ///
    /// # Panics
    ///
    /// Panics if no variable has the id of `i`.
    fn get_element_variable(&self, i: usize) -> Element;

    /// Returns the number of integer numeric variables;
    fn get_number_of_integer_variables(&self) -> usize;

    /// Returns the value of an integer numeric variable.
    ///
    /// # Panics
    ///
    /// Panics if no variable has the id of `i`.
    fn get_integer_variable(&self, i: usize) -> Integer;

    /// Returns the number of continuous numeric variables;
    fn get_number_of_continuous_variables(&self) -> usize;

    /// Returns the value of a continuous numeric variable.
    ///
    /// # Panics
    ///
    /// Panics if no variable has the id of `i`.
    fn get_continuous_variable(&self, i: usize) -> Continuous;

    /// Returns the number of element resource variables;
    fn get_number_of_element_resource_variables(&self) -> usize;

    /// Returns the value of an element resource variable.
    ///
    /// # Panics
    ///
    /// Panics if no variable has the id of `i`.
    fn get_element_resource_variable(&self, i: usize) -> Element;

    /// Returns the number of integer resource variables;
    fn get_number_of_integer_resource_variables(&self) -> usize;

    /// Returns the value of an integer resource variable.
    ///
    /// # Panics
    ///
    /// Panics if no variable has the id of `i`.
    fn get_integer_resource_variable(&self, i: usize) -> Integer;

    /// Returns the number of continuous resource variables;
    fn get_number_of_continuous_resource_variables(&self) -> usize;

    /// Returns the value of a continuous resource variable.
    ///
    /// # Panics
    ///
    /// Panics if no variable has the id of `i`.
    fn get_continuous_resource_variable(&self, i: usize) -> Continuous;

    /// Returns the transitioned state by the effect.
    ///
    /// # Panics
    ///
    /// Panics if the cost of the transition state is used or a min/max reduce operation is performed on an empty set or vector.
    fn apply_effect<T: From<State>>(
        &self,
        effect: &effect::Effect,
        registry: &table_registry::TableRegistry,
    ) -> T {
        let len = self.get_number_of_set_variables();
        let mut set_variables = Vec::with_capacity(len);
        let mut i = 0;
        for e in &effect.set_effects {
            while i < e.0 {
                set_variables.push(self.get_set_variable(i).clone());
                i += 1;
            }
            set_variables.push(e.1.eval(self, registry));
            i += 1;
        }
        while i < len {
            set_variables.push(self.get_set_variable(i).clone());
            i += 1;
        }

        let len = self.get_number_of_vector_variables();
        let mut vector_variables = Vec::with_capacity(len);
        for e in &effect.vector_effects {
            while i < e.0 {
                vector_variables.push(self.get_vector_variable(i).clone());
                i += 1;
            }
            vector_variables.push(e.1.eval(self, registry));
            i += 1;
        }
        while i < len {
            vector_variables.push(self.get_vector_variable(i).clone());
            i += 1;
        }

        let mut element_variables: Vec<usize> = (0..self.get_number_of_element_variables())
            .map(|i| self.get_element_variable(i))
            .collect();
        for e in &effect.element_effects {
            element_variables[e.0] = e.1.eval(self, registry);
        }

        let mut integer_variables: Vec<Integer> = (0..self.get_number_of_integer_variables())
            .map(|i| self.get_integer_variable(i))
            .collect();
        for e in &effect.integer_effects {
            integer_variables[e.0] = e.1.eval(self, registry);
        }

        let mut continuous_variables: Vec<Continuous> = (0..self
            .get_number_of_continuous_variables())
            .map(|i| self.get_continuous_variable(i))
            .collect();
        for e in &effect.continuous_effects {
            continuous_variables[e.0] = e.1.eval(self, registry);
        }

        let mut element_resource_variables: Vec<usize> = (0..self
            .get_number_of_element_resource_variables())
            .map(|i| self.get_element_resource_variable(i))
            .collect();
        for e in &effect.element_resource_effects {
            element_resource_variables[e.0] = e.1.eval(self, registry);
        }

        let mut integer_resource_variables: Vec<Integer> = (0..self
            .get_number_of_integer_resource_variables())
            .map(|i| self.get_integer_resource_variable(i))
            .collect();
        for e in &effect.integer_resource_effects {
            integer_resource_variables[e.0] = e.1.eval(self, registry);
        }

        let mut continuous_resource_variables: Vec<Continuous> = (0..self
            .get_number_of_continuous_resource_variables())
            .map(|i| self.get_continuous_resource_variable(i))
            .collect();
        for e in &effect.continuous_resource_effects {
            continuous_resource_variables[e.0] = e.1.eval(self, registry);
        }

        T::from(State {
            signature_variables: SignatureVariables {
                set_variables,
                vector_variables,
                element_variables,
                integer_variables,
                continuous_variables,
            },
            resource_variables: ResourceVariables {
                element_variables: element_resource_variables,
                integer_variables: integer_resource_variables,
                continuous_variables: continuous_resource_variables,
            },
        })
    }
}

/// State variables other than resource variables.
#[derive(Debug, PartialEq, Clone, Default)]
pub struct SignatureVariables {
    /// Set variables.
    pub set_variables: Vec<Set>,
    /// Vector variables.
    pub vector_variables: Vec<Vector>,
    /// Element variables.
    pub element_variables: Vec<Element>,
    /// Integer numeric variables.
    pub integer_variables: Vec<Integer>,
    /// Continuous numeric variables.
    pub continuous_variables: Vec<Continuous>,
}

/// Resource variables.
#[derive(Debug, PartialEq, Clone, Default)]
pub struct ResourceVariables {
    /// Element variables.
    pub element_variables: Vec<Element>,
    /// Integer numeric variables.
    pub integer_variables: Vec<Integer>,
    /// Continuous numeric variables.
    pub continuous_variables: Vec<Continuous>,
}

/// State in DyPDL.
#[derive(Debug, PartialEq, Clone, Default)]
pub struct State {
    /// Variables other than resource variables
    pub signature_variables: SignatureVariables,
    /// Resource variables
    pub resource_variables: ResourceVariables,
}

impl StateInterface for State {
    /// Returns the number of set variables.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// let set = model.create_set(object_type, &[0, 1, 2, 3]).unwrap();
    /// model.add_set_variable("variable", object_type, set).unwrap();
    /// let state = model.target.clone();
    ///
    /// assert_eq!(state.get_number_of_set_variables(), 1);
    /// ```
    #[inline]
    fn get_number_of_set_variables(&self) -> usize {
        self.signature_variables.set_variables.len()
    }

    /// Returns the value of a set variable.
    ///
    /// # Panics
    ///
    /// Panics if no variable has the id of `i`.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// let set = model.create_set(object_type, &[0, 1, 2, 3]).unwrap();
    /// let variable = model.add_set_variable("variable", object_type, set.clone()).unwrap();
    /// let state = model.target.clone();
    ///
    /// assert_eq!(state.get_set_variable(variable.id()), &set);
    /// ```
    #[inline]
    fn get_set_variable(&self, i: usize) -> &Set {
        &self.signature_variables.set_variables[i]
    }

    /// Returns the number of vector variables;
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// model.add_vector_variable("variable", object_type, vec![0, 1, 2, 3]).unwrap();
    /// let state = model.target.clone();
    ///
    /// assert_eq!(state.get_number_of_vector_variables(), 1);
    /// ```
    #[inline]
    fn get_number_of_vector_variables(&self) -> usize {
        self.signature_variables.vector_variables.len()
    }

    /// Returns the value of a vector variable.
    ///
    /// # Panics
    ///
    /// Panics if no variable has the id of `i`.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// let variable = model.add_vector_variable("variable", object_type, vec![0, 1, 2, 3]).unwrap();
    /// let state = model.target.clone();
    ///
    /// assert_eq!(state.get_vector_variable(variable.id()), &vec![0, 1, 2, 3]);
    #[inline]
    fn get_vector_variable(&self, i: usize) -> &Vector {
        &self.signature_variables.vector_variables[i]
    }

    /// Returns the number of vector variables;
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// model.add_element_variable("variable", object_type, 0).unwrap();
    /// let state = model.target.clone();
    ///
    /// assert_eq!(state.get_number_of_element_variables(), 1);
    /// ```
    #[inline]
    fn get_number_of_element_variables(&self) -> usize {
        self.signature_variables.element_variables.len()
    }

    /// Returns the value of an element variable.
    ///
    /// # Panics
    ///
    /// Panics if no variable has the id of `i`.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// let variable = model.add_element_variable("variable", object_type, 0).unwrap();
    /// let state = model.target.clone();
    ///
    /// assert_eq!(state.get_element_variable(variable.id()), 0);
    /// ```
    #[inline]
    fn get_element_variable(&self, i: usize) -> Element {
        self.signature_variables.element_variables[i]
    }

    /// Returns the number of integer numeric variables;
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// model.add_integer_variable("variable", 0).unwrap();
    /// let state = model.target.clone();
    ///
    /// assert_eq!(state.get_number_of_integer_variables(), 1);
    /// ```
    #[inline]
    fn get_number_of_integer_variables(&self) -> usize {
        self.signature_variables.integer_variables.len()
    }

    /// Returns the value of an integer numeric variable.
    ///
    /// # Panics
    ///
    /// Panics if no variable has the id of `i`.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let variable = model.add_integer_variable("variable", 0).unwrap();
    /// let state = model.target.clone();
    ///
    /// assert_eq!(state.get_integer_variable(variable.id()), 0);
    /// ```
    #[inline]
    fn get_integer_variable(&self, i: usize) -> Integer {
        self.signature_variables.integer_variables[i]
    }

    /// Returns the number of continuous numeric variables;
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// model.add_continuous_variable("variable", 0.5).unwrap();
    /// let state = model.target.clone();
    ///
    /// assert_eq!(state.get_number_of_continuous_variables(), 1);
    /// ```
    #[inline]
    fn get_number_of_continuous_variables(&self) -> usize {
        self.signature_variables.continuous_variables.len()
    }

    /// Returns the value of a continuous numeric variable.
    ///
    /// # Panics
    ///
    /// Panics if no variable has the id of `i`.
    ///
    /// # Examples
    ///
    /// ```
    /// use approx::assert_relative_eq;
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let variable = model.add_continuous_variable("variable", 0.5).unwrap();
    /// let state = model.target.clone();
    ///
    /// assert_relative_eq!(state.get_continuous_variable(variable.id()), 0.5);
    /// ```
    #[inline]
    fn get_continuous_variable(&self, i: usize) -> Continuous {
        self.signature_variables.continuous_variables[i]
    }

    /// Returns the number of element resource variables;
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// model.add_element_resource_variable("variable", object_type, false, 0).unwrap();
    /// let state = model.target.clone();
    ///
    /// assert_eq!(state.get_number_of_element_resource_variables(), 1);
    /// ```
    #[inline]
    fn get_number_of_element_resource_variables(&self) -> usize {
        self.resource_variables.element_variables.len()
    }

    /// Returns the value of an element resource variable.
    ///
    /// # Panics
    ///
    /// Panics if no variable has the id of `i`.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// let variable = model.add_element_resource_variable("variable", object_type, false, 0).unwrap();
    /// let state = model.target.clone();
    ///
    /// assert_eq!(state.get_element_resource_variable(variable.id()), 0);
    /// ```
    #[inline]
    fn get_element_resource_variable(&self, i: usize) -> Element {
        self.resource_variables.element_variables[i]
    }

    /// Returns the number of integer resource variables;
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// model.add_integer_resource_variable("variable", false, 0).unwrap();
    /// let state = model.target.clone();
    ///
    /// assert_eq!(state.get_number_of_integer_resource_variables(), 1);
    /// ```
    #[inline]
    fn get_number_of_integer_resource_variables(&self) -> usize {
        self.resource_variables.integer_variables.len()
    }

    /// Returns the value of an integer resource variable.
    ///
    /// # Panics
    ///
    /// Panics if no variable has the id of `i`.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let variable = model.add_integer_resource_variable("variable", false, 0).unwrap();
    /// let state = model.target.clone();
    ///
    /// assert_eq!(state.get_integer_resource_variable(variable.id()), 0);
    /// ```
    #[inline]
    fn get_integer_resource_variable(&self, i: usize) -> Integer {
        self.resource_variables.integer_variables[i]
    }

    /// Returns the number of continuous resource variables;
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// model.add_continuous_resource_variable("variable", false, 0.5).unwrap();
    /// let state = model.target.clone();
    ///
    /// assert_eq!(state.get_number_of_continuous_resource_variables(), 1);
    /// ```
    #[inline]
    fn get_number_of_continuous_resource_variables(&self) -> usize {
        self.resource_variables.continuous_variables.len()
    }

    /// Returns the value of a continuous resource variable.
    ///
    /// # Panics
    ///
    /// Panics if no variable has the id of `i`.
    ///
    /// # Examples
    ///
    /// ```
    /// use approx::assert_relative_eq;
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let variable = model.add_continuous_resource_variable("variable", false, 0.5).unwrap();
    /// let state = model.target.clone();
    ///
    /// assert_relative_eq!(state.get_continuous_resource_variable(variable.id()), 0.5);
    /// ```
    #[inline]
    fn get_continuous_resource_variable(&self, i: usize) -> Continuous {
        self.resource_variables.continuous_variables[i]
    }
}

impl State {
    /// Returns if the given state is equal to the current state.
    ///
    /// # Panics
    ///
    /// Panics if the state metadata is wrong.
    pub fn is_satisfied<U: StateInterface>(&self, state: &U, metadata: &StateMetadata) -> bool {
        for i in 0..metadata.number_of_element_variables() {
            if self.get_element_variable(i) != state.get_element_variable(i) {
                return false;
            }
        }
        for i in 0..metadata.number_of_element_resource_variables() {
            if self.get_element_resource_variable(i) != state.get_element_resource_variable(i) {
                return false;
            }
        }
        for i in 0..metadata.number_of_integer_variables() {
            if self.get_integer_variable(i) != state.get_integer_variable(i) {
                return false;
            }
        }
        for i in 0..metadata.number_of_integer_resource_variables() {
            if self.get_integer_resource_variable(i) != state.get_integer_resource_variable(i) {
                return false;
            }
        }
        for i in 0..metadata.number_of_continuous_variables() {
            if (self.get_continuous_variable(i) - state.get_continuous_variable(i)).abs()
                > Continuous::EPSILON
            {
                return false;
            }
        }
        for i in 0..metadata.number_of_continuous_resource_variables() {
            if (self.get_continuous_resource_variable(i)
                - state.get_continuous_resource_variable(i))
            .abs()
                > Continuous::EPSILON
            {
                return false;
            }
        }
        for i in 0..metadata.number_of_set_variables() {
            if self.get_set_variable(i) != state.get_set_variable(i) {
                return false;
            }
        }
        for i in 0..metadata.number_of_vector_variables() {
            if self.get_vector_variable(i) != state.get_vector_variable(i) {
                return false;
            }
        }
        true
    }
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

define_handle!(ObjectType);
define_handle!(ElementVariable);
define_handle!(ElementResourceVariable);
define_handle!(SetVariable);
define_handle!(VectorVariable);
define_handle!(IntegerVariable);
define_handle!(IntegerResourceVariable);
define_handle!(ContinuousVariable);
define_handle!(ContinuousResourceVariable);

/// Information about state variables.
#[derive(Debug, PartialEq, Clone, Eq, Default)]
pub struct StateMetadata {
    /// Map from an object type id to the name.
    pub object_type_names: Vec<String>,
    /// Map from a name to its object type id.
    pub name_to_object_type: FxHashMap<String, usize>,
    /// Map from an object type id to the number of objects.
    pub object_numbers: Vec<usize>,

    /// Map from a set variable id to the name.
    pub set_variable_names: Vec<String>,
    /// Map from a name to the set variable id.
    pub name_to_set_variable: FxHashMap<String, usize>,
    /// Map from a set variable id to its object type id.
    pub set_variable_to_object: Vec<usize>,

    /// Map from a vector variable id to the name.
    pub vector_variable_names: Vec<String>,
    /// Map from a name to a set variable id.
    pub name_to_vector_variable: FxHashMap<String, usize>,
    /// Map from a vector variable id to its object type id.
    pub vector_variable_to_object: Vec<usize>,

    /// Map from an element variable id to the name.
    pub element_variable_names: Vec<String>,
    /// Map from a name to an element variable id.
    pub name_to_element_variable: FxHashMap<String, usize>,
    /// Map from an element variable id to its object type id.
    pub element_variable_to_object: Vec<usize>,

    /// Map from an integer variable id to the name.
    pub integer_variable_names: Vec<String>,
    /// Map from a name to an integer variable id.
    pub name_to_integer_variable: FxHashMap<String, usize>,

    /// Map from a continuous variable id to the name.
    pub continuous_variable_names: Vec<String>,
    /// Map from a name to a continuous variable id.
    pub name_to_continuous_variable: FxHashMap<String, usize>,

    /// Map from an element resource variable id to the name.
    pub element_resource_variable_names: Vec<String>,
    /// Map from a name to an element resource variable id.
    pub name_to_element_resource_variable: FxHashMap<String, usize>,
    /// Map from an element resource variable id to its object type id.
    pub element_resource_variable_to_object: Vec<usize>,
    /// Map from an element resource variable id to its preference.
    pub element_less_is_better: Vec<bool>,

    /// Map from an integer resource variable id to the name.
    pub integer_resource_variable_names: Vec<String>,
    /// Map from a name to an integer resource variable id.
    pub name_to_integer_resource_variable: FxHashMap<String, usize>,
    /// Map from an integer resource variable id to its preference.
    pub integer_less_is_better: Vec<bool>,

    /// Map from a continuous resource variable id to the name.
    pub continuous_resource_variable_names: Vec<String>,
    /// Map from a name to a continuous resource variable id.
    pub name_to_continuous_resource_variable: FxHashMap<String, usize>,
    /// Map from a continuous resource variable id to its preference.
    pub continuous_less_is_better: Vec<bool>,
}

impl StateMetadata {
    /// Returns the number of object types.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// model.add_object_type("object", 4).unwrap();
    ///
    /// assert_eq!(model.state_metadata.number_of_object_types(), 1);
    /// ```
    #[inline]
    pub fn number_of_object_types(&self) -> usize {
        self.object_type_names.len()
    }

    /// Returns the number of set variables.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// let set = model.create_set(object_type, &[0, 1, 2, 3]).unwrap();
    /// model.add_set_variable("variable", object_type, set).unwrap();
    ///
    /// assert_eq!(model.state_metadata.number_of_set_variables(), 1);
    /// ```
    #[inline]
    pub fn number_of_set_variables(&self) -> usize {
        self.set_variable_names.len()
    }

    /// Returns the number of vector variables.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// model.add_vector_variable("variable", object_type, vec![0, 1, 2, 3]).unwrap();
    ///
    /// assert_eq!(model.state_metadata.number_of_vector_variables(), 1);
    /// ```
    #[inline]
    pub fn number_of_vector_variables(&self) -> usize {
        self.vector_variable_names.len()
    }

    /// Returns the number of element variables.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// model.add_element_variable("variable", object_type, 0).unwrap();
    ///
    /// assert_eq!(model.state_metadata.number_of_element_variables(), 1);
    /// ```
    #[inline]
    pub fn number_of_element_variables(&self) -> usize {
        self.element_variable_names.len()
    }

    /// Returns the number of integer variables.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// model.add_integer_variable("variable", 0).unwrap();
    ///
    /// assert_eq!(model.state_metadata.number_of_integer_variables(), 1);
    /// ```
    #[inline]
    pub fn number_of_integer_variables(&self) -> usize {
        self.integer_variable_names.len()
    }

    /// Returns the number of continuous variables.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// model.add_continuous_variable("variable", 0.5).unwrap();
    ///
    /// assert_eq!(model.state_metadata.number_of_continuous_variables(), 1);
    /// ```
    #[inline]
    pub fn number_of_continuous_variables(&self) -> usize {
        self.continuous_variable_names.len()
    }

    /// Returns the number of element resource variables.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let object_type = model.add_object_type("object", 4).unwrap();
    /// model.add_element_resource_variable("variable", object_type, false, 0).unwrap();
    ///
    /// assert_eq!(model.state_metadata.number_of_element_resource_variables(), 1);
    /// ```
    #[inline]
    pub fn number_of_element_resource_variables(&self) -> usize {
        self.element_resource_variable_names.len()
    }

    /// Returns the number of integer resource variables.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// model.add_integer_resource_variable("variable", false, 0).unwrap();
    ///
    /// assert_eq!(model.state_metadata.number_of_integer_resource_variables(), 1);
    /// ```
    #[inline]
    pub fn number_of_integer_resource_variables(&self) -> usize {
        self.integer_resource_variable_names.len()
    }

    /// Returns the number of continuous resource variables.
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// model.add_continuous_resource_variable("variable", false, 0.5).unwrap();
    ///
    /// assert_eq!(model.state_metadata.number_of_continuous_resource_variables(), 1);
    /// ```
    #[inline]
    pub fn number_of_continuous_resource_variables(&self) -> usize {
        self.continuous_resource_variable_names.len()
    }

    /// Returns true if there is a resource variable and false otherwise.
    pub fn has_resource_variables(&self) -> bool {
        self.number_of_element_resource_variables() > 0
            || self.number_of_integer_resource_variables() > 0
            || self.number_of_continuous_resource_variables() > 0
    }

    /// Returns the set of names used by object types and variables.
    pub fn get_name_set(&self) -> FxHashSet<String> {
        let mut name_set = FxHashSet::default();
        for name in &self.object_type_names {
            name_set.insert(name.clone());
        }
        for name in &self.set_variable_names {
            name_set.insert(name.clone());
        }
        for name in &self.vector_variable_names {
            name_set.insert(name.clone());
        }
        for name in &self.element_variable_names {
            name_set.insert(name.clone());
        }
        for name in &self.integer_variable_names {
            name_set.insert(name.clone());
        }
        for name in &self.continuous_variable_names {
            name_set.insert(name.clone());
        }
        for name in &self.element_resource_variable_names {
            name_set.insert(name.clone());
        }
        for name in &self.integer_resource_variable_names {
            name_set.insert(name.clone());
        }
        for name in &self.continuous_resource_variable_names {
            name_set.insert(name.clone());
        }
        name_set
    }

    /// Returns the dominance relation between two states.
    ///
    /// # Panics
    ///
    /// Panics the metadata is wrong.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    /// use std::cmp::Ordering;
    ///
    /// let mut model = Model::default();
    /// let v1 = model.add_integer_resource_variable("v1", false, 1).unwrap();
    /// let v2 = model.add_continuous_resource_variable("v2", true, 1.5).unwrap();
    /// let a = model.target.clone();
    ///
    /// model.set_target(v1, 0).unwrap();
    /// model.set_target(v2, 2.5).unwrap();
    /// let b = model.target.clone();
    /// assert_eq!(model.state_metadata.dominance(&a, &b), Some(Ordering::Greater));
    ///
    /// model.set_target(v1, 1).unwrap();
    /// model.set_target(v2, 0.5).unwrap();
    /// let b = model.target.clone();
    /// assert_eq!(model.state_metadata.dominance(&a, &b), Some(Ordering::Less));
    ///
    /// model.set_target(v1, 0).unwrap();
    /// model.set_target(v2, 0.5).unwrap();
    /// let b = model.target.clone();
    /// assert_eq!(model.state_metadata.dominance(&a, &b), None);
    /// ```
    pub fn dominance<U: StateInterface, V: StateInterface>(
        &self,
        a: &U,
        b: &V,
    ) -> Option<Ordering> {
        let status = Some(Ordering::Equal);
        let x = |i| a.get_element_resource_variable(i);
        let y = |i| b.get_element_resource_variable(i);
        let status = Self::compare_resource_variables(&x, &y, &self.element_less_is_better, status);
        status?;
        let x = |i| a.get_integer_resource_variable(i);
        let y = |i| b.get_integer_resource_variable(i);
        let status = Self::compare_resource_variables(&x, &y, &self.integer_less_is_better, status);
        status?;
        let x = |i| a.get_continuous_resource_variable(i);
        let y = |i| b.get_continuous_resource_variable(i);
        Self::compare_resource_variables(&x, &y, &self.continuous_less_is_better, status)
    }

    fn compare_resource_variables<T: PartialOrd, F, G>(
        x: &F,
        y: &G,
        less_is_better: &[bool],
        mut status: Option<Ordering>,
    ) -> Option<Ordering>
    where
        F: Fn(usize) -> T,
        G: Fn(usize) -> T,
    {
        for (i, less_is_better) in less_is_better.iter().enumerate() {
            let v1 = x(i);
            let v2 = y(i);
            match status {
                Some(Ordering::Equal) => {
                    if v1 < v2 {
                        if *less_is_better {
                            status = Some(Ordering::Greater);
                        } else {
                            status = Some(Ordering::Less);
                        }
                    }
                    if v1 > v2 {
                        if *less_is_better {
                            status = Some(Ordering::Less);
                        } else {
                            status = Some(Ordering::Greater);
                        }
                    }
                }
                Some(Ordering::Less) => {
                    if v1 < v2 {
                        if *less_is_better {
                            return None;
                        }
                    } else if v1 > v2 && !less_is_better {
                        return None;
                    }
                }
                Some(Ordering::Greater) => {
                    if v1 > v2 {
                        if *less_is_better {
                            return None;
                        }
                    } else if v1 < v2 && !less_is_better {
                        return None;
                    }
                }
                None => {}
            }
        }
        status
    }

    /// Check if a state is valid.
    ///
    /// # Errors
    ///
    /// If a state is invalid, e.g., it contains variables not existing in this model.
    pub fn check_state<'a, T: StateInterface>(&self, state: &'a T) -> Result<(), ModelErr>
    where
        &'a T: panic::UnwindSafe,
    {
        let n = self.number_of_element_variables();
        for i in 0..n {
            let v = panic::catch_unwind(|| state.get_element_variable(i));
            if v.is_err() {
                return Err(ModelErr::new(format!(
                    "{} th element variable does not exists",
                    i
                )));
            }
            let v = v.unwrap();
            let m = self.object_numbers[self.element_variable_to_object[i]];
            if v >= m {
                return Err(ModelErr::new(format!(
                    "value {} for an element variable >= #objects ({})",
                    i, m
                )));
            }
        }
        let n = self.number_of_element_resource_variables();
        for i in 0..n {
            let v = panic::catch_unwind(|| state.get_element_resource_variable(i));
            if v.is_err() {
                return Err(ModelErr::new(format!(
                    "{} th element resource variable does not exist",
                    i
                )));
            }
            let v = v.unwrap();
            let m = self.object_numbers[self.element_resource_variable_to_object[i]];
            if v >= m {
                return Err(ModelErr::new(format!(
                    "value {} for an element resource variable >= #objects ({})",
                    i, m
                )));
            }
        }
        let n = self.number_of_set_variables();
        for i in 0..n {
            let v = panic::catch_unwind(|| state.get_set_variable(i));
            if v.is_err() {
                return Err(ModelErr::new(format!(
                    "{} th set variable does not exists",
                    i
                )));
            }
            let v = v.unwrap();
            let m = self.object_numbers[self.set_variable_to_object[i]];
            if v.len() > m {
                return Err(ModelErr::new(format!(
                    "set size {} for {} th set variable > #objects ({})",
                    v.len(),
                    i,
                    m
                )));
            }
        }
        let n = self.number_of_vector_variables();
        for i in 0..n {
            let v = panic::catch_unwind(|| state.get_vector_variable(i));
            if v.is_err() {
                return Err(ModelErr::new(format!(
                    "{} th vector variable does not exists",
                    i
                )));
            }
            let v = v.unwrap();
            let m = self.object_numbers[self.vector_variable_to_object[i]];
            if v.iter().any(|v| *v >= m) {
                return Err(ModelErr::new(format!(
                    "vector for {} th vector variable contains a value >= #objects ({})",
                    i, m
                )));
            }
        }
        let n = self.number_of_integer_variables();
        for i in 0..n {
            let v = panic::catch_unwind(|| state.get_integer_variable(i));
            if v.is_err() {
                return Err(ModelErr::new(format!(
                    "{} th integer variable does not exist",
                    i
                )));
            }
        }
        let n = self.number_of_integer_resource_variables();
        for i in 0..n {
            let v = panic::catch_unwind(|| state.get_integer_resource_variable(i));
            if v.is_err() {
                return Err(ModelErr::new(format!(
                    "{} th integer resource variable does not exist",
                    i
                )));
            }
        }
        let n = self.number_of_continuous_variables();
        for i in 0..n {
            let v = panic::catch_unwind(|| state.get_continuous_variable(i));
            if v.is_err() {
                return Err(ModelErr::new(format!(
                    "{} th continuous variable does not exist",
                    i
                )));
            }
        }
        let n = self.number_of_continuous_resource_variables();
        for i in 0..n {
            let v = panic::catch_unwind(|| state.get_continuous_resource_variable(i));
            if v.is_err() {
                return Err(ModelErr::new(format!(
                    "{} th continuous resource variable does not exist",
                    i
                )));
            }
        }
        Ok(())
    }

    /// Returns object type given a name.
    ///
    /// # Errors
    ///
    /// If no object type with the name.
    pub fn get_object_type(&self, name: &str) -> Result<ObjectType, ModelErr> {
        if let Some(id) = self.name_to_object_type.get(name) {
            Ok(ObjectType(*id))
        } else {
            Err(ModelErr::new(format!("no such object `{}`", name)))
        }
    }

    /// Adds an object type and returns it.
    ///
    /// # Errors
    ///
    /// If the name is already used.
    pub fn add_object_type<T>(&mut self, name: T, number: usize) -> Result<ObjectType, ModelErr>
    where
        String: From<T>,
    {
        let name = String::from(name);
        match self.name_to_object_type.entry(name) {
            Entry::Vacant(e) => {
                let id = self.object_type_names.len();
                self.object_type_names.push(e.key().clone());
                self.object_numbers.push(number);
                e.insert(id);
                Ok(ObjectType(id))
            }
            Entry::Occupied(e) => Err(ModelErr::new(format!(
                "object `{}` already exists",
                e.key()
            ))),
        }
    }

    /// Returns the number of objects associated with the type.
    ///
    /// # Errors
    ///
    /// If the object type is not in the model.
    pub fn get_number_of_objects(&self, ob: ObjectType) -> Result<usize, ModelErr> {
        self.check_object(ob)?;
        Ok(self.object_numbers[ob.id()])
    }

    // Disabled because it is inconsistent with the other modeling interfaces.
    // /// Change the number of objects.
    // ///
    // /// # Errors
    // ///
    // /// If the object type is not in the model.
    // pub fn set_number_of_object(&mut self, ob: ObjectType, number: usize) -> Result<(), ModelErr> {
    //     self.check_object(ob)?;
    //     self.object_numbers[ob.id()] = number;
    //     Ok(())
    // }

    /// Create a set of objects associated with the type.
    ///
    /// # Errors
    ///
    /// If the object type is not in the model or an input value is greater than or equal to the number of the objects.
    pub fn create_set(&self, ob: ObjectType, array: &[Element]) -> Result<Set, ModelErr> {
        let n = self.get_number_of_objects(ob)?;
        let mut set = Set::with_capacity(n);
        for v in array {
            if *v >= n {
                return Err(ModelErr::new(format!(
                    "index {} for object >= #objects ({})",
                    *v, n
                )));
            }
            set.insert(*v);
        }
        Ok(set)
    }

    /// Returns an element variable given a name.
    ///
    /// # Errors
    ///
    /// If no such variable.
    #[inline]
    pub fn get_element_variable(&self, name: &str) -> Result<ElementVariable, ModelErr> {
        let id = Self::get_variable(name, &self.name_to_element_variable)?;
        Ok(ElementVariable(id))
    }

    /// Adds and returns an element variable.
    ///
    /// The value in the target state must be specified.
    ///
    /// # Errors
    ///
    /// If the name is already used or the object type is not in the model.
    pub fn add_element_variable<T>(
        &mut self,
        name: T,
        ob: ObjectType,
    ) -> Result<ElementVariable, ModelErr>
    where
        String: From<T>,
    {
        self.check_object(ob)?;
        let id = Self::add_variable(
            name,
            &mut self.element_variable_names,
            &mut self.name_to_element_variable,
        )?;
        self.element_variable_to_object.push(ob.id());
        Ok(ElementVariable(id))
    }

    /// Returns an element resource variable given a name.
    ///
    /// # Errors
    ///
    /// If no such variable.
    #[inline]
    pub fn get_element_resource_variable(
        &self,
        name: &str,
    ) -> Result<ElementResourceVariable, ModelErr> {
        let id = Self::get_variable(name, &self.name_to_element_resource_variable)?;
        Ok(ElementResourceVariable(id))
    }

    /// Adds and returns an element resource variable.
    ///
    /// The value in the target state must be specified.
    ///
    /// # Errors
    ///
    /// If the name is already used or the object type is not in the model.
    pub fn add_element_resource_variable<T>(
        &mut self,
        name: T,
        ob: ObjectType,
        less_is_better: bool,
    ) -> Result<ElementResourceVariable, ModelErr>
    where
        String: From<T>,
    {
        self.check_object(ob)?;
        let id = Self::add_variable(
            name,
            &mut self.element_resource_variable_names,
            &mut self.name_to_element_resource_variable,
        )?;
        self.element_resource_variable_to_object.push(ob.id());
        self.element_less_is_better.push(less_is_better);
        Ok(ElementResourceVariable(id))
    }

    /// Returns a set variable given a name.
    ///
    /// # Errors
    ///
    /// If no such variable.
    #[inline]
    pub fn get_set_variable(&self, name: &str) -> Result<SetVariable, ModelErr> {
        let id = Self::get_variable(name, &self.name_to_set_variable)?;
        Ok(SetVariable(id))
    }

    /// Adds and returns a set variable.
    ///
    /// The value in the target state must be specified.
    ///
    /// # Errors
    ///
    /// If the name is already used or the object type is not in the model.
    pub fn add_set_variable<T>(&mut self, name: T, ob: ObjectType) -> Result<SetVariable, ModelErr>
    where
        String: From<T>,
    {
        self.check_object(ob)?;
        let id = Self::add_variable(
            name,
            &mut self.set_variable_names,
            &mut self.name_to_set_variable,
        )?;
        self.set_variable_to_object.push(ob.id());
        Ok(SetVariable(id))
    }

    /// Returns a vector variable given a name.
    ///
    /// # Errors
    ///
    /// If no such variable.
    #[inline]
    pub fn get_vector_variable(&self, name: &str) -> Result<VectorVariable, ModelErr> {
        let id = Self::get_variable(name, &self.name_to_vector_variable)?;
        Ok(VectorVariable(id))
    }

    /// Adds and returns a vector variable.
    ///
    /// The value in the target state must be specified.
    ///
    /// # Errors
    ///
    /// If the name is already used or the object type is not in the model.
    pub fn add_vector_variable<T>(
        &mut self,
        name: T,
        ob: ObjectType,
    ) -> Result<VectorVariable, ModelErr>
    where
        String: From<T>,
    {
        self.check_object(ob)?;
        let id = Self::add_variable(
            name,
            &mut self.vector_variable_names,
            &mut self.name_to_vector_variable,
        )?;
        self.vector_variable_to_object.push(ob.id());
        Ok(VectorVariable(id))
    }

    /// Returns an integer variable given a name.
    ///
    /// # Errors
    ///
    /// If no such variable.
    #[inline]
    pub fn get_integer_variable(&self, name: &str) -> Result<IntegerVariable, ModelErr> {
        let id = Self::get_variable(name, &self.name_to_integer_variable)?;
        Ok(IntegerVariable(id))
    }

    /// Adds and returns an integer variable.
    ///
    /// The value in the target state must be specified.
    ///
    /// # Errors
    ///
    /// If the name is already used.
    pub fn add_integer_variable<T>(&mut self, name: T) -> Result<IntegerVariable, ModelErr>
    where
        String: From<T>,
    {
        let id = Self::add_variable(
            name,
            &mut self.integer_variable_names,
            &mut self.name_to_integer_variable,
        )?;
        Ok(IntegerVariable(id))
    }

    /// Returns an integer resource variable given a name.
    ///
    /// # Errors
    ///
    /// If no such variable.
    #[inline]
    pub fn get_integer_resource_variable(
        &self,
        name: &str,
    ) -> Result<IntegerResourceVariable, ModelErr> {
        let id = Self::get_variable(name, &self.name_to_integer_resource_variable)?;
        Ok(IntegerResourceVariable(id))
    }

    /// Adds and returns an integer resource variable.
    ///
    /// The value in the target state must be specified.
    ///
    /// # Errors
    ///
    /// If the name is already used.
    pub fn add_integer_resource_variable<T>(
        &mut self,
        name: T,
        less_is_better: bool,
    ) -> Result<IntegerResourceVariable, ModelErr>
    where
        String: From<T>,
    {
        let id = Self::add_variable(
            name,
            &mut self.integer_resource_variable_names,
            &mut self.name_to_integer_resource_variable,
        )?;
        self.integer_less_is_better.push(less_is_better);
        Ok(IntegerResourceVariable(id))
    }

    /// Returns a continuous variable given a name.
    ///
    /// # Errors
    ///
    /// If no such variable.
    #[inline]
    pub fn get_continuous_variable(&self, name: &str) -> Result<ContinuousVariable, ModelErr> {
        let id = Self::get_variable(name, &self.name_to_continuous_variable)?;
        Ok(ContinuousVariable(id))
    }

    /// Adds and returns a continuous variable.
    ///
    /// The value in the target state must be specified.
    ///
    /// # Errors
    ///
    /// If the name is already used.
    pub fn add_continuous_variable<T>(&mut self, name: T) -> Result<ContinuousVariable, ModelErr>
    where
        String: From<T>,
    {
        let id = Self::add_variable(
            name,
            &mut self.continuous_variable_names,
            &mut self.name_to_continuous_variable,
        )?;
        Ok(ContinuousVariable(id))
    }

    /// Returns a continuous resource variable given a name.
    ///
    /// # Errors
    ///
    /// If no such variable.
    #[inline]
    pub fn get_continuous_resource_variable(
        &self,
        name: &str,
    ) -> Result<ContinuousResourceVariable, ModelErr> {
        let id = Self::get_variable(name, &self.name_to_continuous_resource_variable)?;
        Ok(ContinuousResourceVariable(id))
    }

    /// Adds and returns a continuous resource variable.
    ///
    /// The value in the target state must be specified.
    ///
    /// # Errors
    ///
    /// If the name is already used.
    pub fn add_continuous_resource_variable<T>(
        &mut self,
        name: T,
        less_is_better: bool,
    ) -> Result<ContinuousResourceVariable, ModelErr>
    where
        String: From<T>,
    {
        let id = Self::add_variable(
            name,
            &mut self.continuous_resource_variable_names,
            &mut self.name_to_continuous_resource_variable,
        )?;
        self.continuous_less_is_better.push(less_is_better);
        Ok(ContinuousResourceVariable(id))
    }

    fn check_object(&self, ob: ObjectType) -> Result<(), ModelErr> {
        if ob.id() >= self.number_of_object_types() {
            Err(ModelErr::new(format!(
                "object id {} >= #object types ({})",
                ob.id(),
                self.object_numbers.len()
            )))
        } else {
            Ok(())
        }
    }

    fn get_variable(
        name: &str,
        name_to_variable: &FxHashMap<String, usize>,
    ) -> Result<usize, ModelErr> {
        if let Some(id) = name_to_variable.get(name) {
            Ok(*id)
        } else {
            Err(ModelErr::new(format!("no such variable `{}`", name)))
        }
    }

    fn add_variable<T>(
        name: T,
        variable_names: &mut Vec<String>,
        name_to_variable: &mut FxHashMap<String, usize>,
    ) -> Result<usize, ModelErr>
    where
        String: From<T>,
    {
        let name = String::from(name);
        match name_to_variable.entry(name) {
            Entry::Vacant(e) => {
                let id = variable_names.len();
                variable_names.push(e.key().clone());
                e.insert(id);
                Ok(id)
            }
            Entry::Occupied(e) => Err(ModelErr::new(format!(
                "variable `{}` already exists",
                e.key()
            ))),
        }
    }
}

/// Trait for checking if a variable is defined.
pub trait CheckVariable<T> {
    /// Check if the variable is defined.
    ///
    /// # Errors
    /// If the variable is not defined.
    fn check_variable(&self, v: T) -> Result<(), ModelErr>;
}

macro_rules! impl_check_variable {
    ($T:ty,$x:ident) => {
        impl CheckVariable<$T> for StateMetadata {
            fn check_variable(&self, v: $T) -> Result<(), ModelErr> {
                let id = v.id();
                let n = self.$x.len();
                if id >= n {
                    Err(ModelErr::new(format!(
                        "variable id {} >= #variables ({})",
                        id, n
                    )))
                } else {
                    Ok(())
                }
            }
        }
    };
}

impl_check_variable!(ElementVariable, element_variable_names);
impl_check_variable!(ElementResourceVariable, element_resource_variable_names);
impl_check_variable!(SetVariable, set_variable_names);
impl_check_variable!(VectorVariable, vector_variable_names);
impl_check_variable!(IntegerVariable, integer_variable_names);
impl_check_variable!(IntegerResourceVariable, integer_resource_variable_names);
impl_check_variable!(ContinuousVariable, continuous_variable_names);
impl_check_variable!(
    ContinuousResourceVariable,
    continuous_resource_variable_names
);

/// Trait for getting the object type.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
///
/// let mut model = Model::default();
/// let object_type = model.add_object_type("object", 4).unwrap();
/// let variable = model.add_element_variable("variable", object_type, 0).unwrap();
///
/// assert!(model.get_object_type_of(variable).is_ok())
/// ```
pub trait GetObjectTypeOf<T> {
    /// Returns the object type of the variable.
    ///
    /// # Errors
    ///
    /// If the variable is not included in the model.
    fn get_object_type_of(&self, v: T) -> Result<ObjectType, ModelErr>;
}

macro_rules! impl_get_object_type_of {
    ($T:ty,$x:ident) => {
        impl GetObjectTypeOf<$T> for StateMetadata {
            fn get_object_type_of(&self, v: $T) -> Result<ObjectType, ModelErr> {
                self.check_variable(v)?;
                Ok(ObjectType(self.$x[v.id()]))
            }
        }
    };
}

impl_get_object_type_of!(ElementVariable, element_variable_to_object);
impl_get_object_type_of!(ElementResourceVariable, element_resource_variable_to_object);
impl_get_object_type_of!(SetVariable, set_variable_to_object);
impl_get_object_type_of!(VectorVariable, vector_variable_to_object);

/// Trait for accessing preference of resource variables.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
///
/// let mut model = Model::default();
/// let variable = model.add_integer_resource_variable("variable", true, 0).unwrap();
///
/// assert!(model.get_preference(variable).unwrap());
/// assert!(model.set_preference(variable, false).is_ok());
/// assert!(!model.get_preference(variable).unwrap());
/// ```
pub trait AccessPreference<T> {
    /// Returns the preference of a resource variable.
    ///
    /// # Errors
    ///
    /// If the variable is not included in the model.
    fn get_preference(&self, v: T) -> Result<bool, ModelErr>;
    /// Sets the preference of a resource variable.
    ///
    /// # Errors
    ///
    /// If the variable is not included in the model.
    fn set_preference(&mut self, v: T, less_is_better: bool) -> Result<(), ModelErr>;
}

macro_rules! impl_access_preference {
    ($T:ty,$x:ident) => {
        impl AccessPreference<$T> for StateMetadata {
            fn get_preference(&self, v: $T) -> Result<bool, ModelErr> {
                self.check_variable(v)?;
                Ok(self.$x[v.id()])
            }

            fn set_preference(&mut self, v: $T, less_is_better: bool) -> Result<(), ModelErr> {
                self.check_variable(v)?;
                self.$x[v.id()] = less_is_better;
                Ok(())
            }
        }
    };
}

impl_access_preference!(ElementResourceVariable, element_less_is_better);
impl_access_preference!(IntegerResourceVariable, integer_less_is_better);
impl_access_preference!(ContinuousResourceVariable, continuous_less_is_better);

#[cfg(test)]
mod tests {
    use super::super::expression::*;
    use super::*;

    fn generate_metadata() -> StateMetadata {
        let object_names = vec![String::from("object"), String::from("small")];
        let object_numbers = vec![10, 2];
        let mut name_to_object = FxHashMap::default();
        name_to_object.insert(String::from("object"), 0);
        name_to_object.insert(String::from("small"), 1);

        let set_variable_names = vec![
            String::from("s0"),
            String::from("s1"),
            String::from("s2"),
            String::from("s3"),
        ];
        let mut name_to_set_variable = FxHashMap::default();
        name_to_set_variable.insert(String::from("s0"), 0);
        name_to_set_variable.insert(String::from("s1"), 1);
        name_to_set_variable.insert(String::from("s2"), 2);
        name_to_set_variable.insert(String::from("s3"), 3);
        let set_variable_to_object = vec![0, 0, 0, 1];

        let vector_variable_names = vec![
            String::from("p0"),
            String::from("p1"),
            String::from("p2"),
            String::from("p3"),
        ];
        let mut name_to_vector_variable = FxHashMap::default();
        name_to_vector_variable.insert(String::from("p0"), 0);
        name_to_vector_variable.insert(String::from("p1"), 1);
        name_to_vector_variable.insert(String::from("p2"), 2);
        name_to_vector_variable.insert(String::from("p3"), 3);
        let vector_variable_to_object = vec![0, 0, 0, 1];

        let element_variable_names = vec![
            String::from("e0"),
            String::from("e1"),
            String::from("e2"),
            String::from("e3"),
        ];
        let mut name_to_element_variable = FxHashMap::default();
        name_to_element_variable.insert(String::from("e0"), 0);
        name_to_element_variable.insert(String::from("e1"), 1);
        name_to_element_variable.insert(String::from("e2"), 2);
        name_to_element_variable.insert(String::from("e3"), 3);
        let element_variable_to_object = vec![0, 0, 0, 1];

        let integer_variable_names = vec![
            String::from("i0"),
            String::from("i1"),
            String::from("i2"),
            String::from("i3"),
        ];
        let mut name_to_integer_variable = FxHashMap::default();
        name_to_integer_variable.insert(String::from("i0"), 0);
        name_to_integer_variable.insert(String::from("i1"), 1);
        name_to_integer_variable.insert(String::from("i2"), 2);
        name_to_integer_variable.insert(String::from("i3"), 3);

        let continuous_variable_names = vec![
            String::from("c0"),
            String::from("c1"),
            String::from("c2"),
            String::from("c3"),
        ];
        let mut name_to_continuous_variable = FxHashMap::default();
        name_to_continuous_variable.insert(String::from("c0"), 0);
        name_to_continuous_variable.insert(String::from("c1"), 1);
        name_to_continuous_variable.insert(String::from("c2"), 2);
        name_to_continuous_variable.insert(String::from("c3"), 3);

        let element_resource_variable_names = vec![
            String::from("er0"),
            String::from("er1"),
            String::from("er2"),
            String::from("er3"),
        ];
        let mut name_to_element_resource_variable = FxHashMap::default();
        name_to_element_resource_variable.insert(String::from("er0"), 0);
        name_to_element_resource_variable.insert(String::from("er1"), 1);
        name_to_element_resource_variable.insert(String::from("er2"), 2);
        name_to_element_resource_variable.insert(String::from("er3"), 3);
        let element_resource_variable_to_object = vec![0, 0, 0, 1];

        let integer_resource_variable_names = vec![
            String::from("ir0"),
            String::from("ir1"),
            String::from("ir2"),
            String::from("ir3"),
        ];
        let mut name_to_integer_resource_variable = FxHashMap::default();
        name_to_integer_resource_variable.insert(String::from("ir0"), 0);
        name_to_integer_resource_variable.insert(String::from("ir1"), 1);
        name_to_integer_resource_variable.insert(String::from("ir2"), 2);
        name_to_integer_resource_variable.insert(String::from("ir3"), 3);

        let continuous_resource_variable_names = vec![
            String::from("cr0"),
            String::from("cr1"),
            String::from("cr2"),
            String::from("cr3"),
        ];
        let mut name_to_continuous_resource_variable = FxHashMap::default();
        name_to_continuous_resource_variable.insert(String::from("cr0"), 0);
        name_to_continuous_resource_variable.insert(String::from("cr1"), 1);
        name_to_continuous_resource_variable.insert(String::from("cr2"), 2);
        name_to_continuous_resource_variable.insert(String::from("cr3"), 3);

        StateMetadata {
            object_type_names: object_names,
            name_to_object_type: name_to_object,
            object_numbers,
            set_variable_names,
            name_to_set_variable,
            set_variable_to_object,
            vector_variable_names,
            name_to_vector_variable,
            vector_variable_to_object,
            element_variable_names,
            name_to_element_variable,
            element_variable_to_object,
            integer_variable_names,
            name_to_integer_variable,
            continuous_variable_names,
            name_to_continuous_variable,
            element_resource_variable_names,
            name_to_element_resource_variable,
            element_resource_variable_to_object,
            element_less_is_better: vec![false, false, true, false],
            integer_resource_variable_names,
            name_to_integer_resource_variable,
            integer_less_is_better: vec![false, false, true, false],
            continuous_resource_variable_names,
            name_to_continuous_resource_variable,
            continuous_less_is_better: vec![false, false, true, false],
        }
    }

    #[test]
    fn state_get_number_of_set_variables() {
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::default()],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_number_of_set_variables(), 1);
    }

    #[test]
    fn state_get_set_variable() {
        let mut set = Set::with_capacity(2);
        set.insert(1);
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::with_capacity(2), set.clone()],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_set_variable(0), &Set::with_capacity(2));
        assert_eq!(state.get_set_variable(1), &set);
    }

    #[test]
    #[should_panic]
    fn state_get_set_variable_panic() {
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![Set::default()],
                ..Default::default()
            },
            ..Default::default()
        };
        state.get_set_variable(1);
    }

    #[test]
    fn state_get_number_of_vector_variables() {
        let state = State {
            signature_variables: SignatureVariables {
                vector_variables: vec![Vector::default()],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_number_of_vector_variables(), 1);
    }

    #[test]
    fn state_get_vector_variable() {
        let state = State {
            signature_variables: SignatureVariables {
                vector_variables: vec![Vector::default(), vec![1]],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_vector_variable(0), &Vector::default());
        assert_eq!(state.get_vector_variable(1), &vec![1]);
    }

    #[test]
    #[should_panic]
    fn state_get_vector_variable_panic() {
        let state = State {
            signature_variables: SignatureVariables {
                vector_variables: vec![Vector::default()],
                ..Default::default()
            },
            ..Default::default()
        };
        state.get_vector_variable(1);
    }

    #[test]
    fn state_get_number_of_element_variables() {
        let state = State {
            signature_variables: SignatureVariables {
                element_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_number_of_element_variables(), 1);
    }

    #[test]
    fn state_get_element_variable() {
        let state = State {
            signature_variables: SignatureVariables {
                element_variables: vec![0, 1],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_element_variable(0), 0);
        assert_eq!(state.get_element_variable(1), 1);
    }

    #[test]
    #[should_panic]
    fn state_get_element_variable_panic() {
        let state = State {
            signature_variables: SignatureVariables {
                element_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        state.get_element_variable(1);
    }

    #[test]
    fn state_get_number_of_integer_variables() {
        let state = State {
            signature_variables: SignatureVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_number_of_integer_variables(), 1);
    }

    #[test]
    fn state_get_integer_variable() {
        let state = State {
            signature_variables: SignatureVariables {
                integer_variables: vec![0, 1],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_integer_variable(0), 0);
        assert_eq!(state.get_integer_variable(1), 1);
    }

    #[test]
    #[should_panic]
    fn state_get_integer_variable_panic() {
        let state = State {
            signature_variables: SignatureVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        state.get_integer_variable(1);
    }

    #[test]
    fn state_get_number_of_continuous_variables() {
        let state = State {
            signature_variables: SignatureVariables {
                continuous_variables: vec![0.0],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_number_of_continuous_variables(), 1);
    }

    #[test]
    fn state_get_continuous_variable() {
        let state = State {
            signature_variables: SignatureVariables {
                continuous_variables: vec![0.0, 1.0],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_continuous_variable(0), 0.0);
        assert_eq!(state.get_continuous_variable(1), 1.0);
    }

    #[test]
    #[should_panic]
    fn state_get_continuous_variable_panic() {
        let state = State {
            signature_variables: SignatureVariables {
                continuous_variables: vec![0.0],
                ..Default::default()
            },
            ..Default::default()
        };
        state.get_continuous_variable(1);
    }

    #[test]
    fn state_get_number_of_element_resource_variables() {
        let state = State {
            resource_variables: ResourceVariables {
                element_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_number_of_element_resource_variables(), 1);
    }

    #[test]
    fn state_get_element_resource_variable() {
        let state = State {
            resource_variables: ResourceVariables {
                element_variables: vec![0, 1],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_element_resource_variable(0), 0);
        assert_eq!(state.get_element_resource_variable(1), 1);
    }

    #[test]
    #[should_panic]
    fn state_get_element_resource_variable_panic() {
        let state = State {
            resource_variables: ResourceVariables {
                element_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        state.get_element_resource_variable(1);
    }

    #[test]
    fn state_get_number_of_integer_resource_variables() {
        let state = State {
            resource_variables: ResourceVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_number_of_integer_resource_variables(), 1);
    }

    #[test]
    fn state_get_integer_resource_variable() {
        let state = State {
            resource_variables: ResourceVariables {
                integer_variables: vec![0, 1],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_integer_resource_variable(0), 0);
        assert_eq!(state.get_integer_resource_variable(1), 1);
    }

    #[test]
    #[should_panic]
    fn state_get_integer_resource_variable_panic() {
        let state = State {
            resource_variables: ResourceVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        state.get_integer_resource_variable(1);
    }

    #[test]
    fn state_get_number_of_continuous_resource_variables() {
        let state = State {
            resource_variables: ResourceVariables {
                continuous_variables: vec![0.0],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_number_of_continuous_resource_variables(), 1);
    }

    #[test]
    fn state_get_continuous_resource_variable() {
        let state = State {
            resource_variables: ResourceVariables {
                continuous_variables: vec![0.0, 1.0],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_continuous_resource_variable(0), 0.0);
        assert_eq!(state.get_continuous_resource_variable(1), 1.0);
    }

    #[test]
    #[should_panic]
    fn state_get_continuous_resource_variable_panic() {
        let state = State {
            resource_variables: ResourceVariables {
                continuous_variables: vec![0.0],
                ..Default::default()
            },
            ..Default::default()
        };
        state.get_continuous_resource_variable(1);
    }

    #[test]
    fn apply_effect() {
        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(0);
        set2.insert(1);
        let state = State {
            signature_variables: SignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2], vec![1, 2]],
                element_variables: vec![1, 2],
                integer_variables: vec![1, 2, 3],
                continuous_variables: vec![1.0, 2.0, 3.0],
            },
            resource_variables: ResourceVariables {
                element_variables: vec![],
                integer_variables: vec![4, 5, 6],
                continuous_variables: vec![4.0, 5.0, 6.0],
            },
        };
        let registry = table_registry::TableRegistry::default();
        let set_effect1 = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            ElementExpression::Constant(1),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        let set_effect2 = SetExpression::SetElementOperation(
            SetElementOperator::Remove,
            ElementExpression::Constant(0),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
        );
        let vector_effect1 = VectorExpression::Push(
            ElementExpression::Constant(1),
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                0,
            ))),
        );
        let vector_effect2 = VectorExpression::Push(
            ElementExpression::Constant(0),
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                1,
            ))),
        );
        let element_effect1 = ElementExpression::Constant(2);
        let element_effect2 = ElementExpression::Constant(1);
        let integer_effect1 = IntegerExpression::BinaryOperation(
            BinaryOperator::Sub,
            Box::new(IntegerExpression::Variable(0)),
            Box::new(IntegerExpression::Constant(1)),
        );
        let integer_effect2 = IntegerExpression::BinaryOperation(
            BinaryOperator::Mul,
            Box::new(IntegerExpression::Variable(1)),
            Box::new(IntegerExpression::Constant(2)),
        );
        let continuous_effect1 = ContinuousExpression::BinaryOperation(
            BinaryOperator::Sub,
            Box::new(ContinuousExpression::Variable(0)),
            Box::new(ContinuousExpression::Constant(1.0)),
        );
        let continuous_effect2 = ContinuousExpression::BinaryOperation(
            BinaryOperator::Mul,
            Box::new(ContinuousExpression::Variable(1)),
            Box::new(ContinuousExpression::Constant(2.0)),
        );
        let integer_resource_effect1 = IntegerExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(IntegerExpression::ResourceVariable(0)),
            Box::new(IntegerExpression::Constant(1)),
        );
        let integer_resource_effect2 = IntegerExpression::BinaryOperation(
            BinaryOperator::Div,
            Box::new(IntegerExpression::ResourceVariable(1)),
            Box::new(IntegerExpression::Constant(2)),
        );
        let continuous_resource_effect1 = ContinuousExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(ContinuousExpression::ResourceVariable(0)),
            Box::new(ContinuousExpression::Constant(1.0)),
        );
        let continuous_resource_effect2 = ContinuousExpression::BinaryOperation(
            BinaryOperator::Div,
            Box::new(ContinuousExpression::ResourceVariable(1)),
            Box::new(ContinuousExpression::Constant(2.0)),
        );
        let effect = effect::Effect {
            set_effects: vec![(0, set_effect1), (1, set_effect2)],
            vector_effects: vec![(0, vector_effect1), (1, vector_effect2)],
            element_effects: vec![(0, element_effect1), (1, element_effect2)],
            integer_effects: vec![(0, integer_effect1), (1, integer_effect2)],
            continuous_effects: vec![(0, continuous_effect1), (1, continuous_effect2)],
            element_resource_effects: vec![],
            integer_resource_effects: vec![
                (0, integer_resource_effect1),
                (1, integer_resource_effect2),
            ],
            continuous_resource_effects: vec![
                (0, continuous_resource_effect1),
                (1, continuous_resource_effect2),
            ],
        };

        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(1);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(1);
        let expected = State {
            signature_variables: SignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2, 1], vec![1, 2, 0]],
                element_variables: vec![2, 1],
                integer_variables: vec![0, 4, 3],
                continuous_variables: vec![0.0, 4.0, 3.0],
            },
            resource_variables: ResourceVariables {
                element_variables: vec![],
                integer_variables: vec![5, 2, 6],
                continuous_variables: vec![5.0, 2.5, 6.0],
            },
        };
        let successor: State = state.apply_effect(&effect, &registry);
        assert_eq!(successor, expected);
    }

    #[test]
    fn is_satisfied() {
        let state = State {
            signature_variables: SignatureVariables {
                integer_variables: vec![1],
                ..Default::default()
            },
            ..Default::default()
        };
        let mut name_to_integer_variable = FxHashMap::default();
        name_to_integer_variable.insert(String::from("i0"), 0);
        let metadata = StateMetadata {
            integer_variable_names: vec![String::from("i0")],
            name_to_integer_variable,
            ..Default::default()
        };

        let base_state = State {
            signature_variables: SignatureVariables {
                integer_variables: vec![1],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(base_state.is_satisfied(&state, &metadata));

        let base_state = State {
            signature_variables: SignatureVariables {
                integer_variables: vec![2],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(!base_state.is_satisfied(&state, &metadata));
    }

    #[test]
    fn number_of_objects() {
        let metadata = generate_metadata();
        assert_eq!(metadata.number_of_object_types(), 2);
    }

    #[test]
    fn number_of_set_variables() {
        let metadata = generate_metadata();
        assert_eq!(metadata.number_of_set_variables(), 4);
    }

    #[test]
    fn number_of_vector_variables() {
        let metadata = generate_metadata();
        assert_eq!(metadata.number_of_vector_variables(), 4);
    }

    #[test]
    fn number_of_element_variables() {
        let metadata = generate_metadata();
        assert_eq!(metadata.number_of_element_variables(), 4);
    }

    #[test]
    fn number_of_integer_variables() {
        let metadata = generate_metadata();
        assert_eq!(metadata.number_of_integer_variables(), 4);
    }

    #[test]
    fn number_of_continuous_variables() {
        let metadata = generate_metadata();
        assert_eq!(metadata.number_of_continuous_variables(), 4);
    }

    #[test]
    fn number_of_integer_resource_variables() {
        let metadata = generate_metadata();
        assert_eq!(metadata.number_of_integer_resource_variables(), 4);
    }

    #[test]
    fn number_of_continuous_resource_variables() {
        let metadata = generate_metadata();
        assert_eq!(metadata.number_of_continuous_resource_variables(), 4);
    }

    #[test]
    fn has_resource_variable() {
        let metadata = StateMetadata {
            element_variable_names: vec![String::from("v")],
            element_variable_to_object: vec![0],
            integer_variable_names: vec![String::from("v")],
            continuous_variable_names: vec![String::from("v")],
            set_variable_names: vec![String::from("v")],
            set_variable_to_object: vec![0],
            vector_variable_names: vec![String::from("v")],
            vector_variable_to_object: vec![0],
            ..Default::default()
        };
        assert!(!metadata.has_resource_variables());
        let metadata = StateMetadata {
            element_resource_variable_names: vec![String::from("v")],
            element_resource_variable_to_object: vec![0],
            element_less_is_better: vec![true],
            ..Default::default()
        };
        assert!(metadata.has_resource_variables());
        let metadata = StateMetadata {
            integer_resource_variable_names: vec![String::from("v")],
            integer_less_is_better: vec![true],
            ..Default::default()
        };
        assert!(metadata.has_resource_variables());
        let metadata = StateMetadata {
            continuous_resource_variable_names: vec![String::from("v")],
            continuous_less_is_better: vec![true],
            ..Default::default()
        };
        assert!(metadata.has_resource_variables());
    }

    #[test]
    fn get_name_set() {
        let metadata = generate_metadata();
        let mut expected = FxHashSet::default();
        expected.insert(String::from("object"));
        expected.insert(String::from("small"));
        expected.insert(String::from("s0"));
        expected.insert(String::from("s1"));
        expected.insert(String::from("s2"));
        expected.insert(String::from("s3"));
        expected.insert(String::from("p0"));
        expected.insert(String::from("p1"));
        expected.insert(String::from("p2"));
        expected.insert(String::from("p3"));
        expected.insert(String::from("e0"));
        expected.insert(String::from("e1"));
        expected.insert(String::from("e2"));
        expected.insert(String::from("e3"));
        expected.insert(String::from("er0"));
        expected.insert(String::from("er1"));
        expected.insert(String::from("er2"));
        expected.insert(String::from("er3"));
        expected.insert(String::from("i0"));
        expected.insert(String::from("i1"));
        expected.insert(String::from("i2"));
        expected.insert(String::from("i3"));
        expected.insert(String::from("c0"));
        expected.insert(String::from("c1"));
        expected.insert(String::from("c2"));
        expected.insert(String::from("c3"));
        expected.insert(String::from("ir0"));
        expected.insert(String::from("ir1"));
        expected.insert(String::from("ir2"));
        expected.insert(String::from("ir3"));
        expected.insert(String::from("cr0"));
        expected.insert(String::from("cr1"));
        expected.insert(String::from("cr2"));
        expected.insert(String::from("cr3"));
        assert_eq!(metadata.get_name_set(), expected);
    }

    #[test]
    fn dominance() {
        let metadata = generate_metadata();

        let a = State {
            resource_variables: ResourceVariables {
                element_variables: vec![0, 1, 2, 0],
                integer_variables: vec![1, 2, 2, 0],
                continuous_variables: vec![0.0, 0.0, 0.0, 0.0],
            },
            ..Default::default()
        };
        let b = State {
            resource_variables: ResourceVariables {
                element_variables: vec![0, 1, 2, 0],
                integer_variables: vec![1, 2, 2, 0],
                continuous_variables: vec![0.0, 0.0, 0.0, 0.0],
            },
            ..Default::default()
        };
        assert_eq!(metadata.dominance(&a, &b), Some(Ordering::Equal));

        let b = State {
            resource_variables: ResourceVariables {
                element_variables: vec![0, 0, 3, 0],
                integer_variables: vec![1, 2, 2, 0],
                continuous_variables: vec![0.0, 0.0, 0.0, 0.0],
            },
            ..Default::default()
        };
        assert_eq!(metadata.dominance(&a, &b), Some(Ordering::Greater));
        assert_eq!(metadata.dominance(&b, &a), Some(Ordering::Less));

        let b = State {
            resource_variables: ResourceVariables {
                element_variables: vec![0, 1, 2, 0],
                integer_variables: vec![1, 1, 3, 0],
                continuous_variables: vec![0.0, 0.0, 0.0, 0.0],
            },
            ..Default::default()
        };
        assert_eq!(metadata.dominance(&a, &b), Some(Ordering::Greater));
        assert_eq!(metadata.dominance(&b, &a), Some(Ordering::Less));

        let b = State {
            resource_variables: ResourceVariables {
                element_variables: vec![0, 1, 2, 0],
                integer_variables: vec![1, 3, 3, 0],
                continuous_variables: vec![0.0, 0.0, 0.0, 0.0],
            },
            ..Default::default()
        };
        assert!(metadata.dominance(&b, &a).is_none());

        let a = State {
            resource_variables: ResourceVariables {
                element_variables: vec![0, 1, 2, 0],
                integer_variables: vec![1, 2, 2, 0],
                continuous_variables: vec![1.0, 2.0, 2.0, 0.0],
            },
            ..Default::default()
        };
        let b = State {
            resource_variables: ResourceVariables {
                element_variables: vec![0, 1, 2, 0],
                integer_variables: vec![1, 2, 2, 0],
                continuous_variables: vec![1.0, 1.0, 3.0, 0.0],
            },
            ..Default::default()
        };
        assert_eq!(metadata.dominance(&a, &b), Some(Ordering::Greater));
        assert_eq!(metadata.dominance(&b, &a), Some(Ordering::Less));

        let b = State {
            resource_variables: ResourceVariables {
                element_variables: vec![0, 1, 2, 0],
                integer_variables: vec![1, 2, 2, 0],
                continuous_variables: vec![1.0, 3.0, 4.0, 0.0],
            },
            ..Default::default()
        };
        assert!(metadata.dominance(&a, &b).is_none());
    }

    #[test]
    #[should_panic]
    fn dominance_element_length_panic() {
        let metadata = generate_metadata();
        let a = State {
            resource_variables: ResourceVariables {
                element_variables: vec![1, 2, 3],
                integer_variables: vec![1, 2, 2, 2],
                continuous_variables: vec![],
            },
            ..Default::default()
        };
        let b = State {
            resource_variables: ResourceVariables {
                element_variables: vec![1, 2, 3, 0],
                integer_variables: vec![1, 2, 2, 2],
                continuous_variables: vec![],
            },
            ..Default::default()
        };
        metadata.dominance(&b, &a);
    }

    #[test]
    #[should_panic]
    fn dominance_integer_length_panic() {
        let metadata = generate_metadata();
        let a = State {
            resource_variables: ResourceVariables {
                element_variables: vec![],
                integer_variables: vec![1, 2, 2],
                continuous_variables: vec![],
            },
            ..Default::default()
        };
        let b = State {
            resource_variables: ResourceVariables {
                element_variables: vec![],
                integer_variables: vec![1, 2, 2, 0],
                continuous_variables: vec![],
            },
            ..Default::default()
        };
        metadata.dominance(&b, &a);
    }

    #[test]
    #[should_panic]
    fn dominance_continuous_length_panic() {
        let metadata = generate_metadata();
        let a = State {
            resource_variables: ResourceVariables {
                element_variables: vec![],
                integer_variables: vec![1, 2, 2, 0],
                continuous_variables: vec![1.0, 2.0, 2.0, 0.0],
            },
            ..Default::default()
        };
        let b = State {
            resource_variables: ResourceVariables {
                element_variables: vec![],
                integer_variables: vec![1, 2, 2, 0],
                continuous_variables: vec![1.0, 1.0, 3.0],
            },
            ..Default::default()
        };
        metadata.dominance(&b, &a);
    }

    #[test]
    fn check_state_ok() {
        let metadata = generate_metadata();
        let state = State {
            signature_variables: SignatureVariables {
                element_variables: vec![4, 8, 9, 1],
                vector_variables: vec![vec![4], vec![8], vec![9], vec![1]],
                set_variables: vec![
                    Set::with_capacity(10),
                    Set::with_capacity(10),
                    Set::with_capacity(10),
                    Set::with_capacity(2),
                ],
                integer_variables: vec![-1, 2, 4, 5],
                continuous_variables: vec![-1.0, 2.0, 4.0, 5.0],
            },
            resource_variables: ResourceVariables {
                element_variables: vec![4, 8, 9, 1],
                integer_variables: vec![-1, 2, 4, 5],
                continuous_variables: vec![-1.0, 2.0, 4.0, 5.0],
            },
        };
        assert!(metadata.check_state(&state).is_ok())
    }

    #[test]
    fn chech_state_err() {
        let metadata = generate_metadata();
        let state = State {
            signature_variables: SignatureVariables {
                element_variables: vec![4, 8, 9],
                vector_variables: vec![vec![4], vec![8], vec![9], vec![1]],
                set_variables: vec![
                    Set::with_capacity(10),
                    Set::with_capacity(10),
                    Set::with_capacity(10),
                    Set::with_capacity(2),
                ],
                integer_variables: vec![-1, 2, 4, 5],
                continuous_variables: vec![-1.0, 2.0, 4.0, 5.0],
            },
            resource_variables: ResourceVariables {
                element_variables: vec![4, 8, 9, 1],
                integer_variables: vec![-1, 2, 4, 5],
                continuous_variables: vec![-1.0, 2.0, 4.0, 5.0],
            },
        };
        assert!(metadata.check_state(&state).is_err());
        let state = State {
            signature_variables: SignatureVariables {
                element_variables: vec![4, 8, 9, 1],
                vector_variables: vec![vec![4], vec![8], vec![9]],
                set_variables: vec![
                    Set::with_capacity(10),
                    Set::with_capacity(10),
                    Set::with_capacity(10),
                    Set::with_capacity(2),
                ],
                integer_variables: vec![-1, 2, 4, 5],
                continuous_variables: vec![-1.0, 2.0, 4.0, 5.0],
            },
            resource_variables: ResourceVariables {
                element_variables: vec![4, 8, 9, 1],
                integer_variables: vec![-1, 2, 4, 5],
                continuous_variables: vec![-1.0, 2.0, 4.0, 5.0],
            },
        };
        assert!(metadata.check_state(&state).is_err());
        let state = State {
            signature_variables: SignatureVariables {
                element_variables: vec![4, 8, 9, 1],
                vector_variables: vec![vec![4], vec![8], vec![9], vec![1]],
                set_variables: vec![
                    Set::with_capacity(10),
                    Set::with_capacity(10),
                    Set::with_capacity(10),
                ],
                integer_variables: vec![-1, 2, 4, 5],
                continuous_variables: vec![-1.0, 2.0, 4.0, 5.0],
            },
            resource_variables: ResourceVariables {
                element_variables: vec![4, 8, 9, 1],
                integer_variables: vec![-1, 2, 4, 5],
                continuous_variables: vec![-1.0, 2.0, 4.0, 5.0],
            },
        };
        assert!(metadata.check_state(&state).is_err());
        let state = State {
            signature_variables: SignatureVariables {
                element_variables: vec![4, 8, 9, 1],
                vector_variables: vec![vec![4], vec![8], vec![9], vec![1]],
                set_variables: vec![
                    Set::with_capacity(10),
                    Set::with_capacity(10),
                    Set::with_capacity(10),
                    Set::with_capacity(2),
                ],
                integer_variables: vec![-1, 2, 4],
                continuous_variables: vec![-1.0, 2.0, 4.0, 5.0],
            },
            resource_variables: ResourceVariables {
                element_variables: vec![4, 8, 9, 1],
                integer_variables: vec![-1, 2, 4, 5],
                continuous_variables: vec![-1.0, 2.0, 4.0, 5.0],
            },
        };
        assert!(metadata.check_state(&state).is_err());
        let state = State {
            signature_variables: SignatureVariables {
                element_variables: vec![4, 8, 9, 1],
                vector_variables: vec![vec![4], vec![8], vec![9], vec![1]],
                set_variables: vec![
                    Set::with_capacity(10),
                    Set::with_capacity(10),
                    Set::with_capacity(10),
                    Set::with_capacity(2),
                ],
                integer_variables: vec![-1, 2, 4, 5],
                continuous_variables: vec![-1.0, 2.0, 4.0],
            },
            resource_variables: ResourceVariables {
                element_variables: vec![4, 8, 9, 1],
                integer_variables: vec![-1, 2, 4, 5],
                continuous_variables: vec![-1.0, 2.0, 4.0, 5.0],
            },
        };
        assert!(metadata.check_state(&state).is_err());
        let state = State {
            signature_variables: SignatureVariables {
                element_variables: vec![4, 8, 9, 1],
                vector_variables: vec![vec![4], vec![8], vec![9], vec![1]],
                set_variables: vec![
                    Set::with_capacity(10),
                    Set::with_capacity(10),
                    Set::with_capacity(10),
                    Set::with_capacity(2),
                ],
                integer_variables: vec![-1, 2, 4, 5],
                continuous_variables: vec![-1.0, 2.0, 4.0, 5.0],
            },
            resource_variables: ResourceVariables {
                element_variables: vec![4, 8, 9],
                integer_variables: vec![-1, 2, 4, 5],
                continuous_variables: vec![-1.0, 2.0, 4.0, 5.0],
            },
        };
        assert!(metadata.check_state(&state).is_err());
        let state = State {
            signature_variables: SignatureVariables {
                element_variables: vec![4, 8, 9, 1],
                vector_variables: vec![vec![4], vec![8], vec![9], vec![1]],
                set_variables: vec![
                    Set::with_capacity(10),
                    Set::with_capacity(10),
                    Set::with_capacity(10),
                    Set::with_capacity(2),
                ],
                integer_variables: vec![-1, 2, 4, 5],
                continuous_variables: vec![-1.0, 2.0, 4.0, 5.0],
            },
            resource_variables: ResourceVariables {
                element_variables: vec![4, 8, 9, 1],
                integer_variables: vec![-1, 2, 4],
                continuous_variables: vec![-1.0, 2.0, 4.0, 5.0],
            },
        };
        assert!(metadata.check_state(&state).is_err());
        let state = State {
            signature_variables: SignatureVariables {
                element_variables: vec![4, 8, 9, 1],
                vector_variables: vec![vec![4], vec![8], vec![9], vec![1]],
                set_variables: vec![
                    Set::with_capacity(10),
                    Set::with_capacity(10),
                    Set::with_capacity(10),
                    Set::with_capacity(2),
                ],
                integer_variables: vec![-1, 2, 4, 5],
                continuous_variables: vec![-1.0, 2.0, 4.0, 5.0],
            },
            resource_variables: ResourceVariables {
                element_variables: vec![4, 8, 9, 1],
                integer_variables: vec![-1, 2, 4, 5],
                continuous_variables: vec![-1.0, 2.0, 4.0],
            },
        };
        assert!(metadata.check_state(&state).is_err());
        let state = State {
            signature_variables: SignatureVariables {
                element_variables: vec![4, 8, 10, 1],
                vector_variables: vec![vec![4], vec![8], vec![9], vec![1]],
                set_variables: vec![
                    Set::with_capacity(10),
                    Set::with_capacity(10),
                    Set::with_capacity(10),
                    Set::with_capacity(2),
                ],
                integer_variables: vec![-1, 2, 4, 5],
                continuous_variables: vec![-1.0, 2.0, 4.0, 5.0],
            },
            resource_variables: ResourceVariables {
                element_variables: vec![4, 8, 9, 1],
                integer_variables: vec![-1, 2, 4, 5],
                continuous_variables: vec![-1.0, 2.0, 4.0, 5.0],
            },
        };
        assert!(metadata.check_state(&state).is_err());
        let state = State {
            signature_variables: SignatureVariables {
                element_variables: vec![4, 8, 9, 1],
                vector_variables: vec![vec![4], vec![8], vec![10], vec![1]],
                set_variables: vec![
                    Set::with_capacity(10),
                    Set::with_capacity(10),
                    Set::with_capacity(10),
                    Set::with_capacity(2),
                ],
                integer_variables: vec![-1, 2, 4, 5],
                continuous_variables: vec![-1.0, 2.0, 4.0, 5.0],
            },
            resource_variables: ResourceVariables {
                element_variables: vec![4, 8, 9, 1],
                integer_variables: vec![-1, 2, 4, 5],
                continuous_variables: vec![-1.0, 2.0, 4.0, 5.0],
            },
        };
        assert!(metadata.check_state(&state).is_err());
        let state = State {
            signature_variables: SignatureVariables {
                element_variables: vec![4, 8, 9, 1],
                vector_variables: vec![vec![4], vec![8], vec![9], vec![1]],
                set_variables: vec![
                    Set::with_capacity(10),
                    Set::with_capacity(10),
                    Set::with_capacity(11),
                    Set::with_capacity(2),
                ],
                integer_variables: vec![-1, 2, 4, 5],
                continuous_variables: vec![-1.0, 2.0, 4.0, 5.0],
            },
            resource_variables: ResourceVariables {
                element_variables: vec![4, 8, 9, 1],
                integer_variables: vec![-1, 2, 4, 5],
                continuous_variables: vec![-1.0, 2.0, 4.0, 5.0],
            },
        };
        assert!(metadata.check_state(&state).is_err());
        let state = State {
            signature_variables: SignatureVariables {
                element_variables: vec![4, 8, 9, 1],
                vector_variables: vec![vec![4], vec![8], vec![9], vec![1]],
                set_variables: vec![
                    Set::with_capacity(10),
                    Set::with_capacity(10),
                    Set::with_capacity(10),
                    Set::with_capacity(2),
                ],
                integer_variables: vec![-1, 2, 4, 5],
                continuous_variables: vec![-1.0, 2.0, 4.0, 5.0],
            },
            resource_variables: ResourceVariables {
                element_variables: vec![4, 8, 10, 1],
                integer_variables: vec![-1, 2, 4, 5],
                continuous_variables: vec![-1.0, 2.0, 4.0, 5.0],
            },
        };
        assert!(metadata.check_state(&state).is_err());
    }

    #[test]
    fn add_object_ok() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        assert_eq!(ob, ObjectType(0));
        assert_eq!(ob.id(), 0);
        assert_eq!(metadata.object_type_names, vec![String::from("something")]);
        assert_eq!(metadata.object_numbers, vec![10]);
        let mut name_to_object = FxHashMap::default();
        name_to_object.insert(String::from("something"), 0);
        assert_eq!(metadata.name_to_object_type, name_to_object);
        let ob = metadata.add_object_type(String::from("other"), 5);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        assert_eq!(ob, ObjectType(1));
        assert_eq!(ob.id(), 1);
        assert_eq!(
            metadata.object_type_names,
            vec![String::from("something"), String::from("other")]
        );
        assert_eq!(metadata.object_numbers, vec![10, 5]);
        name_to_object.insert(String::from("other"), 1);
        assert_eq!(metadata.name_to_object_type, name_to_object);
    }

    #[test]
    fn add_object_err() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_err());
        assert_eq!(metadata.object_type_names, vec![String::from("something")]);
        assert_eq!(metadata.object_numbers, vec![10]);
        let mut name_to_object = FxHashMap::default();
        name_to_object.insert(String::from("something"), 0);
        assert_eq!(metadata.name_to_object_type, name_to_object);
    }

    #[test]
    fn number_of_object_ok() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let result = metadata.get_number_of_objects(ob);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 10);
        // let result = metadata.set_number_of_object(ob, 5);
        // assert!(result.is_ok());
        // assert_eq!(metadata.object_numbers, vec![5]);
        // let result = metadata.get_number_of_objects(ob);
        // assert!(result.is_ok());
        // assert_eq!(result.unwrap(), 5);
    }

    #[test]
    fn number_of_object_err() {
        let mut metadata1 = StateMetadata::default();
        let ob = metadata1.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());

        let mut metadata2 = StateMetadata::default();
        let ob = metadata2.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = metadata2.add_object_type(String::from("other"), 10);
        assert!(ob.is_ok());
        // let ob = ob.unwrap();

        // let result = metadata1.set_number_of_object(ob, 5);
        // assert!(result.is_err());
        // let result = metadata1.get_number_of_objects(ob);
        // assert!(result.is_err());

        assert_eq!(metadata1.object_type_names, vec![String::from("something")]);
        assert_eq!(metadata1.object_numbers, vec![10]);
        let mut name_to_object = FxHashMap::default();
        name_to_object.insert(String::from("something"), 0);
        assert_eq!(metadata1.name_to_object_type, name_to_object);
    }

    #[test]
    fn create_set_ok() {
        let mut metadata1 = StateMetadata::default();
        let ob = metadata1.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let result = metadata1.create_set(ob, &[0, 1, 9]);
        assert!(result.is_ok());
        let mut set = Set::with_capacity(10);
        set.insert(0);
        set.insert(1);
        set.insert(9);
        assert_eq!(result.unwrap(), set);
    }

    #[test]
    fn create_set_err() {
        let mut metadata1 = StateMetadata::default();
        let ob = metadata1.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let result = metadata1.create_set(ob, &[0, 1, 10]);
        assert!(result.is_err());

        let mut metadata2 = StateMetadata::default();
        let ob = metadata2.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = metadata2.add_object_type(String::from("other"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();

        let result = metadata1.create_set(ob, &[0, 1, 9]);
        assert!(result.is_err());
    }

    #[test]
    fn add_element_variable_ok() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_element_variable(String::from("v"), ob);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(v, ElementVariable(0));
        assert_eq!(v.id(), 0);
        assert_eq!(metadata.element_variable_names, vec![String::from("v")]);
        assert_eq!(metadata.element_variable_to_object, vec![0]);
        let mut name_to_variable = FxHashMap::default();
        name_to_variable.insert(String::from("v"), 0);
        assert_eq!(metadata.name_to_element_variable, name_to_variable);
        let v = metadata.add_element_variable(String::from("u"), ob);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(v, ElementVariable(1));
        assert_eq!(v.id(), 1);
        assert_eq!(
            metadata.element_variable_names,
            vec![String::from("v"), String::from("u")]
        );
        assert_eq!(metadata.element_variable_to_object, vec![0, 0]);
        name_to_variable.insert(String::from("u"), 1);
        assert_eq!(metadata.name_to_element_variable, name_to_variable);
    }

    #[test]
    fn add_element_variable_err() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_element_variable(String::from("v"), ob);
        assert!(v.is_ok());
        let v = metadata.add_element_variable(String::from("v"), ob);
        assert!(v.is_err());
        assert_eq!(metadata.element_variable_names, vec![String::from("v")]);
        assert_eq!(metadata.element_variable_to_object, vec![0]);
        let mut name_to_variable = FxHashMap::default();
        name_to_variable.insert(String::from("v"), 0);
        assert_eq!(metadata.name_to_element_variable, name_to_variable);

        let mut metadata2 = StateMetadata::default();
        let ob = metadata2.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = metadata2.add_object_type(String::from("other"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_element_variable(String::from("u"), ob);
        assert!(v.is_err());
        assert_eq!(metadata.element_variable_names, vec![String::from("v")]);
        assert_eq!(metadata.element_variable_to_object, vec![0]);
        let mut name_to_variable = FxHashMap::default();
        name_to_variable.insert(String::from("v"), 0);
        assert_eq!(metadata.name_to_element_variable, name_to_variable);
    }

    #[test]
    fn get_object_type_of_element_variable_ok() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_element_variable(String::from("v"), ob);
        assert!(v.is_ok());
        let v = v.unwrap();
        let ob1 = metadata.get_object_type_of(v);
        assert!(ob1.is_ok());
        assert_eq!(ob1.unwrap(), ob);
    }

    #[test]
    fn get_object_type_of_element_variable_err() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_element_variable(String::from("v"), ob);
        assert!(v.is_ok());

        let mut metadata2 = StateMetadata::default();
        let ob = metadata2.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata2.add_element_variable(String::from("v"), ob);
        assert!(v.is_ok());
        let v = metadata2.add_element_variable(String::from("u"), ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let ob = metadata.get_object_type_of(v);
        assert!(ob.is_err());
    }

    #[test]
    fn add_element_resource_variable_ok() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_element_resource_variable(String::from("v"), ob, true);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(v, ElementResourceVariable(0));
        assert_eq!(v.id(), 0);
        assert_eq!(
            metadata.element_resource_variable_names,
            vec![String::from("v")]
        );
        assert_eq!(metadata.element_resource_variable_to_object, vec![0]);
        assert_eq!(metadata.element_less_is_better, vec![true]);
        let mut name_to_variable = FxHashMap::default();
        name_to_variable.insert(String::from("v"), 0);
        assert_eq!(metadata.name_to_element_resource_variable, name_to_variable);
        let v = metadata.add_element_resource_variable(String::from("u"), ob, false);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(v, ElementResourceVariable(1));
        assert_eq!(v.id(), 1);
        assert_eq!(
            metadata.element_resource_variable_names,
            vec![String::from("v"), String::from("u")]
        );
        assert_eq!(metadata.element_resource_variable_to_object, vec![0, 0]);
        name_to_variable.insert(String::from("u"), 1);
        assert_eq!(metadata.name_to_element_resource_variable, name_to_variable);
        assert_eq!(metadata.element_less_is_better, vec![true, false]);
    }

    #[test]
    fn add_element_resource_variable_err() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_element_resource_variable(String::from("v"), ob, true);
        assert!(v.is_ok());
        let v = metadata.add_element_resource_variable(String::from("v"), ob, false);
        assert!(v.is_err());
        assert_eq!(
            metadata.element_resource_variable_names,
            vec![String::from("v")]
        );
        assert_eq!(metadata.element_resource_variable_to_object, vec![0]);
        assert_eq!(metadata.element_less_is_better, vec![true]);
        let mut name_to_variable = FxHashMap::default();
        name_to_variable.insert(String::from("v"), 0);
        assert_eq!(metadata.name_to_element_resource_variable, name_to_variable);

        let mut metadata2 = StateMetadata::default();
        let ob = metadata2.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = metadata2.add_object_type(String::from("other"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_element_resource_variable(String::from("u"), ob, true);
        assert!(v.is_err());
        assert_eq!(
            metadata.element_resource_variable_names,
            vec![String::from("v")]
        );
        assert_eq!(metadata.element_resource_variable_to_object, vec![0]);
        assert_eq!(metadata.element_less_is_better, vec![true]);
        let mut name_to_variable = FxHashMap::default();
        name_to_variable.insert(String::from("v"), 0);
        assert_eq!(metadata.name_to_element_resource_variable, name_to_variable);
    }

    #[test]
    fn get_object_type_of_element_resource_variable_ok() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_element_resource_variable(String::from("v"), ob, true);
        assert!(v.is_ok());
        let v = v.unwrap();
        let ob1 = metadata.get_object_type_of(v);
        assert!(ob1.is_ok());
        assert_eq!(ob1.unwrap(), ob);
    }

    #[test]
    fn get_object_type_of_element_resource_variable_err() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_element_resource_variable(String::from("v"), ob, true);
        assert!(v.is_ok());

        let mut metadata2 = StateMetadata::default();
        let ob = metadata2.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata2.add_element_resource_variable(String::from("v"), ob, false);
        assert!(v.is_ok());
        let v = metadata2.add_element_resource_variable(String::from("u"), ob, true);
        assert!(v.is_ok());
        let v = v.unwrap();

        let ob = metadata.get_object_type_of(v);
        assert!(ob.is_err());
    }

    #[test]
    fn element_resource_variable_preference_ok() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_element_resource_variable(String::from("v"), ob, true);
        assert!(v.is_ok());
        let v = v.unwrap();
        let result = metadata.get_preference(v);
        assert!(result.is_ok());
        assert!(result.unwrap());
        let result = metadata.set_preference(v, false);
        assert!(result.is_ok());
        assert_eq!(metadata.element_less_is_better, vec![false]);
        let result = metadata.get_preference(v);
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    #[test]
    fn element_resource_variable_preference_err() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_element_resource_variable(String::from("v"), ob, true);
        assert!(v.is_ok());

        let mut metadata2 = StateMetadata::default();
        let ob = metadata2.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata2.add_element_resource_variable(String::from("v"), ob, true);
        assert!(v.is_ok());
        let v = metadata2.add_element_resource_variable(String::from("u"), ob, true);
        assert!(v.is_ok());
        let v = v.unwrap();
        let result = metadata.get_preference(v);
        assert!(result.is_err());
        let result = metadata.set_preference(v, false);
        assert!(result.is_err());
        assert_eq!(metadata.element_less_is_better, vec![true]);
    }

    #[test]
    fn add_set_variable_ok() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_set_variable(String::from("v"), ob);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(v, SetVariable(0));
        assert_eq!(v.id(), 0);
        assert_eq!(metadata.set_variable_names, vec![String::from("v")]);
        assert_eq!(metadata.set_variable_to_object, vec![0]);
        let mut name_to_variable = FxHashMap::default();
        name_to_variable.insert(String::from("v"), 0);
        assert_eq!(metadata.name_to_set_variable, name_to_variable);
        let v = metadata.add_set_variable(String::from("u"), ob);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(v, SetVariable(1));
        assert_eq!(v.id(), 1);
        assert_eq!(
            metadata.set_variable_names,
            vec![String::from("v"), String::from("u")]
        );
        assert_eq!(metadata.set_variable_to_object, vec![0, 0]);
        name_to_variable.insert(String::from("u"), 1);
        assert_eq!(metadata.name_to_set_variable, name_to_variable);
    }

    #[test]
    fn add_set_variable_err() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_set_variable(String::from("v"), ob);
        assert!(v.is_ok());
        let v = metadata.add_set_variable(String::from("v"), ob);
        assert!(v.is_err());
        assert_eq!(metadata.set_variable_names, vec![String::from("v")]);
        assert_eq!(metadata.set_variable_to_object, vec![0]);
        let mut name_to_variable = FxHashMap::default();
        name_to_variable.insert(String::from("v"), 0);
        assert_eq!(metadata.name_to_set_variable, name_to_variable);

        let mut metadata2 = StateMetadata::default();
        let ob = metadata2.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = metadata2.add_object_type(String::from("other"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_set_variable(String::from("u"), ob);
        assert!(v.is_err());
        assert_eq!(metadata.set_variable_names, vec![String::from("v")]);
        assert_eq!(metadata.set_variable_to_object, vec![0]);
        let mut name_to_variable = FxHashMap::default();
        name_to_variable.insert(String::from("v"), 0);
        assert_eq!(metadata.name_to_set_variable, name_to_variable);
    }

    #[test]
    fn get_object_type_of_set_variable_ok() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_set_variable(String::from("v"), ob);
        assert!(v.is_ok());
        let v = v.unwrap();
        let ob1 = metadata.get_object_type_of(v);
        assert!(ob1.is_ok());
        assert_eq!(ob1.unwrap(), ob);
    }

    #[test]
    fn get_object_type_of_set_variable_err() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_set_variable(String::from("v"), ob);
        assert!(v.is_ok());

        let mut metadata2 = StateMetadata::default();
        let ob = metadata2.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata2.add_set_variable(String::from("v"), ob);
        assert!(v.is_ok());
        let v = metadata2.add_set_variable(String::from("u"), ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let ob = metadata.get_object_type_of(v);
        assert!(ob.is_err());
    }

    #[test]
    fn add_vector_variable_ok() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_vector_variable(String::from("v"), ob);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(v, VectorVariable(0));
        assert_eq!(v.id(), 0);
        assert_eq!(metadata.vector_variable_names, vec![String::from("v")]);
        assert_eq!(metadata.vector_variable_to_object, vec![0]);
        let mut name_to_variable = FxHashMap::default();
        name_to_variable.insert(String::from("v"), 0);
        assert_eq!(metadata.name_to_vector_variable, name_to_variable);
        let v = metadata.add_vector_variable(String::from("u"), ob);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(v, VectorVariable(1));
        assert_eq!(v.id(), 1);
        assert_eq!(
            metadata.vector_variable_names,
            vec![String::from("v"), String::from("u")]
        );
        assert_eq!(metadata.vector_variable_to_object, vec![0, 0]);
        name_to_variable.insert(String::from("u"), 1);
        assert_eq!(metadata.name_to_vector_variable, name_to_variable);
    }

    #[test]
    fn add_vector_variable_err() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_vector_variable(String::from("v"), ob);
        assert!(v.is_ok());
        let v = metadata.add_vector_variable(String::from("v"), ob);
        assert!(v.is_err());
        assert_eq!(metadata.vector_variable_names, vec![String::from("v")]);
        assert_eq!(metadata.vector_variable_to_object, vec![0]);
        let mut name_to_variable = FxHashMap::default();
        name_to_variable.insert(String::from("v"), 0);
        assert_eq!(metadata.name_to_vector_variable, name_to_variable);

        let mut metadata2 = StateMetadata::default();
        let ob = metadata2.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = metadata2.add_object_type(String::from("other"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_vector_variable(String::from("u"), ob);
        assert!(v.is_err());
        assert_eq!(metadata.vector_variable_names, vec![String::from("v")]);
        assert_eq!(metadata.vector_variable_to_object, vec![0]);
        let mut name_to_variable = FxHashMap::default();
        name_to_variable.insert(String::from("v"), 0);
        assert_eq!(metadata.name_to_vector_variable, name_to_variable);
    }

    #[test]
    fn get_object_type_of_vector_variable_ok() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_vector_variable(String::from("v"), ob);
        assert!(v.is_ok());
        let v = v.unwrap();
        let ob1 = metadata.get_object_type_of(v);
        assert!(ob1.is_ok());
        assert_eq!(ob1.unwrap(), ob);
    }

    #[test]
    fn get_object_type_of_vector_variable_err() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_vector_variable(String::from("v"), ob);
        assert!(v.is_ok());

        let mut metadata2 = StateMetadata::default();
        let ob = metadata2.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata2.add_vector_variable(String::from("v"), ob);
        assert!(v.is_ok());
        let v = metadata2.add_vector_variable(String::from("u"), ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let ob = metadata.get_object_type_of(v);
        assert!(ob.is_err());
    }

    #[test]
    fn add_integer_variable_ok() {
        let mut metadata = StateMetadata::default();
        let v = metadata.add_integer_variable(String::from("v"));
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(v, IntegerVariable(0));
        assert_eq!(v.id(), 0);
        assert_eq!(metadata.integer_variable_names, vec![String::from("v")]);
        let mut name_to_variable = FxHashMap::default();
        name_to_variable.insert(String::from("v"), 0);
        assert_eq!(metadata.name_to_integer_variable, name_to_variable);
        let v = metadata.add_integer_variable(String::from("u"));
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(v, IntegerVariable(1));
        assert_eq!(v.id(), 1);
        assert_eq!(
            metadata.integer_variable_names,
            vec![String::from("v"), String::from("u")]
        );
        name_to_variable.insert(String::from("u"), 1);
        assert_eq!(metadata.name_to_integer_variable, name_to_variable);
    }

    #[test]
    fn add_integer_variable_err() {
        let mut metadata = StateMetadata::default();
        let v = metadata.add_integer_variable(String::from("v"));
        assert!(v.is_ok());
        let v = metadata.add_integer_variable(String::from("v"));
        assert!(v.is_err());
        assert_eq!(metadata.integer_variable_names, vec![String::from("v")]);
        let mut name_to_variable = FxHashMap::default();
        name_to_variable.insert(String::from("v"), 0);
        assert_eq!(metadata.name_to_integer_variable, name_to_variable);
    }

    #[test]
    fn add_integer_resource_variable_ok() {
        let mut metadata = StateMetadata::default();
        let v = metadata.add_integer_resource_variable(String::from("v"), true);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(v, IntegerResourceVariable(0));
        assert_eq!(v.id(), 0);
        assert_eq!(
            metadata.integer_resource_variable_names,
            vec![String::from("v")]
        );
        let mut name_to_variable = FxHashMap::default();
        name_to_variable.insert(String::from("v"), 0);
        assert_eq!(metadata.name_to_integer_resource_variable, name_to_variable);
        assert_eq!(metadata.integer_less_is_better, vec![true]);
        let v = metadata.add_integer_resource_variable(String::from("u"), false);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(v, IntegerResourceVariable(1));
        assert_eq!(v.id(), 1);
        assert_eq!(
            metadata.integer_resource_variable_names,
            vec![String::from("v"), String::from("u")]
        );
        name_to_variable.insert(String::from("u"), 1);
        assert_eq!(metadata.name_to_integer_resource_variable, name_to_variable);
        assert_eq!(metadata.integer_less_is_better, vec![true, false]);
    }

    #[test]
    fn add_integer_resource_variable_err() {
        let mut metadata = StateMetadata::default();
        let v = metadata.add_integer_resource_variable(String::from("v"), true);
        assert!(v.is_ok());
        let v = metadata.add_integer_resource_variable(String::from("v"), false);
        assert!(v.is_err());
        assert_eq!(
            metadata.integer_resource_variable_names,
            vec![String::from("v")]
        );
        let mut name_to_variable = FxHashMap::default();
        name_to_variable.insert(String::from("v"), 0);
        assert_eq!(metadata.name_to_integer_resource_variable, name_to_variable);
        assert_eq!(metadata.integer_less_is_better, vec![true]);
    }

    #[test]
    fn integer_resource_variable_preference_ok() {
        let mut metadata = StateMetadata::default();
        let v = metadata.add_integer_resource_variable(String::from("v"), true);
        assert!(v.is_ok());
        let v = v.unwrap();
        let result = metadata.get_preference(v);
        assert!(result.is_ok());
        assert!(result.unwrap());
        let result = metadata.set_preference(v, false);
        assert!(result.is_ok());
        assert_eq!(metadata.integer_less_is_better, vec![false]);
        let result = metadata.get_preference(v);
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    #[test]
    fn integer_resource_variable_preference_err() {
        let mut metadata = StateMetadata::default();
        let v = metadata.add_integer_resource_variable(String::from("v"), true);
        assert!(v.is_ok());

        let mut metadata2 = StateMetadata::default();
        let v = metadata2.add_integer_resource_variable(String::from("v"), true);
        assert!(v.is_ok());
        let v = metadata2.add_integer_resource_variable(String::from("u"), true);
        assert!(v.is_ok());
        let v = v.unwrap();
        let result = metadata.get_preference(v);
        assert!(result.is_err());
        let result = metadata.set_preference(v, false);
        assert!(result.is_err());
        assert_eq!(metadata.integer_less_is_better, vec![true]);
    }

    #[test]
    fn add_continuous_variable_ok() {
        let mut metadata = StateMetadata::default();
        let v = metadata.add_continuous_variable(String::from("v"));
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(v, ContinuousVariable(0));
        assert_eq!(v.id(), 0);
        assert_eq!(metadata.continuous_variable_names, vec![String::from("v")]);
        let mut name_to_variable = FxHashMap::default();
        name_to_variable.insert(String::from("v"), 0);
        assert_eq!(metadata.name_to_continuous_variable, name_to_variable);
        let v = metadata.add_continuous_variable(String::from("u"));
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(v, ContinuousVariable(1));
        assert_eq!(v.id(), 1);
        assert_eq!(
            metadata.continuous_variable_names,
            vec![String::from("v"), String::from("u")]
        );
        name_to_variable.insert(String::from("u"), 1);
        assert_eq!(metadata.name_to_continuous_variable, name_to_variable);
    }

    #[test]
    fn add_continuous_variable_err() {
        let mut metadata = StateMetadata::default();
        let v = metadata.add_continuous_variable(String::from("v"));
        assert!(v.is_ok());
        let v = metadata.add_continuous_variable(String::from("v"));
        assert!(v.is_err());
        assert_eq!(metadata.continuous_variable_names, vec![String::from("v")]);
        let mut name_to_variable = FxHashMap::default();
        name_to_variable.insert(String::from("v"), 0);
        assert_eq!(metadata.name_to_continuous_variable, name_to_variable);
    }

    #[test]
    fn add_continuous_resource_variable_ok() {
        let mut metadata = StateMetadata::default();
        let v = metadata.add_continuous_resource_variable(String::from("v"), true);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(v, ContinuousResourceVariable(0));
        assert_eq!(v.id(), 0);
        assert_eq!(
            metadata.continuous_resource_variable_names,
            vec![String::from("v")]
        );
        let mut name_to_variable = FxHashMap::default();
        name_to_variable.insert(String::from("v"), 0);
        assert_eq!(
            metadata.name_to_continuous_resource_variable,
            name_to_variable
        );
        assert_eq!(metadata.continuous_less_is_better, vec![true]);
        let v = metadata.add_continuous_resource_variable(String::from("u"), false);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(v, ContinuousResourceVariable(1));
        assert_eq!(v.id(), 1);
        assert_eq!(
            metadata.continuous_resource_variable_names,
            vec![String::from("v"), String::from("u")]
        );
        name_to_variable.insert(String::from("u"), 1);
        assert_eq!(
            metadata.name_to_continuous_resource_variable,
            name_to_variable
        );
        assert_eq!(metadata.continuous_less_is_better, vec![true, false]);
    }

    #[test]
    fn add_continuous_resource_variable_err() {
        let mut metadata = StateMetadata::default();
        let v = metadata.add_continuous_resource_variable(String::from("v"), true);
        assert!(v.is_ok());
        let v = metadata.add_continuous_resource_variable(String::from("v"), false);
        assert!(v.is_err());
        assert_eq!(
            metadata.continuous_resource_variable_names,
            vec![String::from("v")]
        );
        let mut name_to_variable = FxHashMap::default();
        name_to_variable.insert(String::from("v"), 0);
        assert_eq!(
            metadata.name_to_continuous_resource_variable,
            name_to_variable
        );
        assert_eq!(metadata.continuous_less_is_better, vec![true]);
    }

    #[test]
    fn continuous_resource_variable_preference_ok() {
        let mut metadata = StateMetadata::default();
        let v = metadata.add_continuous_resource_variable(String::from("v"), true);
        assert!(v.is_ok());
        let v = v.unwrap();
        let result = metadata.get_preference(v);
        assert!(result.is_ok());
        assert!(result.unwrap());
        let result = metadata.set_preference(v, false);
        assert!(result.is_ok());
        assert_eq!(metadata.continuous_less_is_better, vec![false]);
        let result = metadata.get_preference(v);
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    #[test]
    fn continuous_resource_variable_preference_err() {
        let mut metadata = StateMetadata::default();
        let v = metadata.add_continuous_resource_variable(String::from("v"), true);
        assert!(v.is_ok());

        let mut metadata2 = StateMetadata::default();
        let v = metadata2.add_continuous_resource_variable(String::from("v"), true);
        assert!(v.is_ok());
        let v = metadata2.add_continuous_resource_variable(String::from("u"), true);
        assert!(v.is_ok());
        let v = v.unwrap();
        let result = metadata.get_preference(v);
        assert!(result.is_err());
        let result = metadata.set_preference(v, false);
        assert!(result.is_err());
        assert_eq!(metadata.continuous_less_is_better, vec![true]);
    }
}
