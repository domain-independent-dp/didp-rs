//! # DyPDL
//!
//! A library for Dynamic Programming Description Language (DyPDL).

mod base_case;
mod effect;
pub mod expression;
mod grounded_condition;
mod state;
mod table;
mod table_data;
mod table_registry;
mod transition;
mod util;
pub mod variable_type;

pub use base_case::BaseCase;
pub use effect::Effect;
pub use grounded_condition::GroundedCondition;
pub use state::{
    AccessPreference, CheckVariable, ContinuousResourceVariable, ContinuousVariable, DPState,
    ElementResourceVariable, ElementVariable, GetObjectTypeOf, IntegerResourceVariable,
    IntegerVariable, ObjectType, ResourceVariables, SetVariable, SignatureVariables, State,
    StateMetadata, VectorVariable,
};
pub use table::{Table, Table1D, Table2D, Table3D};
pub use table_data::{
    Table1DHandle, Table2DHandle, Table3DHandle, TableData, TableHandle, TableInterface,
};
pub use table_registry::TableRegistry;
pub use transition::{AddEffect, CostExpression, Transition};
pub use util::ModelErr;
pub use variable_type::{Continuous, Element, Integer, Set, Vector};

pub mod prelude {
    //! DyPDL's prelude.

    pub use super::expression::{
        ComparisonOperator, Condition, ContinuousBinaryOperation, ContinuousExpression,
        ElementExpression, IfThenElse, IntegerExpression, MaxMin, SetElementOperation,
        SetExpression, VectorExpression,
    };
    pub use super::{
        AccessPreference, AccessTarget, AddDualBound, AddEffect, CheckExpression, CheckVariable,
        Continuous, ContinuousResourceVariable, ContinuousVariable, CostExpression, CostType,
        Element, ElementResourceVariable, ElementVariable, GetObjectTypeOf, Integer,
        IntegerResourceVariable, IntegerVariable, Model, ObjectType, ReduceFunction,
        ResourceVariables, Set, SetVariable, SignatureVariables, State, StateMetadata,
        Table1DHandle, Table2DHandle, Table3DHandle, TableHandle, TableInterface, Transition,
        Vector, VectorVariable,
    };
}

use rustc_hash::{FxHashMap, FxHashSet};
use std::panic;

/// Type of numeric values to represent the costs of states.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum CostType {
    Integer,
    Continuous,
}

impl Default for CostType {
    #[inline]
    fn default() -> Self {
        Self::Integer
    }
}

/// How to compute the value of a state given applicable transitions.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ReduceFunction {
    /// Minimum of the evaluation values of the cost expressions of applicable transitions.
    Min,
    /// Maximum of the evaluation values of the cost expressions of applicable transitions.
    Max,
    /// Sum of the evaluation values of the cost expressions of applicable transitions.
    Sum,
    /// Product of the evaluation values of the cost expressions of applicable transitions.
    Product,
}

impl Default for ReduceFunction {
    #[inline]
    fn default() -> Self {
        Self::Min
    }
}

/// DyPDL model.
#[derive(Debug, PartialEq, Clone, Default)]
pub struct Model {
    /// Information about state variables.
    pub state_metadata: StateMetadata,
    /// Target state.
    pub target: State,
    /// Tables of constants.
    pub table_registry: TableRegistry,
    /// State constraints.
    pub state_constraints: Vec<GroundedCondition>,
    /// Base cases.
    pub base_cases: Vec<BaseCase>,
    /// Explicit definitions of base states.
    pub base_states: Vec<State>,
    /// Specifying how to compute the value of a state from applicable transitions.
    pub reduce_function: ReduceFunction,
    /// Type of the cost.
    pub cost_type: CostType,
    /// Forward transitions.
    pub forward_transitions: Vec<Transition>,
    /// Forward forced transitions.
    pub forward_forced_transitions: Vec<Transition>,
    /// Backward transitions.
    pub backward_transitions: Vec<Transition>,
    /// Backward forced transitions.
    pub backward_forced_transitions: Vec<Transition>,
    /// Dual bounds.
    pub dual_bounds: Vec<CostExpression>,
}

impl Model {
    /// Returns a model whose cost is integer.
    pub fn integer_cost_model() -> Model {
        Model {
            cost_type: CostType::Integer,
            ..Default::default()
        }
    }

    /// Returns a model whose cost is continuous.
    pub fn continuous_cost_model() -> Model {
        Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        }
    }

    /// Returns true if a state satisfies all state constraints and false otherwise.
    pub fn check_constraints<U: DPState>(&self, state: &U) -> bool {
        self.state_constraints.iter().all(|constraint| {
            constraint
                .is_satisfied(state, &self.table_registry)
                .unwrap_or(true)
        })
    }

    /// Returns true if a state satisfies any of base cases and false otherwise.
    ///
    /// # Panics
    ///
    /// if the cost of the transition state is used or a min/max reduce operation is performed on an empty set or vector.
    pub fn is_base<U: DPState>(&self, state: &U) -> bool {
        self.base_cases
            .iter()
            .any(|case| case.is_satisfied(state, &self.table_registry))
            || self
                .base_states
                .iter()
                .any(|base| base.is_satisfied(state, &self.state_metadata))
    }

    /// Returns true if there is a resource variable and false otherwise.
    pub fn has_resource_variables(&self) -> bool {
        self.state_metadata.has_resource_variables()
    }

    /// Returns true if a dual bound is defined and false otherwise.
    pub fn has_dual_bounds(&self) -> bool {
        !self.dual_bounds.is_empty()
    }

    /// Evaluate the dual bound given a state.
    ///
    /// # Panics
    ///
    /// if the cost of the transition state is used or a min/max reduce operation is performed on an empty set or vector.
    pub fn eval_dual_bound<U: DPState, T: variable_type::Numeric + Ord>(
        &self,
        state: &U,
    ) -> Option<T> {
        match self.reduce_function {
            ReduceFunction::Min => self
                .dual_bounds
                .iter()
                .map(|b| b.eval(state, &self.table_registry))
                .max(),
            ReduceFunction::Max => self
                .dual_bounds
                .iter()
                .map(|b| b.eval(state, &self.table_registry))
                .min(),
            _ => None,
        }
    }

    /// Validate a solution consists of forward transitions.
    ///
    /// # Panics
    ///
    /// if the cost of the transition state is used or a min/max reduce operation is performed on an empty set or vector.
    pub fn validate_forward<T: variable_type::Numeric>(
        &self,
        transitions: &[Transition],
        cost: T,
        show_message: bool,
    ) -> bool {
        let mut state_vec = vec![self.target.clone()];
        for (i, transition) in transitions.iter().enumerate() {
            let state = state_vec.last().unwrap();
            for forced_transition in &self.forward_forced_transitions {
                if forced_transition.is_applicable(state, &self.table_registry) {
                    if forced_transition == transition {
                        break;
                    } else {
                        println!("Forced transition {} is applicable in the {} th state, but transition {} is applied", forced_transition.get_full_name(), i, transition.get_full_name());
                        return false;
                    }
                }
            }
            if !transition.is_applicable(state, &self.table_registry) {
                if show_message {
                    println!(
                        "The {} th transition {} is not applicable.",
                        i,
                        transition.get_full_name()
                    );
                }
                return false;
            }
            let next_state = state.apply_effect(&transition.effect, &self.table_registry);
            if !self.check_constraints(&next_state) {
                if show_message {
                    println!("The {} th state does not satisfy state constraints", i + 1);
                }
                return false;
            }
            state_vec.push(next_state);
        }
        if !self.is_base(state_vec.last().unwrap()) {
            if show_message {
                println!("The last state is not a base state.")
            }
            return false;
        }
        let mut validation_cost = T::zero();
        state_vec.pop();
        for (state, transition) in state_vec.into_iter().zip(transitions).rev() {
            validation_cost = transition.eval_cost(validation_cost, &state, &self.table_registry);
        }
        if cost != validation_cost {
            if show_message {
                println!("The cost does not match the actual cost.")
            }
            return false;
        }
        true
    }

    /// Returns object type given a name.
    ///
    /// # Errors
    ///
    /// if no object type with the name.
    #[inline]
    pub fn get_object_type(&self, name: &str) -> Result<ObjectType, ModelErr> {
        self.state_metadata.get_object_type(name)
    }

    /// Adds an object type and returns it.
    ///
    /// # Errors
    ///
    /// if the name is already used.
    #[inline]
    pub fn add_object_type<T>(&mut self, name: T, number: usize) -> Result<ObjectType, ModelErr>
    where
        String: From<T>,
    {
        self.state_metadata.add_object_type(name, number)
    }

    /// Retunrs the number of objects associated with the type.
    ///
    /// # Errors
    ///
    /// if the object type is not in the model.
    #[inline]
    pub fn get_number_of_objects(&self, ob: ObjectType) -> Result<usize, ModelErr> {
        self.state_metadata.get_number_of_objects(ob)
    }

    /// Change the number of objects.
    ///
    /// # Errors
    ///
    /// if the object type is not in the model.
    #[inline]
    pub fn set_number_of_object(&mut self, ob: ObjectType, number: usize) -> Result<(), ModelErr> {
        self.state_metadata.set_number_of_object(ob, number)
    }

    /// Create a set of objects asociated with the type.
    ///
    /// # Errors
    ///
    /// if the object type is not in the model or an input value is greater than or equal to the number of the objects.
    #[inline]
    pub fn create_set(&self, ob: ObjectType, array: &[Element]) -> Result<Set, ModelErr> {
        self.state_metadata.create_set(ob, array)
    }

    /// Returns an element variable given a name.
    ///
    /// # Errors
    ///
    /// if no such variable.
    #[inline]
    pub fn get_element_variable(&self, name: &str) -> Result<ElementVariable, ModelErr> {
        self.state_metadata.get_element_variable(name)
    }

    /// Adds and returns an element variable.
    ///
    /// The value in the target state must be specified.
    ///
    /// # Errors
    ///
    /// if the name is already used, the object type is not in the model, or the target value is greater than or equal to the number of the objects.
    pub fn add_element_variable<T>(
        &mut self,
        name: T,
        ob: ObjectType,
        target: Element,
    ) -> Result<ElementVariable, ModelErr>
    where
        String: From<T>,
    {
        let n = self.get_number_of_objects(ob)?;
        if target >= n {
            Err(ModelErr::new(format!(
                "target value for element variable {} >= #objects ({})",
                String::from(name),
                n
            )))
        } else {
            let v = self.state_metadata.add_element_variable(name, ob)?;
            self.target
                .signature_variables
                .element_variables
                .push(target);
            Ok(v)
        }
    }

    /// Returns an element resouce variable given a name.
    ///
    /// # Errors
    ///
    /// if no such variable.
    #[inline]
    pub fn get_element_resource_variable(
        &self,
        name: &str,
    ) -> Result<ElementResourceVariable, ModelErr> {
        self.state_metadata.get_element_resource_variable(name)
    }

    /// Adds and returns an element resource variable.
    ///
    /// The value in the target state must be specified.
    ///
    /// # Errors
    ///
    /// if the name is already used, the object type is not in the model, or the target value is greater than or equal to the number of the objects.
    pub fn add_element_resource_variable<T>(
        &mut self,
        name: T,
        ob: ObjectType,
        less_is_better: bool,
        target: Element,
    ) -> Result<ElementResourceVariable, ModelErr>
    where
        String: From<T>,
    {
        let n = self.get_number_of_objects(ob)?;
        if target >= n {
            Err(ModelErr::new(format!(
                "target value for element resource variable {} >= #objects ({})",
                String::from(name),
                n
            )))
        } else {
            let v = self
                .state_metadata
                .add_element_resource_variable(name, ob, less_is_better)?;
            self.target
                .resource_variables
                .element_variables
                .push(target);
            Ok(v)
        }
    }

    /// Returns a set variable given a name.
    ///
    /// # Errors
    ///
    /// if no such variable.
    #[inline]
    pub fn get_set_variable(&self, name: &str) -> Result<SetVariable, ModelErr> {
        self.state_metadata.get_set_variable(name)
    }

    /// Adds and returns a set variable.
    ///
    /// The value in the target state must be specified.
    ///
    /// # Errors
    ///
    /// if the name is already used, the object type is not in the model, or the target contains a value greater than or equal to the number of the objects.
    pub fn add_set_variable<T>(
        &mut self,
        name: T,
        ob: ObjectType,
        target: Set,
    ) -> Result<SetVariable, ModelErr>
    where
        String: From<T>,
    {
        let n = self.get_number_of_objects(ob)?;
        if target.len() != n {
            Err(ModelErr::new(format!(
                "target set size {} for set variable {} != #objects ({})",
                target.len(),
                String::from(name),
                n
            )))
        } else {
            let v = self.state_metadata.add_set_variable(name, ob)?;
            self.target.signature_variables.set_variables.push(target);
            Ok(v)
        }
    }

    /// Returns a vector variable given a name.
    ///
    /// # Errors
    ///
    /// if no such variable.
    #[inline]
    pub fn get_vector_variable(&self, name: &str) -> Result<VectorVariable, ModelErr> {
        self.state_metadata.get_vector_variable(name)
    }

    /// Adds and returns a vector variable.
    ///
    /// The value in the target state must be specified.
    ///
    /// # Errors
    ///
    /// if the name is already used, the object type is not in the model, or the target contains a value greater than or equal to the number of the objects.
    pub fn add_vector_variable<T>(
        &mut self,
        name: T,
        ob: ObjectType,
        target: Vector,
    ) -> Result<VectorVariable, ModelErr>
    where
        String: From<T>,
    {
        let n = self.get_number_of_objects(ob)?;
        if target.iter().any(|v| *v >= n) {
            Err(ModelErr::new(format!(
                "target vector {:?} for vector variable {} contains a value >= #objects ({})",
                target,
                String::from(name),
                n
            )))
        } else {
            let v = self.state_metadata.add_vector_variable(name, ob)?;
            self.target
                .signature_variables
                .vector_variables
                .push(target);
            Ok(v)
        }
    }

    /// Returns an integer variable given a name.
    ///
    /// # Errors
    ///
    /// if no such variable.
    #[inline]
    pub fn get_integer_variable(&self, name: &str) -> Result<IntegerVariable, ModelErr> {
        self.state_metadata.get_integer_variable(name)
    }

    /// Adds and returns an integer variable.
    ///
    /// The value in the target state must be specified.
    ///
    /// # Errors
    ///
    /// if the name is already used.
    pub fn add_integer_variable<T>(
        &mut self,
        name: T,
        target: Integer,
    ) -> Result<IntegerVariable, ModelErr>
    where
        String: From<T>,
    {
        let v = self.state_metadata.add_integer_variable(name)?;
        self.target
            .signature_variables
            .integer_variables
            .push(target);
        Ok(v)
    }

    /// Returns an integer resource variable given a name.
    ///
    /// # Errors
    ///
    /// if no such variable.
    #[inline]
    pub fn get_integer_resource_variable(
        &self,
        name: &str,
    ) -> Result<IntegerResourceVariable, ModelErr> {
        self.state_metadata.get_integer_resource_variable(name)
    }

    /// Adds and returns an integer resource variable.
    ///
    /// The value in the target state must be specified.
    ///
    /// # Errors
    ///
    /// if the name is already used.
    pub fn add_integer_resource_variable<T>(
        &mut self,
        name: T,
        less_is_better: bool,
        target: Integer,
    ) -> Result<IntegerResourceVariable, ModelErr>
    where
        String: From<T>,
    {
        let v = self
            .state_metadata
            .add_integer_resource_variable(name, less_is_better)?;
        self.target
            .resource_variables
            .integer_variables
            .push(target);
        Ok(v)
    }

    /// Returns a continuous variable given a name.
    ///
    /// # Errors
    ///
    /// if no such variable.
    #[inline]
    pub fn get_continuous_variable(&self, name: &str) -> Result<ContinuousVariable, ModelErr> {
        self.state_metadata.get_continuous_variable(name)
    }

    /// Adds and returns a continuous variable.
    ///
    /// The value in the target state must be specified.
    ///
    /// # Errors
    ///
    /// if the name is already used.
    pub fn add_continuous_variable<T>(
        &mut self,
        name: T,
        target: Continuous,
    ) -> Result<ContinuousVariable, ModelErr>
    where
        String: From<T>,
    {
        let v = self.state_metadata.add_continuous_variable(name)?;
        self.target
            .signature_variables
            .continuous_variables
            .push(target);
        Ok(v)
    }

    /// Returns a continuous resource variable given a name.
    ///
    /// # Errors
    ///
    /// if no such variable.
    #[inline]
    pub fn get_continuous_resource_variable(
        &self,
        name: &str,
    ) -> Result<ContinuousResourceVariable, ModelErr> {
        self.state_metadata.get_continuous_resource_variable(name)
    }

    /// Adds and returns a continuous resource variable.
    ///
    /// The value in the target state must be specified.
    ///
    /// # Errors
    ///
    /// if the name is already used.
    pub fn add_continuous_resource_variable<T>(
        &mut self,
        name: T,
        less_is_better: bool,
        target: Continuous,
    ) -> Result<ContinuousResourceVariable, ModelErr>
    where
        String: From<T>,
    {
        let v = self
            .state_metadata
            .add_continuous_resource_variable(name, less_is_better)?;
        self.target
            .resource_variables
            .continuous_variables
            .push(target);
        Ok(v)
    }

    /// Adds a state constraint.
    ///
    /// # Errors
    ///
    /// if the condition is invalid, e.., it uses not existing variables or the state of the transitioned state.
    #[inline]
    pub fn add_state_constraint(
        &mut self,
        condition: expression::Condition,
    ) -> Result<(), ModelErr> {
        self.check_expression(&condition, false)?;
        let simplified = condition.simplify(&self.table_registry);
        match simplified {
            expression::Condition::Constant(true) => {
                eprintln!("constraint {:?} is always satisfied", condition)
            }
            expression::Condition::Constant(false) => {
                eprintln!("constraint {:?} cannot be satisfied", condition)
            }
            _ => {}
        }
        self.state_constraints.push(GroundedCondition {
            condition: simplified,
            ..Default::default()
        });
        Ok(())
    }

    /// Adds a base case.
    ///
    /// # Errors
    ///
    /// if a condition is invalid, e.g., it uses variables not existing in this model or the state of the transitioned state.
    #[inline]
    pub fn add_base_case(
        &mut self,
        conditions: Vec<expression::Condition>,
    ) -> Result<(), ModelErr> {
        let mut simplified_conditions = Vec::with_capacity(conditions.len());
        for condition in &conditions {
            self.check_expression(condition, false)?;
            let simplified = condition.simplify(&self.table_registry);
            match simplified {
                expression::Condition::Constant(true) => {
                    eprintln!("base case condition {:?} is always satisfied", condition)
                }
                expression::Condition::Constant(false) => {
                    eprintln!("base case condition {:?} cannot be satisfied", condition)
                }
                _ => {}
            }
            simplified_conditions.push(simplified);
        }
        self.base_cases.push(BaseCase::new(
            simplified_conditions
                .into_iter()
                .map(|condition| GroundedCondition {
                    condition,
                    ..Default::default()
                })
                .collect(),
        ));
        Ok(())
    }

    /// Check if a state is valid.
    ///
    /// # Errors
    ///
    /// If a state is invalid, e.g., it contains variables not existing in this model.
    pub fn check_state<'a, T: DPState>(&self, state: &'a T) -> Result<(), ModelErr>
    where
        &'a T: panic::UnwindSafe,
    {
        self.state_metadata.check_state(state)
    }

    /// Adds a base state.
    ///
    /// # Errors
    ///
    /// If a state is invalid, e.g., it contains variables not existing in this model.
    #[inline]
    pub fn add_base_state(&mut self, state: State) -> Result<(), ModelErr> {
        self.check_state(&state)?;
        self.base_states.push(state);
        Ok(())
    }

    /// Returns the reduce function.
    #[inline]
    pub fn get_reduce_function(&self) -> ReduceFunction {
        self.reduce_function.clone()
    }

    /// Changes the reduce function.
    #[inline]
    pub fn set_reduce_function(&mut self, reduce_function: ReduceFunction) {
        self.reduce_function = reduce_function
    }

    /// Adds a forward transition.
    ///
    /// # Errors
    ///
    /// if it uses an invalid expression, e.g., it uses variables not existing in this model or the state of the transitioned state.
    #[inline]
    pub fn add_forward_transition(&mut self, transition: Transition) -> Result<(), ModelErr> {
        let transition = self.check_and_simplify_transition(&transition)?;
        self.forward_transitions.push(transition);
        Ok(())
    }

    /// Adds a forward forced transition.
    ///
    /// # Errors
    ///
    /// if it uses an invalid expression, e.g., it uses variables not existing in this model or the state of the transitioned state.
    #[inline]
    pub fn add_forward_forced_transition(
        &mut self,
        transition: Transition,
    ) -> Result<(), ModelErr> {
        let transition = self.check_and_simplify_transition(&transition)?;
        self.forward_forced_transitions.push(transition);
        Ok(())
    }

    /// Adds a backward transition.
    ///
    /// # Errors
    ///
    /// if it uses an invalid expression, e.g., it uses variables not existing in this model or the state of the transitioned state.
    #[inline]
    pub fn add_backward_transition(&mut self, transition: Transition) -> Result<(), ModelErr> {
        let transition = self.check_and_simplify_transition(&transition)?;
        self.backward_transitions.push(transition);
        Ok(())
    }

    /// Adds a backward forced transition.
    ///
    /// # Errors
    ///
    /// if it uses an invalid expression, e.g., it uses variables not existing in this model or the state of the transitioned state.
    #[inline]
    pub fn add_backward_forced_transition(
        &mut self,
        transition: Transition,
    ) -> Result<(), ModelErr> {
        let transition = self.check_and_simplify_transition(&transition)?;
        self.backward_forced_transitions.push(transition);
        Ok(())
    }

    fn check_and_simplify_transition(
        &self,
        transition: &Transition,
    ) -> Result<Transition, ModelErr> {
        let cost = match &transition.cost {
            CostExpression::Integer(expression) => {
                self.check_expression(expression, true)?;
                CostExpression::from(expression.simplify(&self.table_registry))
            }
            CostExpression::Continuous(expression) => {
                if self.cost_type == CostType::Integer {
                    return Err(ModelErr::new(String::from("Could not add a transition with a continuous cost expression for an integer cost model")));
                }
                self.check_expression(expression, true)?;
                CostExpression::from(expression.simplify(&self.table_registry))
            }
        };
        let n = self.state_metadata.number_of_set_variables();
        for (i, e) in &transition.elements_in_set_variable {
            if *i >= n {
                return Err(ModelErr::new(format!(
                    "set variable id {} >= #set variables ({})",
                    *i, n
                )));
            } else {
                let object = self.state_metadata.set_variable_to_object[*i];
                let m = self.state_metadata.object_numbers[object];
                if *e >= m {
                    return Err(ModelErr::new(format!(
                        "element {} >= #objects ({}) for object id {}",
                        *e, n, object
                    )));
                }
            }
        }
        let n = self.state_metadata.number_of_vector_variables();
        for (i, e) in &transition.elements_in_vector_variable {
            if *i >= n {
                return Err(ModelErr::new(format!(
                    "vector variable id {} >= #set variables ({})",
                    *i, n
                )));
            } else {
                let object = self.state_metadata.vector_variable_to_object[*i];
                let m = self.state_metadata.object_numbers[object];
                if *e >= m {
                    return Err(ModelErr::new(format!(
                        "element {} >= #objects ({}) for object id {}",
                        *e, n, object
                    )));
                }
            }
        }
        let mut preconditions = Vec::with_capacity(transition.preconditions.len());
        for condition in &transition.preconditions {
            let n = self.state_metadata.number_of_set_variables();
            for (i, e) in &condition.elements_in_set_variable {
                if *i >= n {
                    return Err(ModelErr::new(format!(
                        "set variable id {} >= #set variables ({})",
                        *i, n
                    )));
                } else {
                    let object = self.state_metadata.set_variable_to_object[*i];
                    let m = self.state_metadata.object_numbers[object];
                    if *e >= m {
                        return Err(ModelErr::new(format!(
                            "element {} >= #objects ({}) for object id {}",
                            *e, n, object
                        )));
                    }
                }
            }
            let n = self.state_metadata.number_of_vector_variables();
            for (i, e) in &condition.elements_in_vector_variable {
                if *i >= n {
                    return Err(ModelErr::new(format!(
                        "vector variable id {} >= #set variables ({})",
                        *i, n
                    )));
                } else {
                    let object = self.state_metadata.vector_variable_to_object[*i];
                    let m = self.state_metadata.object_numbers[object];
                    if *e >= m {
                        return Err(ModelErr::new(format!(
                            "element {} >= #objects ({}) for object id {}",
                            *e, n, object
                        )));
                    }
                }
            }
            self.check_expression(&condition.condition, false)?;
            let simplified = condition.condition.simplify(&self.table_registry);
            match simplified {
                expression::Condition::Constant(true) => {
                    eprintln!("precondition {:?} is always satisfied", condition);
                }
                expression::Condition::Constant(false) => {
                    eprintln!("precondition {:?} is never satisfied", condition);
                }
                _ => {}
            }
            let elements_in_set_variable = condition.elements_in_set_variable.clone();
            let elements_in_vector_variable = condition.elements_in_vector_variable.clone();
            preconditions.push(GroundedCondition {
                elements_in_set_variable,
                elements_in_vector_variable,
                condition: simplified,
            })
        }
        let n = self.state_metadata.number_of_set_variables();
        let mut set_effects = Vec::with_capacity(transition.effect.set_effects.len());
        let mut variable_ids = FxHashSet::default();
        for (i, expression) in &transition.effect.set_effects {
            if *i >= n {
                return Err(ModelErr::new(format!(
                    "set variable id {} >= #set variables ({})",
                    *i, n
                )));
            } else if variable_ids.contains(i) {
                return Err(ModelErr::new(format!(
                    "the transition already has an effect on set variable id {}",
                    *i
                )));
            }
            self.check_expression(expression, false)?;
            set_effects.push((*i, expression.simplify(&self.table_registry)));
            variable_ids.insert(*i);
        }
        let n = self.state_metadata.number_of_vector_variables();
        let mut vector_effects = Vec::with_capacity(transition.effect.vector_effects.len());
        let mut variable_ids = FxHashSet::default();
        for (i, expression) in &transition.effect.vector_effects {
            if *i >= n {
                return Err(ModelErr::new(format!(
                    "vector variable id {} >= #vector variables ({})",
                    *i, n
                )));
            } else if variable_ids.contains(i) {
                return Err(ModelErr::new(format!(
                    "the transition already has an effect on set variable id {}",
                    *i
                )));
            }
            self.check_expression(expression, false)?;
            vector_effects.push((*i, expression.simplify(&self.table_registry)));
            variable_ids.insert(*i);
        }
        let n = self.state_metadata.number_of_element_variables();
        let mut element_effects = Vec::with_capacity(transition.effect.element_effects.len());
        let mut variable_ids = FxHashSet::default();
        for (i, expression) in &transition.effect.element_effects {
            if *i >= n {
                return Err(ModelErr::new(format!(
                    "element variable id {} >= #element variables ({})",
                    *i, n
                )));
            } else if variable_ids.contains(i) {
                return Err(ModelErr::new(format!(
                    "the transition already has an effect on set variable id {}",
                    *i
                )));
            }
            self.check_expression(expression, false)?;
            element_effects.push((*i, expression.simplify(&self.table_registry)));
            variable_ids.insert(*i);
        }
        let n = self.state_metadata.number_of_integer_variables();
        let mut integer_effects = Vec::with_capacity(transition.effect.integer_effects.len());
        let mut variable_ids = FxHashSet::default();
        for (i, expression) in &transition.effect.integer_effects {
            if *i >= n {
                return Err(ModelErr::new(format!(
                    "integer variable id {} >= #integer variables ({})",
                    *i, n
                )));
            } else if variable_ids.contains(i) {
                return Err(ModelErr::new(format!(
                    "the transition already has an effect on set variable id {}",
                    *i
                )));
            }
            self.check_expression(expression, false)?;
            integer_effects.push((*i, expression.simplify(&self.table_registry)));
            variable_ids.insert(*i);
        }
        let n = self.state_metadata.number_of_continuous_variables();
        let mut continuous_effects = Vec::with_capacity(transition.effect.continuous_effects.len());
        let mut variable_ids = FxHashSet::default();
        for (i, expression) in &transition.effect.continuous_effects {
            if *i >= n {
                return Err(ModelErr::new(format!(
                    "continuous variable id {} >= #continuous variables ({})",
                    *i, n
                )));
            } else if variable_ids.contains(i) {
                return Err(ModelErr::new(format!(
                    "the transition already has an effect on set variable id {}",
                    *i
                )));
            }
            self.check_expression(expression, false)?;
            continuous_effects.push((*i, expression.simplify(&self.table_registry)));
            variable_ids.insert(*i);
        }
        let n = self.state_metadata.number_of_element_resource_variables();
        let mut element_resource_effects =
            Vec::with_capacity(transition.effect.element_resource_effects.len());
        let mut variable_ids = FxHashSet::default();
        for (i, expression) in &transition.effect.element_resource_effects {
            if *i >= n {
                return Err(ModelErr::new(format!(
                    "element_resource variable id {} >= #element_resource variables ({})",
                    *i, n
                )));
            } else if variable_ids.contains(i) {
                return Err(ModelErr::new(format!(
                    "the transition already has an effect on set variable id {}",
                    *i
                )));
            }
            self.check_expression(expression, false)?;
            element_resource_effects.push((*i, expression.simplify(&self.table_registry)));
            variable_ids.insert(*i);
        }
        let n = self.state_metadata.number_of_integer_resource_variables();
        let mut integer_resource_effects =
            Vec::with_capacity(transition.effect.integer_resource_effects.len());
        let mut variable_ids = FxHashSet::default();
        for (i, expression) in &transition.effect.integer_resource_effects {
            if *i >= n {
                return Err(ModelErr::new(format!(
                    "integer_resource variable id {} >= #integer_resource variables ({})",
                    *i, n
                )));
            } else if variable_ids.contains(i) {
                return Err(ModelErr::new(format!(
                    "the transition already has an effect on set variable id {}",
                    *i
                )));
            }
            self.check_expression(expression, false)?;
            integer_resource_effects.push((*i, expression.simplify(&self.table_registry)));
            variable_ids.insert(*i);
        }
        let n = self
            .state_metadata
            .number_of_continuous_resource_variables();
        let mut continuous_resource_effects =
            Vec::with_capacity(transition.effect.continuous_resource_effects.len());
        let mut variable_ids = FxHashSet::default();
        for (i, expression) in &transition.effect.continuous_resource_effects {
            if *i >= n {
                return Err(ModelErr::new(format!(
                    "continuous_resource variable id {} >= #continuous_resource variables ({})",
                    *i, n
                )));
            } else if variable_ids.contains(i) {
                return Err(ModelErr::new(format!(
                    "the transition already has an effect on set variable id {}",
                    *i
                )));
            }
            self.check_expression(expression, false)?;
            continuous_resource_effects.push((*i, expression.simplify(&self.table_registry)));
            variable_ids.insert(*i);
        }
        let effect = Effect {
            set_effects,
            vector_effects,
            element_effects,
            integer_effects,
            continuous_effects,
            element_resource_effects,
            integer_resource_effects,
            continuous_resource_effects,
        };
        Ok(Transition {
            name: transition.name.clone(),
            parameter_names: transition.parameter_names.clone(),
            parameter_values: transition.parameter_values.clone(),
            elements_in_set_variable: transition.elements_in_set_variable.clone(),
            elements_in_vector_variable: transition.elements_in_vector_variable.clone(),
            preconditions,
            effect,
            cost,
        })
    }
}

/// A trait for accessing the values in the target state.
pub trait AccessTarget<T, U> {
    /// Returns the value in the target state.
    ///
    /// # Errors
    ///
    /// if the variable is not in the model.
    fn get_target(&self, variable: T) -> Result<U, ModelErr>;
    /// Set the value in the target state
    ///
    /// # Errors
    ///
    /// if the variable is not in the model.
    fn set_target(&mut self, variable: T, target: U) -> Result<(), ModelErr>;
}

impl AccessTarget<ElementVariable, Element> for Model {
    fn get_target(&self, variable: ElementVariable) -> Result<Element, ModelErr> {
        self.state_metadata.check_variable(variable)?;
        Ok(self.target.get_element_variable(variable.id()))
    }

    fn set_target(&mut self, variable: ElementVariable, target: Element) -> Result<(), ModelErr> {
        let ob = self.get_object_type_of(variable)?;
        let n = self.get_number_of_objects(ob)?;
        if target >= n {
            Err(ModelErr::new(format!(
                "target value for element variable id {} >= #objects ({})",
                variable.id(),
                n
            )))
        } else {
            self.target.signature_variables.element_variables[variable.id()] = target;
            Ok(())
        }
    }
}

impl AccessTarget<ElementResourceVariable, Element> for Model {
    fn get_target(&self, variable: ElementResourceVariable) -> Result<Element, ModelErr> {
        self.state_metadata.check_variable(variable)?;
        Ok(self.target.get_element_resource_variable(variable.id()))
    }

    fn set_target(
        &mut self,
        variable: ElementResourceVariable,
        target: Element,
    ) -> Result<(), ModelErr> {
        let ob = self.get_object_type_of(variable)?;
        let n = self.get_number_of_objects(ob)?;
        if target >= n {
            Err(ModelErr::new(format!(
                "target value for element variable id {} >= #objects ({})",
                variable.id(),
                n
            )))
        } else {
            self.target.resource_variables.element_variables[variable.id()] = target;
            Ok(())
        }
    }
}

impl AccessTarget<SetVariable, Set> for Model {
    fn get_target(&self, variable: SetVariable) -> Result<Set, ModelErr> {
        self.state_metadata.check_variable(variable)?;
        Ok(self.target.get_set_variable(variable.id()).clone())
    }

    fn set_target(&mut self, variable: SetVariable, target: Set) -> Result<(), ModelErr> {
        let ob = self.get_object_type_of(variable)?;
        let n = self.get_number_of_objects(ob)?;
        if target.len() != n {
            Err(ModelErr::new(format!(
                "target set size {} for set variable id {} != #objects ({})",
                target.len(),
                variable.id(),
                n
            )))
        } else {
            self.target.signature_variables.set_variables[variable.id()] = target;
            Ok(())
        }
    }
}

impl AccessTarget<VectorVariable, Vector> for Model {
    fn get_target(&self, variable: VectorVariable) -> Result<Vector, ModelErr> {
        self.state_metadata.check_variable(variable)?;
        Ok(self.target.get_vector_variable(variable.id()).clone())
    }

    fn set_target(&mut self, variable: VectorVariable, target: Vector) -> Result<(), ModelErr> {
        let ob = self.get_object_type_of(variable)?;
        let n = self.get_number_of_objects(ob)?;
        if target.iter().any(|v| *v >= n) {
            Err(ModelErr::new(format!(
                "target vector {:?} for vector variable id {} contains a value >= #objects ({})",
                target,
                variable.id(),
                n
            )))
        } else {
            self.target.signature_variables.vector_variables[variable.id()] = target;
            Ok(())
        }
    }
}

macro_rules! impl_access_target {
    ($T:ty,$U:ty,$x:ident,$y:ident) => {
        impl AccessTarget<$T, $U> for Model {
            fn get_target(&self, v: $T) -> Result<$U, ModelErr> {
                self.state_metadata.check_variable(v)?;
                Ok(self.target.$x.$y[v.id()].clone())
            }

            fn set_target(&mut self, v: $T, target: $U) -> Result<(), ModelErr> {
                self.state_metadata.check_variable(v)?;
                self.target.$x.$y[v.id()] = target;
                Ok(())
            }
        }
    };
}

impl_access_target!(
    IntegerVariable,
    Integer,
    signature_variables,
    integer_variables
);
impl_access_target!(
    IntegerResourceVariable,
    Integer,
    resource_variables,
    integer_variables
);
impl_access_target!(
    ContinuousVariable,
    Continuous,
    signature_variables,
    continuous_variables
);
impl_access_target!(
    ContinuousResourceVariable,
    Continuous,
    resource_variables,
    continuous_variables
);

macro_rules! impl_get_object_type_of {
    ($T:ty) => {
        impl GetObjectTypeOf<$T> for Model {
            #[inline]
            fn get_object_type_of(&self, v: $T) -> Result<ObjectType, ModelErr> {
                self.state_metadata.get_object_type_of(v)
            }
        }
    };
}

impl_get_object_type_of!(ElementVariable);
impl_get_object_type_of!(ElementResourceVariable);
impl_get_object_type_of!(SetVariable);
impl_get_object_type_of!(VectorVariable);

macro_rules! impl_access_preference {
    ($T:ty) => {
        impl AccessPreference<$T> for Model {
            #[inline]
            fn get_preference(&self, v: $T) -> Result<bool, ModelErr> {
                self.state_metadata.get_preference(v)
            }

            #[inline]
            fn set_preference(&mut self, v: $T, less_is_better: bool) -> Result<(), ModelErr> {
                self.state_metadata.set_preference(v, less_is_better)
            }
        }
    };
}

impl_access_preference!(ElementResourceVariable);
impl_access_preference!(IntegerResourceVariable);
impl_access_preference!(ContinuousResourceVariable);

macro_rules! impl_table_interface {
    ($T:ty) => {
        impl TableInterface<$T> for Model {
            #[inline]
            fn add_table_1d<U>(
                &mut self,
                name: U,
                v: Vec<$T>,
            ) -> Result<Table1DHandle<$T>, ModelErr>
            where
                String: From<U>,
            {
                self.table_registry.add_table_1d(name, v)
            }

            #[inline]
            fn set_table_1d(
                &mut self,
                t: Table1DHandle<$T>,
                x: Element,
                v: $T,
            ) -> Result<(), ModelErr> {
                self.table_registry.set_table_1d(t, x, v)
            }

            #[inline]
            fn update_table_1d(
                &mut self,
                t: Table1DHandle<$T>,
                v: Vec<$T>,
            ) -> Result<(), ModelErr> {
                self.table_registry.update_table_1d(t, v)
            }

            #[inline]
            fn add_table_2d<U>(
                &mut self,
                name: U,
                v: Vec<Vec<$T>>,
            ) -> Result<Table2DHandle<$T>, ModelErr>
            where
                String: From<U>,
            {
                self.table_registry.add_table_2d(name, v)
            }

            #[inline]
            fn set_table_2d(
                &mut self,
                t: Table2DHandle<$T>,
                x: Element,
                y: Element,
                v: $T,
            ) -> Result<(), ModelErr> {
                self.table_registry.set_table_2d(t, x, y, v)
            }

            #[inline]
            fn update_table_2d(
                &mut self,
                t: Table2DHandle<$T>,
                v: Vec<Vec<$T>>,
            ) -> Result<(), ModelErr> {
                self.table_registry.update_table_2d(t, v)
            }

            #[inline]
            fn add_table_3d<U>(
                &mut self,
                name: U,
                v: Vec<Vec<Vec<$T>>>,
            ) -> Result<Table3DHandle<$T>, ModelErr>
            where
                String: From<U>,
            {
                self.table_registry.add_table_3d(name, v)
            }

            #[inline]
            fn set_table_3d(
                &mut self,
                t: Table3DHandle<$T>,
                x: Element,
                y: Element,
                z: Element,
                v: $T,
            ) -> Result<(), ModelErr> {
                self.table_registry.set_table_3d(t, x, y, z, v)
            }

            #[inline]
            fn update_table_3d(
                &mut self,
                t: Table3DHandle<$T>,
                v: Vec<Vec<Vec<$T>>>,
            ) -> Result<(), ModelErr> {
                self.table_registry.update_table_3d(t, v)
            }

            #[inline]
            fn add_table<U>(
                &mut self,
                name: U,
                map: FxHashMap<Vec<Element>, $T>,
                default: $T,
            ) -> Result<TableHandle<$T>, ModelErr>
            where
                String: From<U>,
            {
                self.table_registry.add_table(name, map, default)
            }

            #[inline]
            fn set_table(
                &mut self,
                t: TableHandle<$T>,
                key: Vec<Element>,
                v: $T,
            ) -> Result<(), ModelErr> {
                self.table_registry.set_table(t, key, v)
            }

            #[inline]
            fn set_default(&mut self, t: TableHandle<$T>, default: $T) -> Result<(), ModelErr> {
                self.table_registry.set_default(t, default)
            }

            #[inline]
            fn update_table(
                &mut self,
                t: TableHandle<$T>,
                map: FxHashMap<Vec<Element>, $T>,
                default: $T,
            ) -> Result<(), ModelErr> {
                self.table_registry.update_table(t, map, default)
            }
        }
    };
}

impl_table_interface!(Element);
impl_table_interface!(Set);
impl_table_interface!(Vector);
impl_table_interface!(Integer);
impl_table_interface!(Continuous);
impl_table_interface!(bool);

/// A trait for adding a dual bound.
pub trait AddDualBound<T> {
    /// Adds a dual bound.
    ///
    /// # Errors
    ///
    /// if the expression is invalid, e.., it uses not existing variables or the state of the transitioned state.
    fn add_dual_bound(&mut self, bound: T) -> Result<(), ModelErr>;
}

impl AddDualBound<expression::IntegerExpression> for Model {
    fn add_dual_bound(&mut self, bound: expression::IntegerExpression) -> Result<(), ModelErr> {
        self.check_expression(&bound, false)?;
        self.dual_bounds.push(CostExpression::Integer(
            bound.simplify(&self.table_registry),
        ));
        Ok(())
    }
}

impl AddDualBound<expression::ContinuousExpression> for Model {
    fn add_dual_bound(&mut self, bound: expression::ContinuousExpression) -> Result<(), ModelErr> {
        if self.cost_type == CostType::Integer {
            Err(ModelErr::new(String::from(
                "Could not add a dual bound with a continuous cost expression for a integer cost model"
            )))
        } else {
            self.check_expression(&bound, false)?;
            self.dual_bounds.push(CostExpression::Continuous(
                bound.simplify(&self.table_registry),
            ));
            Ok(())
        }
    }
}

/// A trait for checking if an expression is valid.
pub trait CheckExpression<T> {
    /// Checks if an expression is valid.
    ///
    /// # Errors
    ///
    /// if the expression is invalid, e.., it uses not existing variables or the state of the transitioned state.
    fn check_expression(&self, expression: &T, allow_cost: bool) -> Result<(), ModelErr>;
}

macro_rules! impl_check_table_expression {
    ($T:ty,$x:ident) => {
        impl CheckExpression<expression::TableExpression<$T>> for Model {
            fn check_expression(
                &self,
                expression: &expression::TableExpression<$T>,
                allow_cost: bool,
            ) -> Result<(), ModelErr> {
                match expression {
                    expression::TableExpression::Constant(_) => Ok(()),
                    expression::TableExpression::Table1D(id, x) => {
                        self.table_registry.$x.check_table_1d(*id)?;
                        self.check_expression(x, allow_cost)
                    }
                    expression::TableExpression::Table2D(id, x, y) => {
                        self.table_registry.$x.check_table_2d(*id)?;
                        self.check_expression(x, allow_cost)?;
                        self.check_expression(y, allow_cost)
                    }
                    expression::TableExpression::Table3D(id, x, y, z) => {
                        self.table_registry.$x.check_table_3d(*id)?;
                        self.check_expression(x, allow_cost)?;
                        self.check_expression(y, allow_cost)?;
                        self.check_expression(z, allow_cost)
                    }
                    expression::TableExpression::Table(id, args) => {
                        self.table_registry.$x.check_table(*id)?;
                        for expression in args {
                            self.check_expression(expression, allow_cost)?;
                        }
                        Ok(())
                    }
                }
            }
        }
    };
}

impl_check_table_expression!(Element, element_tables);
impl_check_table_expression!(Set, set_tables);
impl_check_table_expression!(Vector, vector_tables);
impl_check_table_expression!(bool, bool_tables);

impl CheckExpression<expression::ArgumentExpression> for Model {
    fn check_expression(
        &self,
        expression: &expression::ArgumentExpression,
        allow_cost: bool,
    ) -> Result<(), ModelErr> {
        match expression {
            expression::ArgumentExpression::Set(set) => self.check_expression(set, allow_cost),
            expression::ArgumentExpression::Vector(vector) => {
                self.check_expression(vector, allow_cost)
            }
            expression::ArgumentExpression::Element(element) => {
                self.check_expression(element, allow_cost)
            }
        }
    }
}

macro_rules! impl_check_numeric_table_expression {
    ($T:ty,$x:ident) => {
        impl CheckExpression<expression::NumericTableExpression<$T>> for Model {
            fn check_expression(
                &self,
                expression: &expression::NumericTableExpression<$T>,
                allow_cost: bool,
            ) -> Result<(), ModelErr> {
                match expression {
                    expression::NumericTableExpression::Constant(_) => Ok(()),
                    expression::NumericTableExpression::Table(id, args) => {
                        self.table_registry.$x.check_table(*id)?;
                        for expression in args {
                            self.check_expression(expression, allow_cost)?;
                        }
                        Ok(())
                    }
                    expression::NumericTableExpression::TableReduce(_, id, args) => {
                        self.table_registry.$x.check_table(*id)?;
                        for expression in args {
                            self.check_expression(expression, allow_cost)?;
                        }
                        Ok(())
                    }
                    expression::NumericTableExpression::Table1D(id, x) => {
                        self.table_registry.$x.check_table_1d(*id)?;
                        self.check_expression(x, allow_cost)
                    }
                    expression::NumericTableExpression::Table2D(id, x, y) => {
                        self.table_registry.$x.check_table_2d(*id)?;
                        self.check_expression(x, allow_cost)?;
                        self.check_expression(y, allow_cost)
                    }
                    expression::NumericTableExpression::Table3D(id, x, y, z) => {
                        self.table_registry.$x.check_table_2d(*id)?;
                        self.check_expression(x, allow_cost)?;
                        self.check_expression(y, allow_cost)?;
                        self.check_expression(z, allow_cost)
                    }
                    expression::NumericTableExpression::Table1DReduce(_, id, x) => {
                        self.table_registry.$x.check_table_1d(*id)?;
                        self.check_expression(x, allow_cost)
                    }
                    expression::NumericTableExpression::Table1DVectorReduce(_, id, x) => {
                        self.table_registry.$x.check_table_1d(*id)?;
                        self.check_expression(x, allow_cost)
                    }
                    expression::NumericTableExpression::Table2DReduce(_, id, x, y) => {
                        self.table_registry.$x.check_table_2d(*id)?;
                        self.check_expression(x, allow_cost)?;
                        self.check_expression(y, allow_cost)
                    }
                    expression::NumericTableExpression::Table2DVectorReduce(_, id, x, y) => {
                        self.table_registry.$x.check_table_2d(*id)?;
                        self.check_expression(x, allow_cost)?;
                        self.check_expression(y, allow_cost)
                    }
                    expression::NumericTableExpression::Table2DSetVectorReduce(_, id, x, y) => {
                        self.table_registry.$x.check_table_2d(*id)?;
                        self.check_expression(x, allow_cost)?;
                        self.check_expression(y, allow_cost)
                    }
                    expression::NumericTableExpression::Table2DVectorSetReduce(_, id, x, y) => {
                        self.table_registry.$x.check_table_2d(*id)?;
                        self.check_expression(x, allow_cost)?;
                        self.check_expression(y, allow_cost)
                    }
                    expression::NumericTableExpression::Table2DReduceX(_, id, x, y) => {
                        self.table_registry.$x.check_table_2d(*id)?;
                        self.check_expression(x, allow_cost)?;
                        self.check_expression(y, allow_cost)
                    }
                    expression::NumericTableExpression::Table2DReduceY(_, id, x, y) => {
                        self.table_registry.$x.check_table_2d(*id)?;
                        self.check_expression(x, allow_cost)?;
                        self.check_expression(y, allow_cost)
                    }
                    expression::NumericTableExpression::Table2DVectorReduceX(_, id, x, y) => {
                        self.table_registry.$x.check_table_2d(*id)?;
                        self.check_expression(x, allow_cost)?;
                        self.check_expression(y, allow_cost)
                    }
                    expression::NumericTableExpression::Table2DVectorReduceY(_, id, x, y) => {
                        self.table_registry.$x.check_table_2d(*id)?;
                        self.check_expression(x, allow_cost)?;
                        self.check_expression(y, allow_cost)
                    }
                    expression::NumericTableExpression::Table3DReduce(_, id, x, y, z) => {
                        self.table_registry.$x.check_table_2d(*id)?;
                        self.check_expression(x, allow_cost)?;
                        self.check_expression(y, allow_cost)?;
                        self.check_expression(z, allow_cost)
                    }
                }
            }
        }
    };
}

impl_check_numeric_table_expression!(Integer, integer_tables);
impl_check_numeric_table_expression!(Continuous, continuous_tables);

impl CheckExpression<expression::VectorOrElementExpression> for Model {
    fn check_expression(
        &self,
        expression: &expression::VectorOrElementExpression,
        allow_cost: bool,
    ) -> Result<(), ModelErr> {
        match expression {
            expression::VectorOrElementExpression::Vector(vector) => {
                self.check_expression(vector, allow_cost)
            }
            expression::VectorOrElementExpression::Element(element) => {
                self.check_expression(element, allow_cost)
            }
        }
    }
}

macro_rules! impl_check_table_vector_expression {
    ($T:ty,$x:ident) => {
        impl CheckExpression<expression::TableVectorExpression<$T>> for Model {
            fn check_expression(
                &self,
                expression: &expression::TableVectorExpression<$T>,
                allow_cost: bool,
            ) -> Result<(), ModelErr> {
                match expression {
                    expression::TableVectorExpression::Constant(_) => Ok(()),
                    expression::TableVectorExpression::Table(id, args) => {
                        self.table_registry.$x.check_table(*id)?;
                        for expression in args {
                            self.check_expression(expression, allow_cost)?;
                        }
                        Ok(())
                    }
                    expression::TableVectorExpression::TableReduce(_, id, args) => {
                        self.table_registry.$x.check_table(*id)?;
                        for expression in args {
                            self.check_expression(expression, allow_cost)?;
                        }
                        Ok(())
                    }
                    expression::TableVectorExpression::Table1D(id, x) => {
                        self.table_registry.$x.check_table_1d(*id)?;
                        self.check_expression(x, allow_cost)
                    }
                    expression::TableVectorExpression::Table2D(id, x, y) => {
                        self.table_registry.$x.check_table_2d(*id)?;
                        self.check_expression(x, allow_cost)?;
                        self.check_expression(y, allow_cost)
                    }
                    expression::TableVectorExpression::Table2DX(id, x, y) => {
                        self.table_registry.$x.check_table_2d(*id)?;
                        self.check_expression(x, allow_cost)?;
                        self.check_expression(y, allow_cost)
                    }
                    expression::TableVectorExpression::Table2DY(id, x, y) => {
                        self.table_registry.$x.check_table_2d(*id)?;
                        self.check_expression(x, allow_cost)?;
                        self.check_expression(y, allow_cost)
                    }
                    expression::TableVectorExpression::Table3D(id, x, y, z) => {
                        self.table_registry.$x.check_table_3d(*id)?;
                        self.check_expression(x, allow_cost)?;
                        self.check_expression(y, allow_cost)?;
                        self.check_expression(z, allow_cost)
                    }
                    expression::TableVectorExpression::Table2DXReduce(_, id, x, y) => {
                        self.table_registry.$x.check_table_2d(*id)?;
                        self.check_expression(x, allow_cost)?;
                        self.check_expression(y, allow_cost)
                    }
                    expression::TableVectorExpression::Table2DYReduce(_, id, x, y) => {
                        self.table_registry.$x.check_table_2d(*id)?;
                        self.check_expression(x, allow_cost)?;
                        self.check_expression(y, allow_cost)
                    }
                    expression::TableVectorExpression::Table3DReduce(_, id, x, y, z) => {
                        self.table_registry.$x.check_table_3d(*id)?;
                        self.check_expression(x, allow_cost)?;
                        self.check_expression(y, allow_cost)?;
                        self.check_expression(z, allow_cost)
                    }
                }
            }
        }
    };
}

impl_check_table_vector_expression!(Integer, integer_tables);
impl_check_table_vector_expression!(Continuous, continuous_tables);

impl CheckExpression<expression::ElementExpression> for Model {
    fn check_expression(
        &self,
        expression: &expression::ElementExpression,
        allow_cost: bool,
    ) -> Result<(), ModelErr> {
        match expression {
            expression::ElementExpression::Constant(_) => Ok(()),
            expression::ElementExpression::Variable(id) => {
                let n = self.state_metadata.number_of_element_variables();
                if *id >= n {
                    Err(ModelErr::new(format!(
                        "element variable id {} >= #variables ({})",
                        *id, n
                    )))
                } else {
                    Ok(())
                }
            }
            expression::ElementExpression::ResourceVariable(id) => {
                let n = self.state_metadata.number_of_element_resource_variables();
                if *id >= n {
                    Err(ModelErr::new(format!(
                        "element resource variable id {} >= #variables ({})",
                        *id, n
                    )))
                } else {
                    Ok(())
                }
            }
            expression::ElementExpression::BinaryOperation(_, x, y) => {
                self.check_expression(x.as_ref(), allow_cost)?;
                self.check_expression(y.as_ref(), allow_cost)
            }
            expression::ElementExpression::Last(vector) => {
                self.check_expression(vector.as_ref(), allow_cost)
            }
            expression::ElementExpression::At(vector, element) => {
                self.check_expression(vector.as_ref(), allow_cost)?;
                self.check_expression(element.as_ref(), allow_cost)
            }
            expression::ElementExpression::Table(table) => {
                self.check_expression(table.as_ref(), allow_cost)
            }
            expression::ElementExpression::If(condition, x, y) => {
                self.check_expression(condition.as_ref(), allow_cost)?;
                self.check_expression(x.as_ref(), allow_cost)?;
                self.check_expression(y.as_ref(), allow_cost)
            }
        }
    }
}

impl CheckExpression<expression::SetExpression> for Model {
    fn check_expression(
        &self,
        expression: &expression::SetExpression,
        allow_cost: bool,
    ) -> Result<(), ModelErr> {
        match expression {
            expression::SetExpression::Reference(expression::ReferenceExpression::Constant(_)) => {
                Ok(())
            }
            expression::SetExpression::Reference(expression::ReferenceExpression::Variable(id)) => {
                let n = self.state_metadata.number_of_set_variables();
                if *id >= n {
                    Err(ModelErr::new(format!(
                        "set variable id {} >= #variables ({})",
                        *id, n
                    )))
                } else {
                    Ok(())
                }
            }
            expression::SetExpression::Reference(expression::ReferenceExpression::Table(table)) => {
                self.check_expression(table, allow_cost)
            }
            expression::SetExpression::Complement(expression) => {
                self.check_expression(expression.as_ref(), allow_cost)
            }
            expression::SetExpression::SetOperation(_, x, y) => {
                self.check_expression(x.as_ref(), allow_cost)?;
                self.check_expression(y.as_ref(), allow_cost)
            }
            expression::SetExpression::SetElementOperation(_, x, y) => {
                self.check_expression(x, allow_cost)?;
                self.check_expression(y.as_ref(), allow_cost)
            }
            expression::SetExpression::FromVector(_, vector) => {
                self.check_expression(vector.as_ref(), allow_cost)
            }
            expression::SetExpression::If(condition, x, y) => {
                self.check_expression(condition.as_ref(), allow_cost)?;
                self.check_expression(x.as_ref(), allow_cost)?;
                self.check_expression(y.as_ref(), allow_cost)
            }
        }
    }
}

impl CheckExpression<expression::VectorExpression> for Model {
    fn check_expression(
        &self,
        expression: &expression::VectorExpression,
        allow_cost: bool,
    ) -> Result<(), ModelErr> {
        match expression {
            expression::VectorExpression::Reference(expression::ReferenceExpression::Constant(
                _,
            )) => Ok(()),
            expression::VectorExpression::Reference(expression::ReferenceExpression::Variable(
                id,
            )) => {
                let n = self.state_metadata.number_of_vector_variables();
                if *id >= n {
                    Err(ModelErr::new(format!(
                        "vector variable id {} >= #varaibles ({})",
                        *id, n
                    )))
                } else {
                    Ok(())
                }
            }
            expression::VectorExpression::Reference(expression::ReferenceExpression::Table(
                table,
            )) => self.check_expression(table, allow_cost),
            expression::VectorExpression::Indices(expression)
            | expression::VectorExpression::Reverse(expression)
            | expression::VectorExpression::Pop(expression) => {
                self.check_expression(expression.as_ref(), allow_cost)
            }
            expression::VectorExpression::Set(element, vector, i) => {
                self.check_expression(element, allow_cost)?;
                self.check_expression(vector.as_ref(), allow_cost)?;
                self.check_expression(i, allow_cost)
            }
            expression::VectorExpression::Push(element, vector) => {
                self.check_expression(element, allow_cost)?;
                self.check_expression(vector.as_ref(), allow_cost)
            }
            expression::VectorExpression::FromSet(set) => {
                self.check_expression(set.as_ref(), allow_cost)
            }
            expression::VectorExpression::If(condition, x, y) => {
                self.check_expression(condition.as_ref(), allow_cost)?;
                self.check_expression(x.as_ref(), allow_cost)?;
                self.check_expression(y.as_ref(), allow_cost)
            }
        }
    }
}

impl CheckExpression<expression::IntegerExpression> for Model {
    fn check_expression(
        &self,
        expression: &expression::IntegerExpression,
        allow_cost: bool,
    ) -> Result<(), ModelErr> {
        match expression {
            expression::IntegerExpression::Constant(_) => Ok(()),
            expression::IntegerExpression::Variable(id) => {
                let n = self.state_metadata.number_of_integer_variables();
                if *id >= n {
                    Err(ModelErr::new(format!(
                        "integer variable id {} >= #variables ({})",
                        *id, n
                    )))
                } else {
                    Ok(())
                }
            }
            expression::IntegerExpression::ResourceVariable(id) => {
                let n = self.state_metadata.number_of_integer_resource_variables();
                if *id >= n {
                    Err(ModelErr::new(format!(
                        "integer resource variable id {} >= #variables ({})",
                        *id, n
                    )))
                } else {
                    Ok(())
                }
            }
            expression::IntegerExpression::Cost => {
                if allow_cost {
                    if self.cost_type == CostType::Integer {
                        Ok(())
                    } else {
                        Err(ModelErr::new(format!(
                            "using cost is not allowed in an integer expression as the cost type is {:?}. Please explicitly cast it to an integer expression.",
                            self.cost_type,
                        )))
                    }
                } else {
                    Err(ModelErr::new(format!(
                        "using cost is not allowed in integer expression `{:?}`",
                        expression,
                    )))
                }
            }
            expression::IntegerExpression::UnaryOperation(_, x) => {
                self.check_expression(x.as_ref(), allow_cost)
            }
            expression::IntegerExpression::BinaryOperation(_, x, y) => {
                self.check_expression(x.as_ref(), allow_cost)?;
                self.check_expression(y.as_ref(), allow_cost)
            }
            expression::IntegerExpression::Cardinality(set) => {
                self.check_expression(set, allow_cost)
            }
            expression::IntegerExpression::Length(vector) => {
                self.check_expression(vector, allow_cost)
            }
            expression::IntegerExpression::Table(table) => {
                self.check_expression(table.as_ref(), allow_cost)
            }
            expression::IntegerExpression::If(condition, x, y) => {
                self.check_expression(condition.as_ref(), allow_cost)?;
                self.check_expression(x.as_ref(), allow_cost)?;
                self.check_expression(y.as_ref(), allow_cost)
            }
            expression::IntegerExpression::FromContinuous(_, continuous) => {
                self.check_expression(continuous.as_ref(), allow_cost)
            }
            expression::IntegerExpression::Last(vector)
            | expression::IntegerExpression::Reduce(_, vector) => {
                self.check_expression(vector.as_ref(), allow_cost)
            }
            expression::IntegerExpression::At(vector, element) => {
                self.check_expression(vector.as_ref(), allow_cost)?;
                self.check_expression(element, allow_cost)
            }
        }
    }
}

impl CheckExpression<expression::ContinuousExpression> for Model {
    fn check_expression(
        &self,
        expression: &expression::ContinuousExpression,
        allow_cost: bool,
    ) -> Result<(), ModelErr> {
        match expression {
            expression::ContinuousExpression::Constant(_) => Ok(()),
            expression::ContinuousExpression::Variable(id) => {
                let n = self.state_metadata.number_of_continuous_variables();
                if *id >= n {
                    Err(ModelErr::new(format!(
                        "continuous variable id {} >= #variables ({})",
                        *id, n
                    )))
                } else {
                    Ok(())
                }
            }
            expression::ContinuousExpression::ResourceVariable(id) => {
                let n = self
                    .state_metadata
                    .number_of_continuous_resource_variables();
                if *id >= n {
                    Err(ModelErr::new(format!(
                        "continuous resource variable id {} >= #variables ({})",
                        *id, n
                    )))
                } else {
                    Ok(())
                }
            }
            expression::ContinuousExpression::Cost => {
                if allow_cost {
                    Ok(())
                } else {
                    Err(ModelErr::new(format!(
                        "Using cost is not allowed in continuous expression `{:?}`",
                        expression,
                    )))
                }
            }
            expression::ContinuousExpression::UnaryOperation(_, x)
            | expression::ContinuousExpression::ContinuousUnaryOperation(_, x)
            | expression::ContinuousExpression::Round(_, x) => {
                self.check_expression(x.as_ref(), allow_cost)
            }
            expression::ContinuousExpression::BinaryOperation(_, x, y)
            | expression::ContinuousExpression::ContinuousBinaryOperation(_, x, y) => {
                self.check_expression(x.as_ref(), allow_cost)?;
                self.check_expression(y.as_ref(), allow_cost)
            }
            expression::ContinuousExpression::Cardinality(set) => {
                self.check_expression(set, allow_cost)
            }
            expression::ContinuousExpression::Length(vector) => {
                self.check_expression(vector, allow_cost)
            }
            expression::ContinuousExpression::Table(table) => {
                self.check_expression(table.as_ref(), allow_cost)
            }
            expression::ContinuousExpression::If(condition, x, y) => {
                self.check_expression(condition.as_ref(), allow_cost)?;
                self.check_expression(x.as_ref(), allow_cost)?;
                self.check_expression(y.as_ref(), allow_cost)
            }
            expression::ContinuousExpression::FromInteger(integer) => {
                self.check_expression(integer.as_ref(), allow_cost)
            }
            expression::ContinuousExpression::Last(vector)
            | expression::ContinuousExpression::Reduce(_, vector) => {
                self.check_expression(vector.as_ref(), allow_cost)
            }
            expression::ContinuousExpression::At(vector, element) => {
                self.check_expression(vector.as_ref(), allow_cost)?;
                self.check_expression(element, allow_cost)
            }
        }
    }
}

impl CheckExpression<expression::IntegerVectorExpression> for Model {
    fn check_expression(
        &self,
        expression: &expression::IntegerVectorExpression,
        allow_cost: bool,
    ) -> Result<(), ModelErr> {
        match expression {
            expression::IntegerVectorExpression::Constant(_) => Ok(()),
            expression::IntegerVectorExpression::Reverse(expression)
            | expression::IntegerVectorExpression::Pop(expression)
            | expression::IntegerVectorExpression::UnaryOperation(_, expression) => {
                self.check_expression(expression.as_ref(), allow_cost)
            }
            expression::IntegerVectorExpression::Push(integer, vector)
            | expression::IntegerVectorExpression::BinaryOperationX(_, integer, vector)
            | expression::IntegerVectorExpression::BinaryOperationY(_, vector, integer) => {
                self.check_expression(integer, allow_cost)?;
                self.check_expression(vector.as_ref(), allow_cost)
            }
            expression::IntegerVectorExpression::Set(integer, vector, i) => {
                self.check_expression(integer, allow_cost)?;
                self.check_expression(vector.as_ref(), allow_cost)?;
                self.check_expression(i, allow_cost)
            }
            expression::IntegerVectorExpression::VectorOperation(_, x, y) => {
                self.check_expression(x.as_ref(), allow_cost)?;
                self.check_expression(y.as_ref(), allow_cost)
            }
            expression::IntegerVectorExpression::Table(table) => {
                self.check_expression(table.as_ref(), allow_cost)
            }
            expression::IntegerVectorExpression::If(condition, x, y) => {
                self.check_expression(condition.as_ref(), allow_cost)?;
                self.check_expression(x.as_ref(), allow_cost)?;
                self.check_expression(y.as_ref(), allow_cost)
            }
            expression::IntegerVectorExpression::FromContinuous(_, continuous) => {
                self.check_expression(continuous.as_ref(), allow_cost)
            }
        }
    }
}

impl CheckExpression<expression::ContinuousVectorExpression> for Model {
    fn check_expression(
        &self,
        expression: &expression::ContinuousVectorExpression,
        allow_cost: bool,
    ) -> Result<(), ModelErr> {
        match expression {
            expression::ContinuousVectorExpression::Constant(_) => Ok(()),
            expression::ContinuousVectorExpression::Reverse(expression)
            | expression::ContinuousVectorExpression::Pop(expression)
            | expression::ContinuousVectorExpression::UnaryOperation(_, expression)
            | expression::ContinuousVectorExpression::ContinuousUnaryOperation(_, expression)
            | expression::ContinuousVectorExpression::Round(_, expression) => {
                self.check_expression(expression.as_ref(), allow_cost)
            }
            expression::ContinuousVectorExpression::Push(continuous, vector)
            | expression::ContinuousVectorExpression::BinaryOperationX(_, continuous, vector)
            | expression::ContinuousVectorExpression::BinaryOperationY(_, vector, continuous)
            | expression::ContinuousVectorExpression::ContinuousBinaryOperationX(
                _,
                continuous,
                vector,
            )
            | expression::ContinuousVectorExpression::ContinuousBinaryOperationY(
                _,
                vector,
                continuous,
            ) => {
                self.check_expression(continuous, allow_cost)?;
                self.check_expression(vector.as_ref(), allow_cost)
            }
            expression::ContinuousVectorExpression::Set(continuous, vector, i) => {
                self.check_expression(continuous, allow_cost)?;
                self.check_expression(vector.as_ref(), allow_cost)?;
                self.check_expression(i, allow_cost)
            }
            expression::ContinuousVectorExpression::VectorOperation(_, x, y)
            | expression::ContinuousVectorExpression::ContinuousVectorOperation(_, x, y) => {
                self.check_expression(x.as_ref(), allow_cost)?;
                self.check_expression(y.as_ref(), allow_cost)
            }
            expression::ContinuousVectorExpression::Table(table) => {
                self.check_expression(table.as_ref(), allow_cost)
            }
            expression::ContinuousVectorExpression::If(condition, x, y) => {
                self.check_expression(condition.as_ref(), allow_cost)?;
                self.check_expression(x.as_ref(), allow_cost)?;
                self.check_expression(y.as_ref(), allow_cost)
            }
            expression::ContinuousVectorExpression::FromInteger(integer) => {
                self.check_expression(integer.as_ref(), allow_cost)
            }
        }
    }
}

impl CheckExpression<expression::Condition> for Model {
    fn check_expression(
        &self,
        condition: &expression::Condition,
        allow_cost: bool,
    ) -> Result<(), ModelErr> {
        match condition {
            expression::Condition::Constant(_) => Ok(()),
            expression::Condition::Not(condition) => {
                self.check_expression(condition.as_ref(), allow_cost)
            }
            expression::Condition::And(x, y) | expression::Condition::Or(x, y) => {
                self.check_expression(x.as_ref(), allow_cost)?;
                self.check_expression(y.as_ref(), allow_cost)
            }
            expression::Condition::ComparisonE(_, x, y) => {
                self.check_expression(x.as_ref(), allow_cost)?;
                self.check_expression(y.as_ref(), allow_cost)
            }
            expression::Condition::ComparisonI(_, x, y) => {
                self.check_expression(x.as_ref(), allow_cost)?;
                self.check_expression(y.as_ref(), allow_cost)
            }
            expression::Condition::ComparisonC(_, x, y) => {
                self.check_expression(x.as_ref(), allow_cost)?;
                self.check_expression(y.as_ref(), allow_cost)
            }
            expression::Condition::Set(condition) => match condition.as_ref() {
                expression::SetCondition::Constant(_) => Ok(()),
                expression::SetCondition::IsIn(element, set) => {
                    self.check_expression(element, allow_cost)?;
                    self.check_expression(set, allow_cost)
                }
                expression::SetCondition::IsSubset(x, y) => {
                    self.check_expression(x, allow_cost)?;
                    self.check_expression(y, allow_cost)
                }
                expression::SetCondition::IsEmpty(set) => self.check_expression(set, allow_cost),
            },
            expression::Condition::Table(table) => {
                self.check_expression(table.as_ref(), allow_cost)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use expression::*;

    #[test]
    fn cost_type_default() {
        let reduce = CostType::default();
        assert_eq!(reduce, CostType::Integer);
    }

    #[test]
    fn reduce_functin_default() {
        let reduce = ReduceFunction::default();
        assert_eq!(reduce, ReduceFunction::Min);
    }

    #[test]
    fn integer_cost_model() {
        let model = Model::integer_cost_model();
        assert_eq!(
            model,
            Model {
                cost_type: CostType::Integer,
                ..Default::default()
            }
        );
    }

    #[test]
    fn continuous_cost_model() {
        let model = Model::continuous_cost_model();
        assert_eq!(
            model,
            Model {
                cost_type: CostType::Continuous,
                ..Default::default()
            }
        );
    }

    #[test]
    fn check_constraints() {
        let state = state::State::default();
        let model = Model {
            state_constraints: vec![GroundedCondition {
                condition: Condition::Constant(true),
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(model.check_constraints(&state));
        let model = Model {
            state_constraints: vec![
                GroundedCondition {
                    condition: Condition::Constant(true),
                    ..Default::default()
                },
                GroundedCondition {
                    condition: Condition::Constant(false),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        assert!(!model.check_constraints(&state));
    }

    #[test]
    fn is_goal() {
        let state = state::State::default();
        let model = Model {
            base_cases: vec![BaseCase::new(vec![GroundedCondition {
                condition: Condition::Constant(true),
                ..Default::default()
            }])],
            ..Default::default()
        };
        assert!(model.is_base(&state));
        let model = Model {
            base_cases: vec![BaseCase::new(vec![GroundedCondition {
                condition: Condition::Constant(false),
                ..Default::default()
            }])],
            ..Default::default()
        };
        assert!(!model.is_base(&state));
        let model = Model {
            base_cases: vec![BaseCase::new(vec![GroundedCondition {
                condition: Condition::Constant(false),
                ..Default::default()
            }])],
            base_states: vec![state::State::default()],
            ..Default::default()
        };
        assert!(model.is_base(&state));
    }

    #[test]
    fn has_resource_variables() {
        let model = Model::default();
        assert!(!model.has_resource_variables());
        let model = Model {
            state_metadata: StateMetadata {
                integer_resource_variable_names: vec![String::from("v")],
                integer_less_is_better: vec![true],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.has_resource_variables());
    }

    #[test]
    fn has_dual_bounds() {
        let model = Model::default();
        assert!(!model.has_dual_bounds());
        let model = Model {
            dual_bounds: vec![CostExpression::Integer(IntegerExpression::Constant(0))],
            ..Default::default()
        };
        assert!(model.has_dual_bounds());
    }

    #[test]
    fn eval_dual_bound() {
        let state = State::default();

        let model = Model {
            reduce_function: ReduceFunction::Min,
            ..Default::default()
        };
        assert_eq!(model.eval_dual_bound::<_, Integer>(&state), None);

        let model = Model {
            reduce_function: ReduceFunction::Max,
            ..Default::default()
        };
        assert_eq!(model.eval_dual_bound::<_, Integer>(&state), None);

        let model = Model {
            reduce_function: ReduceFunction::Min,
            dual_bounds: vec![
                CostExpression::Integer(IntegerExpression::Constant(1)),
                CostExpression::Integer(IntegerExpression::Constant(2)),
            ],
            ..Default::default()
        };
        assert_eq!(model.eval_dual_bound::<_, Integer>(&state), Some(2));

        let model = Model {
            reduce_function: ReduceFunction::Max,
            dual_bounds: vec![
                CostExpression::Integer(IntegerExpression::Constant(1)),
                CostExpression::Integer(IntegerExpression::Constant(2)),
            ],
            ..Default::default()
        };
        assert_eq!(model.eval_dual_bound::<_, Integer>(&state), Some(1));

        let model = Model {
            reduce_function: ReduceFunction::Sum,
            dual_bounds: vec![
                CostExpression::Integer(IntegerExpression::Constant(1)),
                CostExpression::Integer(IntegerExpression::Constant(2)),
            ],
            ..Default::default()
        };
        assert_eq!(model.eval_dual_bound::<_, Integer>(&state), None);

        let model = Model {
            reduce_function: ReduceFunction::Product,
            dual_bounds: vec![
                CostExpression::Integer(IntegerExpression::Constant(1)),
                CostExpression::Integer(IntegerExpression::Constant(2)),
            ],
            ..Default::default()
        };
        assert_eq!(model.eval_dual_bound::<_, Integer>(&state), None);
    }

    #[test]
    fn validate_forward_true() {
        let name_to_integer_variable = FxHashMap::default();
        let model = Model {
            state_metadata: StateMetadata {
                integer_variable_names: vec![String::from("v1")],
                name_to_integer_variable,
                ..Default::default()
            },
            target: State {
                signature_variables: SignatureVariables {
                    integer_variables: vec![0],
                    ..Default::default()
                },
                ..Default::default()
            },
            state_constraints: vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Ge,
                    Box::new(IntegerExpression::Variable(0)),
                    Box::new(IntegerExpression::Constant(0)),
                ),
                ..Default::default()
            }],
            base_cases: vec![BaseCase::new(vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Ge,
                    Box::new(IntegerExpression::Variable(0)),
                    Box::new(IntegerExpression::Constant(2)),
                ),
                ..Default::default()
            }])],
            forward_transitions: vec![
                Transition {
                    name: String::from("increase"),
                    effect: Effect {
                        integer_effects: vec![(
                            0,
                            IntegerExpression::BinaryOperation(
                                BinaryOperator::Add,
                                Box::new(IntegerExpression::Variable(0)),
                                Box::new(IntegerExpression::Constant(1)),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                        BinaryOperator::Add,
                        Box::new(IntegerExpression::Cost),
                        Box::new(IntegerExpression::Constant(1)),
                    )),
                    ..Default::default()
                },
                Transition {
                    name: String::from("decrease"),
                    effect: Effect {
                        integer_effects: vec![(
                            0,
                            IntegerExpression::BinaryOperation(
                                BinaryOperator::Sub,
                                Box::new(IntegerExpression::Variable(0)),
                                Box::new(IntegerExpression::Constant(1)),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                        BinaryOperator::Add,
                        Box::new(IntegerExpression::Cost),
                        Box::new(IntegerExpression::Constant(1)),
                    )),
                    ..Default::default()
                },
            ],
            forward_forced_transitions: vec![Transition {
                name: String::from("forced increase"),
                preconditions: vec![GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Eq,
                        Box::new(IntegerExpression::Variable(0)),
                        Box::new(IntegerExpression::Constant(0)),
                    ),
                    ..Default::default()
                }],
                effect: Effect {
                    integer_effects: vec![(0, IntegerExpression::Constant(1))],
                    ..Default::default()
                },
                cost: CostExpression::Integer(IntegerExpression::Cost),
                ..Default::default()
            }],
            ..Default::default()
        };

        let transitions = vec![
            model.forward_forced_transitions[0].clone(),
            model.forward_transitions[0].clone(),
        ];
        assert!(model.validate_forward(&transitions, 1, true));
    }

    #[test]
    fn validate_forward_forced_false() {
        let name_to_integer_variable = FxHashMap::default();
        let model = Model {
            state_metadata: StateMetadata {
                integer_variable_names: vec![String::from("v1")],
                name_to_integer_variable,
                ..Default::default()
            },
            target: State {
                signature_variables: SignatureVariables {
                    integer_variables: vec![0],
                    ..Default::default()
                },
                ..Default::default()
            },
            state_constraints: vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Ge,
                    Box::new(IntegerExpression::Variable(0)),
                    Box::new(IntegerExpression::Constant(0)),
                ),
                ..Default::default()
            }],
            base_cases: vec![BaseCase::new(vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Ge,
                    Box::new(IntegerExpression::Variable(0)),
                    Box::new(IntegerExpression::Constant(2)),
                ),
                ..Default::default()
            }])],
            forward_transitions: vec![
                Transition {
                    name: String::from("increase"),
                    effect: Effect {
                        integer_effects: vec![(
                            0,
                            IntegerExpression::BinaryOperation(
                                BinaryOperator::Add,
                                Box::new(IntegerExpression::Variable(0)),
                                Box::new(IntegerExpression::Constant(1)),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                        BinaryOperator::Add,
                        Box::new(IntegerExpression::Cost),
                        Box::new(IntegerExpression::Constant(1)),
                    )),
                    ..Default::default()
                },
                Transition {
                    name: String::from("decrease"),
                    effect: Effect {
                        integer_effects: vec![(
                            0,
                            IntegerExpression::BinaryOperation(
                                BinaryOperator::Sub,
                                Box::new(IntegerExpression::Variable(0)),
                                Box::new(IntegerExpression::Constant(1)),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                        BinaryOperator::Add,
                        Box::new(IntegerExpression::Cost),
                        Box::new(IntegerExpression::Constant(1)),
                    )),
                    ..Default::default()
                },
            ],
            forward_forced_transitions: vec![Transition {
                name: String::from("forced increase"),
                preconditions: vec![GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Eq,
                        Box::new(IntegerExpression::Variable(0)),
                        Box::new(IntegerExpression::Constant(0)),
                    ),
                    ..Default::default()
                }],
                effect: Effect {
                    integer_effects: vec![(0, IntegerExpression::Constant(1))],
                    ..Default::default()
                },
                cost: CostExpression::Integer(IntegerExpression::Cost),
                ..Default::default()
            }],
            ..Default::default()
        };

        let transitions = vec![
            model.forward_transitions[0].clone(),
            model.forward_transitions[0].clone(),
        ];
        assert!(!model.validate_forward(&transitions, 2, true));
    }

    #[test]
    fn validate_forward_state_constraint_false() {
        let name_to_integer_variable = FxHashMap::default();
        let model = Model {
            state_metadata: StateMetadata {
                integer_variable_names: vec![String::from("v1")],
                name_to_integer_variable,
                ..Default::default()
            },
            target: State {
                signature_variables: SignatureVariables {
                    integer_variables: vec![0],
                    ..Default::default()
                },
                ..Default::default()
            },
            state_constraints: vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Ge,
                    Box::new(IntegerExpression::Variable(0)),
                    Box::new(IntegerExpression::Constant(0)),
                ),
                ..Default::default()
            }],
            base_cases: vec![BaseCase::new(vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Ge,
                    Box::new(IntegerExpression::Variable(0)),
                    Box::new(IntegerExpression::Constant(2)),
                ),
                ..Default::default()
            }])],
            forward_transitions: vec![
                Transition {
                    name: String::from("increase"),
                    effect: Effect {
                        integer_effects: vec![(
                            0,
                            IntegerExpression::BinaryOperation(
                                BinaryOperator::Add,
                                Box::new(IntegerExpression::Variable(0)),
                                Box::new(IntegerExpression::Constant(1)),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                        BinaryOperator::Add,
                        Box::new(IntegerExpression::Cost),
                        Box::new(IntegerExpression::Constant(1)),
                    )),
                    ..Default::default()
                },
                Transition {
                    name: String::from("decrease"),
                    effect: Effect {
                        integer_effects: vec![(
                            0,
                            IntegerExpression::BinaryOperation(
                                BinaryOperator::Sub,
                                Box::new(IntegerExpression::Variable(0)),
                                Box::new(IntegerExpression::Constant(1)),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                        BinaryOperator::Add,
                        Box::new(IntegerExpression::Cost),
                        Box::new(IntegerExpression::Constant(1)),
                    )),
                    ..Default::default()
                },
            ],
            forward_forced_transitions: vec![Transition {
                name: String::from("forced increase"),
                preconditions: vec![GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Eq,
                        Box::new(IntegerExpression::Variable(0)),
                        Box::new(IntegerExpression::Constant(0)),
                    ),
                    ..Default::default()
                }],
                effect: Effect {
                    integer_effects: vec![(0, IntegerExpression::Constant(1))],
                    ..Default::default()
                },
                cost: CostExpression::Integer(IntegerExpression::Cost),
                ..Default::default()
            }],
            ..Default::default()
        };

        let transitions = vec![
            model.forward_forced_transitions[0].clone(),
            model.forward_transitions[1].clone(),
            model.forward_transitions[0].clone(),
            model.forward_transitions[0].clone(),
        ];
        assert!(!model.validate_forward(&transitions, 3, true));
    }

    #[test]
    fn validate_forward_applicable_false() {
        let name_to_integer_variable = FxHashMap::default();
        let model = Model {
            state_metadata: StateMetadata {
                integer_variable_names: vec![String::from("v1")],
                name_to_integer_variable,
                ..Default::default()
            },
            target: State {
                signature_variables: SignatureVariables {
                    integer_variables: vec![0],
                    ..Default::default()
                },
                ..Default::default()
            },
            state_constraints: vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Ge,
                    Box::new(IntegerExpression::Variable(0)),
                    Box::new(IntegerExpression::Constant(0)),
                ),
                ..Default::default()
            }],
            base_cases: vec![BaseCase::new(vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Ge,
                    Box::new(IntegerExpression::Variable(0)),
                    Box::new(IntegerExpression::Constant(2)),
                ),
                ..Default::default()
            }])],
            forward_transitions: vec![
                Transition {
                    name: String::from("increase"),
                    effect: Effect {
                        integer_effects: vec![(
                            0,
                            IntegerExpression::BinaryOperation(
                                BinaryOperator::Add,
                                Box::new(IntegerExpression::Variable(0)),
                                Box::new(IntegerExpression::Constant(1)),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                        BinaryOperator::Add,
                        Box::new(IntegerExpression::Cost),
                        Box::new(IntegerExpression::Constant(1)),
                    )),
                    ..Default::default()
                },
                Transition {
                    name: String::from("decrease"),
                    effect: Effect {
                        integer_effects: vec![(
                            0,
                            IntegerExpression::BinaryOperation(
                                BinaryOperator::Sub,
                                Box::new(IntegerExpression::Variable(0)),
                                Box::new(IntegerExpression::Constant(1)),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                        BinaryOperator::Add,
                        Box::new(IntegerExpression::Cost),
                        Box::new(IntegerExpression::Constant(1)),
                    )),
                    ..Default::default()
                },
            ],
            forward_forced_transitions: vec![Transition {
                name: String::from("forced increase"),
                preconditions: vec![GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Eq,
                        Box::new(IntegerExpression::Variable(0)),
                        Box::new(IntegerExpression::Constant(0)),
                    ),
                    ..Default::default()
                }],
                effect: Effect {
                    integer_effects: vec![(0, IntegerExpression::Constant(1))],
                    ..Default::default()
                },
                cost: CostExpression::Integer(IntegerExpression::Cost),
                ..Default::default()
            }],
            ..Default::default()
        };

        let transitions = vec![
            model.forward_transitions[0].clone(),
            model.forward_forced_transitions[0].clone(),
        ];
        assert!(!model.validate_forward(&transitions, 1, true));
    }

    #[test]
    fn validate_forward_base_false() {
        let name_to_integer_variable = FxHashMap::default();
        let model = Model {
            state_metadata: StateMetadata {
                integer_variable_names: vec![String::from("v1")],
                name_to_integer_variable,
                ..Default::default()
            },
            target: State {
                signature_variables: SignatureVariables {
                    integer_variables: vec![0],
                    ..Default::default()
                },
                ..Default::default()
            },
            state_constraints: vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Ge,
                    Box::new(IntegerExpression::Variable(0)),
                    Box::new(IntegerExpression::Constant(0)),
                ),
                ..Default::default()
            }],
            base_cases: vec![BaseCase::new(vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Ge,
                    Box::new(IntegerExpression::Variable(0)),
                    Box::new(IntegerExpression::Constant(2)),
                ),
                ..Default::default()
            }])],
            forward_transitions: vec![
                Transition {
                    name: String::from("increase"),
                    effect: Effect {
                        integer_effects: vec![(
                            0,
                            IntegerExpression::BinaryOperation(
                                BinaryOperator::Add,
                                Box::new(IntegerExpression::Variable(0)),
                                Box::new(IntegerExpression::Constant(1)),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                        BinaryOperator::Add,
                        Box::new(IntegerExpression::Cost),
                        Box::new(IntegerExpression::Constant(1)),
                    )),
                    ..Default::default()
                },
                Transition {
                    name: String::from("decrease"),
                    effect: Effect {
                        integer_effects: vec![(
                            0,
                            IntegerExpression::BinaryOperation(
                                BinaryOperator::Sub,
                                Box::new(IntegerExpression::Variable(0)),
                                Box::new(IntegerExpression::Constant(1)),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                        BinaryOperator::Add,
                        Box::new(IntegerExpression::Cost),
                        Box::new(IntegerExpression::Constant(1)),
                    )),
                    ..Default::default()
                },
            ],
            forward_forced_transitions: vec![Transition {
                name: String::from("forced increase"),
                preconditions: vec![GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Eq,
                        Box::new(IntegerExpression::Variable(0)),
                        Box::new(IntegerExpression::Constant(0)),
                    ),
                    ..Default::default()
                }],
                effect: Effect {
                    integer_effects: vec![(0, IntegerExpression::Constant(1))],
                    ..Default::default()
                },
                cost: CostExpression::Integer(IntegerExpression::Cost),
                ..Default::default()
            }],
            ..Default::default()
        };

        let transitions = vec![model.forward_transitions[0].clone()];
        assert!(!model.validate_forward(&transitions, 0, true));
    }

    #[test]
    fn validate_forward_cost_false() {
        let name_to_integer_variable = FxHashMap::default();
        let model = Model {
            state_metadata: StateMetadata {
                integer_variable_names: vec![String::from("v1")],
                name_to_integer_variable,
                ..Default::default()
            },
            target: State {
                signature_variables: SignatureVariables {
                    integer_variables: vec![0],
                    ..Default::default()
                },
                ..Default::default()
            },
            state_constraints: vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Ge,
                    Box::new(IntegerExpression::Variable(0)),
                    Box::new(IntegerExpression::Constant(0)),
                ),
                ..Default::default()
            }],
            base_cases: vec![BaseCase::new(vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Ge,
                    Box::new(IntegerExpression::Variable(0)),
                    Box::new(IntegerExpression::Constant(2)),
                ),
                ..Default::default()
            }])],
            forward_transitions: vec![
                Transition {
                    name: String::from("increase"),
                    effect: Effect {
                        integer_effects: vec![(
                            0,
                            IntegerExpression::BinaryOperation(
                                BinaryOperator::Add,
                                Box::new(IntegerExpression::Variable(0)),
                                Box::new(IntegerExpression::Constant(1)),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                        BinaryOperator::Add,
                        Box::new(IntegerExpression::Cost),
                        Box::new(IntegerExpression::Constant(1)),
                    )),
                    ..Default::default()
                },
                Transition {
                    name: String::from("decrease"),
                    effect: Effect {
                        integer_effects: vec![(
                            0,
                            IntegerExpression::BinaryOperation(
                                BinaryOperator::Sub,
                                Box::new(IntegerExpression::Variable(0)),
                                Box::new(IntegerExpression::Constant(1)),
                            ),
                        )],
                        ..Default::default()
                    },
                    cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                        BinaryOperator::Add,
                        Box::new(IntegerExpression::Cost),
                        Box::new(IntegerExpression::Constant(1)),
                    )),
                    ..Default::default()
                },
            ],
            forward_forced_transitions: vec![Transition {
                name: String::from("forced increase"),
                preconditions: vec![GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Eq,
                        Box::new(IntegerExpression::Variable(0)),
                        Box::new(IntegerExpression::Constant(0)),
                    ),
                    ..Default::default()
                }],
                effect: Effect {
                    integer_effects: vec![(0, IntegerExpression::Constant(1))],
                    ..Default::default()
                },
                cost: CostExpression::Integer(IntegerExpression::Cost),
                ..Default::default()
            }],
            ..Default::default()
        };

        let transitions = vec![
            model.forward_forced_transitions[0].clone(),
            model.forward_transitions[0].clone(),
        ];
        assert!(!model.validate_forward(&transitions, 2, true));
    }

    #[test]
    fn object_ok() {
        let mut model = Model::default();
        let ob = model.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let ob2 = model.get_object_type("something");
        assert!(ob2.is_ok());
        let ob2 = ob2.unwrap();
        assert_eq!(ob, ob2);
        let n = model.get_number_of_objects(ob);
        assert!(n.is_ok());
        assert_eq!(n.unwrap(), 10);
        let set = model.create_set(ob, &[1, 2, 4]);
        assert!(set.is_ok());
        let mut expected = Set::with_capacity(10);
        expected.insert(1);
        expected.insert(2);
        expected.insert(4);
        assert_eq!(set.unwrap(), expected);
        assert!(model.set_number_of_object(ob, 11).is_ok());
        let n = model.get_number_of_objects(ob);
        assert!(n.is_ok());
        assert_eq!(n.unwrap(), 11);
    }

    #[test]
    fn object_err() {
        let mut model = Model::default();
        let ob = model.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let set = model.create_set(ob, &[10]);
        assert!(set.is_err());
        let ob2 = model.add_object_type(String::from("something"), 10);
        assert!(ob2.is_err());

        let mut model = Model::default();
        let n = model.get_number_of_objects(ob);
        assert!(n.is_err());
        let set = model.create_set(ob, &[1, 2, 4]);
        assert!(set.is_err());
        let result = model.set_number_of_object(ob, 11);
        assert!(result.is_err());
        let ob = model.get_object_type("something");
        assert!(ob.is_err());
    }

    #[test]
    fn element_variable_ok() {
        let mut model = Model::default();
        let ob = model.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable(String::from("v"), ob, 2);
        assert!(v.is_ok());
        let v = model.add_element_variable(String::from("v2"), ob, 2);
        assert!(v.is_ok());
        let v = v.unwrap();
        let ob2 = model.get_object_type_of(v);
        assert!(ob2.is_ok());
        assert_eq!(ob2.unwrap(), ob);
        let target = model.get_target(v);
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), 2);
        let result = model.set_target(v, 3);
        assert!(result.is_ok());
        let target = model.get_target(v);
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), 3);
    }

    #[test]
    fn element_variable_err() {
        let mut model = Model::default();
        let ob = model.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable(String::from("v"), ob, 2);
        assert!(v.is_ok());
        let v = v.unwrap();
        let v2 = model.add_element_variable(String::from("v"), ob, 2);
        assert!(v2.is_err());
        let v2 = model.add_element_variable(String::from("v2"), ob, 10);
        assert!(v2.is_err());
        let result = model.set_target(v, 10);
        assert!(result.is_err());

        let mut model = Model::default();
        let v2 = model.add_element_variable(String::from("v3"), ob, 2);
        assert!(v2.is_err());
        let ob2 = model.get_object_type_of(v);
        assert!(ob2.is_err());
        let target = model.get_target(v);
        assert!(target.is_err());
        let result = model.set_target(v, 5);
        assert!(result.is_err());
    }

    #[test]
    fn element_resource_variable_ok() {
        let mut model = Model::default();
        let ob = model.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable(String::from("v"), ob, true, 2);
        assert!(v.is_ok());
        let v = model.add_element_resource_variable(String::from("v2"), ob, true, 2);
        assert!(v.is_ok());
        let v = v.unwrap();
        let ob2 = model.get_object_type_of(v);
        assert!(ob2.is_ok());
        assert_eq!(ob2.unwrap(), ob);
        let target = model.get_target(v);
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), 2);
        let result = model.set_target(v, 3);
        assert!(result.is_ok());
        let target = model.get_target(v);
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), 3);
        let preference = model.get_preference(v);
        assert!(preference.is_ok());
        assert!(preference.unwrap());
        let result = model.set_preference(v, false);
        assert!(result.is_ok());
        let preference = model.get_preference(v);
        assert!(preference.is_ok());
        assert!(!preference.unwrap());
    }

    #[test]
    fn element_resource_variable_err() {
        let mut model = Model::default();
        let ob = model.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable(String::from("v"), ob, true, 2);
        assert!(v.is_ok());
        let v = v.unwrap();
        let v2 = model.add_element_resource_variable(String::from("v"), ob, true, 2);
        assert!(v2.is_err());
        let v2 = model.add_element_resource_variable(String::from("v2"), ob, true, 10);
        assert!(v2.is_err());
        let result = model.set_target(v, 10);
        assert!(result.is_err());

        let mut model = Model::default();
        let v2 = model.add_element_resource_variable(String::from("v3"), ob, true, 2);
        assert!(v2.is_err());
        let ob2 = model.get_object_type_of(v);
        assert!(ob2.is_err());
        let target = model.get_target(v);
        assert!(target.is_err());
        let result = model.set_target(v, 5);
        assert!(result.is_err());
        let preference = model.get_preference(v);
        assert!(preference.is_err());
        let result = model.set_preference(v, false);
        assert!(result.is_err());
    }

    #[test]
    fn set_variable_ok() {
        let mut model = Model::default();
        let ob = model.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let target = model.create_set(ob, &[1, 2, 4]);
        assert!(target.is_ok());
        let target = target.unwrap();
        let v = model.add_set_variable(String::from("v"), ob, target.clone());
        assert!(v.is_ok());
        let v = model.add_set_variable(String::from("v2"), ob, target.clone());
        assert!(v.is_ok());
        let v = v.unwrap();
        let ob2 = model.get_object_type_of(v);
        assert!(ob2.is_ok());
        assert_eq!(ob2.unwrap(), ob);
        let target2 = model.get_target(v);
        assert!(target2.is_ok());
        assert_eq!(target2.unwrap(), target);
        let target = model.create_set(ob, &[2, 3, 5]);
        assert!(target.is_ok());
        let target = target.unwrap();
        let result = model.set_target(v, target.clone());
        assert!(result.is_ok());
        let target2 = model.get_target(v);
        assert!(target2.is_ok());
        assert_eq!(target2.unwrap(), target);
    }

    #[test]
    fn set_variable_err() {
        let mut model = Model::default();
        let ob = model.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let target = model.create_set(ob, &[1, 2, 4]);
        assert!(target.is_ok());
        let target = target.unwrap();
        let v = model.add_set_variable(String::from("v"), ob, target.clone());
        assert!(v.is_ok());
        let v = v.unwrap();

        let v2 = model.add_set_variable(String::from("v"), ob, target.clone());
        assert!(v2.is_err());
        let target2 = Set::with_capacity(11);
        let v2 = model.add_set_variable(String::from("v2"), ob, target2.clone());
        assert!(v2.is_err());
        let result = model.set_target(v, target2);
        assert!(result.is_err());

        let mut model = Model::default();
        let v2 = model.add_set_variable(String::from("v3"), ob, target.clone());
        assert!(v2.is_err());
        let ob2 = model.get_object_type_of(v);
        assert!(ob2.is_err());
        let target2 = model.get_target(v);
        assert!(target2.is_err());
        let result = model.set_target(v, target);
        assert!(result.is_err());
    }

    #[test]
    fn vector_variable_ok() {
        let mut model = Model::default();
        let ob = model.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let target = vec![1, 2, 4];
        let v = model.add_vector_variable(String::from("v"), ob, target.clone());
        assert!(v.is_ok());
        let v = model.add_vector_variable(String::from("v2"), ob, target.clone());
        assert!(v.is_ok());
        let v = v.unwrap();
        let ob2 = model.get_object_type_of(v);
        assert!(ob2.is_ok());
        assert_eq!(ob2.unwrap(), ob);
        let target2 = model.get_target(v);
        assert!(target2.is_ok());
        assert_eq!(target2.unwrap(), target);
        let target = vec![2, 4, 5];
        let result = model.set_target(v, target.clone());
        assert!(result.is_ok());
        let target2 = model.get_target(v);
        assert!(target2.is_ok());
        assert_eq!(target2.unwrap(), target);
    }

    #[test]
    fn vector_variable_err() {
        let mut model = Model::default();
        let ob = model.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let target = vec![1, 2, 4];
        let v = model.add_vector_variable(String::from("v"), ob, target.clone());
        assert!(v.is_ok());
        let v = v.unwrap();

        let v2 = model.add_vector_variable(String::from("v"), ob, target.clone());
        assert!(v2.is_err());
        let target2 = vec![10];
        let v2 = model.add_vector_variable(String::from("v2"), ob, target2.clone());
        assert!(v2.is_err());
        let result = model.set_target(v, target2);
        assert!(result.is_err());

        let mut model = Model::default();
        let v2 = model.add_vector_variable(String::from("v3"), ob, target.clone());
        assert!(v2.is_err());
        let ob2 = model.get_object_type_of(v);
        assert!(ob2.is_err());
        let target2 = model.get_target(v);
        assert!(target2.is_err());
        let result = model.set_target(v, target);
        assert!(result.is_err());
    }

    #[test]
    fn integer_variable_ok() {
        let mut model = Model::default();
        let v = model.add_integer_variable(String::from("v"), 2);
        assert!(v.is_ok());
        let v = model.add_integer_variable(String::from("v2"), 2);
        assert!(v.is_ok());
        let v = v.unwrap();
        let target = model.get_target(v);
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), 2);
        let result = model.set_target(v, 3);
        assert!(result.is_ok());
        let target = model.get_target(v);
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), 3);
    }

    #[test]
    fn integer_variable_err() {
        let mut model = Model::default();
        let v = model.add_integer_variable(String::from("v"), 2);
        assert!(v.is_ok());
        let v = v.unwrap();
        let v2 = model.add_integer_variable(String::from("v"), 2);
        assert!(v2.is_err());

        let mut model = Model::default();
        let target = model.get_target(v);
        assert!(target.is_err());
        let result = model.set_target(v, 5);
        assert!(result.is_err());
    }

    #[test]
    fn integer_resource_variable_ok() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable(String::from("v"), true, 2);
        assert!(v.is_ok());
        let v = model.add_integer_resource_variable(String::from("v2"), true, 2);
        assert!(v.is_ok());
        let v = v.unwrap();
        let target = model.get_target(v);
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), 2);
        let result = model.set_target(v, 3);
        assert!(result.is_ok());
        let target = model.get_target(v);
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), 3);
        let preference = model.get_preference(v);
        assert!(preference.is_ok());
        assert!(preference.unwrap());
        let result = model.set_preference(v, false);
        assert!(result.is_ok());
        let preference = model.get_preference(v);
        assert!(preference.is_ok());
        assert!(!preference.unwrap());
    }

    #[test]
    fn integer_resource_variable_err() {
        let mut model = Model::default();
        let v = model.add_integer_resource_variable(String::from("v"), true, 2);
        assert!(v.is_ok());
        let v = v.unwrap();
        let v2 = model.add_integer_resource_variable(String::from("v"), true, 2);
        assert!(v2.is_err());

        let mut model = Model::default();
        let target = model.get_target(v);
        assert!(target.is_err());
        let result = model.set_target(v, 5);
        assert!(result.is_err());
        let preference = model.get_preference(v);
        assert!(preference.is_err());
        let result = model.set_preference(v, false);
        assert!(result.is_err());
    }

    #[test]
    fn continuous_variable_ok() {
        let mut model = Model::default();
        let v = model.add_continuous_variable(String::from("v"), 2.0);
        assert!(v.is_ok());
        let v = model.add_continuous_variable(String::from("v2"), 2.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let target = model.get_target(v);
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), 2.0);
        let result = model.set_target(v, 3.0);
        assert!(result.is_ok());
        let target = model.get_target(v);
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), 3.0);
    }

    #[test]
    fn continuous_variable_err() {
        let mut model = Model::default();
        let v = model.add_continuous_variable(String::from("v"), 2.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let v2 = model.add_continuous_variable(String::from("v"), 2.0);
        assert!(v2.is_err());

        let mut model = Model::default();
        let target = model.get_target(v);
        assert!(target.is_err());
        let result = model.set_target(v, 5.0);
        assert!(result.is_err());
    }

    #[test]
    fn continuous_resource_variable_ok() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable(String::from("v"), true, 2.0);
        assert!(v.is_ok());
        let v = model.add_continuous_resource_variable(String::from("v2"), true, 2.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let target = model.get_target(v);
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), 2.0);
        let result = model.set_target(v, 3.0);
        assert!(result.is_ok());
        let target = model.get_target(v);
        assert!(target.is_ok());
        assert_eq!(target.unwrap(), 3.0);
        let preference = model.get_preference(v);
        assert!(preference.is_ok());
        assert!(preference.unwrap());
        let result = model.set_preference(v, false);
        assert!(result.is_ok());
        let preference = model.get_preference(v);
        assert!(preference.is_ok());
        assert!(!preference.unwrap());
    }

    #[test]
    fn continuous_resource_variable_err() {
        let mut model = Model::default();
        let v = model.add_continuous_resource_variable(String::from("v"), true, 2.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let v2 = model.add_continuous_resource_variable(String::from("v"), true, 2.0);
        assert!(v2.is_err());

        let mut model = Model::default();
        let target = model.get_target(v);
        assert!(target.is_err());
        let result = model.set_target(v, 5.0);
        assert!(result.is_err());
        let preference = model.get_preference(v);
        assert!(preference.is_err());
        let result = model.set_preference(v, false);
        assert!(result.is_err());
    }

    #[test]
    fn add_table_1d_ok() {
        let mut model = Model::default();
        let t = model.add_table_1d(String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let t = model.add_table_1d(String::from("t2"), vec![0, 2]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
        let t = model.add_table_1d(String::from("t1"), vec![0.0, 1.0]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let t = model.add_table_1d(String::from("t2"), vec![0.0, 2.0]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
        let t = model.add_table_1d(String::from("t1"), vec![true, false]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let t = model.add_table_1d(String::from("t2"), vec![true, false]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
        let t = model.add_table_1d(String::from("t1"), vec![vec![1, 2], vec![1, 2]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let t = model.add_table_1d(String::from("t2"), vec![vec![1, 2], vec![1, 2]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
        let t = model.add_table_1d(String::from("t1"), vec![Set::default()]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let t = model.add_table_1d(String::from("t2"), vec![Set::default()]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
        let t: Result<Table1DHandle<Element>, _> =
            model.add_table_1d(String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let t: Result<Table1DHandle<Element>, _> =
            model.add_table_1d(String::from("t2"), vec![0, 2]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
    }

    #[test]
    fn add_table_1d_err() {
        let mut model = Model::default();
        let t = model.add_table_1d(String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());
        let t = model.add_table_1d(String::from("t1"), vec![0, 1]);
        assert!(t.is_err());
        let t = model.add_table_1d(String::from("t1"), vec![0.0, 1.0]);
        assert!(t.is_ok());
        let t = model.add_table_1d(String::from("t1"), vec![0.0, 1.0]);
        assert!(t.is_err());
        let t = model.add_table_1d(String::from("t1"), vec![true]);
        assert!(t.is_ok());
        let t = model.add_table_1d(String::from("t1"), vec![true]);
        assert!(t.is_err());
        let t = model.add_table_1d(String::from("t1"), vec![vec![]]);
        assert!(t.is_ok());
        let t = model.add_table_1d(String::from("t1"), vec![vec![]]);
        assert!(t.is_err());
        let t = model.add_table_1d(String::from("t1"), vec![Set::default()]);
        assert!(t.is_ok());
        let t = model.add_table_1d(String::from("t1"), vec![Set::default()]);
        assert!(t.is_err());
        let t: Result<Table1DHandle<Element>, _> =
            model.add_table_1d(String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());
        let t: Result<Table1DHandle<Element>, _> =
            model.add_table_1d(String::from("t1"), vec![0, 2]);
        assert!(t.is_err());
    }

    #[test]
    fn set_table_1d_ok() {
        let mut model = Model::default();
        let t = model.add_table_1d(String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_1d(t, 0, 1);
        assert!(result.is_ok());
        let t = model.add_table_1d(String::from("t1"), vec![0.0, 1.0]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_1d(t, 0, 1.0);
        assert!(result.is_ok());
        let t = model.add_table_1d(String::from("t1"), vec![true]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_1d(t, 0, false);
        assert!(result.is_ok());
        let t = model.add_table_1d(String::from("t1"), vec![vec![]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_1d(t, 0, vec![0]);
        assert!(result.is_ok());
        let t = model.add_table_1d(String::from("t1"), vec![Set::default()]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_1d(t, 0, Set::with_capacity(2));
        assert!(result.is_ok());
        let t: Result<Table1DHandle<Element>, _> =
            model.add_table_1d(String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_1d(t, 0, 1);
        assert!(result.is_ok());
    }

    #[test]
    fn set_table_1d_err() {
        let mut model = Model::default();
        let t = model.add_table_1d(String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table_1d(String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());
        let t = model1.add_table_1d(String::from("t2"), vec![0, 1]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_1d(t, 0, 1);
        assert!(result.is_err());

        let t = model.add_table_1d(String::from("t1"), vec![0.0, 1.0]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table_1d(String::from("t1"), vec![0.0, 1.0]);
        assert!(t.is_ok());
        let t = model1.add_table_1d(String::from("t2"), vec![0.0, 1.0]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_1d(t, 0, 1.0);
        assert!(result.is_err());

        let t = model.add_table_1d(String::from("t1"), vec![true]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table_1d(String::from("t1"), vec![true]);
        assert!(t.is_ok());
        let t = model1.add_table_1d(String::from("t2"), vec![true]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_1d(t, 0, false);
        assert!(result.is_err());

        let t = model.add_table_1d(String::from("t1"), vec![vec![]]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table_1d(String::from("t1"), vec![vec![]]);
        assert!(t.is_ok());
        let t = model1.add_table_1d(String::from("t2"), vec![vec![]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_1d(t, 0, vec![1]);
        assert!(result.is_err());

        let t = model.add_table_1d(String::from("t1"), vec![Set::default()]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table_1d(String::from("t1"), vec![Set::default()]);
        assert!(t.is_ok());
        let t = model1.add_table_1d(String::from("t2"), vec![Set::default()]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_1d(t, 0, Set::with_capacity(1));
        assert!(result.is_err());

        let t: Result<Table1DHandle<Element>, _> =
            model.add_table_1d(String::from("t1"), vec![1, 2]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t: Result<Table1DHandle<Element>, _> =
            model1.add_table_1d(String::from("t1"), vec![1, 2]);
        assert!(t.is_ok());
        let t: Result<Table1DHandle<Element>, _> =
            model1.add_table_1d(String::from("t2"), vec![2, 3]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_1d(t, 0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn update_table_1d_ok() {
        let mut model = Model::default();
        let t = model.add_table_1d(String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_1d(t, vec![1, 1]);
        assert!(result.is_ok());
        let t = model.add_table_1d(String::from("t1"), vec![0.0, 1.0]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_1d(t, vec![1.0, 1.0]);
        assert!(result.is_ok());
        let t = model.add_table_1d(String::from("t1"), vec![false]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_1d(t, vec![true]);
        assert!(result.is_ok());
        let t = model.add_table_1d(String::from("t1"), vec![vec![]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_1d(t, vec![vec![1]]);
        assert!(result.is_ok());
        let t = model.add_table_1d(String::from("t1"), vec![Set::default()]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_1d(t, vec![Set::with_capacity(1)]);
        assert!(result.is_ok());
        let t: Result<Table1DHandle<Element>, _> = model.add_table_1d(String::from("t1"), vec![0]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_1d(t, vec![1]);
        assert!(result.is_ok());
    }

    #[test]
    fn update_table_1d_err() {
        let mut model = Model::default();
        let t = model.add_table_1d(String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table_1d(String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());
        let t = model1.add_table_1d(String::from("t2"), vec![0, 1]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_1d(t, vec![1, 1]);
        assert!(result.is_err());

        let t = model.add_table_1d(String::from("t1"), vec![0.0, 1.0]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table_1d(String::from("t1"), vec![0.0, 1.0]);
        assert!(t.is_ok());
        let t = model1.add_table_1d(String::from("t2"), vec![0.0, 1.0]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_1d(t, vec![1.0, 1.0]);
        assert!(result.is_err());

        let t = model.add_table_1d(String::from("t1"), vec![true]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table_1d(String::from("t1"), vec![true]);
        assert!(t.is_ok());
        let t = model1.add_table_1d(String::from("t2"), vec![true]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_1d(t, vec![false]);
        assert!(result.is_err());

        let t = model.add_table_1d(String::from("t1"), vec![vec![]]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table_1d(String::from("t1"), vec![vec![]]);
        assert!(t.is_ok());
        let t = model1.add_table_1d(String::from("t2"), vec![vec![]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_1d(t, vec![vec![1]]);
        assert!(result.is_err());

        let t = model.add_table_1d(String::from("t1"), vec![Set::default()]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table_1d(String::from("t1"), vec![Set::default()]);
        assert!(t.is_ok());
        let t = model1.add_table_1d(String::from("t2"), vec![Set::default()]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_1d(t, vec![Set::with_capacity(1)]);
        assert!(result.is_err());

        let t: Result<Table1DHandle<Element>, _> =
            model.add_table_1d(String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t: Result<Table1DHandle<Element>, _> =
            model1.add_table_1d(String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());
        let t: Result<Table1DHandle<Element>, _> =
            model1.add_table_1d(String::from("t2"), vec![0, 1]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_1d(t, vec![1, 1]);
        assert!(result.is_err());
    }

    #[test]
    fn add_table_2d_ok() {
        let mut model = Model::default();
        let t = model.add_table_2d(String::from("t1"), vec![vec![0, 1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("t1"), 0);
        let t = model.add_table_2d(String::from("t2"), vec![vec![0, 2]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
        let t = model.add_table_2d(String::from("t1"), vec![vec![0.0, 1.0]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("t1"), 0);
        let t = model.add_table_2d(String::from("t2"), vec![vec![0.0, 2.0]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
        let t = model.add_table_2d(String::from("t1"), vec![vec![false]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("t1"), 0);
        let t = model.add_table_2d(String::from("t2"), vec![vec![true]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
        let t = model.add_table_2d(String::from("t1"), vec![vec![vec![]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("t1"), 0);
        let t = model.add_table_2d(String::from("t2"), vec![vec![vec![]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
        let t = model.add_table_2d(String::from("t1"), vec![vec![Set::default()]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("t1"), 0);
        let t = model.add_table_2d(String::from("t2"), vec![vec![Set::default()]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
        let t: Result<Table2DHandle<Element>, _> =
            model.add_table_2d(String::from("t1"), vec![vec![1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("t1"), 0);
        let t: Result<Table2DHandle<Element>, _> =
            model.add_table_2d(String::from("t2"), vec![vec![1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
    }

    #[test]
    fn add_table_2d_err() {
        let mut model = Model::default();
        let t = model.add_table_2d(String::from("t1"), vec![vec![0, 1]]);
        assert!(t.is_ok());
        let t = model.add_table_2d(String::from("t1"), vec![vec![0, 1]]);
        assert!(t.is_err());
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("t1"), 0);
        let t = model.add_table_2d(String::from("t1"), vec![vec![0.0, 1.0]]);
        assert!(t.is_ok());
        let t = model.add_table_2d(String::from("t1"), vec![vec![0.0, 1.0]]);
        assert!(t.is_err());
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("t1"), 0);
        let t = model.add_table_2d(String::from("t1"), vec![vec![false]]);
        assert!(t.is_ok());
        let t = model.add_table_2d(String::from("t1"), vec![vec![true]]);
        assert!(t.is_err());
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("t1"), 0);
        let t = model.add_table_2d(String::from("t1"), vec![vec![vec![]]]);
        assert!(t.is_ok());
        let t = model.add_table_2d(String::from("t1"), vec![vec![vec![]]]);
        assert!(t.is_err());
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("t1"), 0);
        let t = model.add_table_2d(String::from("t1"), vec![vec![Set::default()]]);
        assert!(t.is_ok());
        let t = model.add_table_2d(String::from("t1"), vec![vec![Set::default()]]);
        assert!(t.is_err());
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("t1"), 0);
        let t: Result<Table2DHandle<Element>, _> =
            model.add_table_2d(String::from("t1"), vec![vec![1]]);
        assert!(t.is_ok());
        let t: Result<Table2DHandle<Element>, _> =
            model.add_table_2d(String::from("t1"), vec![vec![1]]);
        assert!(t.is_err());
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("t1"), 0);
    }

    #[test]
    fn set_table_2d_ok() {
        let mut model = Model::default();
        let t = model.add_table_2d(String::from("t1"), vec![vec![0, 1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_2d(t, 0, 0, 1);
        assert!(result.is_ok());
        let t = model.add_table_2d(String::from("t1"), vec![vec![0.0, 1.0]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_2d(t, 0, 0, 1.0);
        assert!(result.is_ok());
        let t = model.add_table_2d(String::from("t1"), vec![vec![false]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_2d(t, 0, 0, true);
        assert!(result.is_ok());
        let t = model.add_table_2d(String::from("t1"), vec![vec![vec![]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_2d(t, 0, 0, vec![1]);
        assert!(result.is_ok());
        let t = model.add_table_2d(String::from("t1"), vec![vec![Set::default()]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_2d(t, 0, 0, Set::with_capacity(1));
        assert!(result.is_ok());
        let t: Result<Table2DHandle<Element>, _> =
            model.add_table_2d(String::from("t1"), vec![vec![0]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_2d(t, 0, 0, 1);
        assert!(result.is_ok());
    }

    #[test]
    fn set_table_2d_err() {
        let mut model = Model::default();
        let t = model.add_table_2d(String::from("t1"), vec![vec![0, 1]]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table_2d(String::from("t1"), vec![vec![0, 1]]);
        assert!(t.is_ok());
        let t = model1.add_table_2d(String::from("t2"), vec![vec![0, 1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_2d(t, 0, 0, 1);
        assert!(result.is_err());

        let t = model.add_table_2d(String::from("t1"), vec![vec![0.0, 1.0]]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table_2d(String::from("t1"), vec![vec![0.0, 1.0]]);
        assert!(t.is_ok());
        let t = model1.add_table_2d(String::from("t2"), vec![vec![0.0, 1.0]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_2d(t, 0, 0, 1.0);
        assert!(result.is_err());

        let t = model.add_table_2d(String::from("t1"), vec![vec![false]]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table_2d(String::from("t1"), vec![vec![false]]);
        assert!(t.is_ok());
        let t = model1.add_table_2d(String::from("t2"), vec![vec![false]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_2d(t, 0, 0, true);
        assert!(result.is_err());

        let t = model.add_table_2d(String::from("t1"), vec![vec![vec![]]]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table_2d(String::from("t1"), vec![vec![vec![]]]);
        assert!(t.is_ok());
        let t = model1.add_table_2d(String::from("t2"), vec![vec![vec![]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_2d(t, 0, 0, vec![0]);
        assert!(result.is_err());

        let t = model.add_table_2d(String::from("t1"), vec![vec![Set::default()]]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table_2d(String::from("t1"), vec![vec![Set::default()]]);
        assert!(t.is_ok());
        let t = model1.add_table_2d(String::from("t2"), vec![vec![Set::default()]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_2d(t, 0, 0, Set::with_capacity(1));
        assert!(result.is_err());

        let t: Result<Table2DHandle<Element>, _> =
            model.add_table_2d(String::from("t1"), vec![vec![1]]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t: Result<Table2DHandle<Element>, _> =
            model1.add_table_2d(String::from("t1"), vec![vec![0]]);
        assert!(t.is_ok());
        let t: Result<Table2DHandle<Element>, _> =
            model1.add_table_2d(String::from("t2"), vec![vec![0]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_2d(t, 0, 0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn update_table_2d_ok() {
        let mut model = Model::default();
        let t = model.add_table_2d(String::from("t1"), vec![vec![0, 1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_2d(t, vec![vec![1, 1]]);
        assert!(result.is_ok());
        let t = model.add_table_2d(String::from("t1"), vec![vec![0.0, 1.0]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_2d(t, vec![vec![1.0, 1.0]]);
        assert!(result.is_ok());
        let t = model.add_table_2d(String::from("t1"), vec![vec![false]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_2d(t, vec![vec![true]]);
        assert!(result.is_ok());
        let t = model.add_table_2d(String::from("t1"), vec![vec![vec![]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_2d(t, vec![vec![vec![1]]]);
        assert!(result.is_ok());
        let t = model.add_table_2d(String::from("t1"), vec![vec![Set::default()]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_2d(t, vec![vec![Set::with_capacity(1)]]);
        assert!(result.is_ok());
        let t: Result<Table2DHandle<Element>, _> =
            model.add_table_2d(String::from("t1"), vec![vec![0]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_2d(t, vec![vec![1]]);
        assert!(result.is_ok());
    }

    #[test]
    fn update_table_2d_err() {
        let mut model = Model::default();
        let t = model.add_table_2d(String::from("t1"), vec![vec![0, 1]]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table_2d(String::from("t1"), vec![vec![0, 1]]);
        assert!(t.is_ok());
        let t = model1.add_table_2d(String::from("t2"), vec![vec![0, 1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_2d(t, vec![vec![1, 1]]);
        assert!(result.is_err());

        let t = model.add_table_2d(String::from("t1"), vec![vec![0.0, 1.0]]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table_2d(String::from("t1"), vec![vec![0.0, 1.0]]);
        assert!(t.is_ok());
        let t = model1.add_table_2d(String::from("t2"), vec![vec![0.0, 1.0]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_2d(t, vec![vec![1.0, 1.0]]);
        assert!(result.is_err());

        let t = model.add_table_2d(String::from("t1"), vec![vec![false]]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table_2d(String::from("t1"), vec![vec![false]]);
        assert!(t.is_ok());
        let t = model1.add_table_2d(String::from("t2"), vec![vec![false]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_2d(t, vec![vec![true]]);
        assert!(result.is_err());

        let t = model.add_table_2d(String::from("t1"), vec![vec![vec![]]]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table_2d(String::from("t1"), vec![vec![vec![]]]);
        assert!(t.is_ok());
        let t = model1.add_table_2d(String::from("t2"), vec![vec![vec![]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_2d(t, vec![vec![vec![1]]]);
        assert!(result.is_err());

        let t = model.add_table_2d(String::from("t1"), vec![vec![Set::default()]]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table_2d(String::from("t1"), vec![vec![Set::default()]]);
        assert!(t.is_ok());
        let t = model1.add_table_2d(String::from("t2"), vec![vec![Set::default()]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_2d(t, vec![vec![Set::with_capacity(1)]]);
        assert!(result.is_err());
    }

    #[test]
    fn add_table_3d_ok() {
        let mut model = Model::default();
        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![0, 1]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let t = model.add_table_3d(String::from("t2"), vec![vec![vec![0, 2]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![0.0, 1.0]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let t = model.add_table_3d(String::from("t2"), vec![vec![vec![0.0, 2.0]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![false]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let t = model.add_table_3d(String::from("t2"), vec![vec![vec![true]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![vec![]]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let t = model.add_table_3d(String::from("t2"), vec![vec![vec![vec![]]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![Set::default()]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let t = model.add_table_3d(String::from("t2"), vec![vec![vec![Set::default()]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
        let t: Result<Table3DHandle<Element>, _> =
            model.add_table_3d(String::from("t1"), vec![vec![vec![1]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let t: Result<Table3DHandle<Element>, _> =
            model.add_table_3d(String::from("t2"), vec![vec![vec![1]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
    }

    #[test]
    fn add_table_3d_err() {
        let mut model = Model::default();
        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![0, 1]]]);
        assert!(t.is_ok());
        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![0, 1]]]);
        assert!(t.is_err());
        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![0.0, 1.0]]]);
        assert!(t.is_ok());
        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![0.0, 1.0]]]);
        assert!(t.is_err());
        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![false]]]);
        assert!(t.is_ok());
        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![false]]]);
        assert!(t.is_err());
        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![vec![]]]]);
        assert!(t.is_ok());
        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![vec![]]]]);
        assert!(t.is_err());
        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![Set::default()]]]);
        assert!(t.is_ok());
        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![Set::default()]]]);
        assert!(t.is_err());
        let t: Result<Table3DHandle<Element>, _> =
            model.add_table_3d(String::from("t1"), vec![vec![vec![1]]]);
        assert!(t.is_ok());
        let t: Result<Table3DHandle<Element>, _> =
            model.add_table_3d(String::from("t1"), vec![vec![vec![1]]]);
        assert!(t.is_err());
    }

    #[test]
    fn set_table_3d_ok() {
        let mut model = Model::default();
        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![0, 1]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_3d(t, 0, 0, 0, 1);
        assert!(result.is_ok());
        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![0.0, 1.0]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_3d(t, 0, 0, 0, 1.0);
        assert!(result.is_ok());
        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![false]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_3d(t, 0, 0, 0, true);
        assert!(result.is_ok());
        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![vec![]]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_3d(t, 0, 0, 0, vec![1]);
        assert!(result.is_ok());
        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![Set::default()]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_3d(t, 0, 0, 0, Set::with_capacity(1));
        assert!(result.is_ok());
        let t: Result<Table3DHandle<Element>, _> =
            model.add_table_3d(String::from("t1"), vec![vec![vec![1]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_3d(t, 0, 0, 0, 2);
        assert!(result.is_ok());
    }

    #[test]
    fn set_table_3d_err() {
        let mut model = Model::default();
        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![0, 1]]]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table_3d(String::from("t1"), vec![vec![vec![0, 1]]]);
        assert!(t.is_ok());
        let t = model1.add_table_3d(String::from("t2"), vec![vec![vec![0, 1]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_3d(t, 0, 0, 0, 1);
        assert!(result.is_err());

        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![0.0, 1.0]]]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table_3d(String::from("t1"), vec![vec![vec![0.0, 1.0]]]);
        assert!(t.is_ok());
        let t = model1.add_table_3d(String::from("t2"), vec![vec![vec![0.0, 1.0]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_3d(t, 0, 0, 0, 1.0);
        assert!(result.is_err());

        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![false]]]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table_3d(String::from("t1"), vec![vec![vec![false]]]);
        assert!(t.is_ok());
        let t = model1.add_table_3d(String::from("t2"), vec![vec![vec![false]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_3d(t, 0, 0, 0, true);
        assert!(result.is_err());

        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![vec![]]]]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table_3d(String::from("t1"), vec![vec![vec![vec![]]]]);
        assert!(t.is_ok());
        let t = model1.add_table_3d(String::from("t2"), vec![vec![vec![vec![]]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_3d(t, 0, 0, 0, vec![1]);
        assert!(result.is_err());

        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![Set::default()]]]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table_3d(String::from("t1"), vec![vec![vec![Set::default()]]]);
        assert!(t.is_ok());
        let t = model1.add_table_3d(String::from("t2"), vec![vec![vec![Set::default()]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_3d(t, 0, 0, 0, Set::with_capacity(1));
        assert!(result.is_err());

        let t: Result<Table3DHandle<Element>, _> =
            model.add_table_3d(String::from("t1"), vec![vec![vec![1]]]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t: Result<Table3DHandle<Element>, _> =
            model1.add_table_3d(String::from("t1"), vec![vec![vec![1]]]);
        assert!(t.is_ok());
        let t: Result<Table3DHandle<Element>, _> =
            model1.add_table_3d(String::from("t2"), vec![vec![vec![1]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table_3d(t, 0, 0, 0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn update_table_3d_ok() {
        let mut model = Model::default();
        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![0, 1]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_3d(t, vec![vec![vec![1, 1]]]);
        assert!(result.is_ok());
        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![0.0, 1.0]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_3d(t, vec![vec![vec![1.0, 1.0]]]);
        assert!(result.is_ok());
        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![true]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_3d(t, vec![vec![vec![false]]]);
        assert!(result.is_ok());
        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![vec![]]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_3d(t, vec![vec![vec![vec![1]]]]);
        assert!(result.is_ok());
        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![Set::default()]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_3d(t, vec![vec![vec![Set::with_capacity(1)]]]);
        assert!(result.is_ok());
        let t: Result<Table3DHandle<Element>, _> =
            model.add_table_3d(String::from("t1"), vec![vec![vec![1]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_3d(t, vec![vec![vec![2]]]);
        assert!(result.is_ok());
    }

    #[test]
    fn update_table_3d_err() {
        let mut model = Model::default();
        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![0, 1]]]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table_3d(String::from("t1"), vec![vec![vec![0, 1]]]);
        assert!(t.is_ok());
        let t = model1.add_table_3d(String::from("t2"), vec![vec![vec![0, 1]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_3d(t, vec![vec![vec![1, 1]]]);
        assert!(result.is_err());

        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![0.0, 1.0]]]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table_3d(String::from("t1"), vec![vec![vec![0.0, 1.0]]]);
        assert!(t.is_ok());
        let t = model1.add_table_3d(String::from("t2"), vec![vec![vec![0.0, 1.0]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_3d(t, vec![vec![vec![1.0, 1.0]]]);
        assert!(result.is_err());

        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![false]]]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table_3d(String::from("t1"), vec![vec![vec![false]]]);
        assert!(t.is_ok());
        let t = model1.add_table_3d(String::from("t2"), vec![vec![vec![false]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_3d(t, vec![vec![vec![true]]]);
        assert!(result.is_err());

        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![vec![]]]]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table_3d(String::from("t1"), vec![vec![vec![vec![]]]]);
        assert!(t.is_ok());
        let t = model1.add_table_3d(String::from("t2"), vec![vec![vec![vec![]]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_3d(t, vec![vec![vec![vec![1]]]]);
        assert!(result.is_err());

        let t = model.add_table_3d(String::from("t1"), vec![vec![vec![Set::default()]]]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table_3d(String::from("t1"), vec![vec![vec![Set::default()]]]);
        assert!(t.is_ok());
        let t = model1.add_table_3d(String::from("t2"), vec![vec![vec![Set::default()]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_3d(t, vec![vec![vec![Set::with_capacity(1)]]]);
        assert!(result.is_err());

        let t: Result<Table3DHandle<Element>, _> =
            model.add_table_3d(String::from("t1"), vec![vec![vec![1]]]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t: Result<Table3DHandle<Element>, _> =
            model1.add_table_3d(String::from("t1"), vec![vec![vec![1]]]);
        assert!(t.is_ok());
        let t: Result<Table3DHandle<Element>, _> =
            model1.add_table_3d(String::from("t2"), vec![vec![vec![1]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.update_table_3d(t, vec![vec![vec![0]]]);
        assert!(result.is_err());
    }

    #[test]
    fn add_table_ok() {
        let mut model = Model::default();
        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1);
        let t = model.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let mut map2 = FxHashMap::default();
        map2.insert(vec![0, 0, 0, 1], 2);
        let t = model.add_table(String::from("t2"), map2.clone(), 1);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1.0);
        let t = model.add_table(String::from("t1"), map.clone(), 0.0);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let mut map2 = FxHashMap::default();
        map2.insert(vec![0, 0, 0, 1], 2.0);
        let t = model.add_table(String::from("t2"), map2.clone(), 1.0);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], true);
        let t = model.add_table(String::from("t1"), map.clone(), false);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let mut map2 = FxHashMap::default();
        map2.insert(vec![0, 0, 0, 1], true);
        let t = model.add_table(String::from("t2"), map2.clone(), false);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], vec![1]);
        let t = model.add_table(String::from("t1"), map.clone(), vec![]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let mut map2 = FxHashMap::default();
        map2.insert(vec![0, 0, 0, 1], vec![2]);
        let t = model.add_table(String::from("t2"), map2.clone(), vec![]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], Set::with_capacity(1));
        let t = model.add_table(String::from("t1"), map.clone(), Set::default());
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let mut map2 = FxHashMap::default();
        map2.insert(vec![0, 0, 0, 1], Set::with_capacity(1));
        let t = model.add_table(String::from("t2"), map2.clone(), Set::default());
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);

        let mut map: FxHashMap<_, Element> = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1);
        let t = model.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let mut map2: FxHashMap<_, Element> = FxHashMap::default();
        map2.insert(vec![0, 0, 0, 1], 2);
        let t = model.add_table(String::from("t2"), map2.clone(), 0);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
    }

    #[test]
    fn add_table_err() {
        let mut model = Model::default();
        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1);
        let t = model.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());
        let t = model.add_table(String::from("t1"), map.clone(), 1);
        assert!(t.is_err());

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1.0);
        let t = model.add_table(String::from("t1"), map.clone(), 0.0);
        assert!(t.is_ok());
        let t = model.add_table(String::from("t1"), map.clone(), 1.0);
        assert!(t.is_err());

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], true);
        let t = model.add_table(String::from("t1"), map.clone(), false);
        assert!(t.is_ok());
        let t = model.add_table(String::from("t1"), map.clone(), false);
        assert!(t.is_err());

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], vec![1]);
        let t = model.add_table(String::from("t1"), map.clone(), vec![]);
        assert!(t.is_ok());
        let t = model.add_table(String::from("t1"), map.clone(), vec![]);
        assert!(t.is_err());

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], Set::with_capacity(1));
        let t = model.add_table(String::from("t1"), map.clone(), Set::default());
        assert!(t.is_ok());
        let t = model.add_table(String::from("t1"), map.clone(), Set::default());
        assert!(t.is_err());

        let mut map: FxHashMap<_, Element> = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1);
        let t = model.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());
        let t = model.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_err());
    }

    #[test]
    fn set_table_ok() {
        let mut model = Model::default();
        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1);
        let t = model.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table(t, vec![0, 0, 0, 0], 1);
        assert!(result.is_ok());

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1.0);
        let t = model.add_table(String::from("t1"), map.clone(), 0.0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table(t, vec![0, 0, 0, 0], 1.0);
        assert!(result.is_ok());

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], true);
        let t = model.add_table(String::from("t1"), map.clone(), false);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table(t, vec![0, 0, 0, 0], true);
        assert!(result.is_ok());

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], vec![1]);
        let t = model.add_table(String::from("t1"), map.clone(), vec![]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table(t, vec![0, 0, 0, 0], vec![]);
        assert!(result.is_ok());

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], Set::with_capacity(1));
        let t = model.add_table(String::from("t1"), map.clone(), Set::default());
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table(t, vec![0, 0, 0, 0], Set::default());
        assert!(result.is_ok());

        let mut map: FxHashMap<_, Element> = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1);
        let t = model.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table(t, vec![0, 0, 0, 0], 0);
        assert!(result.is_ok());
    }

    #[test]
    fn set_table_err() {
        let mut model = Model::default();
        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1);
        let t = model.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table(String::from("t1"), map.clone(), 1);
        assert!(t.is_ok());
        let t = model1.add_table(String::from("t2"), map.clone(), 2);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table(t, vec![0, 0, 0, 0], 1);
        assert!(result.is_err());

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1.0);
        let t = model.add_table(String::from("t1"), map.clone(), 0.0);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table(String::from("t1"), map.clone(), 1.0);
        assert!(t.is_ok());
        let t = model1.add_table(String::from("t2"), map.clone(), 2.0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table(t, vec![0, 0, 0, 0], 1.0);
        assert!(result.is_err());

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], true);
        let t = model.add_table(String::from("t1"), map.clone(), false);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table(String::from("t1"), map.clone(), false);
        assert!(t.is_ok());
        let t = model1.add_table(String::from("t2"), map.clone(), false);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table(t, vec![0, 0, 0, 0], true);
        assert!(result.is_err());

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], vec![1]);
        let t = model.add_table(String::from("t1"), map.clone(), vec![]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table(String::from("t1"), map.clone(), vec![]);
        assert!(t.is_ok());
        let t = model1.add_table(String::from("t2"), map.clone(), vec![]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table(t, vec![0, 0, 0, 0], vec![1]);
        assert!(result.is_err());

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], Set::with_capacity(1));
        let t = model.add_table(String::from("t1"), map.clone(), Set::default());
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table(String::from("t1"), map.clone(), Set::default());
        assert!(t.is_ok());
        let t = model1.add_table(String::from("t2"), map.clone(), Set::default());
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table(t, vec![0, 0, 0, 0], Set::with_capacity(1));
        assert!(result.is_err());

        let mut map: FxHashMap<_, Element> = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1);
        let t = model.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());
        let t = model1.add_table(String::from("t2"), map.clone(), 0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_table(t, vec![0, 0, 0, 0], 1);
        assert!(result.is_err());
    }

    #[test]
    fn set_default_ok() {
        let mut model = Model::default();
        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1);
        let t = model.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_default(t, 1);
        assert!(result.is_ok());

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1.0);
        let t = model.add_table(String::from("t1"), map.clone(), 0.0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_default(t, 1.0);
        assert!(result.is_ok());

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], true);
        let t = model.add_table(String::from("t1"), map.clone(), false);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_default(t, true);
        assert!(result.is_ok());

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], vec![1]);
        let t = model.add_table(String::from("t1"), map.clone(), vec![]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_default(t, vec![2]);
        assert!(result.is_ok());

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], Set::with_capacity(1));
        let t = model.add_table(String::from("t1"), map.clone(), Set::default());
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_default(t, Set::with_capacity(2));
        assert!(result.is_ok());

        let mut map: FxHashMap<_, Element> = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1);
        let t = model.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_default(t, 2);
        assert!(result.is_ok());
    }

    #[test]
    fn set_default_err() {
        let mut model = Model::default();
        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1);
        let t = model.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());
        let t = model1.add_table(String::from("t2"), map.clone(), 0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_default(t, 1);
        assert!(result.is_err());

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1.0);
        let t = model.add_table(String::from("t1"), map.clone(), 0.0);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table(String::from("t1"), map.clone(), 0.0);
        assert!(t.is_ok());
        let t = model1.add_table(String::from("t2"), map.clone(), 0.0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_default(t, 1.0);
        assert!(result.is_err());

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], true);
        let t = model.add_table(String::from("t1"), map.clone(), false);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table(String::from("t1"), map.clone(), false);
        assert!(t.is_ok());
        let t = model1.add_table(String::from("t2"), map.clone(), false);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_default(t, true);
        assert!(result.is_err());

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], vec![1]);
        let t = model.add_table(String::from("t1"), map.clone(), vec![]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table(String::from("t1"), map.clone(), vec![]);
        assert!(t.is_ok());
        let t = model1.add_table(String::from("t2"), map.clone(), vec![]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_default(t, vec![1]);
        assert!(result.is_err());

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], Set::with_capacity(1));
        let t = model.add_table(String::from("t1"), map.clone(), Set::default());
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table(String::from("t1"), map.clone(), Set::default());
        assert!(t.is_ok());
        let t = model1.add_table(String::from("t2"), map.clone(), Set::default());
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_default(t, Set::with_capacity(2));
        assert!(result.is_err());

        let mut map: FxHashMap<_, Element> = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1);
        let t = model.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());
        let t = model1.add_table(String::from("t2"), map.clone(), 0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = model.set_default(t, 2);
        assert!(result.is_err());
    }

    #[test]
    fn update_table_ok() {
        let mut model = Model::default();
        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1);
        let t = model.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 2);
        let result = model.update_table(t, map.clone(), 1);
        assert!(result.is_ok());

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1.0);
        let t = model.add_table(String::from("t1"), map.clone(), 0.0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 2.0);
        let result = model.update_table(t, map.clone(), 1.0);
        assert!(result.is_ok());

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], true);
        let t = model.add_table(String::from("t1"), map.clone(), false);
        assert!(t.is_ok());
        let t = t.unwrap();
        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], true);
        let result = model.update_table(t, map.clone(), false);
        assert!(result.is_ok());

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], vec![1]);
        let t = model.add_table(String::from("t1"), map.clone(), vec![]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], vec![1]);
        let result = model.update_table(t, map.clone(), vec![]);
        assert!(result.is_ok());

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], Set::with_capacity(1));
        let t = model.add_table(String::from("t1"), map.clone(), Set::default());
        assert!(t.is_ok());
        let t = t.unwrap();
        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], Set::with_capacity(1));
        let result = model.update_table(t, map.clone(), Set::default());
        assert!(result.is_ok());

        let mut map: FxHashMap<_, Element> = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1);
        let t = model.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1);
        let result = model.update_table(t, map.clone(), 0);
        assert!(result.is_ok());
    }

    #[test]
    fn update_table_err() {
        let mut model = Model::default();
        let mut map: FxHashMap<_, Integer> = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1);
        let t = model.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table(String::from("t1"), map.clone(), 1);
        assert!(t.is_ok());
        let t = model1.add_table(String::from("t2"), map.clone(), 2);
        assert!(t.is_ok());
        let t = t.unwrap();
        let mut map2 = FxHashMap::default();
        map2.insert(vec![0, 0, 0, 1], 2);
        let result = model.update_table(t, map2, 3);
        assert!(result.is_err());

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1.0);
        let t = model.add_table(String::from("t1"), map.clone(), 0.0);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table(String::from("t1"), map.clone(), 1.0);
        assert!(t.is_ok());
        let t = model1.add_table(String::from("t2"), map.clone(), 2.0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let mut map2 = FxHashMap::default();
        map2.insert(vec![0, 0, 0, 1], 2.0);
        let result = model.update_table(t, map2, 3.0);
        assert!(result.is_err());

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], true);
        let t = model.add_table(String::from("t1"), map.clone(), false);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table(String::from("t1"), map.clone(), true);
        assert!(t.is_ok());
        let t = model1.add_table(String::from("t2"), map.clone(), true);
        assert!(t.is_ok());
        let t = t.unwrap();
        let mut map2 = FxHashMap::default();
        map2.insert(vec![0, 0, 0, 1], true);
        let result = model.update_table(t, map2, false);
        assert!(result.is_err());

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], vec![1]);
        let t = model.add_table(String::from("t1"), map.clone(), vec![]);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table(String::from("t1"), map.clone(), vec![]);
        assert!(t.is_ok());
        let t = model1.add_table(String::from("t2"), map.clone(), vec![]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let mut map2 = FxHashMap::default();
        map2.insert(vec![0, 0, 0, 1], vec![]);
        let result = model.update_table(t, map2, vec![]);
        assert!(result.is_err());

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], Set::with_capacity(1));
        let t = model.add_table(String::from("t1"), map.clone(), Set::default());
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table(String::from("t1"), map.clone(), Set::default());
        assert!(t.is_ok());
        let t = model1.add_table(String::from("t2"), map.clone(), Set::default());
        assert!(t.is_ok());
        let t = t.unwrap();
        let mut map2 = FxHashMap::default();
        map2.insert(vec![0, 0, 0, 1], Set::with_capacity(1));
        let result = model.update_table(t, map2, Set::default());
        assert!(result.is_err());

        let mut map: FxHashMap<_, Element> = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1);
        let t = model.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());

        let mut model1 = Model::default();
        let t = model1.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());
        let t = model1.add_table(String::from("t2"), map.clone(), 0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let mut map2 = FxHashMap::default();
        map2.insert(vec![0, 0, 0, 1], 1);
        let result = model.update_table(t, map2, 0);
        assert!(result.is_err());
    }

    #[test]
    fn add_constraint_ok() {
        let mut model = Model::default();
        let v = model.add_integer_variable(String::from("v"), 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let condition =
            Condition::comparison_i(ComparisonOperator::Ge, v, 0) & Condition::Constant(true);
        let result = model.add_state_constraint(condition);
        assert!(result.is_ok());
        assert_eq!(
            model.state_constraints,
            vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Ge,
                    Box::new(IntegerExpression::Variable(v.id())),
                    Box::new(IntegerExpression::Constant(0))
                ),
                ..Default::default()
            }]
        )
    }

    #[test]
    fn add_constraint_err() {
        let mut model = Model::default();
        let v = model.add_integer_variable(String::from("v"), 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let mut model = Model::default();
        let condition = Condition::comparison_i(ComparisonOperator::Ge, v, 0);
        let result = model.add_state_constraint(condition);
        assert!(result.is_err());
        assert_eq!(model.state_constraints, vec![]);
    }

    #[test]
    fn add_base_case_ok() {
        let mut model = Model::default();
        let v = model.add_integer_variable(String::from("v"), 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let condition =
            Condition::comparison_i(ComparisonOperator::Ge, v, 3) & Condition::Constant(true);
        let result = model.add_base_case(vec![condition]);
        assert!(result.is_ok());
        assert_eq!(
            model.base_cases,
            vec![BaseCase::new(vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Ge,
                    Box::new(IntegerExpression::Variable(v.id())),
                    Box::new(IntegerExpression::Constant(3))
                ),
                ..Default::default()
            }])]
        )
    }

    #[test]
    fn add_base_case_err() {
        let mut model = Model::default();
        let v = model.add_integer_variable(String::from("v"), 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let mut model = Model::default();
        let condition = Condition::comparison_i(ComparisonOperator::Ge, v, 0);
        let result = model.add_base_case(vec![condition]);
        assert!(result.is_err());
        assert_eq!(model.base_cases, vec![]);
    }

    #[test]
    fn check_state_ok() {
        let mut model = Model::default();
        let v = model.add_integer_variable(String::from("v"), 0);
        assert!(v.is_ok());
        assert!(model.check_state(&model.target).is_ok());
    }

    #[test]
    fn check_state_err() {
        let mut model = Model::default();
        let v = model.add_integer_variable(String::from("v"), 0);
        assert!(v.is_ok());
        let state = State::default();
        assert!(model.check_state(&state).is_err());
    }

    #[test]
    fn add_base_state_ok() {
        let mut model = Model::default();
        let v = model.add_integer_variable(String::from("v"), 0);
        assert!(v.is_ok());
        let state = model.target.clone();
        assert!(model.add_base_state(state).is_ok());
        assert_eq!(model.base_states, vec![model.target.clone()]);
    }

    #[test]
    fn add_base_state_err() {
        let mut model = Model::default();
        let v = model.add_integer_variable(String::from("v"), 0);
        assert!(v.is_ok());
        let state = State::default();
        assert!(model.add_base_state(state).is_err());
        assert_eq!(model.base_states, vec![]);
    }

    #[test]
    fn reduce_function() {
        let mut model = Model {
            reduce_function: ReduceFunction::Min,
            ..Default::default()
        };
        assert_eq!(model.get_reduce_function(), ReduceFunction::Min);
        model.set_reduce_function(ReduceFunction::Max);
        assert_eq!(model.get_reduce_function(), ReduceFunction::Max);
    }

    #[test]
    fn add_forward_transition_ok() {
        let mut model = Model {
            cost_type: CostType::Integer,
            ..Default::default()
        };
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("sv1", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = model.add_set_variable("sv2", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = model.add_vector_variable("vv1", ob, vec![]);
        assert!(v.is_ok());
        let v = model.add_vector_variable("vv2", ob, vec![]);
        assert!(v.is_ok());
        let v = model.add_element_variable("ev1", ob, 0);
        assert!(v.is_ok());
        let v = model.add_element_variable("ev2", ob, 0);
        assert!(v.is_ok());
        let v = model.add_element_resource_variable("erv1", ob, false, 0);
        assert!(v.is_ok());
        let v = model.add_element_resource_variable("erv2", ob, true, 0);
        assert!(v.is_ok());
        let v = model.add_integer_variable("iv1", 0);
        assert!(v.is_ok());
        let v = model.add_integer_variable("iv2", 0);
        assert!(v.is_ok());
        let v = model.add_integer_resource_variable("irv1", false, 0);
        assert!(v.is_ok());
        let v = model.add_integer_resource_variable("irv2", true, 0);
        assert!(v.is_ok());
        let v = model.add_continuous_variable("cv1", 0.0);
        assert!(v.is_ok());
        let v = model.add_continuous_variable("cv2", 0.0);
        assert!(v.is_ok());
        let v = model.add_continuous_resource_variable("crv1", false, 0.0);
        assert!(v.is_ok());
        let v = model.add_continuous_resource_variable("crv2", true, 0.0);
        assert!(v.is_ok());

        let transition = Transition {
            cost: CostExpression::Integer(IntegerExpression::Cost),
            ..Default::default()
        };
        let result = model.add_forward_transition(transition);
        assert!(result.is_ok());
        assert_eq!(
            model.forward_transitions,
            vec![Transition {
                cost: CostExpression::Integer(IntegerExpression::Cost),
                ..Default::default()
            }]
        );

        let transition = Transition {
            cost: CostExpression::Integer(IntegerExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(IntegerExpression::Constant(-1)),
            )),
            preconditions: vec![
                GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Eq,
                        Box::new(IntegerExpression::Variable(0)),
                        Box::new(IntegerExpression::Constant(0)),
                    ),
                    ..Default::default()
                },
                GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Eq,
                        Box::new(IntegerExpression::Variable(0)),
                        Box::new(IntegerExpression::UnaryOperation(
                            UnaryOperator::Abs,
                            Box::new(IntegerExpression::Constant(0)),
                        )),
                    ),
                    ..Default::default()
                },
            ],
            effect: Effect {
                set_effects: vec![
                    (
                        0,
                        SetExpression::Reference(ReferenceExpression::Constant(
                            Set::with_capacity(10),
                        )),
                    ),
                    (
                        1,
                        SetExpression::SetElementOperation(
                            SetElementOperator::Add,
                            ElementExpression::Constant(0),
                            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                                Set::with_capacity(10),
                            ))),
                        ),
                    ),
                ],
                vector_effects: vec![
                    (
                        0,
                        VectorExpression::Reference(ReferenceExpression::Constant(vec![1, 2])),
                    ),
                    (
                        1,
                        VectorExpression::Push(
                            ElementExpression::Constant(0),
                            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                                vec![1, 2],
                            ))),
                        ),
                    ),
                ],
                element_effects: vec![
                    (0, ElementExpression::Constant(0)),
                    (
                        1,
                        ElementExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ElementExpression::Constant(1)),
                            Box::new(ElementExpression::Constant(1)),
                        ),
                    ),
                ],
                integer_effects: vec![
                    (0, IntegerExpression::Constant(0)),
                    (
                        1,
                        IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::Constant(1)),
                            Box::new(IntegerExpression::Constant(1)),
                        ),
                    ),
                ],
                continuous_effects: vec![
                    (0, ContinuousExpression::Constant(0.0)),
                    (
                        1,
                        ContinuousExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ContinuousExpression::Constant(1.0)),
                            Box::new(ContinuousExpression::Constant(1.0)),
                        ),
                    ),
                ],
                element_resource_effects: vec![
                    (0, ElementExpression::Constant(0)),
                    (
                        1,
                        ElementExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ElementExpression::Constant(1)),
                            Box::new(ElementExpression::Constant(1)),
                        ),
                    ),
                ],
                integer_resource_effects: vec![
                    (0, IntegerExpression::Constant(0)),
                    (
                        1,
                        IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::Constant(1)),
                            Box::new(IntegerExpression::Constant(1)),
                        ),
                    ),
                ],
                continuous_resource_effects: vec![
                    (0, ContinuousExpression::Constant(0.0)),
                    (
                        1,
                        ContinuousExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ContinuousExpression::Constant(1.0)),
                            Box::new(ContinuousExpression::Constant(1.0)),
                        ),
                    ),
                ],
            },
            ..Default::default()
        };
        let result = model.add_forward_transition(transition);
        assert!(result.is_ok());
        assert_eq!(
            model.forward_transitions,
            vec![
                Transition {
                    cost: CostExpression::Integer(IntegerExpression::Cost),
                    ..Default::default()
                },
                Transition {
                    cost: CostExpression::Integer(IntegerExpression::Constant(1)),
                    preconditions: vec![
                        GroundedCondition {
                            condition: Condition::ComparisonI(
                                ComparisonOperator::Eq,
                                Box::new(IntegerExpression::Variable(0)),
                                Box::new(IntegerExpression::Constant(0)),
                            ),
                            ..Default::default()
                        },
                        GroundedCondition {
                            condition: Condition::ComparisonI(
                                ComparisonOperator::Eq,
                                Box::new(IntegerExpression::Variable(0)),
                                Box::new(IntegerExpression::Constant(0),),
                            ),
                            ..Default::default()
                        },
                    ],
                    effect: Effect {
                        set_effects: vec![
                            (
                                0,
                                SetExpression::Reference(ReferenceExpression::Constant(
                                    Set::with_capacity(10),
                                )),
                            ),
                            (
                                1,
                                SetExpression::Reference(ReferenceExpression::Constant({
                                    let mut set = Set::with_capacity(10);
                                    set.insert(0);
                                    set
                                },)),
                            ),
                        ],
                        vector_effects: vec![
                            (
                                0,
                                VectorExpression::Reference(ReferenceExpression::Constant(vec![
                                    1, 2
                                ])),
                            ),
                            (
                                1,
                                VectorExpression::Reference(ReferenceExpression::Constant(vec![
                                    1, 2, 0
                                ])),
                            ),
                        ],
                        element_effects: vec![
                            (0, ElementExpression::Constant(0)),
                            (1, ElementExpression::Constant(2)),
                        ],
                        integer_effects: vec![
                            (0, IntegerExpression::Constant(0)),
                            (1, IntegerExpression::Constant(2)),
                        ],
                        continuous_effects: vec![
                            (0, ContinuousExpression::Constant(0.0)),
                            (1, ContinuousExpression::Constant(2.0)),
                        ],
                        element_resource_effects: vec![
                            (0, ElementExpression::Constant(0)),
                            (1, ElementExpression::Constant(2)),
                        ],
                        integer_resource_effects: vec![
                            (0, IntegerExpression::Constant(0)),
                            (1, IntegerExpression::Constant(2)),
                        ],
                        continuous_resource_effects: vec![
                            (0, ContinuousExpression::Constant(0.0)),
                            (1, ContinuousExpression::Constant(2.0)),
                        ],
                    },
                    ..Default::default()
                }
            ]
        );

        let mut model = Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        };
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("sv1", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = model.add_set_variable("sv2", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = model.add_vector_variable("vv1", ob, vec![]);
        assert!(v.is_ok());
        let v = model.add_vector_variable("vv2", ob, vec![]);
        assert!(v.is_ok());
        let v = model.add_element_variable("ev1", ob, 0);
        assert!(v.is_ok());
        let v = model.add_element_variable("ev2", ob, 0);
        assert!(v.is_ok());
        let v = model.add_element_resource_variable("erv1", ob, false, 0);
        assert!(v.is_ok());
        let v = model.add_element_resource_variable("erv2", ob, true, 0);
        assert!(v.is_ok());
        let v = model.add_integer_variable("iv1", 0);
        assert!(v.is_ok());
        let v = model.add_integer_variable("iv2", 0);
        assert!(v.is_ok());
        let v = model.add_integer_resource_variable("irv1", false, 0);
        assert!(v.is_ok());
        let v = model.add_integer_resource_variable("irv2", true, 0);
        assert!(v.is_ok());
        let v = model.add_continuous_variable("cv1", 0.0);
        assert!(v.is_ok());
        let v = model.add_continuous_variable("cv2", 0.0);
        assert!(v.is_ok());
        let v = model.add_continuous_resource_variable("crv1", false, 0.0);
        assert!(v.is_ok());
        let v = model.add_continuous_resource_variable("crv2", true, 0.0);
        assert!(v.is_ok());

        let transition = Transition {
            cost: CostExpression::Continuous(ContinuousExpression::Cost),
            ..Default::default()
        };
        let result = model.add_forward_transition(transition);
        assert!(result.is_ok());
        assert_eq!(
            model.forward_transitions,
            vec![Transition {
                cost: CostExpression::Continuous(ContinuousExpression::Cost),
                ..Default::default()
            }]
        );

        let transition = Transition {
            cost: CostExpression::Continuous(ContinuousExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(ContinuousExpression::Constant(-1.0)),
            )),
            preconditions: vec![
                GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Eq,
                        Box::new(IntegerExpression::Variable(0)),
                        Box::new(IntegerExpression::Constant(0)),
                    ),
                    ..Default::default()
                },
                GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Eq,
                        Box::new(IntegerExpression::Variable(0)),
                        Box::new(IntegerExpression::UnaryOperation(
                            UnaryOperator::Abs,
                            Box::new(IntegerExpression::Constant(0)),
                        )),
                    ),
                    ..Default::default()
                },
            ],
            effect: Effect {
                set_effects: vec![
                    (
                        0,
                        SetExpression::Reference(ReferenceExpression::Constant(
                            Set::with_capacity(10),
                        )),
                    ),
                    (
                        1,
                        SetExpression::SetElementOperation(
                            SetElementOperator::Add,
                            ElementExpression::Constant(0),
                            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                                Set::with_capacity(10),
                            ))),
                        ),
                    ),
                ],
                vector_effects: vec![
                    (
                        0,
                        VectorExpression::Reference(ReferenceExpression::Constant(vec![1, 2])),
                    ),
                    (
                        1,
                        VectorExpression::Push(
                            ElementExpression::Constant(0),
                            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                                vec![1, 2],
                            ))),
                        ),
                    ),
                ],
                element_effects: vec![
                    (0, ElementExpression::Constant(0)),
                    (
                        1,
                        ElementExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ElementExpression::Constant(1)),
                            Box::new(ElementExpression::Constant(1)),
                        ),
                    ),
                ],
                integer_effects: vec![
                    (0, IntegerExpression::Constant(0)),
                    (
                        1,
                        IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::Constant(1)),
                            Box::new(IntegerExpression::Constant(1)),
                        ),
                    ),
                ],
                continuous_effects: vec![
                    (0, ContinuousExpression::Constant(0.0)),
                    (
                        1,
                        ContinuousExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ContinuousExpression::Constant(1.0)),
                            Box::new(ContinuousExpression::Constant(1.0)),
                        ),
                    ),
                ],
                element_resource_effects: vec![
                    (0, ElementExpression::Constant(0)),
                    (
                        1,
                        ElementExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ElementExpression::Constant(1)),
                            Box::new(ElementExpression::Constant(1)),
                        ),
                    ),
                ],
                integer_resource_effects: vec![
                    (0, IntegerExpression::Constant(0)),
                    (
                        1,
                        IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::Constant(1)),
                            Box::new(IntegerExpression::Constant(1)),
                        ),
                    ),
                ],
                continuous_resource_effects: vec![
                    (0, ContinuousExpression::Constant(0.0)),
                    (
                        1,
                        ContinuousExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ContinuousExpression::Constant(1.0)),
                            Box::new(ContinuousExpression::Constant(1.0)),
                        ),
                    ),
                ],
            },
            ..Default::default()
        };
        let result = model.add_forward_transition(transition);
        assert!(result.is_ok());
        assert_eq!(
            model.forward_transitions,
            vec![
                Transition {
                    cost: CostExpression::Continuous(ContinuousExpression::Cost),
                    ..Default::default()
                },
                Transition {
                    cost: CostExpression::Continuous(ContinuousExpression::Constant(1.0)),
                    preconditions: vec![
                        GroundedCondition {
                            condition: Condition::ComparisonI(
                                ComparisonOperator::Eq,
                                Box::new(IntegerExpression::Variable(0)),
                                Box::new(IntegerExpression::Constant(0)),
                            ),
                            ..Default::default()
                        },
                        GroundedCondition {
                            condition: Condition::ComparisonI(
                                ComparisonOperator::Eq,
                                Box::new(IntegerExpression::Variable(0)),
                                Box::new(IntegerExpression::Constant(0),),
                            ),
                            ..Default::default()
                        },
                    ],
                    effect: Effect {
                        set_effects: vec![
                            (
                                0,
                                SetExpression::Reference(ReferenceExpression::Constant(
                                    Set::with_capacity(10),
                                )),
                            ),
                            (
                                1,
                                SetExpression::Reference(ReferenceExpression::Constant({
                                    let mut set = Set::with_capacity(10);
                                    set.insert(0);
                                    set
                                },)),
                            ),
                        ],
                        vector_effects: vec![
                            (
                                0,
                                VectorExpression::Reference(ReferenceExpression::Constant(vec![
                                    1, 2
                                ])),
                            ),
                            (
                                1,
                                VectorExpression::Reference(ReferenceExpression::Constant(vec![
                                    1, 2, 0
                                ])),
                            ),
                        ],
                        element_effects: vec![
                            (0, ElementExpression::Constant(0)),
                            (1, ElementExpression::Constant(2)),
                        ],
                        integer_effects: vec![
                            (0, IntegerExpression::Constant(0)),
                            (1, IntegerExpression::Constant(2)),
                        ],
                        continuous_effects: vec![
                            (0, ContinuousExpression::Constant(0.0)),
                            (1, ContinuousExpression::Constant(2.0)),
                        ],
                        element_resource_effects: vec![
                            (0, ElementExpression::Constant(0)),
                            (1, ElementExpression::Constant(2)),
                        ],
                        integer_resource_effects: vec![
                            (0, IntegerExpression::Constant(0)),
                            (1, IntegerExpression::Constant(2)),
                        ],
                        continuous_resource_effects: vec![
                            (0, ContinuousExpression::Constant(0.0)),
                            (1, ContinuousExpression::Constant(2.0)),
                        ],
                    },
                    ..Default::default()
                }
            ]
        );
    }

    #[test]
    fn add_forward_transition_err() {
        let mut model = Model {
            cost_type: CostType::Integer,
            ..Default::default()
        };
        let transition = Transition {
            cost: CostExpression::Continuous(ContinuousExpression::Cost),
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model {
            cost_type: CostType::Integer,
            ..Default::default()
        };
        let transition = Transition {
            cost: CostExpression::Integer(IntegerExpression::Variable(0)),
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        };
        let transition = Transition {
            cost: CostExpression::Integer(IntegerExpression::Cost),
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        };
        let transition = Transition {
            cost: CostExpression::Continuous(ContinuousExpression::Variable(0)),
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            elements_in_set_variable: vec![(0, 0)],
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            elements_in_set_variable: vec![(v.id(), 11)],
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            elements_in_vector_variable: vec![(0, 0)],
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_vector_variable("v", ob, vec![0, 1]);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            elements_in_vector_variable: vec![(v.id(), 11)],
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            preconditions: vec![GroundedCondition {
                elements_in_set_variable: vec![(0, 0)],
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            preconditions: vec![GroundedCondition {
                elements_in_set_variable: vec![(v.id(), 11)],
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            preconditions: vec![GroundedCondition {
                elements_in_vector_variable: vec![(0, 0)],
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_vector_variable("v", ob, vec![0, 1]);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            preconditions: vec![GroundedCondition {
                elements_in_vector_variable: vec![(v.id(), 11)],
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            preconditions: vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Eq,
                    Box::new(IntegerExpression::Variable(0)),
                    Box::new(IntegerExpression::Constant(0)),
                ),
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                set_effects: vec![(
                    0,
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v1", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                set_effects: vec![(
                    v.id(),
                    SetExpression::Reference(ReferenceExpression::Variable(1)),
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v1", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                set_effects: vec![
                    (
                        v.id(),
                        SetExpression::Reference(ReferenceExpression::Constant(
                            Set::with_capacity(10),
                        )),
                    ),
                    (
                        v.id(),
                        SetExpression::Reference(ReferenceExpression::Constant(
                            Set::with_capacity(10),
                        )),
                    ),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                vector_effects: vec![(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_vector_variable("v1", ob, vec![0, 1]);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                vector_effects: vec![(
                    v.id(),
                    VectorExpression::Reference(ReferenceExpression::Variable(1)),
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_vector_variable("v1", ob, vec![0, 1]);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                vector_effects: vec![
                    (
                        v.id(),
                        VectorExpression::Reference(ReferenceExpression::Variable(1)),
                    ),
                    (
                        v.id(),
                        VectorExpression::Reference(ReferenceExpression::Variable(1)),
                    ),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                element_effects: vec![(0, ElementExpression::Constant(0))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v1", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                element_effects: vec![(v.id(), ElementExpression::Variable(1))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v1", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                element_effects: vec![
                    (v.id(), ElementExpression::Constant(0)),
                    (v.id(), ElementExpression::Constant(0)),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                element_resource_effects: vec![(0, ElementExpression::Constant(0))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v1", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                element_resource_effects: vec![(v.id(), ElementExpression::ResourceVariable(1))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v1", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                element_resource_effects: vec![
                    (v.id(), ElementExpression::Constant(0)),
                    (v.id(), ElementExpression::Constant(0)),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                integer_effects: vec![(0, IntegerExpression::Constant(0))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_integer_variable("v1", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                integer_effects: vec![(v.id(), IntegerExpression::Variable(1))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_integer_variable("v1", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                integer_effects: vec![
                    (v.id(), IntegerExpression::Constant(0)),
                    (v.id(), IntegerExpression::Constant(0)),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                integer_resource_effects: vec![(0, IntegerExpression::Constant(0))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v1", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                integer_resource_effects: vec![(v.id(), IntegerExpression::ResourceVariable(1))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v1", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                integer_resource_effects: vec![
                    (v.id(), IntegerExpression::Constant(0)),
                    (v.id(), IntegerExpression::Constant(0)),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                continuous_effects: vec![(0, ContinuousExpression::Constant(0.0))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_continuous_variable("v1", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                continuous_effects: vec![(v.id(), ContinuousExpression::Variable(1))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_continuous_variable("v1", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                continuous_effects: vec![
                    (v.id(), ContinuousExpression::Constant(0.0)),
                    (v.id(), ContinuousExpression::Constant(0.0)),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                continuous_resource_effects: vec![(0, ContinuousExpression::Constant(0.0))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v1", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                continuous_resource_effects: vec![(
                    v.id(),
                    ContinuousExpression::ResourceVariable(1),
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v1", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                continuous_resource_effects: vec![
                    (v.id(), ContinuousExpression::Constant(0.0)),
                    (v.id(), ContinuousExpression::Constant(0.0)),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_transition(transition).is_err());
    }

    #[test]
    fn add_forward_forced_transition_ok() {
        let mut model = Model {
            cost_type: CostType::Integer,
            ..Default::default()
        };
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("sv1", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = model.add_set_variable("sv2", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = model.add_vector_variable("vv1", ob, vec![]);
        assert!(v.is_ok());
        let v = model.add_vector_variable("vv2", ob, vec![]);
        assert!(v.is_ok());
        let v = model.add_element_variable("ev1", ob, 0);
        assert!(v.is_ok());
        let v = model.add_element_variable("ev2", ob, 0);
        assert!(v.is_ok());
        let v = model.add_element_resource_variable("erv1", ob, false, 0);
        assert!(v.is_ok());
        let v = model.add_element_resource_variable("erv2", ob, true, 0);
        assert!(v.is_ok());
        let v = model.add_integer_variable("iv1", 0);
        assert!(v.is_ok());
        let v = model.add_integer_variable("iv2", 0);
        assert!(v.is_ok());
        let v = model.add_integer_resource_variable("irv1", false, 0);
        assert!(v.is_ok());
        let v = model.add_integer_resource_variable("irv2", true, 0);
        assert!(v.is_ok());
        let v = model.add_continuous_variable("cv1", 0.0);
        assert!(v.is_ok());
        let v = model.add_continuous_variable("cv2", 0.0);
        assert!(v.is_ok());
        let v = model.add_continuous_resource_variable("crv1", false, 0.0);
        assert!(v.is_ok());
        let v = model.add_continuous_resource_variable("crv2", true, 0.0);
        assert!(v.is_ok());

        let transition = Transition {
            cost: CostExpression::Integer(IntegerExpression::Cost),
            ..Default::default()
        };
        let result = model.add_forward_forced_transition(transition);
        assert!(result.is_ok());
        assert_eq!(
            model.forward_forced_transitions,
            vec![Transition {
                cost: CostExpression::Integer(IntegerExpression::Cost),
                ..Default::default()
            }]
        );

        let transition = Transition {
            cost: CostExpression::Integer(IntegerExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(IntegerExpression::Constant(-1)),
            )),
            preconditions: vec![
                GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Eq,
                        Box::new(IntegerExpression::Variable(0)),
                        Box::new(IntegerExpression::Constant(0)),
                    ),
                    ..Default::default()
                },
                GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Eq,
                        Box::new(IntegerExpression::Variable(0)),
                        Box::new(IntegerExpression::UnaryOperation(
                            UnaryOperator::Abs,
                            Box::new(IntegerExpression::Constant(0)),
                        )),
                    ),
                    ..Default::default()
                },
            ],
            effect: Effect {
                set_effects: vec![
                    (
                        0,
                        SetExpression::Reference(ReferenceExpression::Constant(
                            Set::with_capacity(10),
                        )),
                    ),
                    (
                        1,
                        SetExpression::SetElementOperation(
                            SetElementOperator::Add,
                            ElementExpression::Constant(0),
                            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                                Set::with_capacity(10),
                            ))),
                        ),
                    ),
                ],
                vector_effects: vec![
                    (
                        0,
                        VectorExpression::Reference(ReferenceExpression::Constant(vec![1, 2])),
                    ),
                    (
                        1,
                        VectorExpression::Push(
                            ElementExpression::Constant(0),
                            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                                vec![1, 2],
                            ))),
                        ),
                    ),
                ],
                element_effects: vec![
                    (0, ElementExpression::Constant(0)),
                    (
                        1,
                        ElementExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ElementExpression::Constant(1)),
                            Box::new(ElementExpression::Constant(1)),
                        ),
                    ),
                ],
                integer_effects: vec![
                    (0, IntegerExpression::Constant(0)),
                    (
                        1,
                        IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::Constant(1)),
                            Box::new(IntegerExpression::Constant(1)),
                        ),
                    ),
                ],
                continuous_effects: vec![
                    (0, ContinuousExpression::Constant(0.0)),
                    (
                        1,
                        ContinuousExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ContinuousExpression::Constant(1.0)),
                            Box::new(ContinuousExpression::Constant(1.0)),
                        ),
                    ),
                ],
                element_resource_effects: vec![
                    (0, ElementExpression::Constant(0)),
                    (
                        1,
                        ElementExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ElementExpression::Constant(1)),
                            Box::new(ElementExpression::Constant(1)),
                        ),
                    ),
                ],
                integer_resource_effects: vec![
                    (0, IntegerExpression::Constant(0)),
                    (
                        1,
                        IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::Constant(1)),
                            Box::new(IntegerExpression::Constant(1)),
                        ),
                    ),
                ],
                continuous_resource_effects: vec![
                    (0, ContinuousExpression::Constant(0.0)),
                    (
                        1,
                        ContinuousExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ContinuousExpression::Constant(1.0)),
                            Box::new(ContinuousExpression::Constant(1.0)),
                        ),
                    ),
                ],
            },
            ..Default::default()
        };
        let result = model.add_forward_forced_transition(transition);
        assert!(result.is_ok());
        assert_eq!(
            model.forward_forced_transitions,
            vec![
                Transition {
                    cost: CostExpression::Integer(IntegerExpression::Cost),
                    ..Default::default()
                },
                Transition {
                    cost: CostExpression::Integer(IntegerExpression::Constant(1)),
                    preconditions: vec![
                        GroundedCondition {
                            condition: Condition::ComparisonI(
                                ComparisonOperator::Eq,
                                Box::new(IntegerExpression::Variable(0)),
                                Box::new(IntegerExpression::Constant(0)),
                            ),
                            ..Default::default()
                        },
                        GroundedCondition {
                            condition: Condition::ComparisonI(
                                ComparisonOperator::Eq,
                                Box::new(IntegerExpression::Variable(0)),
                                Box::new(IntegerExpression::Constant(0),),
                            ),
                            ..Default::default()
                        },
                    ],
                    effect: Effect {
                        set_effects: vec![
                            (
                                0,
                                SetExpression::Reference(ReferenceExpression::Constant(
                                    Set::with_capacity(10),
                                )),
                            ),
                            (
                                1,
                                SetExpression::Reference(ReferenceExpression::Constant({
                                    let mut set = Set::with_capacity(10);
                                    set.insert(0);
                                    set
                                },)),
                            ),
                        ],
                        vector_effects: vec![
                            (
                                0,
                                VectorExpression::Reference(ReferenceExpression::Constant(vec![
                                    1, 2
                                ])),
                            ),
                            (
                                1,
                                VectorExpression::Reference(ReferenceExpression::Constant(vec![
                                    1, 2, 0
                                ])),
                            ),
                        ],
                        element_effects: vec![
                            (0, ElementExpression::Constant(0)),
                            (1, ElementExpression::Constant(2)),
                        ],
                        integer_effects: vec![
                            (0, IntegerExpression::Constant(0)),
                            (1, IntegerExpression::Constant(2)),
                        ],
                        continuous_effects: vec![
                            (0, ContinuousExpression::Constant(0.0)),
                            (1, ContinuousExpression::Constant(2.0)),
                        ],
                        element_resource_effects: vec![
                            (0, ElementExpression::Constant(0)),
                            (1, ElementExpression::Constant(2)),
                        ],
                        integer_resource_effects: vec![
                            (0, IntegerExpression::Constant(0)),
                            (1, IntegerExpression::Constant(2)),
                        ],
                        continuous_resource_effects: vec![
                            (0, ContinuousExpression::Constant(0.0)),
                            (1, ContinuousExpression::Constant(2.0)),
                        ],
                    },
                    ..Default::default()
                }
            ]
        );

        let mut model = Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        };
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("sv1", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = model.add_set_variable("sv2", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = model.add_vector_variable("vv1", ob, vec![]);
        assert!(v.is_ok());
        let v = model.add_vector_variable("vv2", ob, vec![]);
        assert!(v.is_ok());
        let v = model.add_element_variable("ev1", ob, 0);
        assert!(v.is_ok());
        let v = model.add_element_variable("ev2", ob, 0);
        assert!(v.is_ok());
        let v = model.add_element_resource_variable("erv1", ob, false, 0);
        assert!(v.is_ok());
        let v = model.add_element_resource_variable("erv2", ob, true, 0);
        assert!(v.is_ok());
        let v = model.add_integer_variable("iv1", 0);
        assert!(v.is_ok());
        let v = model.add_integer_variable("iv2", 0);
        assert!(v.is_ok());
        let v = model.add_integer_resource_variable("irv1", false, 0);
        assert!(v.is_ok());
        let v = model.add_integer_resource_variable("irv2", true, 0);
        assert!(v.is_ok());
        let v = model.add_continuous_variable("cv1", 0.0);
        assert!(v.is_ok());
        let v = model.add_continuous_variable("cv2", 0.0);
        assert!(v.is_ok());
        let v = model.add_continuous_resource_variable("crv1", false, 0.0);
        assert!(v.is_ok());
        let v = model.add_continuous_resource_variable("crv2", true, 0.0);
        assert!(v.is_ok());

        let transition = Transition {
            cost: CostExpression::Continuous(ContinuousExpression::Cost),
            ..Default::default()
        };
        let result = model.add_forward_forced_transition(transition);
        assert!(result.is_ok());
        assert_eq!(
            model.forward_forced_transitions,
            vec![Transition {
                cost: CostExpression::Continuous(ContinuousExpression::Cost),
                ..Default::default()
            }]
        );

        let transition = Transition {
            cost: CostExpression::Continuous(ContinuousExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(ContinuousExpression::Constant(-1.0)),
            )),
            preconditions: vec![
                GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Eq,
                        Box::new(IntegerExpression::Variable(0)),
                        Box::new(IntegerExpression::Constant(0)),
                    ),
                    ..Default::default()
                },
                GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Eq,
                        Box::new(IntegerExpression::Variable(0)),
                        Box::new(IntegerExpression::UnaryOperation(
                            UnaryOperator::Abs,
                            Box::new(IntegerExpression::Constant(0)),
                        )),
                    ),
                    ..Default::default()
                },
            ],
            effect: Effect {
                set_effects: vec![
                    (
                        0,
                        SetExpression::Reference(ReferenceExpression::Constant(
                            Set::with_capacity(10),
                        )),
                    ),
                    (
                        1,
                        SetExpression::SetElementOperation(
                            SetElementOperator::Add,
                            ElementExpression::Constant(0),
                            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                                Set::with_capacity(10),
                            ))),
                        ),
                    ),
                ],
                vector_effects: vec![
                    (
                        0,
                        VectorExpression::Reference(ReferenceExpression::Constant(vec![1, 2])),
                    ),
                    (
                        1,
                        VectorExpression::Push(
                            ElementExpression::Constant(0),
                            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                                vec![1, 2],
                            ))),
                        ),
                    ),
                ],
                element_effects: vec![
                    (0, ElementExpression::Constant(0)),
                    (
                        1,
                        ElementExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ElementExpression::Constant(1)),
                            Box::new(ElementExpression::Constant(1)),
                        ),
                    ),
                ],
                integer_effects: vec![
                    (0, IntegerExpression::Constant(0)),
                    (
                        1,
                        IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::Constant(1)),
                            Box::new(IntegerExpression::Constant(1)),
                        ),
                    ),
                ],
                continuous_effects: vec![
                    (0, ContinuousExpression::Constant(0.0)),
                    (
                        1,
                        ContinuousExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ContinuousExpression::Constant(1.0)),
                            Box::new(ContinuousExpression::Constant(1.0)),
                        ),
                    ),
                ],
                element_resource_effects: vec![
                    (0, ElementExpression::Constant(0)),
                    (
                        1,
                        ElementExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ElementExpression::Constant(1)),
                            Box::new(ElementExpression::Constant(1)),
                        ),
                    ),
                ],
                integer_resource_effects: vec![
                    (0, IntegerExpression::Constant(0)),
                    (
                        1,
                        IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::Constant(1)),
                            Box::new(IntegerExpression::Constant(1)),
                        ),
                    ),
                ],
                continuous_resource_effects: vec![
                    (0, ContinuousExpression::Constant(0.0)),
                    (
                        1,
                        ContinuousExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ContinuousExpression::Constant(1.0)),
                            Box::new(ContinuousExpression::Constant(1.0)),
                        ),
                    ),
                ],
            },
            ..Default::default()
        };
        let result = model.add_forward_forced_transition(transition);
        assert!(result.is_ok());
        assert_eq!(
            model.forward_forced_transitions,
            vec![
                Transition {
                    cost: CostExpression::Continuous(ContinuousExpression::Cost),
                    ..Default::default()
                },
                Transition {
                    cost: CostExpression::Continuous(ContinuousExpression::Constant(1.0)),
                    preconditions: vec![
                        GroundedCondition {
                            condition: Condition::ComparisonI(
                                ComparisonOperator::Eq,
                                Box::new(IntegerExpression::Variable(0)),
                                Box::new(IntegerExpression::Constant(0)),
                            ),
                            ..Default::default()
                        },
                        GroundedCondition {
                            condition: Condition::ComparisonI(
                                ComparisonOperator::Eq,
                                Box::new(IntegerExpression::Variable(0)),
                                Box::new(IntegerExpression::Constant(0),),
                            ),
                            ..Default::default()
                        },
                    ],
                    effect: Effect {
                        set_effects: vec![
                            (
                                0,
                                SetExpression::Reference(ReferenceExpression::Constant(
                                    Set::with_capacity(10),
                                )),
                            ),
                            (
                                1,
                                SetExpression::Reference(ReferenceExpression::Constant({
                                    let mut set = Set::with_capacity(10);
                                    set.insert(0);
                                    set
                                },)),
                            ),
                        ],
                        vector_effects: vec![
                            (
                                0,
                                VectorExpression::Reference(ReferenceExpression::Constant(vec![
                                    1, 2
                                ])),
                            ),
                            (
                                1,
                                VectorExpression::Reference(ReferenceExpression::Constant(vec![
                                    1, 2, 0
                                ])),
                            ),
                        ],
                        element_effects: vec![
                            (0, ElementExpression::Constant(0)),
                            (1, ElementExpression::Constant(2)),
                        ],
                        integer_effects: vec![
                            (0, IntegerExpression::Constant(0)),
                            (1, IntegerExpression::Constant(2)),
                        ],
                        continuous_effects: vec![
                            (0, ContinuousExpression::Constant(0.0)),
                            (1, ContinuousExpression::Constant(2.0)),
                        ],
                        element_resource_effects: vec![
                            (0, ElementExpression::Constant(0)),
                            (1, ElementExpression::Constant(2)),
                        ],
                        integer_resource_effects: vec![
                            (0, IntegerExpression::Constant(0)),
                            (1, IntegerExpression::Constant(2)),
                        ],
                        continuous_resource_effects: vec![
                            (0, ContinuousExpression::Constant(0.0)),
                            (1, ContinuousExpression::Constant(2.0)),
                        ],
                    },
                    ..Default::default()
                }
            ]
        );
    }

    #[test]
    fn add_forward_forced_transition_err() {
        let mut model = Model {
            cost_type: CostType::Integer,
            ..Default::default()
        };
        let transition = Transition {
            cost: CostExpression::Continuous(ContinuousExpression::Cost),
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model {
            cost_type: CostType::Integer,
            ..Default::default()
        };
        let transition = Transition {
            cost: CostExpression::Integer(IntegerExpression::Variable(0)),
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        };
        let transition = Transition {
            cost: CostExpression::Integer(IntegerExpression::Cost),
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        };
        let transition = Transition {
            cost: CostExpression::Continuous(ContinuousExpression::Variable(0)),
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            elements_in_set_variable: vec![(0, 0)],
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            elements_in_set_variable: vec![(v.id(), 11)],
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            elements_in_vector_variable: vec![(0, 0)],
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_vector_variable("v", ob, vec![0, 1]);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            elements_in_vector_variable: vec![(v.id(), 11)],
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            preconditions: vec![GroundedCondition {
                elements_in_set_variable: vec![(0, 0)],
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            preconditions: vec![GroundedCondition {
                elements_in_set_variable: vec![(v.id(), 11)],
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            preconditions: vec![GroundedCondition {
                elements_in_vector_variable: vec![(0, 0)],
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_vector_variable("v", ob, vec![0, 1]);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            preconditions: vec![GroundedCondition {
                elements_in_vector_variable: vec![(v.id(), 11)],
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            preconditions: vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Eq,
                    Box::new(IntegerExpression::Variable(0)),
                    Box::new(IntegerExpression::Constant(0)),
                ),
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                set_effects: vec![(
                    0,
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v1", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                set_effects: vec![(
                    v.id(),
                    SetExpression::Reference(ReferenceExpression::Variable(1)),
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v1", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                set_effects: vec![
                    (
                        v.id(),
                        SetExpression::Reference(ReferenceExpression::Constant(
                            Set::with_capacity(10),
                        )),
                    ),
                    (
                        v.id(),
                        SetExpression::Reference(ReferenceExpression::Constant(
                            Set::with_capacity(10),
                        )),
                    ),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                vector_effects: vec![(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_vector_variable("v1", ob, vec![0, 1]);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                vector_effects: vec![(
                    v.id(),
                    VectorExpression::Reference(ReferenceExpression::Variable(1)),
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_vector_variable("v1", ob, vec![0, 1]);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                vector_effects: vec![
                    (
                        v.id(),
                        VectorExpression::Reference(ReferenceExpression::Variable(1)),
                    ),
                    (
                        v.id(),
                        VectorExpression::Reference(ReferenceExpression::Variable(1)),
                    ),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                element_effects: vec![(0, ElementExpression::Constant(0))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v1", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                element_effects: vec![(v.id(), ElementExpression::Variable(1))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v1", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                element_effects: vec![
                    (v.id(), ElementExpression::Constant(0)),
                    (v.id(), ElementExpression::Constant(0)),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                element_resource_effects: vec![(0, ElementExpression::Constant(0))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v1", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                element_resource_effects: vec![(v.id(), ElementExpression::ResourceVariable(1))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v1", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                element_resource_effects: vec![
                    (v.id(), ElementExpression::Constant(0)),
                    (v.id(), ElementExpression::Constant(0)),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                integer_effects: vec![(0, IntegerExpression::Constant(0))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_integer_variable("v1", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                integer_effects: vec![(v.id(), IntegerExpression::Variable(1))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_integer_variable("v1", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                integer_effects: vec![
                    (v.id(), IntegerExpression::Constant(0)),
                    (v.id(), IntegerExpression::Constant(0)),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                integer_resource_effects: vec![(0, IntegerExpression::Constant(0))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v1", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                integer_resource_effects: vec![(v.id(), IntegerExpression::ResourceVariable(1))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v1", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                integer_resource_effects: vec![
                    (v.id(), IntegerExpression::Constant(0)),
                    (v.id(), IntegerExpression::Constant(0)),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                continuous_effects: vec![(0, ContinuousExpression::Constant(0.0))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_continuous_variable("v1", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                continuous_effects: vec![(v.id(), ContinuousExpression::Variable(1))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_continuous_variable("v1", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                continuous_effects: vec![
                    (v.id(), ContinuousExpression::Constant(0.0)),
                    (v.id(), ContinuousExpression::Constant(0.0)),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                continuous_resource_effects: vec![(0, ContinuousExpression::Constant(0.0))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v1", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                continuous_resource_effects: vec![(
                    v.id(),
                    ContinuousExpression::ResourceVariable(1),
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v1", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                continuous_resource_effects: vec![
                    (v.id(), ContinuousExpression::Constant(0.0)),
                    (v.id(), ContinuousExpression::Constant(0.0)),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_forward_forced_transition(transition).is_err());
    }

    #[test]
    fn add_backward_transition_ok() {
        let mut model = Model {
            cost_type: CostType::Integer,
            ..Default::default()
        };
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("sv1", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = model.add_set_variable("sv2", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = model.add_vector_variable("vv1", ob, vec![]);
        assert!(v.is_ok());
        let v = model.add_vector_variable("vv2", ob, vec![]);
        assert!(v.is_ok());
        let v = model.add_element_variable("ev1", ob, 0);
        assert!(v.is_ok());
        let v = model.add_element_variable("ev2", ob, 0);
        assert!(v.is_ok());
        let v = model.add_element_resource_variable("erv1", ob, false, 0);
        assert!(v.is_ok());
        let v = model.add_element_resource_variable("erv2", ob, true, 0);
        assert!(v.is_ok());
        let v = model.add_integer_variable("iv1", 0);
        assert!(v.is_ok());
        let v = model.add_integer_variable("iv2", 0);
        assert!(v.is_ok());
        let v = model.add_integer_resource_variable("irv1", false, 0);
        assert!(v.is_ok());
        let v = model.add_integer_resource_variable("irv2", true, 0);
        assert!(v.is_ok());
        let v = model.add_continuous_variable("cv1", 0.0);
        assert!(v.is_ok());
        let v = model.add_continuous_variable("cv2", 0.0);
        assert!(v.is_ok());
        let v = model.add_continuous_resource_variable("crv1", false, 0.0);
        assert!(v.is_ok());
        let v = model.add_continuous_resource_variable("crv2", true, 0.0);
        assert!(v.is_ok());

        let transition = Transition {
            cost: CostExpression::Integer(IntegerExpression::Cost),
            ..Default::default()
        };
        let result = model.add_backward_transition(transition);
        assert!(result.is_ok());
        assert_eq!(
            model.backward_transitions,
            vec![Transition {
                cost: CostExpression::Integer(IntegerExpression::Cost),
                ..Default::default()
            }]
        );

        let transition = Transition {
            cost: CostExpression::Integer(IntegerExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(IntegerExpression::Constant(-1)),
            )),
            preconditions: vec![
                GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Eq,
                        Box::new(IntegerExpression::Variable(0)),
                        Box::new(IntegerExpression::Constant(0)),
                    ),
                    ..Default::default()
                },
                GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Eq,
                        Box::new(IntegerExpression::Variable(0)),
                        Box::new(IntegerExpression::UnaryOperation(
                            UnaryOperator::Abs,
                            Box::new(IntegerExpression::Constant(0)),
                        )),
                    ),
                    ..Default::default()
                },
            ],
            effect: Effect {
                set_effects: vec![
                    (
                        0,
                        SetExpression::Reference(ReferenceExpression::Constant(
                            Set::with_capacity(10),
                        )),
                    ),
                    (
                        1,
                        SetExpression::SetElementOperation(
                            SetElementOperator::Add,
                            ElementExpression::Constant(0),
                            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                                Set::with_capacity(10),
                            ))),
                        ),
                    ),
                ],
                vector_effects: vec![
                    (
                        0,
                        VectorExpression::Reference(ReferenceExpression::Constant(vec![1, 2])),
                    ),
                    (
                        1,
                        VectorExpression::Push(
                            ElementExpression::Constant(0),
                            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                                vec![1, 2],
                            ))),
                        ),
                    ),
                ],
                element_effects: vec![
                    (0, ElementExpression::Constant(0)),
                    (
                        1,
                        ElementExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ElementExpression::Constant(1)),
                            Box::new(ElementExpression::Constant(1)),
                        ),
                    ),
                ],
                integer_effects: vec![
                    (0, IntegerExpression::Constant(0)),
                    (
                        1,
                        IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::Constant(1)),
                            Box::new(IntegerExpression::Constant(1)),
                        ),
                    ),
                ],
                continuous_effects: vec![
                    (0, ContinuousExpression::Constant(0.0)),
                    (
                        1,
                        ContinuousExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ContinuousExpression::Constant(1.0)),
                            Box::new(ContinuousExpression::Constant(1.0)),
                        ),
                    ),
                ],
                element_resource_effects: vec![
                    (0, ElementExpression::Constant(0)),
                    (
                        1,
                        ElementExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ElementExpression::Constant(1)),
                            Box::new(ElementExpression::Constant(1)),
                        ),
                    ),
                ],
                integer_resource_effects: vec![
                    (0, IntegerExpression::Constant(0)),
                    (
                        1,
                        IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::Constant(1)),
                            Box::new(IntegerExpression::Constant(1)),
                        ),
                    ),
                ],
                continuous_resource_effects: vec![
                    (0, ContinuousExpression::Constant(0.0)),
                    (
                        1,
                        ContinuousExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ContinuousExpression::Constant(1.0)),
                            Box::new(ContinuousExpression::Constant(1.0)),
                        ),
                    ),
                ],
            },
            ..Default::default()
        };
        let result = model.add_backward_transition(transition);
        assert!(result.is_ok());
        assert_eq!(
            model.backward_transitions,
            vec![
                Transition {
                    cost: CostExpression::Integer(IntegerExpression::Cost),
                    ..Default::default()
                },
                Transition {
                    cost: CostExpression::Integer(IntegerExpression::Constant(1)),
                    preconditions: vec![
                        GroundedCondition {
                            condition: Condition::ComparisonI(
                                ComparisonOperator::Eq,
                                Box::new(IntegerExpression::Variable(0)),
                                Box::new(IntegerExpression::Constant(0)),
                            ),
                            ..Default::default()
                        },
                        GroundedCondition {
                            condition: Condition::ComparisonI(
                                ComparisonOperator::Eq,
                                Box::new(IntegerExpression::Variable(0)),
                                Box::new(IntegerExpression::Constant(0),),
                            ),
                            ..Default::default()
                        },
                    ],
                    effect: Effect {
                        set_effects: vec![
                            (
                                0,
                                SetExpression::Reference(ReferenceExpression::Constant(
                                    Set::with_capacity(10),
                                )),
                            ),
                            (
                                1,
                                SetExpression::Reference(ReferenceExpression::Constant({
                                    let mut set = Set::with_capacity(10);
                                    set.insert(0);
                                    set
                                },)),
                            ),
                        ],
                        vector_effects: vec![
                            (
                                0,
                                VectorExpression::Reference(ReferenceExpression::Constant(vec![
                                    1, 2
                                ])),
                            ),
                            (
                                1,
                                VectorExpression::Reference(ReferenceExpression::Constant(vec![
                                    1, 2, 0
                                ])),
                            ),
                        ],
                        element_effects: vec![
                            (0, ElementExpression::Constant(0)),
                            (1, ElementExpression::Constant(2)),
                        ],
                        integer_effects: vec![
                            (0, IntegerExpression::Constant(0)),
                            (1, IntegerExpression::Constant(2)),
                        ],
                        continuous_effects: vec![
                            (0, ContinuousExpression::Constant(0.0)),
                            (1, ContinuousExpression::Constant(2.0)),
                        ],
                        element_resource_effects: vec![
                            (0, ElementExpression::Constant(0)),
                            (1, ElementExpression::Constant(2)),
                        ],
                        integer_resource_effects: vec![
                            (0, IntegerExpression::Constant(0)),
                            (1, IntegerExpression::Constant(2)),
                        ],
                        continuous_resource_effects: vec![
                            (0, ContinuousExpression::Constant(0.0)),
                            (1, ContinuousExpression::Constant(2.0)),
                        ],
                    },
                    ..Default::default()
                }
            ]
        );

        let mut model = Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        };
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("sv1", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = model.add_set_variable("sv2", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = model.add_vector_variable("vv1", ob, vec![]);
        assert!(v.is_ok());
        let v = model.add_vector_variable("vv2", ob, vec![]);
        assert!(v.is_ok());
        let v = model.add_element_variable("ev1", ob, 0);
        assert!(v.is_ok());
        let v = model.add_element_variable("ev2", ob, 0);
        assert!(v.is_ok());
        let v = model.add_element_resource_variable("erv1", ob, false, 0);
        assert!(v.is_ok());
        let v = model.add_element_resource_variable("erv2", ob, true, 0);
        assert!(v.is_ok());
        let v = model.add_integer_variable("iv1", 0);
        assert!(v.is_ok());
        let v = model.add_integer_variable("iv2", 0);
        assert!(v.is_ok());
        let v = model.add_integer_resource_variable("irv1", false, 0);
        assert!(v.is_ok());
        let v = model.add_integer_resource_variable("irv2", true, 0);
        assert!(v.is_ok());
        let v = model.add_continuous_variable("cv1", 0.0);
        assert!(v.is_ok());
        let v = model.add_continuous_variable("cv2", 0.0);
        assert!(v.is_ok());
        let v = model.add_continuous_resource_variable("crv1", false, 0.0);
        assert!(v.is_ok());
        let v = model.add_continuous_resource_variable("crv2", true, 0.0);
        assert!(v.is_ok());

        let transition = Transition {
            cost: CostExpression::Continuous(ContinuousExpression::Cost),
            ..Default::default()
        };
        let result = model.add_backward_transition(transition);
        assert!(result.is_ok());
        assert_eq!(
            model.backward_transitions,
            vec![Transition {
                cost: CostExpression::Continuous(ContinuousExpression::Cost),
                ..Default::default()
            }]
        );

        let transition = Transition {
            cost: CostExpression::Continuous(ContinuousExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(ContinuousExpression::Constant(-1.0)),
            )),
            preconditions: vec![
                GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Eq,
                        Box::new(IntegerExpression::Variable(0)),
                        Box::new(IntegerExpression::Constant(0)),
                    ),
                    ..Default::default()
                },
                GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Eq,
                        Box::new(IntegerExpression::Variable(0)),
                        Box::new(IntegerExpression::UnaryOperation(
                            UnaryOperator::Abs,
                            Box::new(IntegerExpression::Constant(0)),
                        )),
                    ),
                    ..Default::default()
                },
            ],
            effect: Effect {
                set_effects: vec![
                    (
                        0,
                        SetExpression::Reference(ReferenceExpression::Constant(
                            Set::with_capacity(10),
                        )),
                    ),
                    (
                        1,
                        SetExpression::SetElementOperation(
                            SetElementOperator::Add,
                            ElementExpression::Constant(0),
                            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                                Set::with_capacity(10),
                            ))),
                        ),
                    ),
                ],
                vector_effects: vec![
                    (
                        0,
                        VectorExpression::Reference(ReferenceExpression::Constant(vec![1, 2])),
                    ),
                    (
                        1,
                        VectorExpression::Push(
                            ElementExpression::Constant(0),
                            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                                vec![1, 2],
                            ))),
                        ),
                    ),
                ],
                element_effects: vec![
                    (0, ElementExpression::Constant(0)),
                    (
                        1,
                        ElementExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ElementExpression::Constant(1)),
                            Box::new(ElementExpression::Constant(1)),
                        ),
                    ),
                ],
                integer_effects: vec![
                    (0, IntegerExpression::Constant(0)),
                    (
                        1,
                        IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::Constant(1)),
                            Box::new(IntegerExpression::Constant(1)),
                        ),
                    ),
                ],
                continuous_effects: vec![
                    (0, ContinuousExpression::Constant(0.0)),
                    (
                        1,
                        ContinuousExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ContinuousExpression::Constant(1.0)),
                            Box::new(ContinuousExpression::Constant(1.0)),
                        ),
                    ),
                ],
                element_resource_effects: vec![
                    (0, ElementExpression::Constant(0)),
                    (
                        1,
                        ElementExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ElementExpression::Constant(1)),
                            Box::new(ElementExpression::Constant(1)),
                        ),
                    ),
                ],
                integer_resource_effects: vec![
                    (0, IntegerExpression::Constant(0)),
                    (
                        1,
                        IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::Constant(1)),
                            Box::new(IntegerExpression::Constant(1)),
                        ),
                    ),
                ],
                continuous_resource_effects: vec![
                    (0, ContinuousExpression::Constant(0.0)),
                    (
                        1,
                        ContinuousExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ContinuousExpression::Constant(1.0)),
                            Box::new(ContinuousExpression::Constant(1.0)),
                        ),
                    ),
                ],
            },
            ..Default::default()
        };
        let result = model.add_backward_transition(transition);
        assert!(result.is_ok());
        assert_eq!(
            model.backward_transitions,
            vec![
                Transition {
                    cost: CostExpression::Continuous(ContinuousExpression::Cost),
                    ..Default::default()
                },
                Transition {
                    cost: CostExpression::Continuous(ContinuousExpression::Constant(1.0)),
                    preconditions: vec![
                        GroundedCondition {
                            condition: Condition::ComparisonI(
                                ComparisonOperator::Eq,
                                Box::new(IntegerExpression::Variable(0)),
                                Box::new(IntegerExpression::Constant(0)),
                            ),
                            ..Default::default()
                        },
                        GroundedCondition {
                            condition: Condition::ComparisonI(
                                ComparisonOperator::Eq,
                                Box::new(IntegerExpression::Variable(0)),
                                Box::new(IntegerExpression::Constant(0),),
                            ),
                            ..Default::default()
                        },
                    ],
                    effect: Effect {
                        set_effects: vec![
                            (
                                0,
                                SetExpression::Reference(ReferenceExpression::Constant(
                                    Set::with_capacity(10),
                                )),
                            ),
                            (
                                1,
                                SetExpression::Reference(ReferenceExpression::Constant({
                                    let mut set = Set::with_capacity(10);
                                    set.insert(0);
                                    set
                                },)),
                            ),
                        ],
                        vector_effects: vec![
                            (
                                0,
                                VectorExpression::Reference(ReferenceExpression::Constant(vec![
                                    1, 2
                                ])),
                            ),
                            (
                                1,
                                VectorExpression::Reference(ReferenceExpression::Constant(vec![
                                    1, 2, 0
                                ])),
                            ),
                        ],
                        element_effects: vec![
                            (0, ElementExpression::Constant(0)),
                            (1, ElementExpression::Constant(2)),
                        ],
                        integer_effects: vec![
                            (0, IntegerExpression::Constant(0)),
                            (1, IntegerExpression::Constant(2)),
                        ],
                        continuous_effects: vec![
                            (0, ContinuousExpression::Constant(0.0)),
                            (1, ContinuousExpression::Constant(2.0)),
                        ],
                        element_resource_effects: vec![
                            (0, ElementExpression::Constant(0)),
                            (1, ElementExpression::Constant(2)),
                        ],
                        integer_resource_effects: vec![
                            (0, IntegerExpression::Constant(0)),
                            (1, IntegerExpression::Constant(2)),
                        ],
                        continuous_resource_effects: vec![
                            (0, ContinuousExpression::Constant(0.0)),
                            (1, ContinuousExpression::Constant(2.0)),
                        ],
                    },
                    ..Default::default()
                }
            ]
        );
    }

    #[test]
    fn add_backward_transition_err() {
        let mut model = Model {
            cost_type: CostType::Integer,
            ..Default::default()
        };
        let transition = Transition {
            cost: CostExpression::Continuous(ContinuousExpression::Cost),
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model {
            cost_type: CostType::Integer,
            ..Default::default()
        };
        let transition = Transition {
            cost: CostExpression::Integer(IntegerExpression::Variable(0)),
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        };
        let transition = Transition {
            cost: CostExpression::Integer(IntegerExpression::Cost),
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        };
        let transition = Transition {
            cost: CostExpression::Continuous(ContinuousExpression::Variable(0)),
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            elements_in_set_variable: vec![(0, 0)],
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            elements_in_set_variable: vec![(v.id(), 11)],
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            elements_in_vector_variable: vec![(0, 0)],
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_vector_variable("v", ob, vec![0, 1]);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            elements_in_vector_variable: vec![(v.id(), 11)],
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            preconditions: vec![GroundedCondition {
                elements_in_set_variable: vec![(0, 0)],
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            preconditions: vec![GroundedCondition {
                elements_in_set_variable: vec![(v.id(), 11)],
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            preconditions: vec![GroundedCondition {
                elements_in_vector_variable: vec![(0, 0)],
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_vector_variable("v", ob, vec![0, 1]);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            preconditions: vec![GroundedCondition {
                elements_in_vector_variable: vec![(v.id(), 11)],
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            preconditions: vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Eq,
                    Box::new(IntegerExpression::Variable(0)),
                    Box::new(IntegerExpression::Constant(0)),
                ),
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                set_effects: vec![(
                    0,
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v1", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                set_effects: vec![(
                    v.id(),
                    SetExpression::Reference(ReferenceExpression::Variable(1)),
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v1", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                set_effects: vec![
                    (
                        v.id(),
                        SetExpression::Reference(ReferenceExpression::Constant(
                            Set::with_capacity(10),
                        )),
                    ),
                    (
                        v.id(),
                        SetExpression::Reference(ReferenceExpression::Constant(
                            Set::with_capacity(10),
                        )),
                    ),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                vector_effects: vec![(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_vector_variable("v1", ob, vec![0, 1]);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                vector_effects: vec![(
                    v.id(),
                    VectorExpression::Reference(ReferenceExpression::Variable(1)),
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_vector_variable("v1", ob, vec![0, 1]);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                vector_effects: vec![
                    (
                        v.id(),
                        VectorExpression::Reference(ReferenceExpression::Variable(1)),
                    ),
                    (
                        v.id(),
                        VectorExpression::Reference(ReferenceExpression::Variable(1)),
                    ),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                element_effects: vec![(0, ElementExpression::Constant(0))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v1", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                element_effects: vec![(v.id(), ElementExpression::Variable(1))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v1", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                element_effects: vec![
                    (v.id(), ElementExpression::Constant(0)),
                    (v.id(), ElementExpression::Constant(0)),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                element_resource_effects: vec![(0, ElementExpression::Constant(0))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v1", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                element_resource_effects: vec![(v.id(), ElementExpression::ResourceVariable(1))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v1", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                element_resource_effects: vec![
                    (v.id(), ElementExpression::Constant(0)),
                    (v.id(), ElementExpression::Constant(0)),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                integer_effects: vec![(0, IntegerExpression::Constant(0))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_integer_variable("v1", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                integer_effects: vec![(v.id(), IntegerExpression::Variable(1))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_integer_variable("v1", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                integer_effects: vec![
                    (v.id(), IntegerExpression::Constant(0)),
                    (v.id(), IntegerExpression::Constant(0)),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                integer_resource_effects: vec![(0, IntegerExpression::Constant(0))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v1", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                integer_resource_effects: vec![(v.id(), IntegerExpression::ResourceVariable(1))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v1", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                integer_resource_effects: vec![
                    (v.id(), IntegerExpression::Constant(0)),
                    (v.id(), IntegerExpression::Constant(0)),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                continuous_effects: vec![(0, ContinuousExpression::Constant(0.0))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_continuous_variable("v1", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                continuous_effects: vec![(v.id(), ContinuousExpression::Variable(1))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_continuous_variable("v1", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                continuous_effects: vec![
                    (v.id(), ContinuousExpression::Constant(0.0)),
                    (v.id(), ContinuousExpression::Constant(0.0)),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                continuous_resource_effects: vec![(0, ContinuousExpression::Constant(0.0))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v1", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                continuous_resource_effects: vec![(
                    v.id(),
                    ContinuousExpression::ResourceVariable(1),
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v1", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                continuous_resource_effects: vec![
                    (v.id(), ContinuousExpression::Constant(0.0)),
                    (v.id(), ContinuousExpression::Constant(0.0)),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_transition(transition).is_err());
    }

    #[test]
    fn add_backward_forced_transition_ok() {
        let mut model = Model {
            cost_type: CostType::Integer,
            ..Default::default()
        };
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("sv1", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = model.add_set_variable("sv2", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = model.add_vector_variable("vv1", ob, vec![]);
        assert!(v.is_ok());
        let v = model.add_vector_variable("vv2", ob, vec![]);
        assert!(v.is_ok());
        let v = model.add_element_variable("ev1", ob, 0);
        assert!(v.is_ok());
        let v = model.add_element_variable("ev2", ob, 0);
        assert!(v.is_ok());
        let v = model.add_element_resource_variable("erv1", ob, false, 0);
        assert!(v.is_ok());
        let v = model.add_element_resource_variable("erv2", ob, true, 0);
        assert!(v.is_ok());
        let v = model.add_integer_variable("iv1", 0);
        assert!(v.is_ok());
        let v = model.add_integer_variable("iv2", 0);
        assert!(v.is_ok());
        let v = model.add_integer_resource_variable("irv1", false, 0);
        assert!(v.is_ok());
        let v = model.add_integer_resource_variable("irv2", true, 0);
        assert!(v.is_ok());
        let v = model.add_continuous_variable("cv1", 0.0);
        assert!(v.is_ok());
        let v = model.add_continuous_variable("cv2", 0.0);
        assert!(v.is_ok());
        let v = model.add_continuous_resource_variable("crv1", false, 0.0);
        assert!(v.is_ok());
        let v = model.add_continuous_resource_variable("crv2", true, 0.0);
        assert!(v.is_ok());

        let transition = Transition {
            cost: CostExpression::Integer(IntegerExpression::Cost),
            ..Default::default()
        };
        let result = model.add_backward_forced_transition(transition);
        assert!(result.is_ok());
        assert_eq!(
            model.backward_forced_transitions,
            vec![Transition {
                cost: CostExpression::Integer(IntegerExpression::Cost),
                ..Default::default()
            }]
        );

        let transition = Transition {
            cost: CostExpression::Integer(IntegerExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(IntegerExpression::Constant(-1)),
            )),
            preconditions: vec![
                GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Eq,
                        Box::new(IntegerExpression::Variable(0)),
                        Box::new(IntegerExpression::Constant(0)),
                    ),
                    ..Default::default()
                },
                GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Eq,
                        Box::new(IntegerExpression::Variable(0)),
                        Box::new(IntegerExpression::UnaryOperation(
                            UnaryOperator::Abs,
                            Box::new(IntegerExpression::Constant(0)),
                        )),
                    ),
                    ..Default::default()
                },
            ],
            effect: Effect {
                set_effects: vec![
                    (
                        0,
                        SetExpression::Reference(ReferenceExpression::Constant(
                            Set::with_capacity(10),
                        )),
                    ),
                    (
                        1,
                        SetExpression::SetElementOperation(
                            SetElementOperator::Add,
                            ElementExpression::Constant(0),
                            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                                Set::with_capacity(10),
                            ))),
                        ),
                    ),
                ],
                vector_effects: vec![
                    (
                        0,
                        VectorExpression::Reference(ReferenceExpression::Constant(vec![1, 2])),
                    ),
                    (
                        1,
                        VectorExpression::Push(
                            ElementExpression::Constant(0),
                            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                                vec![1, 2],
                            ))),
                        ),
                    ),
                ],
                element_effects: vec![
                    (0, ElementExpression::Constant(0)),
                    (
                        1,
                        ElementExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ElementExpression::Constant(1)),
                            Box::new(ElementExpression::Constant(1)),
                        ),
                    ),
                ],
                integer_effects: vec![
                    (0, IntegerExpression::Constant(0)),
                    (
                        1,
                        IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::Constant(1)),
                            Box::new(IntegerExpression::Constant(1)),
                        ),
                    ),
                ],
                continuous_effects: vec![
                    (0, ContinuousExpression::Constant(0.0)),
                    (
                        1,
                        ContinuousExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ContinuousExpression::Constant(1.0)),
                            Box::new(ContinuousExpression::Constant(1.0)),
                        ),
                    ),
                ],
                element_resource_effects: vec![
                    (0, ElementExpression::Constant(0)),
                    (
                        1,
                        ElementExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ElementExpression::Constant(1)),
                            Box::new(ElementExpression::Constant(1)),
                        ),
                    ),
                ],
                integer_resource_effects: vec![
                    (0, IntegerExpression::Constant(0)),
                    (
                        1,
                        IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::Constant(1)),
                            Box::new(IntegerExpression::Constant(1)),
                        ),
                    ),
                ],
                continuous_resource_effects: vec![
                    (0, ContinuousExpression::Constant(0.0)),
                    (
                        1,
                        ContinuousExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ContinuousExpression::Constant(1.0)),
                            Box::new(ContinuousExpression::Constant(1.0)),
                        ),
                    ),
                ],
            },
            ..Default::default()
        };
        let result = model.add_backward_forced_transition(transition);
        assert!(result.is_ok());
        assert_eq!(
            model.backward_forced_transitions,
            vec![
                Transition {
                    cost: CostExpression::Integer(IntegerExpression::Cost),
                    ..Default::default()
                },
                Transition {
                    cost: CostExpression::Integer(IntegerExpression::Constant(1)),
                    preconditions: vec![
                        GroundedCondition {
                            condition: Condition::ComparisonI(
                                ComparisonOperator::Eq,
                                Box::new(IntegerExpression::Variable(0)),
                                Box::new(IntegerExpression::Constant(0)),
                            ),
                            ..Default::default()
                        },
                        GroundedCondition {
                            condition: Condition::ComparisonI(
                                ComparisonOperator::Eq,
                                Box::new(IntegerExpression::Variable(0)),
                                Box::new(IntegerExpression::Constant(0),),
                            ),
                            ..Default::default()
                        },
                    ],
                    effect: Effect {
                        set_effects: vec![
                            (
                                0,
                                SetExpression::Reference(ReferenceExpression::Constant(
                                    Set::with_capacity(10),
                                )),
                            ),
                            (
                                1,
                                SetExpression::Reference(ReferenceExpression::Constant({
                                    let mut set = Set::with_capacity(10);
                                    set.insert(0);
                                    set
                                },)),
                            ),
                        ],
                        vector_effects: vec![
                            (
                                0,
                                VectorExpression::Reference(ReferenceExpression::Constant(vec![
                                    1, 2
                                ])),
                            ),
                            (
                                1,
                                VectorExpression::Reference(ReferenceExpression::Constant(vec![
                                    1, 2, 0
                                ])),
                            ),
                        ],
                        element_effects: vec![
                            (0, ElementExpression::Constant(0)),
                            (1, ElementExpression::Constant(2)),
                        ],
                        integer_effects: vec![
                            (0, IntegerExpression::Constant(0)),
                            (1, IntegerExpression::Constant(2)),
                        ],
                        continuous_effects: vec![
                            (0, ContinuousExpression::Constant(0.0)),
                            (1, ContinuousExpression::Constant(2.0)),
                        ],
                        element_resource_effects: vec![
                            (0, ElementExpression::Constant(0)),
                            (1, ElementExpression::Constant(2)),
                        ],
                        integer_resource_effects: vec![
                            (0, IntegerExpression::Constant(0)),
                            (1, IntegerExpression::Constant(2)),
                        ],
                        continuous_resource_effects: vec![
                            (0, ContinuousExpression::Constant(0.0)),
                            (1, ContinuousExpression::Constant(2.0)),
                        ],
                    },
                    ..Default::default()
                }
            ]
        );

        let mut model = Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        };
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("sv1", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = model.add_set_variable("sv2", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = model.add_vector_variable("vv1", ob, vec![]);
        assert!(v.is_ok());
        let v = model.add_vector_variable("vv2", ob, vec![]);
        assert!(v.is_ok());
        let v = model.add_element_variable("ev1", ob, 0);
        assert!(v.is_ok());
        let v = model.add_element_variable("ev2", ob, 0);
        assert!(v.is_ok());
        let v = model.add_element_resource_variable("erv1", ob, false, 0);
        assert!(v.is_ok());
        let v = model.add_element_resource_variable("erv2", ob, true, 0);
        assert!(v.is_ok());
        let v = model.add_integer_variable("iv1", 0);
        assert!(v.is_ok());
        let v = model.add_integer_variable("iv2", 0);
        assert!(v.is_ok());
        let v = model.add_integer_resource_variable("irv1", false, 0);
        assert!(v.is_ok());
        let v = model.add_integer_resource_variable("irv2", true, 0);
        assert!(v.is_ok());
        let v = model.add_continuous_variable("cv1", 0.0);
        assert!(v.is_ok());
        let v = model.add_continuous_variable("cv2", 0.0);
        assert!(v.is_ok());
        let v = model.add_continuous_resource_variable("crv1", false, 0.0);
        assert!(v.is_ok());
        let v = model.add_continuous_resource_variable("crv2", true, 0.0);
        assert!(v.is_ok());

        let transition = Transition {
            cost: CostExpression::Continuous(ContinuousExpression::Cost),
            ..Default::default()
        };
        let result = model.add_backward_forced_transition(transition);
        assert!(result.is_ok());
        assert_eq!(
            model.backward_forced_transitions,
            vec![Transition {
                cost: CostExpression::Continuous(ContinuousExpression::Cost),
                ..Default::default()
            }]
        );

        let transition = Transition {
            cost: CostExpression::Continuous(ContinuousExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(ContinuousExpression::Constant(-1.0)),
            )),
            preconditions: vec![
                GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Eq,
                        Box::new(IntegerExpression::Variable(0)),
                        Box::new(IntegerExpression::Constant(0)),
                    ),
                    ..Default::default()
                },
                GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Eq,
                        Box::new(IntegerExpression::Variable(0)),
                        Box::new(IntegerExpression::UnaryOperation(
                            UnaryOperator::Abs,
                            Box::new(IntegerExpression::Constant(0)),
                        )),
                    ),
                    ..Default::default()
                },
            ],
            effect: Effect {
                set_effects: vec![
                    (
                        0,
                        SetExpression::Reference(ReferenceExpression::Constant(
                            Set::with_capacity(10),
                        )),
                    ),
                    (
                        1,
                        SetExpression::SetElementOperation(
                            SetElementOperator::Add,
                            ElementExpression::Constant(0),
                            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                                Set::with_capacity(10),
                            ))),
                        ),
                    ),
                ],
                vector_effects: vec![
                    (
                        0,
                        VectorExpression::Reference(ReferenceExpression::Constant(vec![1, 2])),
                    ),
                    (
                        1,
                        VectorExpression::Push(
                            ElementExpression::Constant(0),
                            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                                vec![1, 2],
                            ))),
                        ),
                    ),
                ],
                element_effects: vec![
                    (0, ElementExpression::Constant(0)),
                    (
                        1,
                        ElementExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ElementExpression::Constant(1)),
                            Box::new(ElementExpression::Constant(1)),
                        ),
                    ),
                ],
                integer_effects: vec![
                    (0, IntegerExpression::Constant(0)),
                    (
                        1,
                        IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::Constant(1)),
                            Box::new(IntegerExpression::Constant(1)),
                        ),
                    ),
                ],
                continuous_effects: vec![
                    (0, ContinuousExpression::Constant(0.0)),
                    (
                        1,
                        ContinuousExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ContinuousExpression::Constant(1.0)),
                            Box::new(ContinuousExpression::Constant(1.0)),
                        ),
                    ),
                ],
                element_resource_effects: vec![
                    (0, ElementExpression::Constant(0)),
                    (
                        1,
                        ElementExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ElementExpression::Constant(1)),
                            Box::new(ElementExpression::Constant(1)),
                        ),
                    ),
                ],
                integer_resource_effects: vec![
                    (0, IntegerExpression::Constant(0)),
                    (
                        1,
                        IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::Constant(1)),
                            Box::new(IntegerExpression::Constant(1)),
                        ),
                    ),
                ],
                continuous_resource_effects: vec![
                    (0, ContinuousExpression::Constant(0.0)),
                    (
                        1,
                        ContinuousExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(ContinuousExpression::Constant(1.0)),
                            Box::new(ContinuousExpression::Constant(1.0)),
                        ),
                    ),
                ],
            },
            ..Default::default()
        };
        let result = model.add_backward_forced_transition(transition);
        assert!(result.is_ok());
        assert_eq!(
            model.backward_forced_transitions,
            vec![
                Transition {
                    cost: CostExpression::Continuous(ContinuousExpression::Cost),
                    ..Default::default()
                },
                Transition {
                    cost: CostExpression::Continuous(ContinuousExpression::Constant(1.0)),
                    preconditions: vec![
                        GroundedCondition {
                            condition: Condition::ComparisonI(
                                ComparisonOperator::Eq,
                                Box::new(IntegerExpression::Variable(0)),
                                Box::new(IntegerExpression::Constant(0)),
                            ),
                            ..Default::default()
                        },
                        GroundedCondition {
                            condition: Condition::ComparisonI(
                                ComparisonOperator::Eq,
                                Box::new(IntegerExpression::Variable(0)),
                                Box::new(IntegerExpression::Constant(0),),
                            ),
                            ..Default::default()
                        },
                    ],
                    effect: Effect {
                        set_effects: vec![
                            (
                                0,
                                SetExpression::Reference(ReferenceExpression::Constant(
                                    Set::with_capacity(10),
                                )),
                            ),
                            (
                                1,
                                SetExpression::Reference(ReferenceExpression::Constant({
                                    let mut set = Set::with_capacity(10);
                                    set.insert(0);
                                    set
                                },)),
                            ),
                        ],
                        vector_effects: vec![
                            (
                                0,
                                VectorExpression::Reference(ReferenceExpression::Constant(vec![
                                    1, 2
                                ])),
                            ),
                            (
                                1,
                                VectorExpression::Reference(ReferenceExpression::Constant(vec![
                                    1, 2, 0
                                ])),
                            ),
                        ],
                        element_effects: vec![
                            (0, ElementExpression::Constant(0)),
                            (1, ElementExpression::Constant(2)),
                        ],
                        integer_effects: vec![
                            (0, IntegerExpression::Constant(0)),
                            (1, IntegerExpression::Constant(2)),
                        ],
                        continuous_effects: vec![
                            (0, ContinuousExpression::Constant(0.0)),
                            (1, ContinuousExpression::Constant(2.0)),
                        ],
                        element_resource_effects: vec![
                            (0, ElementExpression::Constant(0)),
                            (1, ElementExpression::Constant(2)),
                        ],
                        integer_resource_effects: vec![
                            (0, IntegerExpression::Constant(0)),
                            (1, IntegerExpression::Constant(2)),
                        ],
                        continuous_resource_effects: vec![
                            (0, ContinuousExpression::Constant(0.0)),
                            (1, ContinuousExpression::Constant(2.0)),
                        ],
                    },
                    ..Default::default()
                }
            ]
        );
    }

    #[test]
    fn add_backward_forced_transition_err() {
        let mut model = Model {
            cost_type: CostType::Integer,
            ..Default::default()
        };
        let transition = Transition {
            cost: CostExpression::Continuous(ContinuousExpression::Cost),
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model {
            cost_type: CostType::Integer,
            ..Default::default()
        };
        let transition = Transition {
            cost: CostExpression::Integer(IntegerExpression::Variable(0)),
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        };
        let transition = Transition {
            cost: CostExpression::Integer(IntegerExpression::Cost),
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        };
        let transition = Transition {
            cost: CostExpression::Continuous(ContinuousExpression::Variable(0)),
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            elements_in_set_variable: vec![(0, 0)],
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            elements_in_set_variable: vec![(v.id(), 11)],
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            elements_in_vector_variable: vec![(0, 0)],
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_vector_variable("v", ob, vec![0, 1]);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            elements_in_vector_variable: vec![(v.id(), 11)],
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            preconditions: vec![GroundedCondition {
                elements_in_set_variable: vec![(0, 0)],
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            preconditions: vec![GroundedCondition {
                elements_in_set_variable: vec![(v.id(), 11)],
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            preconditions: vec![GroundedCondition {
                elements_in_vector_variable: vec![(0, 0)],
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_vector_variable("v", ob, vec![0, 1]);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            preconditions: vec![GroundedCondition {
                elements_in_vector_variable: vec![(v.id(), 11)],
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            preconditions: vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Eq,
                    Box::new(IntegerExpression::Variable(0)),
                    Box::new(IntegerExpression::Constant(0)),
                ),
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                set_effects: vec![(
                    0,
                    SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(10))),
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v1", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                set_effects: vec![(
                    v.id(),
                    SetExpression::Reference(ReferenceExpression::Variable(1)),
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_set_variable("v1", ob, Set::with_capacity(10));
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                set_effects: vec![
                    (
                        v.id(),
                        SetExpression::Reference(ReferenceExpression::Constant(
                            Set::with_capacity(10),
                        )),
                    ),
                    (
                        v.id(),
                        SetExpression::Reference(ReferenceExpression::Constant(
                            Set::with_capacity(10),
                        )),
                    ),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                vector_effects: vec![(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_vector_variable("v1", ob, vec![0, 1]);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                vector_effects: vec![(
                    v.id(),
                    VectorExpression::Reference(ReferenceExpression::Variable(1)),
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_vector_variable("v1", ob, vec![0, 1]);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                vector_effects: vec![
                    (
                        v.id(),
                        VectorExpression::Reference(ReferenceExpression::Variable(1)),
                    ),
                    (
                        v.id(),
                        VectorExpression::Reference(ReferenceExpression::Variable(1)),
                    ),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                element_effects: vec![(0, ElementExpression::Constant(0))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v1", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                element_effects: vec![(v.id(), ElementExpression::Variable(1))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable("v1", ob, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                element_effects: vec![
                    (v.id(), ElementExpression::Constant(0)),
                    (v.id(), ElementExpression::Constant(0)),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                element_resource_effects: vec![(0, ElementExpression::Constant(0))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v1", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                element_resource_effects: vec![(v.id(), ElementExpression::ResourceVariable(1))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let ob = model.add_object_type("something", 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_resource_variable("v1", ob, false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                element_resource_effects: vec![
                    (v.id(), ElementExpression::Constant(0)),
                    (v.id(), ElementExpression::Constant(0)),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                integer_effects: vec![(0, IntegerExpression::Constant(0))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_integer_variable("v1", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                integer_effects: vec![(v.id(), IntegerExpression::Variable(1))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_integer_variable("v1", 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                integer_effects: vec![
                    (v.id(), IntegerExpression::Constant(0)),
                    (v.id(), IntegerExpression::Constant(0)),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                integer_resource_effects: vec![(0, IntegerExpression::Constant(0))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v1", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                integer_resource_effects: vec![(v.id(), IntegerExpression::ResourceVariable(1))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_integer_resource_variable("v1", false, 0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                integer_resource_effects: vec![
                    (v.id(), IntegerExpression::Constant(0)),
                    (v.id(), IntegerExpression::Constant(0)),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                continuous_effects: vec![(0, ContinuousExpression::Constant(0.0))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_continuous_variable("v1", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                continuous_effects: vec![(v.id(), ContinuousExpression::Variable(1))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_continuous_variable("v1", 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                continuous_effects: vec![
                    (v.id(), ContinuousExpression::Constant(0.0)),
                    (v.id(), ContinuousExpression::Constant(0.0)),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let transition = Transition {
            effect: Effect {
                continuous_resource_effects: vec![(0, ContinuousExpression::Constant(0.0))],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v1", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                continuous_resource_effects: vec![(
                    v.id(),
                    ContinuousExpression::ResourceVariable(1),
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());

        let mut model = Model::default();
        let v = model.add_continuous_resource_variable("v1", false, 0.0);
        assert!(v.is_ok());
        let v = v.unwrap();
        let transition = Transition {
            effect: Effect {
                continuous_resource_effects: vec![
                    (v.id(), ContinuousExpression::Constant(0.0)),
                    (v.id(), ContinuousExpression::Constant(0.0)),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(model.add_backward_forced_transition(transition).is_err());
    }

    #[test]
    fn add_dual_bound_ok() {
        let mut model = Model {
            cost_type: CostType::Integer,
            ..Default::default()
        };
        assert!(model
            .add_dual_bound(IntegerExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(IntegerExpression::Constant(0))
            ))
            .is_ok());
        assert!(model
            .add_dual_bound(IntegerExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(IntegerExpression::Constant(-1))
            ))
            .is_ok());
        assert_eq!(
            model.dual_bounds,
            vec![
                CostExpression::Integer(IntegerExpression::Constant(0)),
                CostExpression::Integer(IntegerExpression::Constant(1)),
            ]
        );

        let mut model = Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        };
        assert!(model
            .add_dual_bound(ContinuousExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(ContinuousExpression::Constant(0.0))
            ))
            .is_ok());
        assert!(model
            .add_dual_bound(ContinuousExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(ContinuousExpression::Constant(-1.0))
            ))
            .is_ok());
        assert_eq!(
            model.dual_bounds,
            vec![
                CostExpression::Continuous(ContinuousExpression::Constant(0.0)),
                CostExpression::Continuous(ContinuousExpression::Constant(1.0)),
            ]
        );
        let mut model = Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        };
        assert!(model.add_dual_bound(IntegerExpression::Constant(0)).is_ok());
        assert_eq!(
            model.dual_bounds,
            vec![CostExpression::Integer(IntegerExpression::Constant(0)),]
        );
    }

    #[test]
    fn add_dual_bound_err() {
        let mut model = Model {
            cost_type: CostType::Integer,
            ..Default::default()
        };
        assert!(model
            .add_dual_bound(ContinuousExpression::Constant(0.0))
            .is_err());
        assert!(model
            .add_dual_bound(IntegerExpression::Variable(0))
            .is_err());

        let mut model = Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        };
        assert!(model
            .add_dual_bound(ContinuousExpression::Variable(0))
            .is_err());
    }

    #[test]
    fn check_element_table_expression_ok() {
        let mut model = Model::default();
        let t = TableInterface::<Element>::add_table_1d(&mut model, String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());
        let t = TableInterface::<Element>::add_table_2d(
            &mut model,
            String::from("t2"),
            vec![vec![0, 1]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Element>::add_table_3d(
            &mut model,
            String::from("t3"),
            vec![vec![vec![0, 1]]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Element>::add_table(
            &mut model,
            String::from("t3"),
            FxHashMap::default(),
            0,
        );
        assert!(t.is_ok());

        let expression = expression::TableExpression::Constant(0);
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression =
            expression::TableExpression::<Element>::Table1D(0, ElementExpression::Constant(0));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableExpression::<Element>::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableExpression::<Element>::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableExpression::<Element>::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            ],
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());
    }

    #[test]
    fn check_element_table_expression_err() {
        let mut model = Model::default();
        let t = TableInterface::<Element>::add_table_1d(&mut model, String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());
        let t = TableInterface::<Element>::add_table_2d(
            &mut model,
            String::from("t2"),
            vec![vec![0, 1]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Element>::add_table_3d(
            &mut model,
            String::from("t3"),
            vec![vec![vec![0, 1]]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Element>::add_table(
            &mut model,
            String::from("t4"),
            FxHashMap::default(),
            0,
        );
        assert!(t.is_ok());

        let expression =
            expression::TableExpression::<Element>::Table1D(1, ElementExpression::Constant(0));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression =
            expression::TableExpression::<Element>::Table1D(0, ElementExpression::Variable(0));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<Element>::Table2D(
            1,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<Element>::Table2D(
            0,
            ElementExpression::Variable(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<Element>::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<Element>::Table3D(
            1,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<Element>::Table3D(
            0,
            ElementExpression::Variable(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<Element>::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<Element>::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<Element>::Table(
            1,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            ],
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<Element>::Table(
            0,
            vec![
                ElementExpression::Variable(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            ],
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());
    }

    #[test]
    fn check_set_table_expression_ok() {
        let mut model = Model::default();
        let t = TableInterface::<Set>::add_table_1d(
            &mut model,
            String::from("t1"),
            vec![Set::with_capacity(2), Set::with_capacity(2)],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Set>::add_table_2d(
            &mut model,
            String::from("t2"),
            vec![vec![Set::with_capacity(2), Set::with_capacity(2)]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Set>::add_table_3d(
            &mut model,
            String::from("t3"),
            vec![vec![vec![Set::with_capacity(2), Set::with_capacity(2)]]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Set>::add_table(
            &mut model,
            String::from("t4"),
            FxHashMap::default(),
            Set::with_capacity(2),
        );
        assert!(t.is_ok());

        let expression = expression::TableExpression::Constant(0);
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression =
            expression::TableExpression::<Set>::Table1D(0, ElementExpression::Constant(0));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableExpression::<Set>::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableExpression::<Set>::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableExpression::<Set>::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            ],
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());
    }

    #[test]
    fn check_set_table_expression_err() {
        let mut model = Model::default();
        let t = TableInterface::<Set>::add_table_1d(
            &mut model,
            String::from("t1"),
            vec![Set::with_capacity(2), Set::with_capacity(2)],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Set>::add_table_2d(
            &mut model,
            String::from("t2"),
            vec![vec![Set::with_capacity(2), Set::with_capacity(2)]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Set>::add_table_3d(
            &mut model,
            String::from("t3"),
            vec![vec![vec![Set::with_capacity(2), Set::with_capacity(2)]]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Set>::add_table(
            &mut model,
            String::from("t4"),
            FxHashMap::default(),
            Set::with_capacity(2),
        );
        assert!(t.is_ok());

        let expression = expression::TableExpression::Constant(0);
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression =
            expression::TableExpression::<Set>::Table1D(0, ElementExpression::Constant(0));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression =
            expression::TableExpression::<Set>::Table1D(1, ElementExpression::Constant(0));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression =
            expression::TableExpression::<Set>::Table1D(0, ElementExpression::Variable(0));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<Set>::Table2D(
            1,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<Set>::Table2D(
            0,
            ElementExpression::Variable(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<Set>::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<Set>::Table3D(
            1,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<Set>::Table3D(
            0,
            ElementExpression::Variable(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<Set>::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<Set>::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<Set>::Table(
            1,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            ],
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<Set>::Table(
            0,
            vec![
                ElementExpression::Variable(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            ],
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());
    }

    #[test]
    fn check_vector_table_expression_ok() {
        let mut model = Model::default();
        let t = TableInterface::<Vector>::add_table_1d(
            &mut model,
            String::from("t1"),
            vec![vec![0], vec![1]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Vector>::add_table_2d(
            &mut model,
            String::from("t2"),
            vec![vec![vec![0], vec![1]]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Vector>::add_table_3d(
            &mut model,
            String::from("t3"),
            vec![vec![vec![vec![0], vec![1]]]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Vector>::add_table(
            &mut model,
            String::from("t3"),
            FxHashMap::default(),
            vec![0],
        );
        assert!(t.is_ok());

        let expression = expression::TableExpression::Constant(0);
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression =
            expression::TableExpression::<Vector>::Table1D(0, ElementExpression::Constant(0));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableExpression::<Vector>::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableExpression::<Vector>::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableExpression::<Vector>::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            ],
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());
    }

    #[test]
    fn check_vector_table_expression_err() {
        let mut model = Model::default();
        let t = TableInterface::<Vector>::add_table_1d(
            &mut model,
            String::from("t1"),
            vec![vec![0], vec![1]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Vector>::add_table_2d(
            &mut model,
            String::from("t2"),
            vec![vec![vec![0], vec![1]]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Vector>::add_table_3d(
            &mut model,
            String::from("t3"),
            vec![vec![vec![vec![0], vec![1]]]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Vector>::add_table(
            &mut model,
            String::from("t4"),
            FxHashMap::default(),
            vec![0],
        );
        assert!(t.is_ok());

        let expression =
            expression::TableExpression::<Vector>::Table1D(1, ElementExpression::Constant(0));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression =
            expression::TableExpression::<Vector>::Table1D(0, ElementExpression::Variable(0));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<Vector>::Table2D(
            1,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<Vector>::Table2D(
            0,
            ElementExpression::Variable(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<Vector>::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<Vector>::Table3D(
            1,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<Vector>::Table3D(
            0,
            ElementExpression::Variable(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<Vector>::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<Vector>::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<Vector>::Table(
            1,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            ],
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<Vector>::Table(
            0,
            vec![
                ElementExpression::Variable(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            ],
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());
    }

    #[test]
    fn check_bool_table_expression_ok() {
        let mut model = Model::default();
        let t =
            TableInterface::<bool>::add_table_1d(&mut model, String::from("t1"), vec![false, true]);
        assert!(t.is_ok());
        let t = TableInterface::<bool>::add_table_2d(
            &mut model,
            String::from("t2"),
            vec![vec![false, true]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<bool>::add_table_3d(
            &mut model,
            String::from("t3"),
            vec![vec![vec![false, true]]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<bool>::add_table(
            &mut model,
            String::from("t3"),
            FxHashMap::default(),
            false,
        );
        assert!(t.is_ok());

        let expression = expression::TableExpression::Constant(0);
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression =
            expression::TableExpression::<bool>::Table1D(0, ElementExpression::Constant(0));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableExpression::<bool>::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableExpression::<bool>::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableExpression::<bool>::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            ],
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());
    }

    #[test]
    fn check_bool_table_expression_err() {
        let mut model = Model::default();
        let t =
            TableInterface::<bool>::add_table_1d(&mut model, String::from("t1"), vec![false, true]);
        assert!(t.is_ok());
        let t = TableInterface::<bool>::add_table_2d(
            &mut model,
            String::from("t2"),
            vec![vec![false, true]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<bool>::add_table_3d(
            &mut model,
            String::from("t3"),
            vec![vec![vec![false, true]]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<bool>::add_table(
            &mut model,
            String::from("t4"),
            FxHashMap::default(),
            false,
        );
        assert!(t.is_ok());

        let expression =
            expression::TableExpression::<bool>::Table1D(1, ElementExpression::Constant(0));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression =
            expression::TableExpression::<bool>::Table1D(0, ElementExpression::Variable(0));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<bool>::Table2D(
            1,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<bool>::Table2D(
            0,
            ElementExpression::Variable(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<bool>::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<bool>::Table3D(
            1,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<bool>::Table3D(
            0,
            ElementExpression::Variable(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<bool>::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<bool>::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<bool>::Table(
            1,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            ],
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableExpression::<bool>::Table(
            0,
            vec![
                ElementExpression::Variable(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            ],
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());
    }

    #[test]
    fn check_argument_expression_ok() {
        let model = Model::default();
        let expression = ArgumentExpression::Element(ElementExpression::Constant(0));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());
        let expression = ArgumentExpression::Set(SetExpression::Reference(
            ReferenceExpression::Constant(Set::with_capacity(2)),
        ));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());
        let expression = ArgumentExpression::Vector(VectorExpression::Reference(
            ReferenceExpression::Constant(vec![0, 1]),
        ));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());
    }

    #[test]
    fn check_argument_expression_err() {
        let model = Model::default();
        let expression = ArgumentExpression::Element(ElementExpression::Variable(0));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());
        let expression =
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Variable(0)));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());
        let expression = ArgumentExpression::Vector(VectorExpression::Reference(
            ReferenceExpression::Variable(0),
        ));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());
    }

    #[test]
    fn check_integer_table_expression_ok() {
        let mut model = Model::default();
        let t = TableInterface::<Integer>::add_table_1d(&mut model, String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());
        let t = TableInterface::<Integer>::add_table_2d(
            &mut model,
            String::from("t2"),
            vec![vec![0, 1]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Integer>::add_table_3d(
            &mut model,
            String::from("t3"),
            vec![vec![vec![0, 1]]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Integer>::add_table(
            &mut model,
            String::from("t3"),
            FxHashMap::default(),
            0,
        );
        assert!(t.is_ok());

        let expression = expression::NumericTableExpression::Constant(0);
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Integer>::Table1D(
            0,
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Integer>::Table1DReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Integer>::Table1DVectorReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Integer>::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Integer>::Table2DReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Integer>::Table2DVectorReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Integer>::Table2DSetVectorReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Integer>::Table2DVectorSetReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Integer>::Table2DReduceX(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Integer>::Table2DReduceY(
            ReduceOperator::Sum,
            0,
            ElementExpression::Constant(0),
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Integer>::Table2DVectorReduceX(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Integer>::Table2DVectorReduceY(
            ReduceOperator::Sum,
            0,
            ElementExpression::Constant(0),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Integer>::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Integer>::Table3DReduce(
            ReduceOperator::Sum,
            0,
            ArgumentExpression::Element(ElementExpression::Constant(0)),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                Set::with_capacity(2),
            ))),
            ArgumentExpression::Vector(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Integer>::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            ],
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Integer>::TableReduce(
            ReduceOperator::Sum,
            0,
            vec![
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                    Set::with_capacity(2),
                ))),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 1]),
                )),
            ],
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());
    }

    #[test]
    fn check_integer_table_expression_err() {
        let mut model = Model::default();
        let t = TableInterface::<Integer>::add_table_1d(&mut model, String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());
        let t = TableInterface::<Integer>::add_table_2d(
            &mut model,
            String::from("t2"),
            vec![vec![0, 1]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Integer>::add_table_3d(
            &mut model,
            String::from("t3"),
            vec![vec![vec![0, 1]]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Integer>::add_table(
            &mut model,
            String::from("t4"),
            FxHashMap::default(),
            0,
        );
        assert!(t.is_ok());

        let expression = expression::NumericTableExpression::<Integer>::Table1D(
            1,
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table1D(
            0,
            ElementExpression::Variable(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table1DReduce(
            ReduceOperator::Sum,
            1,
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table1DReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table1DVectorReduce(
            ReduceOperator::Sum,
            1,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table1DVectorReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table2D(
            1,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table2D(
            0,
            ElementExpression::Variable(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table2DReduce(
            ReduceOperator::Sum,
            1,
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table2DReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table2DReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table2DVectorReduce(
            ReduceOperator::Sum,
            1,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table2DVectorReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table2DVectorReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table2DSetVectorReduce(
            ReduceOperator::Sum,
            1,
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table2DSetVectorReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table2DSetVectorReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table2DVectorSetReduce(
            ReduceOperator::Sum,
            1,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table2DVectorSetReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table2DVectorSetReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table2DReduceX(
            ReduceOperator::Sum,
            1,
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table2DReduceX(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table2DReduceX(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
            ElementExpression::Variable(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table2DReduceY(
            ReduceOperator::Sum,
            1,
            ElementExpression::Constant(0),
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table2DReduceY(
            ReduceOperator::Sum,
            0,
            ElementExpression::Variable(0),
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table2DReduceY(
            ReduceOperator::Sum,
            0,
            ElementExpression::Constant(0),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table2DVectorReduceX(
            ReduceOperator::Sum,
            1,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table2DVectorReduceX(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table2DVectorReduceX(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            ElementExpression::Variable(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table2DVectorReduceY(
            ReduceOperator::Sum,
            1,
            ElementExpression::Constant(0),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table2DVectorReduceY(
            ReduceOperator::Sum,
            0,
            ElementExpression::Variable(0),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table2DVectorReduceY(
            ReduceOperator::Sum,
            0,
            ElementExpression::Constant(0),
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table3D(
            1,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table3D(
            0,
            ElementExpression::Variable(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table3DReduce(
            ReduceOperator::Sum,
            1,
            ArgumentExpression::Element(ElementExpression::Constant(0)),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                Set::with_capacity(2),
            ))),
            ArgumentExpression::Vector(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table3DReduce(
            ReduceOperator::Sum,
            0,
            ArgumentExpression::Element(ElementExpression::Variable(0)),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                Set::with_capacity(2),
            ))),
            ArgumentExpression::Vector(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table3DReduce(
            ReduceOperator::Sum,
            0,
            ArgumentExpression::Element(ElementExpression::Constant(0)),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Variable(0))),
            ArgumentExpression::Vector(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table3DReduce(
            ReduceOperator::Sum,
            0,
            ArgumentExpression::Element(ElementExpression::Constant(0)),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                Set::with_capacity(2),
            ))),
            ArgumentExpression::Vector(VectorExpression::Reference(ReferenceExpression::Variable(
                0,
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table(
            1,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            ],
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::Table(
            0,
            vec![
                ElementExpression::Variable(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            ],
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::TableReduce(
            ReduceOperator::Sum,
            1,
            vec![
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                    Set::with_capacity(2),
                ))),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 1]),
                )),
            ],
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Integer>::TableReduce(
            ReduceOperator::Sum,
            0,
            vec![
                ArgumentExpression::Element(ElementExpression::Variable(0)),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                    Set::with_capacity(2),
                ))),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 1]),
                )),
            ],
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());
    }

    #[test]
    fn check_continuous_table_expression_ok() {
        let mut model = Model::default();
        let t = TableInterface::<Continuous>::add_table_1d(
            &mut model,
            String::from("t1"),
            vec![0.0, 1.0],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Continuous>::add_table_2d(
            &mut model,
            String::from("t2"),
            vec![vec![0.0, 1.0]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Continuous>::add_table_3d(
            &mut model,
            String::from("t3"),
            vec![vec![vec![0.0, 1.0]]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Continuous>::add_table(
            &mut model,
            String::from("t3"),
            FxHashMap::default(),
            0.0,
        );
        assert!(t.is_ok());

        let expression = expression::NumericTableExpression::Constant(0);
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Continuous>::Table1D(
            0,
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Continuous>::Table1DReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Continuous>::Table1DVectorReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Continuous>::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DVectorReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DSetVectorReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DVectorSetReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DReduceX(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DReduceY(
            ReduceOperator::Sum,
            0,
            ElementExpression::Constant(0),
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DVectorReduceX(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DVectorReduceY(
            ReduceOperator::Sum,
            0,
            ElementExpression::Constant(0),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Continuous>::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Continuous>::Table3DReduce(
            ReduceOperator::Sum,
            0,
            ArgumentExpression::Element(ElementExpression::Constant(0)),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                Set::with_capacity(2),
            ))),
            ArgumentExpression::Vector(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Continuous>::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            ],
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::NumericTableExpression::<Continuous>::TableReduce(
            ReduceOperator::Sum,
            0,
            vec![
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                    Set::with_capacity(2),
                ))),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 1]),
                )),
            ],
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());
    }

    #[test]
    fn check_continuous_table_expression_err() {
        let mut model = Model::default();
        let t = TableInterface::<Continuous>::add_table_1d(
            &mut model,
            String::from("t1"),
            vec![0.0, 1.0],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Continuous>::add_table_2d(
            &mut model,
            String::from("t2"),
            vec![vec![0.0, 1.0]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Continuous>::add_table_3d(
            &mut model,
            String::from("t3"),
            vec![vec![vec![0.0, 1.0]]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Continuous>::add_table(
            &mut model,
            String::from("t4"),
            FxHashMap::default(),
            0.0,
        );
        assert!(t.is_ok());

        let expression = expression::NumericTableExpression::<Continuous>::Table1D(
            1,
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table1D(
            0,
            ElementExpression::Variable(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table1DReduce(
            ReduceOperator::Sum,
            1,
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table1DReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table1DVectorReduce(
            ReduceOperator::Sum,
            1,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table1DVectorReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table2D(
            1,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table2D(
            0,
            ElementExpression::Variable(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DReduce(
            ReduceOperator::Sum,
            1,
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DVectorReduce(
            ReduceOperator::Sum,
            1,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DVectorReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DVectorReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DSetVectorReduce(
            ReduceOperator::Sum,
            1,
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DSetVectorReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DSetVectorReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DVectorSetReduce(
            ReduceOperator::Sum,
            1,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DVectorSetReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DVectorSetReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DReduceX(
            ReduceOperator::Sum,
            1,
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DReduceX(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DReduceX(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
            ElementExpression::Variable(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DReduceY(
            ReduceOperator::Sum,
            1,
            ElementExpression::Constant(0),
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DReduceY(
            ReduceOperator::Sum,
            0,
            ElementExpression::Variable(0),
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DReduceY(
            ReduceOperator::Sum,
            0,
            ElementExpression::Constant(0),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DVectorReduceX(
            ReduceOperator::Sum,
            1,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DVectorReduceX(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DVectorReduceX(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            ElementExpression::Variable(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DVectorReduceY(
            ReduceOperator::Sum,
            1,
            ElementExpression::Constant(0),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DVectorReduceY(
            ReduceOperator::Sum,
            0,
            ElementExpression::Variable(0),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table2DVectorReduceY(
            ReduceOperator::Sum,
            0,
            ElementExpression::Constant(0),
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table3D(
            1,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table3D(
            0,
            ElementExpression::Variable(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table3DReduce(
            ReduceOperator::Sum,
            1,
            ArgumentExpression::Element(ElementExpression::Constant(0)),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                Set::with_capacity(2),
            ))),
            ArgumentExpression::Vector(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table3DReduce(
            ReduceOperator::Sum,
            0,
            ArgumentExpression::Element(ElementExpression::Variable(0)),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                Set::with_capacity(2),
            ))),
            ArgumentExpression::Vector(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table3DReduce(
            ReduceOperator::Sum,
            0,
            ArgumentExpression::Element(ElementExpression::Constant(0)),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Variable(0))),
            ArgumentExpression::Vector(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table3DReduce(
            ReduceOperator::Sum,
            0,
            ArgumentExpression::Element(ElementExpression::Constant(0)),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                Set::with_capacity(2),
            ))),
            ArgumentExpression::Vector(VectorExpression::Reference(ReferenceExpression::Variable(
                0,
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table(
            1,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            ],
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::Table(
            0,
            vec![
                ElementExpression::Variable(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            ],
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::TableReduce(
            ReduceOperator::Sum,
            1,
            vec![
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                    Set::with_capacity(2),
                ))),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 1]),
                )),
            ],
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::NumericTableExpression::<Continuous>::TableReduce(
            ReduceOperator::Sum,
            0,
            vec![
                ArgumentExpression::Element(ElementExpression::Variable(0)),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                    Set::with_capacity(2),
                ))),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 1]),
                )),
            ],
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());
    }

    #[test]
    fn check_vector_or_element_expression_ok() {
        let model = Model::default();
        let expression = VectorOrElementExpression::Element(ElementExpression::Constant(0));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());
        let expression = VectorOrElementExpression::Vector(VectorExpression::Reference(
            ReferenceExpression::Constant(vec![0, 1]),
        ));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());
    }

    #[test]
    fn check_vector_or_element_expression_err() {
        let model = Model::default();
        let expression = VectorOrElementExpression::Element(ElementExpression::Variable(0));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());
        let expression = VectorOrElementExpression::Vector(VectorExpression::Reference(
            ReferenceExpression::Variable(0),
        ));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());
    }

    #[test]
    fn check_integer_table_vector_expression_ok() {
        let mut model = Model::default();
        let t = TableInterface::<Integer>::add_table_1d(&mut model, String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());
        let t = TableInterface::<Integer>::add_table_2d(
            &mut model,
            String::from("t2"),
            vec![vec![0, 1]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Integer>::add_table_3d(
            &mut model,
            String::from("t3"),
            vec![vec![vec![0, 1]]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Integer>::add_table(
            &mut model,
            String::from("t3"),
            FxHashMap::default(),
            0,
        );
        assert!(t.is_ok());

        let expression = expression::TableVectorExpression::Constant(vec![0, 1]);
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableVectorExpression::<Integer>::Table1D(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableVectorExpression::<Integer>::Table2D(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableVectorExpression::<Integer>::Table2DX(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableVectorExpression::<Integer>::Table2DY(
            0,
            ElementExpression::Constant(0),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableVectorExpression::<Integer>::Table3D(
            0,
            VectorOrElementExpression::Element(ElementExpression::Constant(0)),
            VectorOrElementExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            )),
            VectorOrElementExpression::Element(ElementExpression::Constant(0)),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableVectorExpression::<Integer>::Table(
            0,
            vec![
                VectorOrElementExpression::Element(ElementExpression::Constant(0)),
                VectorOrElementExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 1]),
                )),
                VectorOrElementExpression::Element(ElementExpression::Constant(0)),
                VectorOrElementExpression::Element(ElementExpression::Constant(0)),
            ],
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableVectorExpression::<Integer>::Table2DXReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableVectorExpression::<Integer>::Table2DYReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableVectorExpression::<Integer>::Table3DReduce(
            ReduceOperator::Sum,
            0,
            ArgumentExpression::Element(ElementExpression::Constant(0)),
            ArgumentExpression::Vector(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                Set::with_capacity(2),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableVectorExpression::<Integer>::TableReduce(
            ReduceOperator::Sum,
            0,
            vec![
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 1]),
                )),
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                    Set::with_capacity(2),
                ))),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
            ],
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());
    }

    #[test]
    fn check_integer_table_vector_expression_err() {
        let mut model = Model::default();
        let t = TableInterface::<Integer>::add_table_1d(&mut model, String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());
        let t = TableInterface::<Integer>::add_table_2d(
            &mut model,
            String::from("t2"),
            vec![vec![0, 1]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Integer>::add_table_3d(
            &mut model,
            String::from("t3"),
            vec![vec![vec![0, 1]]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Integer>::add_table(
            &mut model,
            String::from("t3"),
            FxHashMap::default(),
            0,
        );
        assert!(t.is_ok());

        let expression = expression::TableVectorExpression::<Integer>::Table1D(
            1,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Integer>::Table1D(
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Integer>::Table2D(
            1,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Integer>::Table2D(
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Integer>::Table2D(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Integer>::Table2DX(
            1,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Integer>::Table2DX(
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Integer>::Table2DX(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            ElementExpression::Variable(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Integer>::Table2DY(
            1,
            ElementExpression::Constant(0),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Integer>::Table2DY(
            0,
            ElementExpression::Variable(0),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Integer>::Table2DY(
            0,
            ElementExpression::Constant(0),
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Integer>::Table3D(
            1,
            VectorOrElementExpression::Element(ElementExpression::Constant(0)),
            VectorOrElementExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            )),
            VectorOrElementExpression::Element(ElementExpression::Constant(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Integer>::Table3D(
            0,
            VectorOrElementExpression::Element(ElementExpression::Variable(0)),
            VectorOrElementExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            )),
            VectorOrElementExpression::Element(ElementExpression::Constant(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Integer>::Table3D(
            0,
            VectorOrElementExpression::Element(ElementExpression::Constant(0)),
            VectorOrElementExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Variable(0),
            )),
            VectorOrElementExpression::Element(ElementExpression::Constant(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Integer>::Table3D(
            0,
            VectorOrElementExpression::Element(ElementExpression::Constant(0)),
            VectorOrElementExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            )),
            VectorOrElementExpression::Element(ElementExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Integer>::Table(
            1,
            vec![
                VectorOrElementExpression::Element(ElementExpression::Constant(0)),
                VectorOrElementExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 1]),
                )),
                VectorOrElementExpression::Element(ElementExpression::Constant(0)),
                VectorOrElementExpression::Element(ElementExpression::Constant(0)),
            ],
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Integer>::Table(
            0,
            vec![
                VectorOrElementExpression::Element(ElementExpression::Variable(0)),
                VectorOrElementExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 1]),
                )),
                VectorOrElementExpression::Element(ElementExpression::Constant(0)),
                VectorOrElementExpression::Element(ElementExpression::Constant(0)),
            ],
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Integer>::Table2DXReduce(
            ReduceOperator::Sum,
            1,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Integer>::Table2DXReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Integer>::Table2DXReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Integer>::Table2DYReduce(
            ReduceOperator::Sum,
            1,
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Integer>::Table2DYReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Integer>::Table2DYReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Integer>::Table3DReduce(
            ReduceOperator::Sum,
            1,
            ArgumentExpression::Element(ElementExpression::Constant(0)),
            ArgumentExpression::Vector(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                Set::with_capacity(2),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Integer>::Table3DReduce(
            ReduceOperator::Sum,
            0,
            ArgumentExpression::Element(ElementExpression::Variable(0)),
            ArgumentExpression::Vector(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                Set::with_capacity(2),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Integer>::Table3DReduce(
            ReduceOperator::Sum,
            0,
            ArgumentExpression::Element(ElementExpression::Constant(0)),
            ArgumentExpression::Vector(VectorExpression::Reference(ReferenceExpression::Variable(
                0,
            ))),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                Set::with_capacity(2),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Integer>::Table3DReduce(
            ReduceOperator::Sum,
            0,
            ArgumentExpression::Element(ElementExpression::Constant(0)),
            ArgumentExpression::Vector(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Integer>::TableReduce(
            ReduceOperator::Sum,
            1,
            vec![
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 1]),
                )),
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                    Set::with_capacity(2),
                ))),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
            ],
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Integer>::TableReduce(
            ReduceOperator::Sum,
            0,
            vec![
                ArgumentExpression::Element(ElementExpression::Variable(0)),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 1]),
                )),
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                    Set::with_capacity(2),
                ))),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
            ],
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());
    }

    #[test]
    fn check_continuous_table_vector_expression_ok() {
        let mut model = Model::default();
        let t = TableInterface::<Continuous>::add_table_1d(
            &mut model,
            String::from("t1"),
            vec![0.0, 1.0],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Continuous>::add_table_2d(
            &mut model,
            String::from("t2"),
            vec![vec![0.0, 1.0]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Continuous>::add_table_3d(
            &mut model,
            String::from("t3"),
            vec![vec![vec![0.0, 1.0]]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Continuous>::add_table(
            &mut model,
            String::from("t3"),
            FxHashMap::default(),
            0.0,
        );
        assert!(t.is_ok());

        let expression = expression::TableVectorExpression::Constant(vec![0, 1]);
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableVectorExpression::<Continuous>::Table1D(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableVectorExpression::<Continuous>::Table2D(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableVectorExpression::<Continuous>::Table2DX(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableVectorExpression::<Continuous>::Table2DY(
            0,
            ElementExpression::Constant(0),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableVectorExpression::<Continuous>::Table3D(
            0,
            VectorOrElementExpression::Element(ElementExpression::Constant(0)),
            VectorOrElementExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            )),
            VectorOrElementExpression::Element(ElementExpression::Constant(0)),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableVectorExpression::<Continuous>::Table(
            0,
            vec![
                VectorOrElementExpression::Element(ElementExpression::Constant(0)),
                VectorOrElementExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 1]),
                )),
                VectorOrElementExpression::Element(ElementExpression::Constant(0)),
                VectorOrElementExpression::Element(ElementExpression::Constant(0)),
            ],
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableVectorExpression::<Continuous>::Table2DXReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableVectorExpression::<Continuous>::Table2DYReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableVectorExpression::<Continuous>::Table3DReduce(
            ReduceOperator::Sum,
            0,
            ArgumentExpression::Element(ElementExpression::Constant(0)),
            ArgumentExpression::Vector(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                Set::with_capacity(2),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = expression::TableVectorExpression::<Continuous>::TableReduce(
            ReduceOperator::Sum,
            0,
            vec![
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 1]),
                )),
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                    Set::with_capacity(2),
                ))),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
            ],
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());
    }

    #[test]
    fn check_continuous_table_vector_expression_err() {
        let mut model = Model::default();
        let t = TableInterface::<Continuous>::add_table_1d(
            &mut model,
            String::from("t1"),
            vec![0.0, 1.0],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Continuous>::add_table_2d(
            &mut model,
            String::from("t2"),
            vec![vec![0.0, 1.0]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Continuous>::add_table_3d(
            &mut model,
            String::from("t3"),
            vec![vec![vec![0.0, 1.0]]],
        );
        assert!(t.is_ok());
        let t = TableInterface::<Continuous>::add_table(
            &mut model,
            String::from("t3"),
            FxHashMap::default(),
            0.0,
        );
        assert!(t.is_ok());

        let expression = expression::TableVectorExpression::<Continuous>::Table1D(
            1,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Continuous>::Table1D(
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Continuous>::Table2D(
            1,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Continuous>::Table2D(
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Continuous>::Table2D(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Continuous>::Table2DX(
            1,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Continuous>::Table2DX(
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Continuous>::Table2DX(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            ElementExpression::Variable(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Continuous>::Table2DY(
            1,
            ElementExpression::Constant(0),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Continuous>::Table2DY(
            0,
            ElementExpression::Variable(0),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Continuous>::Table2DY(
            0,
            ElementExpression::Constant(0),
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Continuous>::Table3D(
            1,
            VectorOrElementExpression::Element(ElementExpression::Constant(0)),
            VectorOrElementExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            )),
            VectorOrElementExpression::Element(ElementExpression::Constant(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Continuous>::Table3D(
            0,
            VectorOrElementExpression::Element(ElementExpression::Variable(0)),
            VectorOrElementExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            )),
            VectorOrElementExpression::Element(ElementExpression::Constant(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Continuous>::Table3D(
            0,
            VectorOrElementExpression::Element(ElementExpression::Constant(0)),
            VectorOrElementExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Variable(0),
            )),
            VectorOrElementExpression::Element(ElementExpression::Constant(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Continuous>::Table3D(
            0,
            VectorOrElementExpression::Element(ElementExpression::Constant(0)),
            VectorOrElementExpression::Vector(VectorExpression::Reference(
                ReferenceExpression::Constant(vec![0, 1]),
            )),
            VectorOrElementExpression::Element(ElementExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Continuous>::Table(
            1,
            vec![
                VectorOrElementExpression::Element(ElementExpression::Constant(0)),
                VectorOrElementExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 1]),
                )),
                VectorOrElementExpression::Element(ElementExpression::Constant(0)),
                VectorOrElementExpression::Element(ElementExpression::Constant(0)),
            ],
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Continuous>::Table(
            0,
            vec![
                VectorOrElementExpression::Element(ElementExpression::Variable(0)),
                VectorOrElementExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 1]),
                )),
                VectorOrElementExpression::Element(ElementExpression::Constant(0)),
                VectorOrElementExpression::Element(ElementExpression::Constant(0)),
            ],
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Continuous>::Table2DXReduce(
            ReduceOperator::Sum,
            1,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Continuous>::Table2DXReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Continuous>::Table2DXReduce(
            ReduceOperator::Sum,
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Continuous>::Table2DYReduce(
            ReduceOperator::Sum,
            1,
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Continuous>::Table2DYReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Continuous>::Table2DYReduce(
            ReduceOperator::Sum,
            0,
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
            VectorExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Continuous>::Table3DReduce(
            ReduceOperator::Sum,
            1,
            ArgumentExpression::Element(ElementExpression::Constant(0)),
            ArgumentExpression::Vector(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                Set::with_capacity(2),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Continuous>::Table3DReduce(
            ReduceOperator::Sum,
            0,
            ArgumentExpression::Element(ElementExpression::Variable(0)),
            ArgumentExpression::Vector(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                Set::with_capacity(2),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Continuous>::Table3DReduce(
            ReduceOperator::Sum,
            0,
            ArgumentExpression::Element(ElementExpression::Constant(0)),
            ArgumentExpression::Vector(VectorExpression::Reference(ReferenceExpression::Variable(
                0,
            ))),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                Set::with_capacity(2),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Continuous>::Table3DReduce(
            ReduceOperator::Sum,
            0,
            ArgumentExpression::Element(ElementExpression::Constant(0)),
            ArgumentExpression::Vector(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Continuous>::TableReduce(
            ReduceOperator::Sum,
            1,
            vec![
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 1]),
                )),
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                    Set::with_capacity(2),
                ))),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
            ],
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = expression::TableVectorExpression::<Continuous>::TableReduce(
            ReduceOperator::Sum,
            0,
            vec![
                ArgumentExpression::Element(ElementExpression::Variable(0)),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 1]),
                )),
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Constant(
                    Set::with_capacity(2),
                ))),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
            ],
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());
    }

    #[test]
    fn check_element_expression_ok() {
        let mut model = Model::default();
        let ob = model.add_object_type(String::from("something"), 2);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_element_variable(String::from("v"), ob, 0);
        assert!(v.is_ok());
        let rv = model.add_element_resource_variable(String::from("rv"), ob, true, 0);
        assert!(rv.is_ok());

        let expression = ElementExpression::Constant(0);
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ElementExpression::Variable(0);
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ElementExpression::ResourceVariable(0);
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ElementExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(ElementExpression::Constant(0)),
            Box::new(ElementExpression::Constant(0)),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ElementExpression::Last(Box::new(VectorExpression::Reference(
            ReferenceExpression::Constant(vec![0, 1]),
        )));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ElementExpression::At(
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
            Box::new(ElementExpression::Constant(0)),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ElementExpression::Table(Box::new(TableExpression::Constant(0)));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ElementExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(ElementExpression::Constant(1)),
            Box::new(ElementExpression::Constant(0)),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());
    }

    #[test]
    fn check_element_expression_err() {
        let model = Model::default();

        let expression = ElementExpression::Variable(0);
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ElementExpression::ResourceVariable(0);
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ElementExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(ElementExpression::Variable(0)),
            Box::new(ElementExpression::Constant(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ElementExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(ElementExpression::Constant(0)),
            Box::new(ElementExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ElementExpression::Last(Box::new(VectorExpression::Reference(
            ReferenceExpression::Variable(0),
        )));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ElementExpression::At(
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                0,
            ))),
            Box::new(ElementExpression::Constant(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ElementExpression::At(
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
            Box::new(ElementExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ElementExpression::Table(Box::new(TableExpression::Table1D(
            0,
            ElementExpression::Constant(0),
        )));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ElementExpression::If(
            Box::new(Condition::ComparisonE(
                ComparisonOperator::Eq,
                Box::new(ElementExpression::Variable(0)),
                Box::new(ElementExpression::Constant(0)),
            )),
            Box::new(ElementExpression::Constant(1)),
            Box::new(ElementExpression::Constant(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ElementExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(ElementExpression::Variable(1)),
            Box::new(ElementExpression::Constant(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ElementExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(ElementExpression::Constant(1)),
            Box::new(ElementExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());
    }

    #[test]
    fn check_set_expression_ok() {
        let mut model = Model::default();
        let ob = model.add_object_type(String::from("something"), 2);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let set = model.create_set(ob, &[]);
        assert!(set.is_ok());
        let set = set.unwrap();
        let v = model.add_set_variable(String::from("v"), ob, set);
        assert!(v.is_ok());

        let expression =
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2)));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = SetExpression::Reference(ReferenceExpression::Variable(0));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = SetExpression::Reference(ReferenceExpression::Table(
            TableExpression::Constant(Set::with_capacity(2)),
        ));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = SetExpression::Complement(Box::new(SetExpression::Reference(
            ReferenceExpression::Constant(Set::with_capacity(2)),
        )));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = SetExpression::SetOperation(
            SetOperator::Union,
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                Set::with_capacity(2),
            ))),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                Set::with_capacity(2),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            ElementExpression::Constant(0),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                Set::with_capacity(2),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = SetExpression::FromVector(
            2,
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = SetExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                Set::with_capacity(2),
            ))),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                Set::with_capacity(2),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());
    }

    #[test]
    fn check_set_expression_err() {
        let model = Model::default();

        let expression = SetExpression::Reference(ReferenceExpression::Variable(0));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = SetExpression::Reference(ReferenceExpression::Table(
            TableExpression::Table1D(0, ElementExpression::Constant(0)),
        ));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = SetExpression::Complement(Box::new(SetExpression::Reference(
            ReferenceExpression::Variable(0),
        )));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = SetExpression::SetOperation(
            SetOperator::Union,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                Set::with_capacity(2),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = SetExpression::SetOperation(
            SetOperator::Union,
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                Set::with_capacity(2),
            ))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            ElementExpression::Variable(0),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                Set::with_capacity(2),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            ElementExpression::Constant(0),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = SetExpression::FromVector(
            2,
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                0,
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = SetExpression::If(
            Box::new(Condition::Set(Box::new(SetCondition::IsEmpty(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                Set::with_capacity(2),
            ))),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                Set::with_capacity(2),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = SetExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                Set::with_capacity(2),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = SetExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                Set::with_capacity(2),
            ))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());
    }

    #[test]
    fn check_vector_expression_ok() {
        let mut model = Model::default();
        let ob = model.add_object_type(String::from("something"), 2);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = model.add_vector_variable(String::from("v"), ob, vec![]);
        assert!(v.is_ok());

        let expression = VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1]));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = VectorExpression::Reference(ReferenceExpression::Variable(0));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = VectorExpression::Reference(ReferenceExpression::Table(
            TableExpression::Constant(vec![0, 1]),
        ));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = VectorExpression::Indices(Box::new(VectorExpression::Reference(
            ReferenceExpression::Variable(0),
        )));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = VectorExpression::Reverse(Box::new(VectorExpression::Reference(
            ReferenceExpression::Variable(0),
        )));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = VectorExpression::Pop(Box::new(VectorExpression::Reference(
            ReferenceExpression::Variable(0),
        )));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = VectorExpression::Set(
            ElementExpression::Constant(0),
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                0,
            ))),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = VectorExpression::Push(
            ElementExpression::Constant(0),
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                0,
            ))),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = VectorExpression::FromSet(Box::new(SetExpression::Reference(
            ReferenceExpression::Constant(Set::with_capacity(2)),
        )));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = VectorExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                0,
            ))),
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                0,
            ))),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());
    }

    #[test]
    fn check_vector_expression_err() {
        let model = Model::default();

        let expression = VectorExpression::Reference(ReferenceExpression::Variable(0));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = VectorExpression::Reference(ReferenceExpression::Table(
            TableExpression::Table1D(0, ElementExpression::Constant(0)),
        ));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = VectorExpression::Indices(Box::new(VectorExpression::Reference(
            ReferenceExpression::Variable(0),
        )));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = VectorExpression::Reverse(Box::new(VectorExpression::Reference(
            ReferenceExpression::Variable(0),
        )));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = VectorExpression::Pop(Box::new(VectorExpression::Reference(
            ReferenceExpression::Variable(0),
        )));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = VectorExpression::Set(
            ElementExpression::Variable(0),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = VectorExpression::Set(
            ElementExpression::Constant(0),
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                0,
            ))),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = VectorExpression::Set(
            ElementExpression::Constant(0),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
            ElementExpression::Variable(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = VectorExpression::Push(
            ElementExpression::Variable(0),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = VectorExpression::Push(
            ElementExpression::Constant(0),
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                0,
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = VectorExpression::FromSet(Box::new(SetExpression::Reference(
            ReferenceExpression::Variable(0),
        )));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = VectorExpression::If(
            Box::new(Condition::Set(Box::new(SetCondition::IsEmpty(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = VectorExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                0,
            ))),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = VectorExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                0,
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());
    }

    #[test]
    fn check_integer_expression_ok() {
        let mut model = Model::default();
        let v = model.add_integer_variable(String::from("v"), 0);
        assert!(v.is_ok());
        let rv = model.add_integer_resource_variable(String::from("rv"), true, 0);
        assert!(rv.is_ok());

        let expression = IntegerExpression::Constant(0);
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = IntegerExpression::Variable(0);
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = IntegerExpression::ResourceVariable(0);
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = IntegerExpression::Cost;
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = IntegerExpression::UnaryOperation(
            UnaryOperator::Abs,
            Box::new(IntegerExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = IntegerExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(IntegerExpression::Constant(0)),
            Box::new(IntegerExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = IntegerExpression::Cardinality(SetExpression::Reference(
            ReferenceExpression::Constant(Set::with_capacity(2)),
        ));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = IntegerExpression::Length(VectorExpression::Reference(
            ReferenceExpression::Constant(vec![0, 1]),
        ));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = IntegerExpression::Table(Box::new(NumericTableExpression::Constant(0)));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = IntegerExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(IntegerExpression::Constant(0)),
            Box::new(IntegerExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = IntegerExpression::FromContinuous(
            CastOperator::Ceil,
            Box::new(ContinuousExpression::Constant(0.0)),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression =
            IntegerExpression::Last(Box::new(IntegerVectorExpression::Constant(vec![0, 1])));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = IntegerExpression::Reduce(
            ReduceOperator::Sum,
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = IntegerExpression::At(
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());
    }

    #[test]
    fn check_integer_expression_err() {
        let model = Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        };

        let expression = IntegerExpression::Cost;
        assert!(model.check_expression(&expression, true).is_err());

        let model = Model::default();

        let expression = IntegerExpression::Variable(0);
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerExpression::ResourceVariable(0);
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerExpression::Cost;
        assert!(model.check_expression(&expression, false).is_err());

        let expression = IntegerExpression::UnaryOperation(
            UnaryOperator::Abs,
            Box::new(IntegerExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(IntegerExpression::Variable(0)),
            Box::new(IntegerExpression::Constant(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(IntegerExpression::Constant(0)),
            Box::new(IntegerExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerExpression::Cardinality(SetExpression::Reference(
            ReferenceExpression::Variable(0),
        ));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerExpression::Length(VectorExpression::Reference(
            ReferenceExpression::Variable(0),
        ));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerExpression::Table(Box::new(NumericTableExpression::Table1D(
            0,
            ElementExpression::Constant(0),
        )));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerExpression::If(
            Box::new(Condition::ComparisonI(
                ComparisonOperator::Eq,
                Box::new(IntegerExpression::Variable(0)),
                Box::new(IntegerExpression::Constant(0)),
            )),
            Box::new(IntegerExpression::Constant(0)),
            Box::new(IntegerExpression::Constant(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(IntegerExpression::Variable(0)),
            Box::new(IntegerExpression::Constant(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(IntegerExpression::Constant(0)),
            Box::new(IntegerExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerExpression::FromContinuous(
            CastOperator::Ceil,
            Box::new(ContinuousExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerExpression::Last(Box::new(IntegerVectorExpression::Table(
            Box::new(TableVectorExpression::Table1D(
                0,
                VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            )),
        )));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerExpression::Reduce(
            ReduceOperator::Sum,
            Box::new(IntegerVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                ),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerExpression::At(
            Box::new(IntegerVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                ),
            ))),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerExpression::At(
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
            ElementExpression::Variable(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());
    }

    #[test]
    fn check_continuous_expression_ok() {
        let model = Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        };

        let expression = ContinuousExpression::Cost;
        assert!(model.check_expression(&expression, true).is_ok());

        let mut model = Model::default();
        let v = model.add_continuous_variable(String::from("v"), 0.0);
        assert!(v.is_ok());
        let rv = model.add_continuous_resource_variable(String::from("rv"), true, 0.0);
        assert!(rv.is_ok());

        let expression = ContinuousExpression::Constant(0.0);
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ContinuousExpression::Variable(0);
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ContinuousExpression::ResourceVariable(0);
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ContinuousExpression::Cost;
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ContinuousExpression::UnaryOperation(
            UnaryOperator::Abs,
            Box::new(ContinuousExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ContinuousExpression::ContinuousUnaryOperation(
            ContinuousUnaryOperator::Sqrt,
            Box::new(ContinuousExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ContinuousExpression::Round(
            CastOperator::Ceil,
            Box::new(ContinuousExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ContinuousExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(ContinuousExpression::Constant(0.0)),
            Box::new(ContinuousExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ContinuousExpression::ContinuousBinaryOperation(
            ContinuousBinaryOperator::Pow,
            Box::new(ContinuousExpression::Constant(0.0)),
            Box::new(ContinuousExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ContinuousExpression::Cardinality(SetExpression::Reference(
            ReferenceExpression::Constant(Set::with_capacity(2)),
        ));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ContinuousExpression::Length(VectorExpression::Reference(
            ReferenceExpression::Constant(vec![0, 1]),
        ));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression =
            ContinuousExpression::Table(Box::new(NumericTableExpression::Constant(0.0)));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ContinuousExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(ContinuousExpression::Constant(0.0)),
            Box::new(ContinuousExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression =
            ContinuousExpression::FromInteger(Box::new(IntegerExpression::Constant(0)));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression =
            ContinuousExpression::Last(Box::new(ContinuousVectorExpression::Constant(vec![
                0.0, 1.0,
            ])));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ContinuousExpression::Reduce(
            ReduceOperator::Sum,
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ContinuousExpression::At(
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());
    }

    #[test]
    fn check_continuous_expression_err() {
        let model = Model::default();

        let expression = ContinuousExpression::Variable(0);
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousExpression::ResourceVariable(0);
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousExpression::Cost;
        assert!(model.check_expression(&expression, false).is_err());

        let expression = ContinuousExpression::UnaryOperation(
            UnaryOperator::Abs,
            Box::new(ContinuousExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousExpression::ContinuousUnaryOperation(
            ContinuousUnaryOperator::Sqrt,
            Box::new(ContinuousExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousExpression::Round(
            CastOperator::Ceil,
            Box::new(ContinuousExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(ContinuousExpression::Variable(0)),
            Box::new(ContinuousExpression::Constant(0.0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(ContinuousExpression::Constant(0.0)),
            Box::new(ContinuousExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousExpression::ContinuousBinaryOperation(
            ContinuousBinaryOperator::Pow,
            Box::new(ContinuousExpression::Variable(0)),
            Box::new(ContinuousExpression::Constant(0.0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousExpression::ContinuousBinaryOperation(
            ContinuousBinaryOperator::Pow,
            Box::new(ContinuousExpression::Constant(0.0)),
            Box::new(ContinuousExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousExpression::Cardinality(SetExpression::Reference(
            ReferenceExpression::Variable(0),
        ));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousExpression::Length(VectorExpression::Reference(
            ReferenceExpression::Variable(0),
        ));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousExpression::Table(Box::new(NumericTableExpression::Table1D(
            0,
            ElementExpression::Constant(0),
        )));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousExpression::If(
            Box::new(Condition::ComparisonC(
                ComparisonOperator::Eq,
                Box::new(ContinuousExpression::Variable(0)),
                Box::new(ContinuousExpression::Constant(0.0)),
            )),
            Box::new(ContinuousExpression::Constant(0.0)),
            Box::new(ContinuousExpression::Constant(0.0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(ContinuousExpression::Variable(0)),
            Box::new(ContinuousExpression::Constant(0.0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(ContinuousExpression::Constant(0.0)),
            Box::new(ContinuousExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression =
            ContinuousExpression::FromInteger(Box::new(IntegerExpression::Variable(0)));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousExpression::Last(Box::new(ContinuousVectorExpression::Table(
            Box::new(TableVectorExpression::Table1D(
                0,
                VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            )),
        )));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousExpression::Reduce(
            ReduceOperator::Sum,
            Box::new(ContinuousVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                ),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousExpression::At(
            Box::new(ContinuousVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                ),
            ))),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousExpression::At(
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            ElementExpression::Variable(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());
    }

    #[test]
    fn check_integer_vector_expression_ok() {
        let model = Model::default();

        let expression = IntegerVectorExpression::Constant(vec![0, 1]);
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression =
            IntegerVectorExpression::Reverse(Box::new(IntegerVectorExpression::Constant(vec![
                0, 1,
            ])));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression =
            IntegerVectorExpression::Pop(Box::new(IntegerVectorExpression::Constant(vec![0, 1])));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = IntegerVectorExpression::UnaryOperation(
            UnaryOperator::Abs,
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = IntegerVectorExpression::Push(
            IntegerExpression::Constant(0),
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = IntegerVectorExpression::BinaryOperationX(
            BinaryOperator::Add,
            IntegerExpression::Constant(0),
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = IntegerVectorExpression::BinaryOperationY(
            BinaryOperator::Add,
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
            IntegerExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = IntegerVectorExpression::Set(
            IntegerExpression::Constant(0),
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = IntegerVectorExpression::VectorOperation(
            BinaryOperator::Add,
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression =
            IntegerVectorExpression::Table(Box::new(TableVectorExpression::Constant(vec![0, 1])));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = IntegerVectorExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = IntegerVectorExpression::FromContinuous(
            CastOperator::Ceil,
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());
    }

    #[test]
    fn check_integer_vector_expression_err() {
        let model = Model::default();

        let expression = IntegerVectorExpression::Reverse(Box::new(
            IntegerVectorExpression::Table(Box::new(TableVectorExpression::Table1D(
                0,
                VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            ))),
        ));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerVectorExpression::Pop(Box::new(IntegerVectorExpression::Table(
            Box::new(TableVectorExpression::Table1D(
                0,
                VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            )),
        )));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerVectorExpression::UnaryOperation(
            UnaryOperator::Abs,
            Box::new(IntegerVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                ),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerVectorExpression::Push(
            IntegerExpression::Variable(0),
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerVectorExpression::Push(
            IntegerExpression::Constant(0),
            Box::new(IntegerVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                ),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerVectorExpression::BinaryOperationX(
            BinaryOperator::Add,
            IntegerExpression::Variable(0),
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerVectorExpression::BinaryOperationX(
            BinaryOperator::Add,
            IntegerExpression::Constant(0),
            Box::new(IntegerVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                ),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerVectorExpression::BinaryOperationY(
            BinaryOperator::Add,
            Box::new(IntegerVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                ),
            ))),
            IntegerExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerVectorExpression::BinaryOperationY(
            BinaryOperator::Add,
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
            IntegerExpression::Variable(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerVectorExpression::Set(
            IntegerExpression::Variable(0),
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerVectorExpression::Set(
            IntegerExpression::Constant(0),
            Box::new(IntegerVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                ),
            ))),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerVectorExpression::Set(
            IntegerExpression::Constant(0),
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
            ElementExpression::Variable(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerVectorExpression::VectorOperation(
            BinaryOperator::Add,
            Box::new(IntegerVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                ),
            ))),
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerVectorExpression::VectorOperation(
            BinaryOperator::Add,
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
            Box::new(IntegerVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                ),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerVectorExpression::Table(Box::new(TableVectorExpression::Table1D(
            0,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
        )));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerVectorExpression::If(
            Box::new(Condition::ComparisonI(
                ComparisonOperator::Eq,
                Box::new(IntegerExpression::Constant(0)),
                Box::new(IntegerExpression::Variable(0)),
            )),
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerVectorExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(IntegerVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                ),
            ))),
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerVectorExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
            Box::new(IntegerVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                ),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = IntegerVectorExpression::FromContinuous(
            CastOperator::Ceil,
            Box::new(ContinuousVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                ),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());
    }

    #[test]
    fn check_continuous_vector_expression_ok() {
        let model = Model::default();

        let expression = ContinuousVectorExpression::Constant(vec![0.0, 1.0]);
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ContinuousVectorExpression::Reverse(Box::new(
            ContinuousVectorExpression::Constant(vec![0.0, 1.0]),
        ));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression =
            ContinuousVectorExpression::Pop(Box::new(ContinuousVectorExpression::Constant(vec![
                0.0, 1.0,
            ])));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ContinuousVectorExpression::UnaryOperation(
            UnaryOperator::Abs,
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ContinuousVectorExpression::ContinuousUnaryOperation(
            ContinuousUnaryOperator::Sqrt,
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ContinuousVectorExpression::Round(
            CastOperator::Ceil,
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ContinuousVectorExpression::Push(
            ContinuousExpression::Constant(0.0),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ContinuousVectorExpression::BinaryOperationX(
            BinaryOperator::Add,
            ContinuousExpression::Constant(0.0),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ContinuousVectorExpression::BinaryOperationY(
            BinaryOperator::Add,
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            ContinuousExpression::Constant(0.0),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ContinuousVectorExpression::ContinuousBinaryOperationX(
            ContinuousBinaryOperator::Pow,
            ContinuousExpression::Constant(0.0),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ContinuousVectorExpression::ContinuousBinaryOperationY(
            ContinuousBinaryOperator::Pow,
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            ContinuousExpression::Constant(0.0),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ContinuousVectorExpression::Set(
            ContinuousExpression::Constant(0.0),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ContinuousVectorExpression::VectorOperation(
            BinaryOperator::Add,
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ContinuousVectorExpression::ContinuousVectorOperation(
            ContinuousBinaryOperator::Pow,
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression =
            ContinuousVectorExpression::Table(Box::new(TableVectorExpression::Constant(vec![
                0.0, 1.0,
            ])));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ContinuousVectorExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = ContinuousVectorExpression::FromInteger(Box::new(
            IntegerVectorExpression::Constant(vec![0, 1]),
        ));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());
    }

    #[test]
    fn check_continuous_vector_expression_err() {
        let model = Model::default();

        let expression = ContinuousVectorExpression::Reverse(Box::new(
            ContinuousVectorExpression::Table(Box::new(TableVectorExpression::Table1D(
                0,
                VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            ))),
        ));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousVectorExpression::Pop(Box::new(
            ContinuousVectorExpression::Table(Box::new(TableVectorExpression::Table1D(
                0,
                VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            ))),
        ));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousVectorExpression::UnaryOperation(
            UnaryOperator::Abs,
            Box::new(ContinuousVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                ),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousVectorExpression::ContinuousUnaryOperation(
            ContinuousUnaryOperator::Sqrt,
            Box::new(ContinuousVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                ),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousVectorExpression::Round(
            CastOperator::Ceil,
            Box::new(ContinuousVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                ),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousVectorExpression::Push(
            ContinuousExpression::Variable(0),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousVectorExpression::Push(
            ContinuousExpression::Constant(0.0),
            Box::new(ContinuousVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                ),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousVectorExpression::BinaryOperationX(
            BinaryOperator::Add,
            ContinuousExpression::Variable(0),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousVectorExpression::BinaryOperationX(
            BinaryOperator::Add,
            ContinuousExpression::Constant(0.0),
            Box::new(ContinuousVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                ),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousVectorExpression::BinaryOperationY(
            BinaryOperator::Add,
            Box::new(ContinuousVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                ),
            ))),
            ContinuousExpression::Constant(0.0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousVectorExpression::BinaryOperationY(
            BinaryOperator::Add,
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            ContinuousExpression::Variable(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousVectorExpression::ContinuousBinaryOperationX(
            ContinuousBinaryOperator::Pow,
            ContinuousExpression::Variable(0),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousVectorExpression::ContinuousBinaryOperationX(
            ContinuousBinaryOperator::Pow,
            ContinuousExpression::Constant(0.0),
            Box::new(ContinuousVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                ),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousVectorExpression::ContinuousBinaryOperationY(
            ContinuousBinaryOperator::Pow,
            Box::new(ContinuousVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                ),
            ))),
            ContinuousExpression::Constant(0.0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousVectorExpression::ContinuousBinaryOperationY(
            ContinuousBinaryOperator::Pow,
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            ContinuousExpression::Variable(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousVectorExpression::Set(
            ContinuousExpression::Variable(0),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousVectorExpression::Set(
            ContinuousExpression::Constant(0.0),
            Box::new(ContinuousVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                ),
            ))),
            ElementExpression::Constant(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousVectorExpression::Set(
            ContinuousExpression::Constant(0.0),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            ElementExpression::Variable(0),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousVectorExpression::VectorOperation(
            BinaryOperator::Add,
            Box::new(ContinuousVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                ),
            ))),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousVectorExpression::VectorOperation(
            BinaryOperator::Add,
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            Box::new(ContinuousVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                ),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousVectorExpression::ContinuousVectorOperation(
            ContinuousBinaryOperator::Pow,
            Box::new(ContinuousVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                ),
            ))),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousVectorExpression::ContinuousVectorOperation(
            ContinuousBinaryOperator::Pow,
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            Box::new(ContinuousVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                ),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression =
            ContinuousVectorExpression::Table(Box::new(TableVectorExpression::Table1D(
                0,
                VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            )));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousVectorExpression::If(
            Box::new(Condition::ComparisonC(
                ComparisonOperator::Eq,
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Variable(0)),
            )),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousVectorExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(ContinuousVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                ),
            ))),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousVectorExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
            Box::new(ContinuousVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                ),
            ))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = ContinuousVectorExpression::FromInteger(Box::new(
            IntegerVectorExpression::Table(Box::new(TableVectorExpression::Table1D(
                0,
                VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            ))),
        ));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());
    }

    #[test]
    fn check_condition_ok() {
        let model = Model::default();

        let expression = Condition::Constant(true);
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = Condition::Not(Box::new(Condition::Constant(true)));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = Condition::And(
            Box::new(Condition::Constant(true)),
            Box::new(Condition::Constant(true)),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = Condition::Or(
            Box::new(Condition::Constant(true)),
            Box::new(Condition::Constant(true)),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = Condition::ComparisonE(
            ComparisonOperator::Eq,
            Box::new(ElementExpression::Constant(0)),
            Box::new(ElementExpression::Constant(0)),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = Condition::ComparisonI(
            ComparisonOperator::Eq,
            Box::new(IntegerExpression::Constant(0)),
            Box::new(IntegerExpression::Constant(0)),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = Condition::ComparisonC(
            ComparisonOperator::Eq,
            Box::new(ContinuousExpression::Constant(0.0)),
            Box::new(ContinuousExpression::Constant(0.0)),
        );
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = Condition::Set(Box::new(SetCondition::Constant(true)));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = Condition::Set(Box::new(SetCondition::IsIn(
            ElementExpression::Constant(0),
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        )));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = Condition::Set(Box::new(SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        )));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = Condition::Set(Box::new(SetCondition::IsEmpty(SetExpression::Reference(
            ReferenceExpression::Constant(Set::with_capacity(2)),
        ))));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());

        let expression = Condition::Table(Box::new(TableExpression::Constant(true)));
        assert!(model.check_expression(&expression, false).is_ok());
        assert!(model.check_expression(&expression, true).is_ok());
    }

    #[test]
    fn check_condition_err() {
        let model = Model::default();

        let expression = Condition::Not(Box::new(Condition::Table(Box::new(
            TableExpression::Table1D(0, ElementExpression::Constant(0)),
        ))));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = Condition::And(
            Box::new(Condition::Table(Box::new(TableExpression::Table1D(
                0,
                ElementExpression::Constant(0),
            )))),
            Box::new(Condition::Constant(true)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = Condition::And(
            Box::new(Condition::Constant(true)),
            Box::new(Condition::Table(Box::new(TableExpression::Table1D(
                0,
                ElementExpression::Constant(0),
            )))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = Condition::Or(
            Box::new(Condition::Table(Box::new(TableExpression::Table1D(
                0,
                ElementExpression::Constant(0),
            )))),
            Box::new(Condition::Constant(true)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = Condition::Or(
            Box::new(Condition::Constant(true)),
            Box::new(Condition::Table(Box::new(TableExpression::Table1D(
                0,
                ElementExpression::Constant(0),
            )))),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = Condition::ComparisonE(
            ComparisonOperator::Eq,
            Box::new(ElementExpression::Variable(0)),
            Box::new(ElementExpression::Constant(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = Condition::ComparisonE(
            ComparisonOperator::Eq,
            Box::new(ElementExpression::Constant(0)),
            Box::new(ElementExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = Condition::ComparisonI(
            ComparisonOperator::Eq,
            Box::new(IntegerExpression::Variable(0)),
            Box::new(IntegerExpression::Constant(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = Condition::ComparisonI(
            ComparisonOperator::Eq,
            Box::new(IntegerExpression::Constant(0)),
            Box::new(IntegerExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = Condition::ComparisonC(
            ComparisonOperator::Eq,
            Box::new(ContinuousExpression::Variable(0)),
            Box::new(ContinuousExpression::Constant(0.0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = Condition::ComparisonC(
            ComparisonOperator::Eq,
            Box::new(ContinuousExpression::Constant(0.0)),
            Box::new(ContinuousExpression::Variable(0)),
        );
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = Condition::Set(Box::new(SetCondition::IsIn(
            ElementExpression::Variable(0),
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        )));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = Condition::Set(Box::new(SetCondition::IsIn(
            ElementExpression::Constant(0),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        )));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = Condition::Set(Box::new(SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
        )));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = Condition::Set(Box::new(SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(2))),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        )));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = Condition::Set(Box::new(SetCondition::IsEmpty(SetExpression::Reference(
            ReferenceExpression::Variable(0),
        ))));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());

        let expression = Condition::Table(Box::new(TableExpression::Table1D(
            0,
            ElementExpression::Constant(0),
        )));
        assert!(model.check_expression(&expression, false).is_err());
        assert!(model.check_expression(&expression, true).is_err());
    }
}
