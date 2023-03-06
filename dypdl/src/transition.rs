use crate::effect::Effect;
use crate::expression::{
    Condition, ContinuousExpression, ElementExpression, IntegerExpression, ReferenceExpression,
    SetCondition, SetExpression, VectorExpression,
};
use crate::grounded_condition;
use crate::state::{
    ContinuousResourceVariable, ContinuousVariable, ElementResourceVariable, ElementVariable,
    IntegerResourceVariable, IntegerVariable, SetVariable, State, StateInterface, VectorVariable,
};
use crate::table_registry;
use crate::util::ModelErr;
use crate::variable_type::{Element, FromNumeric, Numeric};
use std::fmt::Debug;

/// Wrapper for an integer expression or a continuous expression.
#[derive(Debug, PartialEq, Clone)]
pub enum CostExpression {
    /// Integer numeric expression.
    Integer(IntegerExpression),
    /// Continuous numeric expression.
    Continuous(ContinuousExpression),
}

impl Default for CostExpression {
    /// Returns `CostExpression::Integer(IntegerExpression::Cost)`.
    fn default() -> Self {
        Self::Integer(IntegerExpression::Cost)
    }
}

impl From<IntegerExpression> for CostExpression {
    fn from(cost: IntegerExpression) -> Self {
        Self::Integer(cost)
    }
}

impl From<ContinuousExpression> for CostExpression {
    fn from(cost: ContinuousExpression) -> Self {
        Self::Continuous(cost)
    }
}

impl CostExpression {
    /// Returns the evaluation result.
    ///
    /// # Panics
    ///
    /// Panics if the cost of the transition state is used or a min/max reduce operation is performed on an empty set or vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let variable = model.add_integer_variable("x", 2).unwrap();
    /// let state = model.target.clone();
    ///
    /// let expression = CostExpression::from(IntegerExpression::from(variable));
    /// assert_eq!(expression.eval::<Integer, _>(&state, &model.table_registry), 2);
    /// ```
    #[inline]
    pub fn eval<T: Numeric, U: StateInterface>(
        &self,
        state: &U,
        registry: &table_registry::TableRegistry,
    ) -> T {
        match self {
            Self::Integer(expression) => T::from(expression.eval(state, registry)),
            Self::Continuous(expression) => T::from(expression.eval(state, registry)),
        }
    }

    /// Returns the evaluation result.
    ///
    /// # Panics
    ///
    /// Panics if a min/max reduce operation is performed on an empty set or vector.
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let variable = model.add_integer_variable("x", 2).unwrap();
    /// let state = model.target.clone();
    ///
    /// let expression = CostExpression::from(variable + IntegerExpression::Cost);
    /// assert_eq!(expression.eval_cost(1, &state, &model.table_registry), 3);
    /// ```
    pub fn eval_cost<T: Numeric, U: StateInterface>(
        &self,
        cost: T,
        state: &U,
        registry: &table_registry::TableRegistry,
    ) -> T {
        match self {
            Self::Integer(expression) => {
                T::from(expression.eval_cost(FromNumeric::from(cost), state, registry))
            }
            Self::Continuous(expression) => {
                T::from(expression.eval_cost(FromNumeric::from(cost), state, registry))
            }
        }
    }

    /// Returns a simplified version by precomputation.
    ///
    /// # Panics
    ///
    /// Panics if a min/max reduce operation is performed on an empty set or vector.
    pub fn simplify(&self, registry: &table_registry::TableRegistry) -> CostExpression {
        match self {
            Self::Integer(expression) => Self::Integer(expression.simplify(registry)),
            Self::Continuous(expression) => Self::Continuous(expression.simplify(registry)),
        }
    }
}

/// Trait representing a transition.
pub trait TransitionInterface {
    /// Returns true if the transition is applicable and false otherwise.
    fn is_applicable<T: StateInterface>(
        &self,
        state: &T,
        registry: &table_registry::TableRegistry,
    ) -> bool;

    /// Returns the transitioned state.
    fn apply<S: StateInterface, T: From<State>>(
        &self,
        state: &S,
        registry: &table_registry::TableRegistry,
    ) -> T;

    /// Returns the evaluation result of the cost expression.
    fn eval_cost<U: Numeric, T: StateInterface>(
        &self,
        cost: U,
        state: &T,
        registry: &table_registry::TableRegistry,
    ) -> U;
}

/// Transition.
#[derive(Debug, PartialEq, Clone, Default)]
pub struct Transition {
    /// Name of the transition.
    pub name: String,
    //// Names of parameters.
    pub parameter_names: Vec<String>,
    /// The values of parameters.
    pub parameter_values: Vec<Element>,
    /// Pairs of an index of a set variable and a parameter.
    /// A parameter must be included in the corresponding variable to be applicable.
    pub elements_in_set_variable: Vec<(usize, Element)>,
    /// Triplets of an index of a vector variable, a parameter, and the capacity.
    /// A parameter must be included in the corresponding variable to be applicable.
    pub elements_in_vector_variable: Vec<(usize, Element, usize)>,
    /// Preconditions.
    pub preconditions: Vec<grounded_condition::GroundedCondition>,
    /// Effect.
    pub effect: Effect,
    /// Cost expression.
    pub cost: CostExpression,
}

impl TransitionInterface for Transition {
    /// Returns true if the transition is applicable and false otherwise.
    ///
    /// # Panics
    ///
    /// Panics if the cost of the transition state is used or a min/max reduce operation is performed on an empty set or vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let variable = model.add_integer_variable("x", 2).unwrap();
    /// let state = model.target.clone();
    ///
    /// let mut transition = Transition::new("transition");
    /// assert!(transition.is_applicable(&state, &model.table_registry));
    ///
    /// let condition = Condition::comparison_i(ComparisonOperator::Ge, variable, 0);;
    /// transition.add_precondition(condition);
    /// assert!(transition.is_applicable(&state, &model.table_registry));
    ///
    /// let condition = Condition::comparison_i(ComparisonOperator::Le, variable, 1);;
    /// transition.add_precondition(condition);
    /// assert!(!transition.is_applicable(&state, &model.table_registry));
    /// ```
    fn is_applicable<S: StateInterface>(
        &self,
        state: &S,
        registry: &table_registry::TableRegistry,
    ) -> bool {
        for (i, v) in &self.elements_in_set_variable {
            if !state.get_set_variable(*i).contains(*v) {
                return false;
            }
        }
        for (i, v, _) in &self.elements_in_vector_variable {
            if !state.get_vector_variable(*i).contains(v) {
                return false;
            }
        }
        self.preconditions
            .iter()
            .all(|c| c.is_satisfied(state, registry))
    }

    /// Returns the transitioned state.
    ///
    /// # Panics
    ///
    /// Panics if the cost of the transition state is used or a min/max reduce operation is performed on an empty set or vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let variable = model.add_integer_variable("x", 2).unwrap();
    /// let state = model.target.clone();
    ///
    /// let mut transition = Transition::new("transition");
    /// transition.add_effect(variable, variable + 1);
    /// let state: State = transition.apply(&state, &model.table_registry);
    /// assert_eq!(state.get_integer_variable(variable.id()), 3);
    /// ```
    #[inline]
    fn apply<S: StateInterface, T: From<State>>(
        &self,
        state: &S,
        registry: &table_registry::TableRegistry,
    ) -> T {
        state.apply_effect(&self.effect, registry)
    }

    /// Returns the evaluation result of the cost expression.
    ///
    /// # Panics
    ///
    /// Panics if a min/max reduce operation is performed on an empty set or vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let variable = model.add_integer_variable("x", 2).unwrap();
    /// let state = model.target.clone();
    ///
    /// let mut transition = Transition::new("transition");
    /// transition.set_cost(IntegerExpression::Cost + variable);
    /// assert_eq!(transition.eval_cost(1, &state, &model.table_registry), 3);
    /// ```
    #[inline]
    fn eval_cost<T: Numeric, S: StateInterface>(
        &self,
        cost: T,
        state: &S,
        registry: &table_registry::TableRegistry,
    ) -> T {
        self.cost.eval_cost(cost, state, registry)
    }
}

impl Transition {
    /// Returns the name of transition considering parameters.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let transition = Transition::new("transition");
    /// assert_eq!(transition.get_full_name(), String::from("transition"));
    /// ```
    pub fn get_full_name(&self) -> String {
        let mut full_name = self.name.clone();
        for (name, value) in self
            .parameter_names
            .iter()
            .zip(self.parameter_values.iter())
        {
            full_name += format!(" {}:{}", name, value).as_str();
        }
        full_name
    }

    /// Returns a new transition with the name.
    pub fn new<T>(name: T) -> Transition
    where
        String: From<T>,
    {
        Transition {
            name: String::from(name),
            ..Default::default()
        }
    }

    /// Sets the cost expression;
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut transition = Transition::new("transition");
    /// transition.set_cost(IntegerExpression::Cost + 1);
    /// ```
    pub fn set_cost<T>(&mut self, cost: T)
    where
        CostExpression: From<T>,
    {
        self.cost = CostExpression::from(cost)
    }

    /// Gets preconditions.
    ///
    /// Note that the preconditions and their order might be changed due to internal optimizations.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let variable = model.add_integer_variable("x", 2).unwrap();
    /// let state = model.target.clone();
    ///
    /// let mut transition = Transition::new("transition");
    /// let condition = Condition::comparison_i(ComparisonOperator::Ge, variable, 0);;
    /// transition.add_precondition(condition.clone());
    ///
    /// let preconditions = transition.get_preconditions();
    /// assert_eq!(preconditions.len(), 1);
    /// assert!(preconditions[0].eval(&state, &model.table_registry));
    /// ```
    pub fn get_preconditions(&self) -> Vec<Condition> {
        let mut result = Vec::with_capacity(
            self.elements_in_set_variable.len()
                + self.elements_in_vector_variable.len()
                + self.preconditions.len(),
        );

        for (i, e) in &self.elements_in_set_variable {
            result.push(Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Constant(*e),
                SetExpression::Reference(ReferenceExpression::Variable(*i)),
            ))));
        }

        for (i, e, capacity) in &self.elements_in_vector_variable {
            result.push(Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Constant(*e),
                SetExpression::FromVector(
                    *capacity,
                    Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                        *i,
                    ))),
                ),
            ))));
        }

        for condition in &self.preconditions {
            result.push(Condition::from(condition.clone()));
        }

        result
    }

    /// Adds a precondition;
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    ///
    /// let mut model = Model::default();
    /// let variable = model.add_integer_variable("x", 2).unwrap();
    /// let state = model.target.clone();
    ///
    /// let mut transition = Transition::new("transition");
    /// let condition = Condition::comparison_i(ComparisonOperator::Ge, variable, 0);;
    /// transition.add_precondition(condition);
    /// ```
    pub fn add_precondition(&mut self, condition: Condition) {
        match &condition {
            Condition::Set(condition) => match condition.as_ref() {
                SetCondition::IsIn(
                    ElementExpression::Constant(e),
                    SetExpression::Reference(ReferenceExpression::Variable(i)),
                ) => {
                    self.elements_in_set_variable.push((*i, *e));
                    return;
                }
                SetCondition::IsIn(
                    ElementExpression::Constant(e),
                    SetExpression::FromVector(capacity, v),
                ) => {
                    if let VectorExpression::Reference(ReferenceExpression::Variable(i)) =
                        v.as_ref()
                    {
                        self.elements_in_vector_variable.push((*i, *e, *capacity));
                        return;
                    }
                }
                _ => {}
            },
            Condition::Constant(true) => {
                eprintln!("a precondition is always satisfied");
            }
            Condition::Constant(false) => {
                eprintln!("a precondition is never satisfied");
            }
            _ => {}
        }
        self.preconditions
            .push(grounded_condition::GroundedCondition::from(condition))
    }
}

/// Trait for adding an effect.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
///
/// let mut model = Model::default();
/// let variable = model.add_integer_variable("variable", 0).unwrap();
///
/// let mut transition = Transition::new("transition");
/// assert!(transition.add_effect(variable, variable + 1).is_ok());
/// ```
pub trait AddEffect<T, U> {
    /// Adds an effect.
    ///
    /// # Errors
    ///
    /// If an effect is already defined for the variable.
    fn add_effect<V>(&mut self, v: T, expression: V) -> Result<(), ModelErr>
    where
        U: From<V>;
}

macro_rules! impl_add_effect {
    ($T:ty,$U:ty,$x:ident) => {
        impl AddEffect<$T, $U> for Transition {
            fn add_effect<V>(&mut self, v: $T, expression: V) -> Result<(), ModelErr>
            where
                $U: From<V>,
            {
                let expression = <$U>::from(expression);
                for (i, _) in &self.effect.$x {
                    if *i == v.id() {
                        return Err(ModelErr::new(format!(
                            "the transition already has an effect on variable id {}",
                            *i
                        )));
                    }
                }
                self.effect.$x.push((v.id(), expression));
                Ok(())
            }
        }
    };
}

impl_add_effect!(SetVariable, SetExpression, set_effects);
impl_add_effect!(VectorVariable, VectorExpression, vector_effects);
impl_add_effect!(ElementVariable, ElementExpression, element_effects);
impl_add_effect!(
    ElementResourceVariable,
    ElementExpression,
    element_resource_effects
);
impl_add_effect!(IntegerVariable, IntegerExpression, integer_effects);
impl_add_effect!(
    IntegerResourceVariable,
    IntegerExpression,
    integer_resource_effects
);
impl_add_effect!(ContinuousVariable, ContinuousExpression, continuous_effects);
impl_add_effect!(
    ContinuousResourceVariable,
    ContinuousExpression,
    continuous_resource_effects
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::*;
    use crate::state;
    use crate::table;
    use crate::table_data;
    use crate::variable_type::*;
    use rustc_hash::FxHashMap;

    fn generate_registry() -> table_registry::TableRegistry {
        let tables_1d = vec![table::Table1D::new(vec![10, 20, 30])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("f1"), 0);

        let tables_2d = vec![table::Table2D::new(vec![
            vec![10, 20, 30],
            vec![40, 50, 60],
        ])];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("f2"), 0);

        table_registry::TableRegistry {
            integer_tables: table_data::TableData {
                tables_1d,
                name_to_table_1d,
                tables_2d,
                name_to_table_2d,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn generate_state() -> state::State {
        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(0);
        set2.insert(1);
        state::State {
            signature_variables: state::SignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2], vec![1, 2]],
                element_variables: vec![1, 2],
                integer_variables: vec![1, 2, 3],
                continuous_variables: vec![1.0, 2.0, 3.0],
            },
            resource_variables: state::ResourceVariables {
                element_variables: vec![0, 1],
                integer_variables: vec![4, 5, 6],
                continuous_variables: vec![4.0, 5.0, 6.0],
            },
        }
    }

    #[test]
    fn cost_expression_default() {
        let expression = CostExpression::default();
        assert_eq!(expression, CostExpression::Integer(IntegerExpression::Cost));
    }

    #[test]
    fn cost_expression_from() {
        let expression = CostExpression::from(IntegerExpression::Cost);
        assert_eq!(expression, CostExpression::Integer(IntegerExpression::Cost));
        let expression = CostExpression::from(ContinuousExpression::Cost);
        assert_eq!(
            expression,
            CostExpression::Continuous(ContinuousExpression::Cost)
        );
    }

    #[test]
    fn cost_expression_eval() {
        let state = generate_state();
        let registry = generate_registry();

        let expression = CostExpression::Integer(IntegerExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(IntegerExpression::Constant(1)),
            Box::new(IntegerExpression::Constant(1)),
        ));
        assert_eq!(expression.eval_cost(0, &state, &registry), 2);

        let expression = CostExpression::Continuous(ContinuousExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(ContinuousExpression::Constant(1.0)),
            Box::new(ContinuousExpression::Constant(1.0)),
        ));
        assert_eq!(expression.eval_cost(0.0, &state, &registry), 2.0);
    }

    #[test]
    fn cost_expression_eval_cost() {
        let state = generate_state();
        let registry = generate_registry();

        let expression = CostExpression::Integer(IntegerExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(IntegerExpression::Cost),
            Box::new(IntegerExpression::Constant(1)),
        ));
        assert_eq!(expression.eval_cost(0, &state, &registry), 1);

        let expression = CostExpression::Continuous(ContinuousExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(ContinuousExpression::Cost),
            Box::new(ContinuousExpression::Constant(1.0)),
        ));
        assert_eq!(expression.eval_cost(0.0, &state, &registry), 1.0);
    }

    #[test]
    fn cost_expression_simplify_integer() {
        let registry = generate_registry();
        let expression = CostExpression::Integer(IntegerExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(IntegerExpression::Constant(1)),
            Box::new(IntegerExpression::Constant(1)),
        ));
        assert_eq!(
            expression.simplify(&registry),
            CostExpression::Integer(IntegerExpression::Constant(2))
        )
    }

    #[test]
    fn cost_expression_simplify_continuous() {
        let registry = generate_registry();
        let expression = CostExpression::Continuous(ContinuousExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(ContinuousExpression::Constant(1.0)),
            Box::new(ContinuousExpression::Constant(1.0)),
        ));
        assert_eq!(
            expression.simplify(&registry),
            CostExpression::Continuous(ContinuousExpression::Constant(2.0))
        )
    }

    #[test]
    fn applicable() {
        let state = generate_state();
        let registry = generate_registry();
        let set_condition = grounded_condition::GroundedCondition {
            condition: Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Constant(0),
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            ))),
            ..Default::default()
        };
        let numeric_condition = grounded_condition::GroundedCondition {
            condition: Condition::ComparisonI(
                ComparisonOperator::Ge,
                Box::new(IntegerExpression::Variable(0)),
                Box::new(IntegerExpression::Constant(1)),
            ),
            ..Default::default()
        };

        let transition = Transition {
            name: String::from(""),
            preconditions: vec![set_condition, numeric_condition],
            cost: CostExpression::Integer(IntegerExpression::Constant(0)),
            ..Default::default()
        };
        assert!(transition.is_applicable(&state, &registry));

        let transition = Transition {
            name: String::from(""),
            elements_in_set_variable: vec![(0, 0), (1, 1)],
            elements_in_vector_variable: vec![(0, 0, 3), (1, 2, 3)],
            cost: CostExpression::Integer(IntegerExpression::Constant(0)),
            ..Default::default()
        };
        assert!(transition.is_applicable(&state, &registry));
    }

    #[test]
    fn not_applicable() {
        let state = generate_state();
        let registry = generate_registry();
        let set_condition = grounded_condition::GroundedCondition {
            condition: Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Constant(0),
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            ))),
            ..Default::default()
        };
        let numeric_condition = grounded_condition::GroundedCondition {
            condition: Condition::ComparisonI(
                ComparisonOperator::Le,
                Box::new(IntegerExpression::Variable(0)),
                Box::new(IntegerExpression::Constant(1)),
            ),
            ..Default::default()
        };

        let transition = Transition {
            name: String::from(""),
            preconditions: vec![set_condition, numeric_condition],
            cost: CostExpression::Integer(IntegerExpression::Constant(0)),
            ..Default::default()
        };
        assert!(transition.is_applicable(&state, &registry));

        let transition = Transition {
            name: String::from(""),
            elements_in_set_variable: vec![(0, 1), (1, 1)],
            elements_in_vector_variable: vec![(0, 0, 3), (1, 2, 3)],
            cost: CostExpression::Integer(IntegerExpression::Constant(0)),
            ..Default::default()
        };
        assert!(!transition.is_applicable(&state, &registry));

        let transition = Transition {
            name: String::from(""),
            elements_in_set_variable: vec![(0, 1), (1, 1)],
            elements_in_vector_variable: vec![(0, 1, 3), (1, 2, 3)],
            cost: CostExpression::Integer(IntegerExpression::Constant(0)),
            ..Default::default()
        };
        assert!(!transition.is_applicable(&state, &registry));
    }

    #[test]
    fn apply_effects() {
        let state = generate_state();
        let registry = generate_registry();
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
        let element_resource_effect1 = ElementExpression::Constant(1);
        let element_resource_effect2 = ElementExpression::Constant(0);
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
        let transition = Transition {
            name: String::from(""),
            effect: Effect {
                set_effects: vec![(0, set_effect1), (1, set_effect2)],
                vector_effects: vec![(0, vector_effect1), (1, vector_effect2)],
                element_effects: vec![(0, element_effect1), (1, element_effect2)],
                integer_effects: vec![(0, integer_effect1), (1, integer_effect2)],
                continuous_effects: vec![(0, continuous_effect1), (1, continuous_effect2)],
                element_resource_effects: vec![
                    (0, element_resource_effect1),
                    (1, element_resource_effect2),
                ],
                integer_resource_effects: vec![
                    (0, integer_resource_effect1),
                    (1, integer_resource_effect2),
                ],
                continuous_resource_effects: vec![
                    (0, continuous_resource_effect1),
                    (1, continuous_resource_effect2),
                ],
            },
            cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(IntegerExpression::Cost),
                Box::new(IntegerExpression::Constant(1)),
            )),
            ..Default::default()
        };

        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(1);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(1);
        let expected = state::State {
            signature_variables: state::SignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2, 1], vec![1, 2, 0]],
                element_variables: vec![2, 1],
                integer_variables: vec![0, 4, 3],
                continuous_variables: vec![0.0, 4.0, 3.0],
            },
            resource_variables: state::ResourceVariables {
                element_variables: vec![1, 0],
                integer_variables: vec![5, 2, 6],
                continuous_variables: vec![5.0, 2.5, 6.0],
            },
        };
        let successor: State = transition.apply(&state, &registry);
        assert_eq!(successor, expected);
    }

    #[test]
    fn eval_cost() {
        let state = generate_state();
        let registry = generate_registry();

        let transition = Transition {
            cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(IntegerExpression::Cost),
                Box::new(IntegerExpression::Constant(1)),
            )),
            ..Default::default()
        };
        assert_eq!(transition.eval_cost(0, &state, &registry), 1);

        let transition = Transition {
            cost: CostExpression::Continuous(ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Cost),
                Box::new(ContinuousExpression::Constant(1.0)),
            )),
            ..Default::default()
        };
        assert_eq!(transition.eval_cost(0.0, &state, &registry), 1.0);
    }

    #[test]
    fn get_full_name() {
        let transition = Transition {
            name: String::from("transition"),
            parameter_names: vec![String::from("param1"), String::from("param2")],
            parameter_values: vec![0, 1],
            ..Default::default()
        };
        assert_eq!(
            transition.get_full_name(),
            String::from("transition param1:0 param2:1")
        );
    }

    #[test]
    fn new() {
        let transition = Transition::new("t");
        assert_eq!(
            transition,
            Transition {
                name: String::from("t"),
                ..Default::default()
            }
        );
    }

    #[test]
    fn set_cost() {
        let mut transition = Transition::default();
        transition.set_cost(ContinuousExpression::Cost);
        assert_eq!(
            transition,
            Transition {
                cost: CostExpression::Continuous(ContinuousExpression::Cost),
                ..Default::default()
            }
        );
    }

    #[test]
    fn get_preconditions() {
        let transition = Transition {
            elements_in_set_variable: vec![(0, 1), (1, 2)],
            elements_in_vector_variable: vec![(2, 3, 4), (3, 4, 5)],
            preconditions: vec![
                grounded_condition::GroundedCondition {
                    condition: Condition::Set(Box::new(SetCondition::IsIn(
                        ElementExpression::Variable(0),
                        SetExpression::Reference(ReferenceExpression::Variable(0)),
                    ))),
                    ..Default::default()
                },
                grounded_condition::GroundedCondition {
                    condition: Condition::Set(Box::new(SetCondition::IsIn(
                        ElementExpression::Variable(1),
                        SetExpression::Reference(ReferenceExpression::Variable(1)),
                    ))),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        assert_eq!(
            transition.get_preconditions(),
            vec![
                Condition::Set(Box::new(SetCondition::IsIn(
                    ElementExpression::Constant(1),
                    SetExpression::Reference(ReferenceExpression::Variable(0)),
                ))),
                Condition::Set(Box::new(SetCondition::IsIn(
                    ElementExpression::Constant(2),
                    SetExpression::Reference(ReferenceExpression::Variable(1)),
                ))),
                Condition::Set(Box::new(SetCondition::IsIn(
                    ElementExpression::Constant(3),
                    SetExpression::FromVector(
                        4,
                        Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                            2
                        )))
                    ),
                ))),
                Condition::Set(Box::new(SetCondition::IsIn(
                    ElementExpression::Constant(4),
                    SetExpression::FromVector(
                        5,
                        Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                            3
                        )))
                    ),
                ))),
                Condition::Set(Box::new(SetCondition::IsIn(
                    ElementExpression::Variable(0),
                    SetExpression::Reference(ReferenceExpression::Variable(0)),
                ))),
                Condition::Set(Box::new(SetCondition::IsIn(
                    ElementExpression::Variable(1),
                    SetExpression::Reference(ReferenceExpression::Variable(1)),
                ))),
            ]
        );
    }

    #[test]
    fn add_precondition() {
        let mut transition = Transition::default();
        transition.add_precondition(Condition::Set(Box::new(SetCondition::IsIn(
            ElementExpression::Constant(0),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        ))));
        assert_eq!(
            transition,
            Transition {
                elements_in_set_variable: vec![(0, 0)],
                ..Default::default()
            }
        );
        transition.add_precondition(Condition::Set(Box::new(SetCondition::IsIn(
            ElementExpression::Constant(1),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        ))));
        assert_eq!(
            transition,
            Transition {
                elements_in_set_variable: vec![(0, 0), (1, 1)],
                ..Default::default()
            }
        );
        transition.add_precondition(Condition::Set(Box::new(SetCondition::IsIn(
            ElementExpression::Constant(0),
            SetExpression::FromVector(
                10,
                Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                    0,
                ))),
            ),
        ))));
        assert_eq!(
            transition,
            Transition {
                elements_in_set_variable: vec![(0, 0), (1, 1)],
                elements_in_vector_variable: vec![(0, 0, 10)],
                ..Default::default()
            }
        );
        transition.add_precondition(Condition::Set(Box::new(SetCondition::IsIn(
            ElementExpression::Constant(1),
            SetExpression::FromVector(
                10,
                Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                    1,
                ))),
            ),
        ))));
        assert_eq!(
            transition,
            Transition {
                elements_in_set_variable: vec![(0, 0), (1, 1)],
                elements_in_vector_variable: vec![(0, 0, 10), (1, 1, 10)],
                ..Default::default()
            }
        );
        transition.add_precondition(Condition::ComparisonE(
            ComparisonOperator::Eq,
            Box::new(ElementExpression::Variable(0)),
            Box::new(ElementExpression::Constant(0)),
        ));
        assert_eq!(
            transition,
            Transition {
                elements_in_set_variable: vec![(0, 0), (1, 1)],
                elements_in_vector_variable: vec![(0, 0, 10), (1, 1, 10)],
                preconditions: vec![grounded_condition::GroundedCondition {
                    condition: Condition::ComparisonE(
                        ComparisonOperator::Eq,
                        Box::new(ElementExpression::Variable(0)),
                        Box::new(ElementExpression::Constant(0))
                    ),
                    ..Default::default()
                }],
                ..Default::default()
            }
        );
        transition.add_precondition(Condition::ComparisonE(
            ComparisonOperator::Eq,
            Box::new(ElementExpression::Variable(1)),
            Box::new(ElementExpression::Constant(1)),
        ));
        assert_eq!(
            transition,
            Transition {
                elements_in_set_variable: vec![(0, 0), (1, 1)],
                elements_in_vector_variable: vec![(0, 0, 10), (1, 1, 10)],
                preconditions: vec![
                    grounded_condition::GroundedCondition {
                        condition: Condition::ComparisonE(
                            ComparisonOperator::Eq,
                            Box::new(ElementExpression::Variable(0)),
                            Box::new(ElementExpression::Constant(0))
                        ),
                        ..Default::default()
                    },
                    grounded_condition::GroundedCondition {
                        condition: Condition::ComparisonE(
                            ComparisonOperator::Eq,
                            Box::new(ElementExpression::Variable(1)),
                            Box::new(ElementExpression::Constant(1))
                        ),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }
        );
    }

    #[test]
    fn add_set_effect_ok() {
        let mut metadata = state::StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v1 = metadata.add_set_variable(String::from("v1"), ob);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = metadata.add_set_variable(String::from("v2"), ob);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, Set::with_capacity(10));
        assert!(result.is_ok());
        assert_eq!(
            transition,
            Transition {
                effect: Effect {
                    set_effects: vec![(
                        v1.id(),
                        SetExpression::Reference(ReferenceExpression::Constant(
                            Set::with_capacity(10)
                        ))
                    )],
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        let result = transition.add_effect(v2, v1);
        assert!(result.is_ok());
        assert_eq!(
            transition,
            Transition {
                effect: Effect {
                    set_effects: vec![
                        (
                            v1.id(),
                            SetExpression::Reference(ReferenceExpression::Constant(
                                Set::with_capacity(10)
                            ))
                        ),
                        (
                            v2.id(),
                            SetExpression::Reference(ReferenceExpression::Variable(v1.id()))
                        )
                    ],
                    ..Default::default()
                },
                ..Default::default()
            },
        );
    }

    #[test]
    fn add_set_effect_err() {
        let mut metadata = state::StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_set_variable(String::from("v"), ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(v, Set::with_capacity(10));
        assert!(result.is_ok());
        let result = transition.add_effect(v, Set::with_capacity(10));
        assert!(result.is_err());
    }

    #[test]
    fn add_vector_effect_ok() {
        let mut metadata = state::StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v1 = metadata.add_vector_variable(String::from("v1"), ob);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = metadata.add_vector_variable(String::from("v2"), ob);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(
            v1,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![1, 2])),
        );
        assert!(result.is_ok());
        assert_eq!(
            transition,
            Transition {
                effect: Effect {
                    vector_effects: vec![(
                        v1.id(),
                        VectorExpression::Reference(ReferenceExpression::Constant(vec![1, 2]))
                    )],
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        let result = transition.add_effect(
            v2,
            VectorExpression::Reference(ReferenceExpression::Variable(v1.id())),
        );
        assert!(result.is_ok());
        assert_eq!(
            transition,
            Transition {
                effect: Effect {
                    vector_effects: vec![
                        (
                            v1.id(),
                            VectorExpression::Reference(ReferenceExpression::Constant(vec![1, 2]))
                        ),
                        (
                            v2.id(),
                            VectorExpression::Reference(ReferenceExpression::Variable(v1.id()))
                        )
                    ],
                    ..Default::default()
                },
                ..Default::default()
            },
        );
    }

    #[test]
    fn add_vector_effect_err() {
        let mut metadata = state::StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_vector_variable(String::from("v"), ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(
            v,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![1, 2])),
        );
        assert!(result.is_ok());
        let result = transition.add_effect(
            v,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![1, 2])),
        );
        assert!(result.is_err());
    }

    #[test]
    fn add_element_effect_ok() {
        let mut metadata = state::StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v1 = metadata.add_element_variable(String::from("v1"), ob);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = metadata.add_element_variable(String::from("v2"), ob);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, 0);
        assert!(result.is_ok());
        assert_eq!(
            transition,
            Transition {
                effect: Effect {
                    element_effects: vec![(v1.id(), ElementExpression::Constant(0))],
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        let result = transition.add_effect(v2, v1);
        assert!(result.is_ok());
        assert_eq!(
            transition,
            Transition {
                effect: Effect {
                    element_effects: vec![
                        (v1.id(), ElementExpression::Constant(0)),
                        (v2.id(), ElementExpression::Variable(v1.id()))
                    ],
                    ..Default::default()
                },
                ..Default::default()
            },
        );
    }

    #[test]
    fn add_element_effect_err() {
        let mut metadata = state::StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_element_variable(String::from("v"), ob);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(v, 0);
        assert!(result.is_ok());
        let result = transition.add_effect(v, 0);
        assert!(result.is_err());
    }

    #[test]
    fn add_element_resource_effect_ok() {
        let mut metadata = state::StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v1 = metadata.add_element_resource_variable(String::from("v1"), ob, false);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = metadata.add_element_resource_variable(String::from("v2"), ob, true);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, 0);
        assert!(result.is_ok());
        assert_eq!(
            transition,
            Transition {
                effect: Effect {
                    element_resource_effects: vec![(v1.id(), ElementExpression::Constant(0))],
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        let result = transition.add_effect(v2, v1);
        assert!(result.is_ok());
        assert_eq!(
            transition,
            Transition {
                effect: Effect {
                    element_resource_effects: vec![
                        (v1.id(), ElementExpression::Constant(0)),
                        (v2.id(), ElementExpression::ResourceVariable(v1.id()))
                    ],
                    ..Default::default()
                },
                ..Default::default()
            },
        );
    }

    #[test]
    fn add_element_resource_effect_err() {
        let mut metadata = state::StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_element_resource_variable(String::from("v"), ob, false);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(v, 0);
        assert!(result.is_ok());
        let result = transition.add_effect(v, 0);
        assert!(result.is_err());
    }

    #[test]
    fn add_integer_effect_ok() {
        let mut metadata = state::StateMetadata::default();
        let v1 = metadata.add_integer_variable(String::from("v1"));
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = metadata.add_integer_variable(String::from("v2"));
        assert!(v2.is_ok());
        let v2 = v2.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, 0);
        assert!(result.is_ok());
        assert_eq!(
            transition,
            Transition {
                effect: Effect {
                    integer_effects: vec![(v1.id(), IntegerExpression::Constant(0))],
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        let result = transition.add_effect(v2, v1);
        assert!(result.is_ok());
        assert_eq!(
            transition,
            Transition {
                effect: Effect {
                    integer_effects: vec![
                        (v1.id(), IntegerExpression::Constant(0)),
                        (v2.id(), IntegerExpression::Variable(v1.id()))
                    ],
                    ..Default::default()
                },
                ..Default::default()
            },
        );
    }

    #[test]
    fn add_integer_effect_err() {
        let mut metadata = state::StateMetadata::default();
        let v = metadata.add_integer_variable(String::from("v"));
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(v, 0);
        assert!(result.is_ok());
        let result = transition.add_effect(v, 0);
        assert!(result.is_err());
    }

    #[test]
    fn add_integer_resource_effect_ok() {
        let mut metadata = state::StateMetadata::default();
        let v1 = metadata.add_integer_resource_variable(String::from("v1"), false);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = metadata.add_integer_resource_variable(String::from("v2"), true);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, 0);
        assert!(result.is_ok());
        assert_eq!(
            transition,
            Transition {
                effect: Effect {
                    integer_resource_effects: vec![(v1.id(), IntegerExpression::Constant(0))],
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        let result = transition.add_effect(v2, v1);
        assert!(result.is_ok());
        assert_eq!(
            transition,
            Transition {
                effect: Effect {
                    integer_resource_effects: vec![
                        (v1.id(), IntegerExpression::Constant(0)),
                        (v2.id(), IntegerExpression::ResourceVariable(v1.id()))
                    ],
                    ..Default::default()
                },
                ..Default::default()
            },
        );
    }

    #[test]
    fn add_integer_resource_effect_err() {
        let mut metadata = state::StateMetadata::default();
        let v = metadata.add_integer_resource_variable(String::from("v"), false);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(v, 0);
        assert!(result.is_ok());
        let result = transition.add_effect(v, 0);
        assert!(result.is_err());
    }

    #[test]
    fn add_continuous_effect_ok() {
        let mut metadata = state::StateMetadata::default();
        let v1 = metadata.add_continuous_variable(String::from("v1"));
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = metadata.add_continuous_variable(String::from("v2"));
        assert!(v2.is_ok());
        let v2 = v2.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, 0.0);
        assert!(result.is_ok());
        assert_eq!(
            transition,
            Transition {
                effect: Effect {
                    continuous_effects: vec![(v1.id(), ContinuousExpression::Constant(0.0))],
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        let result = transition.add_effect(v2, v1);
        assert!(result.is_ok());
        assert_eq!(
            transition,
            Transition {
                effect: Effect {
                    continuous_effects: vec![
                        (v1.id(), ContinuousExpression::Constant(0.0)),
                        (v2.id(), ContinuousExpression::Variable(v1.id()))
                    ],
                    ..Default::default()
                },
                ..Default::default()
            },
        );
    }

    #[test]
    fn add_continuous_effect_err() {
        let mut metadata = state::StateMetadata::default();
        let v = metadata.add_continuous_variable(String::from("v"));
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(v, 0.0);
        assert!(result.is_ok());
        let result = transition.add_effect(v, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn add_continuous_resource_effect() {
        let mut metadata = state::StateMetadata::default();
        let v1 = metadata.add_continuous_resource_variable(String::from("v1"), false);
        assert!(v1.is_ok());
        let v1 = v1.unwrap();
        let v2 = metadata.add_continuous_resource_variable(String::from("v2"), true);
        assert!(v2.is_ok());
        let v2 = v2.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(v1, 0.0);
        assert!(result.is_ok());
        assert_eq!(
            transition,
            Transition {
                effect: Effect {
                    continuous_resource_effects: vec![(
                        v1.id(),
                        ContinuousExpression::Constant(0.0)
                    )],
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        let result = transition.add_effect(v2, v1);
        assert!(result.is_ok());
        assert_eq!(
            transition,
            Transition {
                effect: Effect {
                    continuous_resource_effects: vec![
                        (v1.id(), ContinuousExpression::Constant(0.0)),
                        (v2.id(), ContinuousExpression::ResourceVariable(v1.id()))
                    ],
                    ..Default::default()
                },
                ..Default::default()
            },
        );
    }

    #[test]
    fn add_continuous_resource_effect_err() {
        let mut metadata = state::StateMetadata::default();
        let v = metadata.add_continuous_resource_variable(String::from("v"), false);
        assert!(v.is_ok());
        let v = v.unwrap();

        let mut transition = Transition::default();
        let result = transition.add_effect(v, 0.0);
        assert!(result.is_ok());
        let result = transition.add_effect(v, 0.0);
        assert!(result.is_err());
    }
}
