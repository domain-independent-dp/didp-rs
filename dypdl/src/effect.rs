use crate::expression;

/// Effect in a transition.
#[derive(Debug, PartialEq, Clone, Default)]
pub struct Effect {
    /// Pairs of the index of a set variable and a set expression.
    /// Must be sorted by the indices of the set variables.
    pub set_effects: Vec<(usize, expression::SetExpression)>,
    /// Pairs of the index of a vector variable and a vector expression.
    /// Must be sorted by the indices of the vector variables.
    pub vector_effects: Vec<(usize, expression::VectorExpression)>,
    /// Pairs of the index of an element variable and an element expression.
    pub element_effects: Vec<(usize, expression::ElementExpression)>,
    /// Pairs of the index of an integer variable and an integer expression.
    pub integer_effects: Vec<(usize, expression::IntegerExpression)>,
    /// Pairs of the index of a continuous variable and a continuous expression.
    pub continuous_effects: Vec<(usize, expression::ContinuousExpression)>,
    /// Pairs of the index of an element resource variable and an element expression.
    pub element_resource_effects: Vec<(usize, expression::ElementExpression)>,
    /// Pairs of the index of an integer resource variable and an integer expression.
    pub integer_resource_effects: Vec<(usize, expression::IntegerExpression)>,
    /// Pairs of the index of a continuous resource variable and a continuous expression.
    pub continuous_resource_effects: Vec<(usize, expression::ContinuousExpression)>,
}
