use super::*;
use crate::variable_type::{Element, Numeric, Set};

pub(super) trait SubstituteLocalVariable: Sized {
    fn substitute_local_variable(&self, id: usize, value: Element) -> Self;
}

impl SubstituteLocalVariable for ElementExpression {
    fn substitute_local_variable(&self, id: usize, value: Element) -> Self {
        match self {
            Self::LocalVariable(i) if *i == id => Self::Constant(value),
            Self::BinaryOperation(op, x, y) => Self::BinaryOperation(
                op.clone(),
                Box::new(x.substitute_local_variable(id, value)),
                Box::new(y.substitute_local_variable(id, value)),
            ),
            Self::Table(table) => Self::Table(Box::new(table.substitute_local_variable(id, value))),
            Self::If(condition, x, y) => Self::If(
                Box::new(condition.substitute_local_variable(id, value)),
                Box::new(x.substitute_local_variable(id, value)),
                Box::new(y.substitute_local_variable(id, value)),
            ),
            _ => self.clone(),
        }
    }
}

impl<T: Clone> SubstituteLocalVariable for TableExpression<T> {
    fn substitute_local_variable(&self, id: usize, value: Element) -> Self {
        match self {
            Self::Table1D(i, x) => Self::Table1D(*i, x.substitute_local_variable(id, value)),
            Self::Table2D(i, x, y) => Self::Table2D(
                *i,
                x.substitute_local_variable(id, value),
                y.substitute_local_variable(id, value),
            ),
            Self::Table3D(i, x, y, z) => Self::Table3D(
                *i,
                x.substitute_local_variable(id, value),
                y.substitute_local_variable(id, value),
                z.substitute_local_variable(id, value),
            ),
            Self::Table(i, args) => Self::Table(
                *i,
                args.iter()
                    .map(|x| x.substitute_local_variable(id, value))
                    .collect(),
            ),
            _ => self.clone(),
        }
    }
}

impl SubstituteLocalVariable for ArgumentExpression {
    fn substitute_local_variable(&self, id: usize, value: Element) -> Self {
        match self {
            Self::Set(x) => Self::Set(x.substitute_local_variable(id, value)),
            Self::Element(x) => Self::Element(x.substitute_local_variable(id, value)),
        }
    }
}

impl SubstituteLocalVariable for ReferenceExpression<Set> {
    fn substitute_local_variable(&self, id: usize, value: Element) -> Self {
        match self {
            Self::Table(table) => Self::Table(table.substitute_local_variable(id, value)),
            _ => self.clone(),
        }
    }
}

impl SubstituteLocalVariable for SetReduceExpression {
    fn substitute_local_variable(&self, id: usize, value: Element) -> Self {
        match self {
            Self::Table1D(op, capacity, i, x) => Self::Table1D(
                op.clone(),
                *capacity,
                *i,
                Box::new(x.substitute_local_variable(id, value)),
            ),
            Self::Table2D(op, capacity, i, x, y) => Self::Table2D(
                op.clone(),
                *capacity,
                *i,
                Box::new(x.substitute_local_variable(id, value)),
                Box::new(y.substitute_local_variable(id, value)),
            ),
            Self::Table3D(op, capacity, i, x, y, z) => Self::Table3D(
                op.clone(),
                *capacity,
                *i,
                Box::new(x.substitute_local_variable(id, value)),
                Box::new(y.substitute_local_variable(id, value)),
                Box::new(z.substitute_local_variable(id, value)),
            ),
            Self::Table(op, capacity, i, args) => Self::Table(
                op.clone(),
                *capacity,
                *i,
                args.iter()
                    .map(|x| x.substitute_local_variable(id, value))
                    .collect(),
            ),
            _ => self.clone(),
        }
    }
}

impl SubstituteLocalVariable for SetExpression {
    fn substitute_local_variable(&self, id: usize, value: Element) -> Self {
        match self {
            Self::Reference(x) => Self::Reference(x.substitute_local_variable(id, value)),
            Self::Complement(x) => {
                Self::Complement(Box::new(x.substitute_local_variable(id, value)))
            }
            Self::SetOperation(op, x, y) => Self::SetOperation(
                op.clone(),
                Box::new(x.substitute_local_variable(id, value)),
                Box::new(y.substitute_local_variable(id, value)),
            ),
            Self::SetElementOperation(op, x, y) => Self::SetElementOperation(
                op.clone(),
                x.substitute_local_variable(id, value),
                Box::new(y.substitute_local_variable(id, value)),
            ),
            Self::Reduce(x) => Self::Reduce(x.substitute_local_variable(id, value)),
            Self::If(condition, x, y) => Self::If(
                Box::new(condition.substitute_local_variable(id, value)),
                Box::new(x.substitute_local_variable(id, value)),
                Box::new(y.substitute_local_variable(id, value)),
            ),
            Self::Filter(set, binder, condition) => Self::Filter(
                Box::new(set.substitute_local_variable(id, value)),
                *binder,
                if *binder == id {
                    condition.clone()
                } else {
                    Box::new(condition.substitute_local_variable(id, value))
                },
            ),
            _ => self.clone(),
        }
    }
}

impl<T: Numeric> SubstituteLocalVariable for NumericTableExpression<T> {
    fn substitute_local_variable(&self, id: usize, value: Element) -> Self {
        match self {
            Self::Table(i, args) => Self::Table(
                *i,
                args.iter()
                    .map(|x| x.substitute_local_variable(id, value))
                    .collect(),
            ),
            Self::TableReduce(op, i, args) => Self::TableReduce(
                op.clone(),
                *i,
                args.iter()
                    .map(|x| x.substitute_local_variable(id, value))
                    .collect(),
            ),
            Self::Table1D(i, x) => Self::Table1D(*i, x.substitute_local_variable(id, value)),
            Self::Table2D(i, x, y) => Self::Table2D(
                *i,
                x.substitute_local_variable(id, value),
                y.substitute_local_variable(id, value),
            ),
            Self::Table3D(i, x, y, z) => Self::Table3D(
                *i,
                x.substitute_local_variable(id, value),
                y.substitute_local_variable(id, value),
                z.substitute_local_variable(id, value),
            ),
            Self::Table1DReduce(op, i, x) => {
                Self::Table1DReduce(op.clone(), *i, x.substitute_local_variable(id, value))
            }
            Self::Table2DReduce(op, i, x, y) => Self::Table2DReduce(
                op.clone(),
                *i,
                x.substitute_local_variable(id, value),
                y.substitute_local_variable(id, value),
            ),
            Self::Table2DReduceX(op, i, x, y) => Self::Table2DReduceX(
                op.clone(),
                *i,
                x.substitute_local_variable(id, value),
                y.substitute_local_variable(id, value),
            ),
            Self::Table2DReduceY(op, i, x, y) => Self::Table2DReduceY(
                op.clone(),
                *i,
                x.substitute_local_variable(id, value),
                y.substitute_local_variable(id, value),
            ),
            Self::Table3DReduce(op, i, x, y, z) => Self::Table3DReduce(
                op.clone(),
                *i,
                x.substitute_local_variable(id, value),
                y.substitute_local_variable(id, value),
                z.substitute_local_variable(id, value),
            ),
            _ => self.clone(),
        }
    }
}

impl SubstituteLocalVariable for IntegerExpression {
    fn substitute_local_variable(&self, id: usize, value: Element) -> Self {
        match self {
            Self::UnaryOperation(op, x) => {
                Self::UnaryOperation(op.clone(), Box::new(x.substitute_local_variable(id, value)))
            }
            Self::BinaryOperation(op, x, y) => Self::BinaryOperation(
                op.clone(),
                Box::new(x.substitute_local_variable(id, value)),
                Box::new(y.substitute_local_variable(id, value)),
            ),
            Self::Cardinality(x) => Self::Cardinality(x.substitute_local_variable(id, value)),
            Self::Table(x) => Self::Table(Box::new(x.substitute_local_variable(id, value))),
            Self::MinimumSpanningTree(set, table) => Self::MinimumSpanningTree(
                Box::new(set.substitute_local_variable(id, value)),
                *table,
            ),
            Self::MinimumSpanningTreeWithConnectivity(set, weights, connectivity) => {
                Self::MinimumSpanningTreeWithConnectivity(
                    Box::new(set.substitute_local_variable(id, value)),
                    *weights,
                    *connectivity,
                )
            }
            Self::MinimumSpanningTreeWithEdges(set, edges) => Self::MinimumSpanningTreeWithEdges(
                Box::new(set.substitute_local_variable(id, value)),
                edges
                    .iter()
                    .map(|(i, j, weight)| (*i, *j, weight.substitute_local_variable(id, value)))
                    .collect(),
            ),
            Self::MinimumSpanningTreeWithEdgesAndConnectivity(set, edges) => {
                Self::MinimumSpanningTreeWithEdgesAndConnectivity(
                    Box::new(set.substitute_local_variable(id, value)),
                    edges
                        .iter()
                        .map(|(i, j, weight, condition)| {
                            (
                                *i,
                                *j,
                                weight.substitute_local_variable(id, value),
                                condition.substitute_local_variable(id, value),
                            )
                        })
                        .collect(),
                )
            }
            Self::MinimumSpanningTreeWithSortedEdges(set, edges) => {
                Self::MinimumSpanningTreeWithSortedEdges(
                    Box::new(set.substitute_local_variable(id, value)),
                    edges.clone(),
                )
            }
            Self::If(condition, x, y) => Self::If(
                Box::new(condition.substitute_local_variable(id, value)),
                Box::new(x.substitute_local_variable(id, value)),
                Box::new(y.substitute_local_variable(id, value)),
            ),
            Self::FromContinuous(op, x) => {
                Self::FromContinuous(op.clone(), Box::new(x.substitute_local_variable(id, value)))
            }
            Self::Reduce(op, set, binder, body) => Self::Reduce(
                op.clone(),
                Box::new(set.substitute_local_variable(id, value)),
                *binder,
                if *binder == id {
                    body.clone()
                } else {
                    Box::new(body.substitute_local_variable(id, value))
                },
            ),
            Self::FilterReduce(op, set, filter_binder, reduce_binder, condition, body) => {
                Self::FilterReduce(
                    op.clone(),
                    Box::new(set.substitute_local_variable(id, value)),
                    *filter_binder,
                    *reduce_binder,
                    if *filter_binder == id {
                        condition.clone()
                    } else {
                        Box::new(condition.substitute_local_variable(id, value))
                    },
                    if *reduce_binder == id {
                        body.clone()
                    } else {
                        Box::new(body.substitute_local_variable(id, value))
                    },
                )
            }
            _ => self.clone(),
        }
    }
}

impl SubstituteLocalVariable for ContinuousExpression {
    fn substitute_local_variable(&self, id: usize, value: Element) -> Self {
        match self {
            Self::UnaryOperation(op, x) => {
                Self::UnaryOperation(op.clone(), Box::new(x.substitute_local_variable(id, value)))
            }
            Self::ContinuousUnaryOperation(op, x) => Self::ContinuousUnaryOperation(
                op.clone(),
                Box::new(x.substitute_local_variable(id, value)),
            ),
            Self::Round(op, x) => {
                Self::Round(op.clone(), Box::new(x.substitute_local_variable(id, value)))
            }
            Self::BinaryOperation(op, x, y) => Self::BinaryOperation(
                op.clone(),
                Box::new(x.substitute_local_variable(id, value)),
                Box::new(y.substitute_local_variable(id, value)),
            ),
            Self::ContinuousBinaryOperation(op, x, y) => Self::ContinuousBinaryOperation(
                op.clone(),
                Box::new(x.substitute_local_variable(id, value)),
                Box::new(y.substitute_local_variable(id, value)),
            ),
            Self::Cardinality(x) => Self::Cardinality(x.substitute_local_variable(id, value)),
            Self::Table(x) => Self::Table(Box::new(x.substitute_local_variable(id, value))),
            Self::MinimumSpanningTree(set, table) => Self::MinimumSpanningTree(
                Box::new(set.substitute_local_variable(id, value)),
                *table,
            ),
            Self::MinimumSpanningTreeWithConnectivity(set, weights, connectivity) => {
                Self::MinimumSpanningTreeWithConnectivity(
                    Box::new(set.substitute_local_variable(id, value)),
                    *weights,
                    *connectivity,
                )
            }
            Self::MinimumSpanningTreeWithEdges(set, edges) => Self::MinimumSpanningTreeWithEdges(
                Box::new(set.substitute_local_variable(id, value)),
                edges
                    .iter()
                    .map(|(i, j, weight)| (*i, *j, weight.substitute_local_variable(id, value)))
                    .collect(),
            ),
            Self::MinimumSpanningTreeWithEdgesAndConnectivity(set, edges) => {
                Self::MinimumSpanningTreeWithEdgesAndConnectivity(
                    Box::new(set.substitute_local_variable(id, value)),
                    edges
                        .iter()
                        .map(|(i, j, weight, condition)| {
                            (
                                *i,
                                *j,
                                weight.substitute_local_variable(id, value),
                                condition.substitute_local_variable(id, value),
                            )
                        })
                        .collect(),
                )
            }
            Self::MinimumSpanningTreeWithSortedEdges(set, edges) => {
                Self::MinimumSpanningTreeWithSortedEdges(
                    Box::new(set.substitute_local_variable(id, value)),
                    edges.clone(),
                )
            }
            Self::If(condition, x, y) => Self::If(
                Box::new(condition.substitute_local_variable(id, value)),
                Box::new(x.substitute_local_variable(id, value)),
                Box::new(y.substitute_local_variable(id, value)),
            ),
            Self::FromInteger(x) => {
                Self::FromInteger(Box::new(x.substitute_local_variable(id, value)))
            }
            Self::Reduce(op, set, binder, body) => Self::Reduce(
                op.clone(),
                Box::new(set.substitute_local_variable(id, value)),
                *binder,
                if *binder == id {
                    body.clone()
                } else {
                    Box::new(body.substitute_local_variable(id, value))
                },
            ),
            Self::FilterReduce(op, set, filter_binder, reduce_binder, condition, body) => {
                Self::FilterReduce(
                    op.clone(),
                    Box::new(set.substitute_local_variable(id, value)),
                    *filter_binder,
                    *reduce_binder,
                    if *filter_binder == id {
                        condition.clone()
                    } else {
                        Box::new(condition.substitute_local_variable(id, value))
                    },
                    if *reduce_binder == id {
                        body.clone()
                    } else {
                        Box::new(body.substitute_local_variable(id, value))
                    },
                )
            }
            Self::FractionalKnapsackSorted(set, capacity, items) => Self::FractionalKnapsackSorted(
                Box::new(set.substitute_local_variable(id, value)),
                Box::new(capacity.substitute_local_variable(id, value)),
                items.clone(),
            ),
            Self::FractionalKnapsack(set, capacity, items) => Self::FractionalKnapsack(
                Box::new(set.substitute_local_variable(id, value)),
                Box::new(capacity.substitute_local_variable(id, value)),
                items
                    .iter()
                    .map(|(item, item_value, weight)| {
                        (
                            *item,
                            item_value.substitute_local_variable(id, value),
                            weight.substitute_local_variable(id, value),
                        )
                    })
                    .collect(),
            ),
            Self::FractionalKnapsackIntegerTable(set, capacity, values, weights) => {
                Self::FractionalKnapsackIntegerTable(
                    Box::new(set.substitute_local_variable(id, value)),
                    Box::new(capacity.substitute_local_variable(id, value)),
                    *values,
                    *weights,
                )
            }
            Self::FractionalKnapsackContinuousTable(set, capacity, values, weights) => {
                Self::FractionalKnapsackContinuousTable(
                    Box::new(set.substitute_local_variable(id, value)),
                    Box::new(capacity.substitute_local_variable(id, value)),
                    *values,
                    *weights,
                )
            }
            Self::FractionalKnapsackIntegerValueContinuousWeightTable(
                set,
                capacity,
                values,
                weights,
            ) => Self::FractionalKnapsackIntegerValueContinuousWeightTable(
                Box::new(set.substitute_local_variable(id, value)),
                Box::new(capacity.substitute_local_variable(id, value)),
                *values,
                *weights,
            ),
            Self::FractionalKnapsackContinuousValueIntegerWeightTable(
                set,
                capacity,
                values,
                weights,
            ) => Self::FractionalKnapsackContinuousValueIntegerWeightTable(
                Box::new(set.substitute_local_variable(id, value)),
                Box::new(capacity.substitute_local_variable(id, value)),
                *values,
                *weights,
            ),
            _ => self.clone(),
        }
    }
}

impl SubstituteLocalVariable for SetCondition {
    fn substitute_local_variable(&self, id: usize, value: Element) -> Self {
        match self {
            Self::IsEqual(x, y) => Self::IsEqual(
                x.substitute_local_variable(id, value),
                y.substitute_local_variable(id, value),
            ),
            Self::IsNotEqual(x, y) => Self::IsNotEqual(
                x.substitute_local_variable(id, value),
                y.substitute_local_variable(id, value),
            ),
            Self::IsIn(x, y) => Self::IsIn(
                x.substitute_local_variable(id, value),
                y.substitute_local_variable(id, value),
            ),
            Self::IsSubset(x, y) => Self::IsSubset(
                x.substitute_local_variable(id, value),
                y.substitute_local_variable(id, value),
            ),
            Self::IsEmpty(x) => Self::IsEmpty(x.substitute_local_variable(id, value)),
            _ => self.clone(),
        }
    }
}

impl SubstituteLocalVariable for Condition {
    fn substitute_local_variable(&self, id: usize, value: Element) -> Self {
        match self {
            Self::Not(x) => Self::Not(Box::new(x.substitute_local_variable(id, value))),
            Self::And(x, y) => Self::And(
                Box::new(x.substitute_local_variable(id, value)),
                Box::new(y.substitute_local_variable(id, value)),
            ),
            Self::Or(x, y) => Self::Or(
                Box::new(x.substitute_local_variable(id, value)),
                Box::new(y.substitute_local_variable(id, value)),
            ),
            Self::Quantified(quantifier, set, binder, condition) => Self::Quantified(
                *quantifier,
                Box::new(set.substitute_local_variable(id, value)),
                *binder,
                if *binder == id {
                    condition.clone()
                } else {
                    Box::new(condition.substitute_local_variable(id, value))
                },
            ),
            Self::ComparisonE(op, x, y) => Self::ComparisonE(
                op.clone(),
                Box::new(x.substitute_local_variable(id, value)),
                Box::new(y.substitute_local_variable(id, value)),
            ),
            Self::ComparisonI(op, x, y) => Self::ComparisonI(
                op.clone(),
                Box::new(x.substitute_local_variable(id, value)),
                Box::new(y.substitute_local_variable(id, value)),
            ),
            Self::ComparisonC(op, x, y) => Self::ComparisonC(
                op.clone(),
                Box::new(x.substitute_local_variable(id, value)),
                Box::new(y.substitute_local_variable(id, value)),
            ),
            Self::Set(x) => Self::Set(Box::new(x.substitute_local_variable(id, value))),
            Self::Table(x) => Self::Table(Box::new(x.substitute_local_variable(id, value))),
            _ => self.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn constant_set(capacity: usize) -> SetExpression {
        SetExpression::Reference(ReferenceExpression::Constant(Set::with_capacity(capacity)))
    }

    #[test]
    fn substitute_local_variable_recursively() {
        let expression = Condition::ComparisonE(
            ComparisonOperator::Eq,
            Box::new(ElementExpression::Table(Box::new(
                TableExpression::Table2D(
                    0,
                    ElementExpression::LocalVariable(0),
                    ElementExpression::LocalVariable(1),
                ),
            ))),
            Box::new(ElementExpression::LocalVariable(0)),
        );

        assert_eq!(
            expression.substitute_local_variable(0, 2),
            Condition::ComparisonE(
                ComparisonOperator::Eq,
                Box::new(ElementExpression::Table(Box::new(
                    TableExpression::Table2D(
                        0,
                        ElementExpression::Constant(2),
                        ElementExpression::LocalVariable(1),
                    )
                ))),
                Box::new(ElementExpression::Constant(2)),
            )
        );
    }

    #[test]
    fn substitute_local_variable_in_filter_reduce() {
        let expression = IntegerExpression::FilterReduce(
            ReduceOperator::Sum,
            Box::new(SetExpression::SetElementOperation(
                SetElementOperator::Add,
                ElementExpression::LocalVariable(0),
                Box::new(constant_set(3)),
            )),
            1,
            1,
            Box::new(Condition::ComparisonE(
                ComparisonOperator::Eq,
                Box::new(ElementExpression::LocalVariable(0)),
                Box::new(ElementExpression::LocalVariable(1)),
            )),
            Box::new(IntegerExpression::Table(Box::new(
                NumericTableExpression::Table1D(0, ElementExpression::LocalVariable(0)),
            ))),
        );

        assert_eq!(
            expression.substitute_local_variable(0, 2),
            IntegerExpression::FilterReduce(
                ReduceOperator::Sum,
                Box::new(SetExpression::SetElementOperation(
                    SetElementOperator::Add,
                    ElementExpression::Constant(2),
                    Box::new(constant_set(3)),
                )),
                1,
                1,
                Box::new(Condition::ComparisonE(
                    ComparisonOperator::Eq,
                    Box::new(ElementExpression::Constant(2)),
                    Box::new(ElementExpression::LocalVariable(1)),
                )),
                Box::new(IntegerExpression::Table(Box::new(
                    NumericTableExpression::Table1D(0, ElementExpression::Constant(2)),
                ))),
            )
        );
    }

    #[test]
    fn substitute_local_variable_respects_binder_scope() {
        let set = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            ElementExpression::LocalVariable(0),
            Box::new(constant_set(3)),
        );
        let condition = Condition::ComparisonE(
            ComparisonOperator::Eq,
            Box::new(ElementExpression::LocalVariable(0)),
            Box::new(ElementExpression::LocalVariable(1)),
        );
        let expression = Condition::Quantified(
            Quantifier::Any,
            Box::new(set),
            0,
            Box::new(condition.clone()),
        );

        assert_eq!(
            expression.substitute_local_variable(0, 2),
            Condition::Quantified(
                Quantifier::Any,
                Box::new(SetExpression::SetElementOperation(
                    SetElementOperator::Add,
                    ElementExpression::Constant(2),
                    Box::new(constant_set(3)),
                )),
                0,
                Box::new(condition),
            )
        );
    }
}
