use super::condition::Condition;
use super::numeric_operator::NumericOperator;
use super::reference_expression::ReferenceExpression;
use crate::state::State;
use crate::table_data::TableData;
use crate::table_registry::TableRegistry;
use crate::variable::{Element, Set, Vector};

#[derive(Debug, PartialEq, Clone)]
pub enum ElementExpression {
    Constant(Element),
    Variable(usize),
    NumericOperation(
        NumericOperator,
        Box<ElementExpression>,
        Box<ElementExpression>,
    ),
    Last(Box<VectorExpression>),
    At(Box<VectorExpression>, Box<ElementExpression>),
    Table(Box<TableExpression<Element>>),
    If(
        Box<Condition>,
        Box<ElementExpression>,
        Box<ElementExpression>,
    ),
}

impl ElementExpression {
    pub fn eval(&self, state: &State, registry: &TableRegistry) -> Element {
        match self {
            Self::Constant(x) => *x,
            Self::Variable(i) => state.signature_variables.element_variables[*i],
            Self::NumericOperation(op, x, y) => {
                op.eval(x.eval(state, registry), y.eval(state, registry))
            }
            Self::Last(vector) => match vector.as_ref() {
                VectorExpression::Reference(vector) => *vector
                    .eval(
                        state,
                        registry,
                        &state.signature_variables.vector_variables,
                        &registry.vector_tables,
                    )
                    .last()
                    .unwrap(),
                vector => *vector.eval(state, registry).last().unwrap(),
            },
            Self::At(vector, i) => match vector.as_ref() {
                VectorExpression::Reference(vector) => vector.eval(
                    state,
                    registry,
                    &state.signature_variables.vector_variables,
                    &registry.vector_tables,
                )[i.eval(state, registry)],
                vector => vector.eval(state, registry)[i.eval(state, registry)],
            },
            Self::Table(table) => *table.eval(state, registry, &registry.element_tables),
            Self::If(condition, x, y) => {
                if condition.eval(state, registry) {
                    x.eval(state, registry)
                } else {
                    y.eval(state, registry)
                }
            }
        }
    }

    pub fn simplify(&self, registry: &TableRegistry) -> ElementExpression {
        match self {
            Self::Last(vector) => match vector.simplify(registry) {
                VectorExpression::Reference(ReferenceExpression::Constant(vector)) => {
                    Self::Constant(*vector.last().unwrap())
                }
                vector => Self::Last(Box::new(vector)),
            },
            Self::At(vector, i) => match (vector.simplify(registry), i.simplify(registry)) {
                (
                    VectorExpression::Reference(ReferenceExpression::Constant(vector)),
                    Self::Constant(i),
                ) => Self::Constant(vector[i]),
                (vector, i) => Self::At(Box::new(vector), Box::new(i)),
            },
            Self::NumericOperation(op, x, y) => {
                match (x.simplify(registry), y.simplify(registry)) {
                    (Self::Constant(x), Self::Constant(y)) => Self::Constant(op.eval(x, y)),
                    (x, y) => Self::NumericOperation(op.clone(), Box::new(x), Box::new(y)),
                }
            }
            Self::Table(table) => match table.simplify(registry, &registry.element_tables) {
                TableExpression::Constant(value) => Self::Constant(value),
                expression => Self::Table(Box::new(expression)),
            },
            Self::If(condition, x, y) => match condition.simplify(registry) {
                Condition::Constant(true) => x.simplify(registry),
                Condition::Constant(false) => y.simplify(registry),
                condition => Self::If(
                    Box::new(condition),
                    Box::new(x.simplify(registry)),
                    Box::new(y.simplify(registry)),
                ),
            },
            _ => self.clone(),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum SetExpression {
    Reference(ReferenceExpression<Set>),
    Complement(Box<SetExpression>),
    SetOperation(SetOperator, Box<SetExpression>, Box<SetExpression>),
    SetElementOperation(SetElementOperator, ElementExpression, Box<SetExpression>),
    FromVector(usize, Box<VectorExpression>),
    If(Box<Condition>, Box<SetExpression>, Box<SetExpression>),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum SetOperator {
    Union,
    Difference,
    Intersection,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum SetElementOperator {
    Add,
    Remove,
}

impl SetExpression {
    pub fn eval(&self, state: &State, registry: &TableRegistry) -> Set {
        match self {
            Self::Reference(expression) => expression
                .eval(
                    state,
                    registry,
                    &state.signature_variables.set_variables,
                    &registry.set_tables,
                )
                .clone(),
            Self::Complement(set) => {
                let mut set = set.eval(state, registry);
                set.toggle_range(..);
                set
            }
            Self::SetOperation(op, x, y) => match (op, x.as_ref(), y.as_ref()) {
                (op, x, SetExpression::Reference(y)) => {
                    let x = x.eval(state, registry);
                    let y = y.eval(
                        state,
                        registry,
                        &state.signature_variables.set_variables,
                        &registry.set_tables,
                    );
                    Self::eval_set_operation(op, x, y)
                }
                (SetOperator::Intersection, SetExpression::Reference(x), y)
                | (SetOperator::Union, SetExpression::Reference(x), y) => {
                    let x = x.eval(
                        state,
                        registry,
                        &state.signature_variables.set_variables,
                        &registry.set_tables,
                    );
                    let y = y.eval(state, registry);
                    Self::eval_set_operation(op, y, x)
                }
                (op, x, y) => {
                    let x = x.eval(state, registry);
                    let y = y.eval(state, registry);
                    Self::eval_set_operation(op, x, &y)
                }
            },
            Self::SetElementOperation(op, element, set) => {
                let set = set.eval(state, registry);
                let element = element.eval(state, registry);
                Self::eval_set_element_operation(op, element, set)
            }
            Self::FromVector(capacity, vector) => match vector.as_ref() {
                VectorExpression::Reference(ReferenceExpression::Constant(vector)) => {
                    let mut set = Set::with_capacity(*capacity);
                    vector.iter().for_each(|v| set.insert(*v));
                    set
                }
                vector => {
                    let mut set = Set::with_capacity(*capacity);
                    vector
                        .eval(state, registry)
                        .into_iter()
                        .for_each(|v| set.insert(v));
                    set
                }
            },
            Self::If(condition, x, y) => {
                if condition.eval(state, registry) {
                    x.eval(state, registry)
                } else {
                    y.eval(state, registry)
                }
            }
        }
    }

    pub fn simplify(&self, registry: &TableRegistry) -> SetExpression {
        match self {
            Self::Reference(expression) => {
                Self::Reference(expression.simplify(registry, &registry.set_tables))
            }
            Self::Complement(expression) => match expression.simplify(registry) {
                Self::Reference(ReferenceExpression::Constant(mut set)) => {
                    set.toggle_range(..);
                    Self::Reference(ReferenceExpression::Constant(set))
                }
                Self::Complement(expression) => *expression,
                expression => Self::Complement(Box::new(expression)),
            },
            Self::SetOperation(op, x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (
                    Self::Reference(ReferenceExpression::Constant(x)),
                    Self::Reference(ReferenceExpression::Constant(y)),
                ) => Self::Reference(ReferenceExpression::Constant(Self::eval_set_operation(
                    op, x, &y,
                ))),
                (
                    Self::Reference(ReferenceExpression::Variable(x)),
                    Self::Reference(ReferenceExpression::Variable(y)),
                ) if x == y => match op {
                    SetOperator::Union | SetOperator::Intersection => {
                        Self::Reference(ReferenceExpression::Variable(x))
                    }
                    SetOperator::Difference => Self::SetOperation(
                        SetOperator::Difference,
                        Box::new(Self::Reference(ReferenceExpression::Variable(x))),
                        Box::new(Self::Reference(ReferenceExpression::Variable(y))),
                    ),
                },
                (x, y) => Self::SetOperation(op.clone(), Box::new(x), Box::new(y)),
            },
            Self::SetElementOperation(op, element, set) => {
                match (set.simplify(registry), element.simplify(registry)) {
                    (
                        Self::Reference(ReferenceExpression::Constant(set)),
                        ElementExpression::Constant(element),
                    ) => Self::Reference(ReferenceExpression::Constant(
                        Self::eval_set_element_operation(op, element, set),
                    )),
                    (set, element) => Self::SetElementOperation(op.clone(), element, Box::new(set)),
                }
            }
            Self::FromVector(capacity, vector) => match vector.simplify(registry) {
                VectorExpression::Reference(ReferenceExpression::Constant(vector)) => {
                    let mut set = Set::with_capacity(*capacity);
                    vector.into_iter().for_each(|v| set.insert(v));
                    Self::Reference(ReferenceExpression::Constant(set))
                }
                vector => Self::FromVector(*capacity, Box::new(vector)),
            },
            Self::If(condition, x, y) => match condition.simplify(registry) {
                Condition::Constant(true) => x.simplify(registry),
                Condition::Constant(false) => y.simplify(registry),
                condition => Self::If(
                    Box::new(condition),
                    Box::new(x.simplify(registry)),
                    Box::new(y.simplify(registry)),
                ),
            },
        }
    }

    fn eval_set_operation(op: &SetOperator, mut x: Set, y: &Set) -> Set {
        match op {
            SetOperator::Union => {
                x.union_with(y);
                x
            }
            SetOperator::Difference => {
                x.difference_with(y);
                x
            }
            SetOperator::Intersection => {
                x.intersect_with(y);
                x
            }
        }
    }

    fn eval_set_element_operation(op: &SetElementOperator, element: Element, mut set: Set) -> Set {
        match op {
            SetElementOperator::Add => {
                set.insert(element);
                set
            }
            SetElementOperator::Remove => {
                set.set(element, false);
                set
            }
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum VectorExpression {
    Reference(ReferenceExpression<Vector>),
    Indices(Box<VectorExpression>),
    Reverse(Box<VectorExpression>),
    Set(ElementExpression, Box<VectorExpression>, ElementExpression),
    Push(ElementExpression, Box<VectorExpression>),
    Pop(Box<VectorExpression>),
    FromSet(Box<SetExpression>),
    If(Box<Condition>, Box<VectorExpression>, Box<VectorExpression>),
}

impl VectorExpression {
    pub fn eval(&self, state: &State, registry: &TableRegistry) -> Vector {
        match self {
            Self::Reference(expression) => expression
                .eval(
                    state,
                    registry,
                    &state.signature_variables.vector_variables,
                    &registry.vector_tables,
                )
                .clone(),
            Self::Indices(vector) => {
                let mut vector = vector.eval(state, registry);
                vector.iter_mut().enumerate().for_each(|(i, v)| *v = i);
                vector
            }
            Self::Reverse(vector) => {
                let mut vector = vector.eval(state, registry);
                vector.reverse();
                vector
            }
            Self::Set(element, vector, i) => {
                let mut vector = vector.eval(state, registry);
                vector[i.eval(state, registry)] = element.eval(state, registry);
                vector
            }
            Self::Push(element, vector) => {
                let element = element.eval(state, registry);
                let mut vector = vector.eval(state, registry);
                vector.push(element);
                vector
            }
            Self::Pop(vector) => {
                let mut vector = vector.eval(state, registry);
                vector.pop();
                vector
            }
            Self::FromSet(set) => match set.as_ref() {
                SetExpression::Reference(set) => set
                    .eval(
                        state,
                        registry,
                        &state.signature_variables.set_variables,
                        &registry.set_tables,
                    )
                    .ones()
                    .collect(),
                set => set.eval(state, registry).ones().collect(),
            },
            Self::If(condition, x, y) => {
                if condition.eval(state, registry) {
                    x.eval(state, registry)
                } else {
                    y.eval(state, registry)
                }
            }
        }
    }

    pub fn simplify(&self, registry: &TableRegistry) -> VectorExpression {
        match self {
            Self::Reference(vector) => {
                Self::Reference(vector.simplify(registry, &registry.vector_tables))
            }
            Self::Indices(vector) => match vector.simplify(registry) {
                VectorExpression::Reference(ReferenceExpression::Constant(mut vector)) => {
                    vector.iter_mut().enumerate().for_each(|(i, v)| *v = i);
                    Self::Reference(ReferenceExpression::Constant(vector))
                }
                vector => Self::Indices(Box::new(vector)),
            },
            Self::Reverse(vector) => match vector.simplify(registry) {
                VectorExpression::Reference(ReferenceExpression::Constant(mut vector)) => {
                    vector.reverse();
                    Self::Reference(ReferenceExpression::Constant(vector))
                }
                vector => Self::Reverse(Box::new(vector)),
            },
            Self::Set(element, vector, i) => match (
                element.simplify(registry),
                vector.simplify(registry),
                i.simplify(registry),
            ) {
                (
                    ElementExpression::Constant(element),
                    VectorExpression::Reference(ReferenceExpression::Constant(mut vector)),
                    ElementExpression::Constant(i),
                ) => {
                    vector[i] = element;
                    Self::Reference(ReferenceExpression::Constant(vector))
                }
                (element, vector, i) => Self::Set(element, Box::new(vector), i),
            },
            Self::Push(element, vector) => {
                match (element.simplify(registry), vector.simplify(registry)) {
                    (
                        ElementExpression::Constant(element),
                        VectorExpression::Reference(ReferenceExpression::Constant(mut vector)),
                    ) => {
                        vector.push(element);
                        Self::Reference(ReferenceExpression::Constant(vector))
                    }
                    (element, vector) => Self::Push(element, Box::new(vector)),
                }
            }
            Self::Pop(vector) => match vector.simplify(registry) {
                VectorExpression::Reference(ReferenceExpression::Constant(mut vector)) => {
                    vector.pop();
                    Self::Reference(ReferenceExpression::Constant(vector))
                }
                vector => Self::Pop(Box::new(vector)),
            },
            Self::FromSet(set) => match set.simplify(registry) {
                SetExpression::Reference(ReferenceExpression::Constant(set)) => {
                    Self::Reference(ReferenceExpression::Constant(set.ones().collect()))
                }
                set => Self::FromSet(Box::new(set)),
            },
            Self::If(condition, x, y) => match condition.simplify(registry) {
                Condition::Constant(true) => x.simplify(registry),
                Condition::Constant(false) => y.simplify(registry),
                condition => Self::If(
                    Box::new(condition),
                    Box::new(x.simplify(registry)),
                    Box::new(y.simplify(registry)),
                ),
            },
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum TableExpression<T: Clone> {
    Constant(T),
    Table1D(usize, ElementExpression),
    Table2D(usize, ElementExpression, ElementExpression),
    Table3D(
        usize,
        ElementExpression,
        ElementExpression,
        ElementExpression,
    ),
    Table(usize, Vec<ElementExpression>),
}

impl<T: Clone> TableExpression<T> {
    pub fn eval<'a>(
        &'a self,
        state: &State,
        registry: &'a TableRegistry,
        tables: &'a TableData<T>,
    ) -> &'a T {
        match self {
            Self::Constant(value) => value,
            Self::Table1D(i, x) => tables.tables_1d[*i].get(x.eval(state, registry)),
            Self::Table2D(i, x, y) => {
                tables.tables_2d[*i].get(x.eval(state, registry), y.eval(state, registry))
            }
            Self::Table3D(i, x, y, z) => tables.tables_3d[*i].get(
                x.eval(state, registry),
                y.eval(state, registry),
                z.eval(state, registry),
            ),
            Self::Table(i, args) => {
                let args: Vec<Element> = args.iter().map(|x| x.eval(state, registry)).collect();
                tables.tables[*i].get(&args)
            }
        }
    }

    pub fn simplify(&self, registry: &TableRegistry, tables: &TableData<T>) -> TableExpression<T> {
        match self {
            Self::Table1D(i, x) => match x.simplify(registry) {
                ElementExpression::Constant(x) => {
                    Self::Constant(tables.tables_1d[*i].get(x).clone())
                }
                x => Self::Table1D(*i, x),
            },
            Self::Table2D(i, x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (ElementExpression::Constant(x), ElementExpression::Constant(y)) => {
                    Self::Constant(tables.tables_2d[*i].get(x, y).clone())
                }
                (x, y) => Self::Table2D(*i, x, y),
            },
            Self::Table3D(i, x, y, z) => match (
                x.simplify(registry),
                y.simplify(registry),
                z.simplify(registry),
            ) {
                (
                    ElementExpression::Constant(x),
                    ElementExpression::Constant(y),
                    ElementExpression::Constant(z),
                ) => Self::Constant(tables.tables_3d[*i].get(x, y, z).clone()),
                (x, y, z) => Self::Table3D(*i, x, y, z),
            },
            Self::Table(i, args) => {
                let mut simplified_args = Vec::with_capacity(args.len());
                for arg in args {
                    match arg.simplify(registry) {
                        ElementExpression::Constant(arg) => simplified_args.push(arg),
                        _ => return self.clone(),
                    }
                }
                Self::Constant(tables.tables[*i].get(&simplified_args).clone())
            }
            _ => self.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::condition::{Comparison, ComparisonOperator};
    use super::super::numeric_expression::NumericExpression;
    use super::*;
    use crate::state::*;
    use crate::table::*;
    use rustc_hash::FxHashMap;
    use std::rc::Rc;

    fn generate_registry() -> TableRegistry {
        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("f0"), 1);

        let tables_1d = vec![Table1D::new(vec![1, 0])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("f1"), 0);

        let tables_2d = vec![Table2D::new(vec![vec![1, 0]])];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("f2"), 0);

        let tables_3d = vec![Table3D::new(vec![vec![vec![1, 0]]])];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("f3"), 0);

        let mut map = FxHashMap::default();
        let key = vec![0, 0, 0, 0];
        map.insert(key, 1);
        let key = vec![0, 0, 0, 1];
        map.insert(key, 0);
        let tables = vec![Table::new(map, 0)];
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("f4"), 0);

        let element_tables = TableData {
            name_to_constant,
            tables_1d,
            name_to_table_1d,
            tables_2d,
            name_to_table_2d,
            tables_3d,
            name_to_table_3d,
            tables,
            name_to_table,
        };

        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("t1"), 0);
        let vector_tables = TableData {
            tables_1d: vec![Table1D::new(vec![vec![0, 1]])],
            name_to_table_1d,
            ..Default::default()
        };

        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        let default = Set::with_capacity(3);
        let tables_1d = vec![Table1D::new(vec![set, default.clone(), default])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("s1"), 0);
        let set_tables = TableData {
            tables_1d,
            name_to_table_1d,
            ..Default::default()
        };

        TableRegistry {
            element_tables,
            set_tables,
            vector_tables,
            ..Default::default()
        }
    }

    fn generate_state() -> State {
        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(0);
        set2.insert(1);
        State {
            signature_variables: Rc::new(SignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2]],
                element_variables: vec![1],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn element_constant_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = ElementExpression::Constant(2);
        assert_eq!(expression.eval(&state, &registry), 2);
    }

    #[test]
    fn element_variable_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = ElementExpression::Variable(0);
        assert_eq!(expression.eval(&state, &registry), 1);
    }

    #[test]
    fn element_numeric_operation_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = ElementExpression::NumericOperation(
            NumericOperator::Add,
            Box::new(ElementExpression::Variable(0)),
            Box::new(ElementExpression::Constant(1)),
        );
        assert_eq!(expression.eval(&state, &registry), 2);
    }

    #[test]
    fn element_last_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = ElementExpression::Last(Box::new(VectorExpression::Reference(
            ReferenceExpression::Constant(vec![0, 1]),
        )));
        assert_eq!(expression.eval(&state, &registry), 1);
    }

    #[test]
    fn element_at_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = ElementExpression::At(
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
            Box::new(ElementExpression::Constant(0)),
        );
        assert_eq!(expression.eval(&state, &registry), 0);
    }

    #[test]
    fn element_table_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = ElementExpression::Table(Box::new(TableExpression::Constant(0)));
        assert_eq!(expression.eval(&state, &registry), 0);
    }

    #[test]
    fn element_if_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = ElementExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(ElementExpression::Constant(1)),
            Box::new(ElementExpression::Constant(0)),
        );
        assert_eq!(expression.eval(&state, &registry), 1);
        let expression = ElementExpression::If(
            Box::new(Condition::Constant(false)),
            Box::new(ElementExpression::Constant(1)),
            Box::new(ElementExpression::Constant(0)),
        );
        assert_eq!(expression.eval(&state, &registry), 0);
    }

    #[test]
    fn element_constant_simplify() {
        let registry = generate_registry();
        let expression = ElementExpression::Constant(2);
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn element_variable_simplify() {
        let registry = generate_registry();
        let expression = ElementExpression::Variable(0);
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn element_numeric_operation_simplify() {
        let registry = generate_registry();
        let expression = ElementExpression::NumericOperation(
            NumericOperator::Add,
            Box::new(ElementExpression::Constant(0)),
            Box::new(ElementExpression::Constant(1)),
        );
        assert_eq!(
            expression.simplify(&registry),
            ElementExpression::Constant(1)
        );
        let expression = ElementExpression::NumericOperation(
            NumericOperator::Add,
            Box::new(ElementExpression::Variable(0)),
            Box::new(ElementExpression::Constant(1)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn element_last_simplify() {
        let registry = generate_registry();
        let expression = ElementExpression::Last(Box::new(VectorExpression::Reference(
            ReferenceExpression::Constant(vec![0, 1]),
        )));
        assert_eq!(
            expression.simplify(&registry),
            ElementExpression::Constant(1)
        );
        let expression = ElementExpression::Last(Box::new(VectorExpression::Reference(
            ReferenceExpression::Variable(0),
        )));
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn element_at_simplify() {
        let registry = generate_registry();
        let expression = ElementExpression::At(
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
            Box::new(ElementExpression::Constant(0)),
        );
        assert_eq!(
            expression.simplify(&registry),
            ElementExpression::Constant(0)
        );
        let expression = ElementExpression::At(
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
            Box::new(ElementExpression::Variable(0)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn element_table_simplify() {
        let registry = generate_registry();
        let expression = ElementExpression::Table(Box::new(TableExpression::Constant(0)));
        assert_eq!(
            expression.simplify(&registry),
            ElementExpression::Constant(0)
        );
        let expression = ElementExpression::Table(Box::new(TableExpression::Table1D(
            0,
            ElementExpression::Variable(0),
        )));
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn element_if_simplify() {
        let registry = generate_registry();
        let expression = ElementExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(ElementExpression::Constant(1)),
            Box::new(ElementExpression::Constant(0)),
        );
        assert_eq!(
            expression.simplify(&registry),
            ElementExpression::Constant(1)
        );
        let expression = ElementExpression::If(
            Box::new(Condition::Constant(false)),
            Box::new(ElementExpression::Constant(1)),
            Box::new(ElementExpression::Constant(0)),
        );
        assert_eq!(
            expression.simplify(&registry),
            ElementExpression::Constant(0)
        );
        let expression = ElementExpression::If(
            Box::new(Condition::Comparison(Box::new(Comparison::ComparisonII(
                ComparisonOperator::Gt,
                NumericExpression::IntegerVariable(0),
                NumericExpression::Constant(1),
            )))),
            Box::new(ElementExpression::Constant(1)),
            Box::new(ElementExpression::Constant(0)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn set_if_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let mut s1 = Set::with_capacity(3);
        s1.insert(1);
        let mut s0 = Set::with_capacity(3);
        s0.insert(0);
        let expression = SetExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                s1.clone(),
            ))),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                s0.clone(),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry), s1);
        let expression = SetExpression::If(
            Box::new(Condition::Constant(false)),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(s1))),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                s0.clone(),
            ))),
        );
        assert_eq!(expression.eval(&state, &registry), s0);
    }

    #[test]
    fn set_reference_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        let expression = SetExpression::Reference(ReferenceExpression::Constant(set.clone()));
        assert_eq!(expression.eval(&state, &registry), set);
        let expression = SetExpression::Reference(ReferenceExpression::Variable(0));
        assert_eq!(
            expression.eval(&state, &registry),
            state.signature_variables.set_variables[0]
        );
        let expression = SetExpression::Reference(ReferenceExpression::Table(
            TableExpression::Table1D(0, ElementExpression::Constant(0)),
        ));
        assert_eq!(expression.eval(&state, &registry), set);
    }

    #[test]
    fn set_complement_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = SetExpression::Complement(Box::new(SetExpression::Reference(
            ReferenceExpression::Variable(0),
        )));
        let mut set = Set::with_capacity(3);
        set.insert(1);
        assert_eq!(expression.eval(&state, &registry), set);
    }

    #[test]
    fn set_union_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = SetExpression::SetOperation(
            SetOperator::Union,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        set.insert(2);
        assert_eq!(expression.eval(&state, &registry), set);
        let expression = SetExpression::SetOperation(
            SetOperator::Union,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert_eq!(
            expression.eval(&state, &registry),
            state.signature_variables.set_variables[0]
        );
    }

    #[test]
    fn set_difference_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = SetExpression::SetOperation(
            SetOperator::Difference,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(2);
        assert_eq!(expression.eval(&state, &registry), set);
        let expression = SetExpression::SetOperation(
            SetOperator::Difference,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert_eq!(expression.eval(&state, &registry), Set::with_capacity(3));
    }

    #[test]
    fn set_intersect_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = SetExpression::SetOperation(
            SetOperator::Intersection,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        assert_eq!(expression.eval(&state, &registry), set);
        let expression = SetExpression::SetOperation(
            SetOperator::Intersection,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert_eq!(
            expression.eval(&state, &registry),
            state.signature_variables.set_variables[0]
        );
    }

    #[test]
    fn set_add_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            ElementExpression::Constant(1),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        set.insert(2);
        assert_eq!(expression.eval(&state, &registry), set);
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            ElementExpression::Constant(0),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert_eq!(
            expression.eval(&state, &registry),
            state.signature_variables.set_variables[0]
        );
    }

    #[test]
    fn set_remove_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Remove,
            ElementExpression::Constant(2),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        assert_eq!(expression.eval(&state, &registry), set);
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Remove,
            ElementExpression::Constant(1),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert_eq!(
            expression.eval(&state, &registry),
            state.signature_variables.set_variables[0]
        );
    }

    #[test]
    fn set_from_vector_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = SetExpression::FromVector(
            3,
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        assert_eq!(expression.eval(&state, &registry), set);
        let expression = SetExpression::FromVector(
            3,
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                0,
            ))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        assert_eq!(expression.eval(&state, &registry), set);
    }

    #[test]
    fn set_reference_simplify() {
        let registry = generate_registry();
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        let expression = SetExpression::Reference(ReferenceExpression::Constant(set.clone()));
        assert_eq!(expression.simplify(&registry), expression);
        let expression = SetExpression::Reference(ReferenceExpression::Variable(0));
        assert_eq!(expression.simplify(&registry), expression);
        let expression = SetExpression::Reference(ReferenceExpression::Table(
            TableExpression::Table1D(0, ElementExpression::Constant(0)),
        ));
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Constant(set))
        );
    }

    #[test]
    fn set_complement_simplify() {
        let registry = generate_registry();
        let expression = SetExpression::Complement(Box::new(SetExpression::Reference(
            ReferenceExpression::Variable(0),
        )));
        assert_eq!(expression.simplify(&registry), expression);
        let mut set = Set::with_capacity(3);
        set.insert(1);
        let expression = SetExpression::Complement(Box::new(SetExpression::Reference(
            ReferenceExpression::Constant(set),
        )));
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Constant(set))
        );
        let expression = SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        ))));
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Variable(0))
        );
    }

    #[test]
    fn set_union_simplify() {
        let registry = generate_registry();
        let expression = SetExpression::SetOperation(
            SetOperator::Union,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
        );
        assert_eq!(expression.simplify(&registry), expression);
        let mut x = Set::with_capacity(3);
        x.insert(0);
        x.insert(2);
        let mut y = Set::with_capacity(3);
        y.insert(0);
        y.insert(1);
        let expression = SetExpression::SetOperation(
            SetOperator::Union,
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(x))),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(y))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        set.insert(2);
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Constant(set))
        );
        let expression = SetExpression::SetOperation(
            SetOperator::Union,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Variable(0))
        );
    }

    #[test]
    fn set_difference_simplify() {
        let registry = generate_registry();
        let expression = SetExpression::SetOperation(
            SetOperator::Difference,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
        );
        assert_eq!(expression.simplify(&registry), expression);
        let mut x = Set::with_capacity(3);
        x.insert(0);
        x.insert(2);
        let mut y = Set::with_capacity(3);
        y.insert(0);
        y.insert(1);
        let expression = SetExpression::SetOperation(
            SetOperator::Difference,
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(x))),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(y))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(2);
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Constant(set))
        );
    }

    #[test]
    fn set_intersect_simplify() {
        let registry = generate_registry();
        let expression = SetExpression::SetOperation(
            SetOperator::Intersection,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
        );
        assert_eq!(expression.simplify(&registry), expression);
        let mut x = Set::with_capacity(3);
        x.insert(0);
        x.insert(2);
        let mut y = Set::with_capacity(3);
        y.insert(0);
        y.insert(1);
        let expression = SetExpression::SetOperation(
            SetOperator::Intersection,
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(x))),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(y))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Constant(set))
        );
        let expression = SetExpression::SetOperation(
            SetOperator::Union,
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Variable(0))
        );
    }

    #[test]
    fn set_add_simplify() {
        let registry = generate_registry();
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            ElementExpression::Constant(1),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert_eq!(expression.simplify(&registry), expression);
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            ElementExpression::Constant(1),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(set))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        set.insert(2);
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Constant(set))
        );
    }

    #[test]
    fn set_remove_simplify() {
        let registry = generate_registry();
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Remove,
            ElementExpression::Constant(2),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        assert_eq!(expression.simplify(&registry), expression);
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Remove,
            ElementExpression::Constant(2),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(set))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Constant(set))
        );
    }

    #[test]
    fn set_from_vector_simplify() {
        let registry = generate_registry();
        let expression = SetExpression::FromVector(
            3,
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Constant(set))
        );
        let expression = SetExpression::FromVector(
            3,
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                0,
            ))),
        );
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn set_if_simplify() {
        let registry = generate_registry();
        let mut s1 = Set::with_capacity(3);
        s1.insert(1);
        let mut s0 = Set::with_capacity(3);
        s0.insert(0);
        let expression = SetExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                s1.clone(),
            ))),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                s0.clone(),
            ))),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Constant(s1.clone()))
        );
        let expression = SetExpression::If(
            Box::new(Condition::Constant(false)),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                s1.clone(),
            ))),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(
                s0.clone(),
            ))),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetExpression::Reference(ReferenceExpression::Constant(s0.clone()))
        );
        let expression = SetExpression::If(
            Box::new(Condition::Comparison(Box::new(Comparison::ComparisonII(
                ComparisonOperator::Gt,
                NumericExpression::IntegerVariable(0),
                NumericExpression::Constant(1),
            )))),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(s1))),
            Box::new(SetExpression::Reference(ReferenceExpression::Constant(s0))),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn vector_reference_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = VectorExpression::Reference(ReferenceExpression::Constant(vec![1, 2]));
        assert_eq!(expression.eval(&state, &registry), vec![1, 2]);
    }

    #[test]
    fn vector_indices_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = VectorExpression::Indices(Box::new(VectorExpression::Reference(
            ReferenceExpression::Constant(vec![1, 2]),
        )));
        assert_eq!(expression.eval(&state, &registry), vec![0, 1]);
    }

    #[test]
    fn vector_reverse_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = VectorExpression::Reverse(Box::new(VectorExpression::Reference(
            ReferenceExpression::Constant(vec![1, 2]),
        )));
        assert_eq!(expression.eval(&state, &registry), vec![2, 1]);
    }

    #[test]
    fn vector_set_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = VectorExpression::Set(
            ElementExpression::Constant(3),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![1, 2],
            ))),
            ElementExpression::Constant(0),
        );
        assert_eq!(expression.eval(&state, &registry), vec![3, 2]);
    }

    #[test]
    fn vector_push_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = VectorExpression::Push(
            ElementExpression::Constant(0),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![1, 2],
            ))),
        );
        assert_eq!(expression.eval(&state, &registry), vec![1, 2, 0]);
    }

    #[test]
    fn vector_pop_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = VectorExpression::Pop(Box::new(VectorExpression::Reference(
            ReferenceExpression::Constant(vec![1, 2]),
        )));
        assert_eq!(expression.eval(&state, &registry), vec![1]);
    }

    #[test]
    fn vector_from_set_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        let expression = VectorExpression::FromSet(Box::new(SetExpression::Reference(
            ReferenceExpression::Constant(set),
        )));
        assert_eq!(expression.eval(&state, &registry), vec![0, 1]);
        let expression = VectorExpression::FromSet(Box::new(SetExpression::Reference(
            ReferenceExpression::Variable(0),
        )));
        assert_eq!(expression.eval(&state, &registry), vec![0, 2]);
    }

    #[test]
    fn vector_if_eval() {
        let state = generate_state();
        let registry = generate_registry();
        let expression = VectorExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![1, 0],
            ))),
        );
        assert_eq!(expression.eval(&state, &registry), vec![0, 1]);
        let expression = VectorExpression::If(
            Box::new(Condition::Constant(false)),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![1, 0],
            ))),
        );
        assert_eq!(expression.eval(&state, &registry), vec![1, 0]);
    }

    #[test]
    fn vector_reference_simplify() {
        let registry = generate_registry();
        let expression = VectorExpression::Reference(ReferenceExpression::Constant(vec![1, 2]));
        assert_eq!(expression.simplify(&registry), expression);
        let expression = VectorExpression::Reference(ReferenceExpression::Table(
            TableExpression::Table1D(0, ElementExpression::Constant(0)),
        ));
        assert_eq!(
            expression.simplify(&registry),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1]))
        );
    }

    #[test]
    fn vector_indices_simplify() {
        let registry = generate_registry();

        let expression = VectorExpression::Indices(Box::new(VectorExpression::Reference(
            ReferenceExpression::Variable(0),
        )));
        assert_eq!(expression.simplify(&registry), expression);

        let expression = VectorExpression::Indices(Box::new(VectorExpression::Reference(
            ReferenceExpression::Constant(vec![1, 2]),
        )));
        assert_eq!(
            expression.simplify(&registry),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1]))
        );
    }

    #[test]
    fn vector_push_simplify() {
        let registry = generate_registry();
        let expression = VectorExpression::Push(
            ElementExpression::Constant(0),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![1, 2],
            ))),
        );
        assert_eq!(
            expression.simplify(&registry),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![1, 2, 0]))
        );
        let expression = VectorExpression::Push(
            ElementExpression::Constant(0),
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                0,
            ))),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn vector_pop_simplify() {
        let registry = generate_registry();
        let expression = VectorExpression::Pop(Box::new(VectorExpression::Reference(
            ReferenceExpression::Constant(vec![1, 2]),
        )));
        assert_eq!(
            expression.simplify(&registry),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![1]))
        );
        let expression = VectorExpression::Pop(Box::new(VectorExpression::Reference(
            ReferenceExpression::Variable(0),
        )));
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn vector_set_simplify() {
        let registry = generate_registry();
        let expression = VectorExpression::Set(
            ElementExpression::Constant(0),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![1, 2],
            ))),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.simplify(&registry),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 2]))
        );
        let expression = VectorExpression::Set(
            ElementExpression::Constant(0),
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                0,
            ))),
            ElementExpression::Variable(0),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn vector_from_set_simplify() {
        let registry = generate_registry();
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        let expression = VectorExpression::FromSet(Box::new(SetExpression::Reference(
            ReferenceExpression::Constant(set),
        )));
        assert_eq!(
            expression.simplify(&registry),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1]))
        );
        let expression = VectorExpression::FromSet(Box::new(SetExpression::Reference(
            ReferenceExpression::Variable(0),
        )));
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn table_constant_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression = TableExpression::Constant(1);
        assert_eq!(
            *expression.eval(&state, &registry, &registry.element_tables),
            1
        );
    }

    #[test]
    fn table_1d_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = TableExpression::Table1D(0, ElementExpression::Constant(0));
        assert_eq!(
            *expression.eval(&state, &registry, &registry.element_tables),
            1
        );
        let expression = TableExpression::Table1D(0, ElementExpression::Constant(1));
        assert_eq!(
            *expression.eval(&state, &registry, &registry.element_tables),
            0
        );
    }

    #[test]
    fn table_2d_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = TableExpression::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            *expression.eval(&state, &registry, &registry.element_tables),
            1
        );
        let expression = TableExpression::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(1),
        );
        assert_eq!(
            *expression.eval(&state, &registry, &registry.element_tables),
            0
        );
    }

    #[test]
    fn table_3d_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = TableExpression::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            *expression.eval(&state, &registry, &registry.element_tables),
            1
        );
        let expression = TableExpression::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(1),
        );
        assert_eq!(
            *expression.eval(&state, &registry, &registry.element_tables),
            0
        );
    }

    #[test]
    fn table_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = TableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            ],
        );
        assert_eq!(
            *expression.eval(&state, &registry, &registry.element_tables),
            1
        );
        let expression = TableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(1),
            ],
        );
        assert_eq!(
            *expression.eval(&state, &registry, &registry.element_tables),
            0
        );
        let expression = TableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(2),
            ],
        );
        assert_eq!(
            *expression.eval(&state, &registry, &registry.element_tables),
            0
        );
    }

    #[test]
    fn table_constant_simplify() {
        let registry = generate_registry();
        let expression = TableExpression::Constant(1);
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            expression
        );
    }

    #[test]
    fn table_1d_simplify() {
        let registry = generate_registry();

        let expression = TableExpression::Table1D(0, ElementExpression::Constant(0));
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            TableExpression::Constant(1)
        );

        let expression = TableExpression::Table1D(0, ElementExpression::Constant(1));
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            TableExpression::Constant(0)
        );

        let expression = TableExpression::Table1D(0, ElementExpression::Variable(0));
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            expression
        );
    }

    #[test]
    fn table_2d_simplify() {
        let registry = generate_registry();

        let expression = TableExpression::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            TableExpression::Constant(1)
        );

        let expression = TableExpression::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(1),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            TableExpression::Constant(0)
        );

        let expression = TableExpression::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            expression
        );
    }

    #[test]
    fn table_3d_simplify() {
        let registry = generate_registry();

        let expression = TableExpression::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            TableExpression::Constant(1)
        );

        let expression = TableExpression::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(1),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            TableExpression::Constant(0)
        );

        let expression = TableExpression::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            expression
        );
    }

    #[test]
    fn table_simplify() {
        let registry = generate_registry();

        let expression = TableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            ],
        );
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            TableExpression::Constant(1)
        );

        let expression = TableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(1),
            ],
        );
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            TableExpression::Constant(0)
        );

        let expression = TableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(2),
            ],
        );
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            TableExpression::Constant(0)
        );

        let expression = TableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Variable(0),
            ],
        );
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            expression
        );
    }

    #[test]
    fn vector_if_simplify() {
        let registry = generate_registry();
        let expression = VectorExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![1, 0],
            ))),
        );
        assert_eq!(
            expression.simplify(&registry),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1]))
        );
        let expression = VectorExpression::If(
            Box::new(Condition::Constant(false)),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![1, 0],
            ))),
        );
        assert_eq!(
            expression.simplify(&registry),
            VectorExpression::Reference(ReferenceExpression::Constant(vec![1, 0]))
        );
        let expression = VectorExpression::If(
            Box::new(Condition::Comparison(Box::new(Comparison::ComparisonII(
                ComparisonOperator::Gt,
                NumericExpression::IntegerVariable(0),
                NumericExpression::Constant(1),
            )))),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![0, 1],
            ))),
            Box::new(VectorExpression::Reference(ReferenceExpression::Constant(
                vec![1, 0],
            ))),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }
}
