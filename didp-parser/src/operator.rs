use crate::expression;
use crate::expression_parser;
use crate::state;
use crate::table_registry;
use crate::variable;
use crate::yaml_util;
use std::collections;
use std::error;
use std::fmt;
use std::rc::Rc;
use std::str;

#[derive(Debug, PartialEq, Eq, Clone, Default)]
pub struct Operator<T: variable::Numeric> {
    pub name: String,
    pub elements_in_set_variable: Vec<(usize, usize)>,
    pub elements_in_permutation_variable: Vec<(usize, usize)>,
    pub preconditions: Vec<expression::Condition<T>>,
    pub set_effects: Vec<(usize, expression::SetExpression)>,
    pub permutation_effects: Vec<(usize, expression::ElementExpression)>,
    pub element_effects: Vec<(usize, expression::ElementExpression)>,
    pub numeric_effects: Vec<(usize, expression::NumericExpression<T>)>,
    pub resource_effects: Vec<(usize, expression::NumericExpression<T>)>,
    pub cost: expression::NumericExpression<T>,
}

impl<T: variable::Numeric> Operator<T> {
    pub fn is_applicable(
        &self,
        state: &state::State<T>,
        metadata: &state::StateMetadata,
        registry: &table_registry::TableRegistry<T>,
    ) -> bool {
        for c in &self.preconditions {
            if !c.eval(state, metadata, registry) {
                return false;
            }
        }
        true
    }

    pub fn apply_effects(
        &self,
        state: &state::State<T>,
        metadata: &state::StateMetadata,
        registry: &table_registry::TableRegistry<T>,
    ) -> state::State<T> {
        let len = state.signature_variables.set_variables.len();
        let mut set_variables = Vec::with_capacity(len);
        let mut i = 0;

        for e in &self.set_effects {
            while i < e.0 {
                set_variables.push(state.signature_variables.set_variables[i].clone());
                i += 1;
            }
            set_variables.push(e.1.eval(state, metadata));
            i += 1;
        }
        while i < len {
            set_variables.push(state.signature_variables.set_variables[i].clone());
            i += 1;
        }

        let mut permutation_variables = state.signature_variables.permutation_variables.clone();
        for e in &self.permutation_effects {
            permutation_variables[e.0].push(e.1.eval(state));
        }

        let mut element_variables = state.signature_variables.element_variables.clone();
        for e in &self.element_effects {
            element_variables[e.0] = e.1.eval(state);
        }

        let mut numeric_variables = state.signature_variables.numeric_variables.clone();
        for e in &self.numeric_effects {
            numeric_variables[e.0] = e.1.eval(state, metadata, registry);
        }

        let mut resource_variables = state.resource_variables.clone();
        for e in &self.resource_effects {
            resource_variables[e.0] = e.1.eval(state, metadata, registry);
        }

        let stage = state.stage + 1;
        let cost = self.cost.eval(state, metadata, registry);

        state::State {
            signature_variables: {
                Rc::new(state::SignatureVariables {
                    set_variables,
                    permutation_variables,
                    element_variables,
                    numeric_variables,
                })
            },
            resource_variables,
            stage,
            cost,
        }
    }
}

pub fn load_operators_from_yaml<T: variable::Numeric>(
    value: &yaml_rust::Yaml,
    metadata: &state::StateMetadata,
    registry: &table_registry::TableRegistry<T>,
) -> Result<Vec<Operator<T>>, Box<dyn error::Error>>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let map = yaml_util::get_map(value)?;
    let lifted_name = yaml_util::get_string_by_key(map, "name")?;

    let (
        parameters_array,
        elements_in_set_variable_array,
        elements_in_permutation_variable_array,
        parameter_names,
    ) = match map.get(&yaml_rust::Yaml::from_str("parameters")) {
        Some(value) => {
            let result = metadata.ground_parameters_from_yaml(value)?;
            let array = yaml_util::get_array(value)?;
            let mut parameter_names = Vec::with_capacity(array.len());
            for map in array {
                let map = yaml_util::get_map(map)?;
                let value = yaml_util::get_string_by_key(&map, "name")?;
                parameter_names.push(value);
            }
            (result.0, result.1, result.2, parameter_names)
        }
        None => (
            vec![collections::HashMap::new()],
            vec![vec![]],
            vec![vec![]],
            vec![],
        ),
    };

    let lifted_preconditions = yaml_util::get_string_array_by_key(map, "preconditions")?;
    let lifted_effects = yaml_util::get_map_by_key(map, "effects")?;
    let lifted_cost = yaml_util::get_string_by_key(map, "cost")?;

    let mut operators = Vec::with_capacity(parameters_array.len());
    'outer: for ((parameters, elements_in_set_variable), elements_in_permutation_variable) in
        parameters_array
            .into_iter()
            .zip(elements_in_set_variable_array.into_iter())
            .zip(elements_in_permutation_variable_array.into_iter())
    {
        let mut name = lifted_name.clone();
        for parameter_name in &parameter_names {
            name += format!(" {}:{}", parameter_name, parameters[parameter_name]).as_str();
        }
        let mut preconditions = Vec::with_capacity(lifted_preconditions.len());
        for condition in &lifted_preconditions {
            let condition = expression_parser::parse_condition(
                condition.clone(),
                metadata,
                registry,
                &parameters,
            )?;
            let condition = condition.simplify(registry);
            match condition {
                expression::Condition::Constant(true) => continue,
                expression::Condition::Constant(false) => continue 'outer,
                _ => preconditions.push(condition),
            }
        }
        let mut set_effects = Vec::new();
        let mut permutation_effects = Vec::new();
        let mut element_effects = Vec::new();
        let mut numeric_effects = Vec::new();
        let mut resource_effects = Vec::new();
        for (variable, effect) in lifted_effects {
            let effect = yaml_util::get_string(effect)?;
            let variable = yaml_util::get_string(variable)?;
            if let Some(i) = metadata.name_to_set_variable.get(&variable) {
                let effect = expression_parser::parse_set(effect, metadata, &parameters)?;
                set_effects.push((*i, effect));
            } else if let Some(i) = metadata.name_to_permutation_variable.get(&variable) {
                let effect = expression_parser::parse_element(effect, metadata, &parameters)?;
                permutation_effects.push((*i, effect));
            } else if let Some(i) = metadata.name_to_element_variable.get(&variable) {
                let effect = expression_parser::parse_element(effect, metadata, &parameters)?;
                element_effects.push((*i, effect));
            } else if let Some(i) = metadata.name_to_numeric_variable.get(&variable) {
                let effect =
                    expression_parser::parse_numeric(effect, metadata, registry, &parameters)?;
                numeric_effects.push((*i, effect.simplify(registry)));
            } else if let Some(i) = metadata.name_to_resource_variable.get(&variable) {
                let effect =
                    expression_parser::parse_numeric(effect, metadata, registry, &parameters)?;
                resource_effects.push((*i, effect.simplify(registry)));
            } else {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "no such variable `{}`",
                    variable
                ))
                .into());
            }
        }
        let cost =
            expression_parser::parse_numeric(lifted_cost.clone(), metadata, registry, &parameters)?;
        let cost = cost.simplify(registry);

        operators.push(Operator {
            name,
            elements_in_set_variable,
            elements_in_permutation_variable,
            preconditions,
            set_effects,
            permutation_effects,
            element_effects,
            numeric_effects,
            resource_effects,
            cost,
        })
    }
    Ok(operators)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table;
    use expression::*;
    use std::collections::HashMap;

    fn generate_metadata() -> state::StateMetadata {
        let object_names = vec![String::from("object")];
        let object_numbers = vec![3];
        let mut name_to_object = HashMap::new();
        name_to_object.insert("object".to_string(), 0);

        let set_variable_names = vec![
            String::from("s0"),
            String::from("s1"),
            String::from("s2"),
            String::from("s3"),
        ];
        let mut name_to_set_variable = HashMap::new();
        name_to_set_variable.insert(String::from("s0"), 0);
        name_to_set_variable.insert(String::from("s1"), 1);
        name_to_set_variable.insert(String::from("s2"), 2);
        name_to_set_variable.insert(String::from("s3"), 3);
        let set_variable_to_object = vec![0, 0, 0, 0];

        let permutation_variable_names = vec![
            "p0".to_string(),
            "p1".to_string(),
            "p2".to_string(),
            "p3".to_string(),
        ];
        let mut name_to_permutation_variable = HashMap::new();
        name_to_permutation_variable.insert("p0".to_string(), 0);
        name_to_permutation_variable.insert("p1".to_string(), 1);
        name_to_permutation_variable.insert("p2".to_string(), 2);
        name_to_permutation_variable.insert("p3".to_string(), 3);
        let permutation_variable_to_object = vec![0, 0, 0, 0];

        let element_variable_names = vec![
            "e0".to_string(),
            "e1".to_string(),
            "e2".to_string(),
            "e3".to_string(),
        ];
        let mut name_to_element_variable = HashMap::new();
        name_to_element_variable.insert("e0".to_string(), 0);
        name_to_element_variable.insert("e1".to_string(), 1);
        name_to_element_variable.insert("e2".to_string(), 2);
        name_to_element_variable.insert("e3".to_string(), 3);
        let element_variable_to_object = vec![0, 0, 0, 0];

        let numeric_variable_names = vec![
            "n0".to_string(),
            "n1".to_string(),
            "n2".to_string(),
            "n3".to_string(),
        ];
        let mut name_to_numeric_variable = HashMap::new();
        name_to_numeric_variable.insert("n0".to_string(), 0);
        name_to_numeric_variable.insert("n1".to_string(), 1);
        name_to_numeric_variable.insert("n2".to_string(), 2);
        name_to_numeric_variable.insert("n3".to_string(), 3);

        let resource_variable_names = vec![
            "r0".to_string(),
            "r1".to_string(),
            "r2".to_string(),
            "r3".to_string(),
        ];
        let mut name_to_resource_variable = HashMap::new();
        name_to_resource_variable.insert("r0".to_string(), 0);
        name_to_resource_variable.insert("r1".to_string(), 1);
        name_to_resource_variable.insert("r2".to_string(), 2);
        name_to_resource_variable.insert("r3".to_string(), 3);

        state::StateMetadata {
            object_names,
            name_to_object,
            object_numbers,
            set_variable_names,
            name_to_set_variable,
            set_variable_to_object,
            permutation_variable_names,
            name_to_permutation_variable,
            permutation_variable_to_object,
            element_variable_names,
            name_to_element_variable,
            element_variable_to_object,
            numeric_variable_names,
            name_to_numeric_variable,
            resource_variable_names,
            name_to_resource_variable,
            less_is_better: vec![false, false, true, false],
        }
    }

    fn generate_registry() -> table_registry::TableRegistry<variable::IntegerVariable> {
        let tables_1d = vec![table::Table1D::new(vec![10, 20, 30])];
        let mut name_to_table_1d = HashMap::new();
        name_to_table_1d.insert(String::from("f1"), 0);

        let tables_2d = vec![table::Table2D::new(vec![
            vec![10, 20, 30],
            vec![40, 50, 60],
        ])];
        let mut name_to_table_2d = HashMap::new();
        name_to_table_2d.insert(String::from("f2"), 0);

        table_registry::TableRegistry {
            numeric_tables: table_registry::TableData {
                tables_1d,
                name_to_table_1d,
                tables_2d,
                name_to_table_2d,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn generate_state() -> state::State<variable::IntegerVariable> {
        let mut set1 = variable::SetVariable::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = variable::SetVariable::with_capacity(3);
        set2.insert(0);
        set2.insert(1);
        state::State {
            signature_variables: Rc::new(state::SignatureVariables {
                set_variables: vec![set1, set2],
                permutation_variables: vec![vec![0, 2], vec![1, 2]],
                element_variables: vec![1, 2],
                numeric_variables: vec![1, 2, 3],
            }),
            resource_variables: vec![4, 5, 6],
            stage: 0,
            cost: 0,
        }
    }

    #[test]
    fn applicable() {
        let state = generate_state();
        let registry = generate_registry();
        let metadata = generate_metadata();
        let set_condition = Condition::Set(SetCondition::IsIn(
            ElementExpression::Constant(0),
            SetExpression::SetVariable(0),
        ));
        let numeric_condition = Condition::Comparison(
            ComparisonOperator::Ge,
            NumericExpression::Variable(0),
            NumericExpression::Constant(1),
        );
        let operator = Operator {
            name: String::from(""),
            elements_in_set_variable: Vec::new(),
            elements_in_permutation_variable: Vec::new(),
            preconditions: vec![set_condition, numeric_condition],
            cost: NumericExpression::Constant(0),
            ..Default::default()
        };
        assert!(operator.is_applicable(&state, &metadata, &registry));
    }

    #[test]
    fn not_applicable() {
        let state = generate_state();
        let registry = generate_registry();
        let metadata = generate_metadata();
        let set_condition = Condition::Set(SetCondition::IsIn(
            ElementExpression::Constant(0),
            SetExpression::SetVariable(0),
        ));
        let numeric_condition = Condition::Comparison(
            ComparisonOperator::Le,
            NumericExpression::Variable(0),
            NumericExpression::Constant(1),
        );
        let operator = Operator {
            name: String::from(""),
            elements_in_set_variable: Vec::new(),
            elements_in_permutation_variable: Vec::new(),
            preconditions: vec![set_condition, numeric_condition],
            cost: NumericExpression::Constant(0),
            ..Default::default()
        };
        assert!(operator.is_applicable(&state, &metadata, &registry));
    }

    #[test]
    fn appy_effects() {
        let state = generate_state();
        let registry = generate_registry();
        let metadata = generate_metadata();
        let set_effect1 = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            Box::new(SetExpression::SetVariable(0)),
            ElementExpression::Constant(1),
        );
        let set_effect2 = SetExpression::SetElementOperation(
            SetElementOperator::Remove,
            Box::new(SetExpression::SetVariable(1)),
            ElementExpression::Constant(0),
        );
        let permutation_effect1 = ElementExpression::Constant(1);
        let permutation_effect2 = ElementExpression::Constant(0);
        let element_effect1 = ElementExpression::Constant(2);
        let element_effect2 = ElementExpression::Constant(1);
        let numeric_effect1 = NumericExpression::NumericOperation(
            NumericOperator::Subtract,
            Box::new(NumericExpression::Variable(0)),
            Box::new(NumericExpression::Constant(1)),
        );
        let numeric_effect2 = NumericExpression::NumericOperation(
            NumericOperator::Multiply,
            Box::new(NumericExpression::Variable(1)),
            Box::new(NumericExpression::Constant(2)),
        );
        let resource_effect1 = NumericExpression::NumericOperation(
            NumericOperator::Add,
            Box::new(NumericExpression::ResourceVariable(0)),
            Box::new(NumericExpression::Constant(1)),
        );
        let resource_effect2 = NumericExpression::NumericOperation(
            NumericOperator::Divide,
            Box::new(NumericExpression::ResourceVariable(1)),
            Box::new(NumericExpression::Constant(2)),
        );
        let operator = Operator {
            name: String::from(""),
            set_effects: vec![(0, set_effect1), (1, set_effect2)],
            permutation_effects: vec![(0, permutation_effect1), (1, permutation_effect2)],
            element_effects: vec![(0, element_effect1), (1, element_effect2)],
            numeric_effects: vec![(0, numeric_effect1), (1, numeric_effect2)],
            resource_effects: vec![(0, resource_effect1), (1, resource_effect2)],
            cost: NumericExpression::NumericOperation(
                NumericOperator::Add,
                Box::new(NumericExpression::Cost),
                Box::new(NumericExpression::Constant(1)),
            ),
            ..Default::default()
        };

        let mut set1 = variable::SetVariable::with_capacity(3);
        set1.insert(0);
        set1.insert(1);
        set1.insert(2);
        let mut set2 = variable::SetVariable::with_capacity(3);
        set2.insert(1);
        let expected = state::State {
            signature_variables: Rc::new(state::SignatureVariables {
                set_variables: vec![set1, set2],
                permutation_variables: vec![vec![0, 2, 1], vec![1, 2, 0]],
                element_variables: vec![2, 1],
                numeric_variables: vec![0, 4, 3],
            }),
            resource_variables: vec![5, 2, 6],
            stage: 1,
            cost: 1,
        };
        let successor = operator.apply_effects(&state, &metadata, &registry);
        assert_eq!(successor, expected);
    }

    #[test]
    fn load_operators_from_yaml_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();

        let operator = r"
name: operator
preconditions: [(>= (f2 0 1) 10)]
effects: {e0: '0'}
cost: '0'
";
        let operator = yaml_rust::YamlLoader::load_from_str(operator);
        assert!(operator.is_ok());
        let operator = operator.unwrap();
        assert_eq!(operator.len(), 1);
        let operator = &operator[0];
        let operators = load_operators_from_yaml(operator, &metadata, &registry);
        println!("{:?}", operators);
        assert!(operators.is_ok());
        let expected = vec![Operator {
            name: String::from("operator"),
            preconditions: Vec::new(),
            element_effects: vec![(0, ElementExpression::Constant(0))],
            cost: NumericExpression::Constant(0),
            ..Default::default()
        }];
        assert_eq!(operators.unwrap(), expected);

        let operator = r"
name: operator
parameters:
        - name: e
          object: s0
preconditions:
        - (>= (f2 e0 e) 10)
        - (is_not e 2)
effects:
        e0: e
        s0: (+ s0 e)
        p0: e
        n0: '1'
        r0: '2'
cost: (+ cost (f1 e))
";
        let operator = yaml_rust::YamlLoader::load_from_str(operator);
        assert!(operator.is_ok());
        let operator = operator.unwrap();
        assert_eq!(operator.len(), 1);
        let operator = &operator[0];
        let operators = load_operators_from_yaml(operator, &metadata, &registry);
        assert!(operators.is_ok());
        let expected = vec![
            Operator {
                name: String::from("operator e:0"),
                elements_in_set_variable: vec![(0, 0)],
                elements_in_permutation_variable: Vec::new(),
                preconditions: vec![Condition::Comparison(
                    ComparisonOperator::Ge,
                    NumericExpression::Table(NumericTableExpression::Table2D(
                        0,
                        ElementExpression::Variable(0),
                        ElementExpression::Constant(0),
                    )),
                    NumericExpression::Constant(10),
                )],
                set_effects: vec![(
                    0,
                    SetExpression::SetElementOperation(
                        SetElementOperator::Add,
                        Box::new(SetExpression::SetVariable(0)),
                        ElementExpression::Constant(0),
                    ),
                )],
                permutation_effects: vec![(0, ElementExpression::Constant(0))],
                element_effects: vec![(0, ElementExpression::Constant(0))],
                numeric_effects: vec![(0, NumericExpression::Constant(1))],
                resource_effects: vec![(0, NumericExpression::Constant(2))],
                cost: NumericExpression::NumericOperation(
                    NumericOperator::Add,
                    Box::new(NumericExpression::Cost),
                    Box::new(NumericExpression::Constant(10)),
                ),
            },
            Operator {
                name: String::from("operator e:1"),
                elements_in_set_variable: vec![(0, 1)],
                elements_in_permutation_variable: Vec::new(),
                preconditions: vec![Condition::Comparison(
                    ComparisonOperator::Ge,
                    NumericExpression::Table(NumericTableExpression::Table2D(
                        0,
                        ElementExpression::Variable(0),
                        ElementExpression::Constant(1),
                    )),
                    NumericExpression::Constant(10),
                )],
                set_effects: vec![(
                    0,
                    SetExpression::SetElementOperation(
                        SetElementOperator::Add,
                        Box::new(SetExpression::SetVariable(0)),
                        ElementExpression::Constant(1),
                    ),
                )],
                permutation_effects: vec![(0, ElementExpression::Constant(1))],
                element_effects: vec![(0, ElementExpression::Constant(1))],
                numeric_effects: vec![(0, NumericExpression::Constant(1))],
                resource_effects: vec![(0, NumericExpression::Constant(2))],
                cost: NumericExpression::NumericOperation(
                    NumericOperator::Add,
                    Box::new(NumericExpression::Cost),
                    Box::new(NumericExpression::Constant(20)),
                ),
            },
        ];
        assert_eq!(operators.unwrap(), expected);
    }

    #[test]
    fn load_operators_from_yaml_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();

        let operator = r"
parameters:
        - name: e
          object: s0
preconditions:
        - (>= (f2 e0 e) 10)
effects:
        e0: e
        s0: (+ s0 e)
        p0: e
        n0: '1'
        r0: '2'
cost: (+ cost (f1 e))
";
        let operator = yaml_rust::YamlLoader::load_from_str(operator);
        assert!(operator.is_ok());
        let operator = operator.unwrap();
        assert_eq!(operator.len(), 1);
        let operator = &operator[0];
        let operators = load_operators_from_yaml(operator, &metadata, &registry);
        assert!(operators.is_err());

        let operator = r"
name: operator
parameters:
        - name: e
          object: s0
effects:
        e0: e
        s0: (+ s0 e)
        p0: e
        n0: '1'
        r0: '2'
cost: (+ cost (f1 e))
";
        let operator = yaml_rust::YamlLoader::load_from_str(operator);
        assert!(operator.is_ok());
        let operator = operator.unwrap();
        assert_eq!(operator.len(), 1);
        let operator = &operator[0];
        let operators = load_operators_from_yaml(operator, &metadata, &registry);
        assert!(operators.is_err());

        let operator = r"
name: operator
parameters:
        - name: e
          object: s0
preconditions:
        - (>= (f2 e0 e) 10)
cost: (+ cost (f1 e))
";
        let operator = yaml_rust::YamlLoader::load_from_str(operator);
        assert!(operator.is_ok());
        let operator = operator.unwrap();
        assert_eq!(operator.len(), 1);
        let operator = &operator[0];
        let operators = load_operators_from_yaml(operator, &metadata, &registry);
        assert!(operators.is_err());

        let operator = r"
name: operator
parameters:
        - name: e
          object: s0
preconditions:
        - (>= (f2 e0 e) 10)
effects:
        e0: e
        s0: (+ s0 e)
        p0: e
        n0: '1'
        r0: '2'
";
        let operator = yaml_rust::YamlLoader::load_from_str(operator);
        assert!(operator.is_ok());
        let operator = operator.unwrap();
        assert_eq!(operator.len(), 1);
        let operator = &operator[0];
        let operators = load_operators_from_yaml(operator, &metadata, &registry);
        assert!(operators.is_err());

        let operator = r"
name: operator
preconditions:
        - (>= (f2 e0 e) 10)
effects:
        e0: e
        s0: (+ s0 e)
        p0: e
        n0: '1'
        r0: '2'
cost: (+ cost (f1 e))
";
        let operator = yaml_rust::YamlLoader::load_from_str(operator);
        assert!(operator.is_ok());
        let operator = operator.unwrap();
        assert_eq!(operator.len(), 1);
        let operator = &operator[0];
        let operators = load_operators_from_yaml(operator, &metadata, &registry);
        assert!(operators.is_err());

        let operator = r"
parameters:
        - name: e
          object: s0
preconditions:
        - (>= (f2 e0 e) 10)
effects:
        e0: e
        s0: (+ s0 e)
        p0: e
        n0: '1'
        r0: '2'
        r5: '5'
cost: (+ cost (f1 e))
";
        let operator = yaml_rust::YamlLoader::load_from_str(operator);
        assert!(operator.is_ok());
        let operator = operator.unwrap();
        assert_eq!(operator.len(), 1);
        let operator = &operator[0];
        let operators = load_operators_from_yaml(operator, &metadata, &registry);
        assert!(operators.is_err());
    }
}
