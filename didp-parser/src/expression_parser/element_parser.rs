use super::util;
use super::util::ParseErr;
use crate::expression::{
    ElementExpression, NumericOperator, ReferenceExpression, SetElementOperator, SetExpression,
    SetOperator, TableExpression, VectorExpression,
};
use crate::state::StateMetadata;
use crate::table_data::TableData;
use crate::table_registry::TableRegistry;
use crate::variable::{Element, Set};
use rustc_hash::FxHashMap;

pub fn parse_expression<'a>(
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<(ElementExpression, &'a [String]), ParseErr> {
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
    match &token[..] {
        "(" => {
            let (name, rest) = rest
                .split_first()
                .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
            if let Some((expression, rest)) = parse_table_expression(
                name,
                rest,
                metadata,
                registry,
                parameters,
                &registry.element_tables,
            )? {
                Ok((ElementExpression::Table(Box::new(expression)), rest))
            } else if let Some((expression, rest)) =
                parse_operation(name, rest, metadata, registry, parameters)?
            {
                Ok((expression, rest))
            } else if let Some((expression, rest)) =
                parse_element_from_vector(name, rest, metadata, registry, parameters)?
            {
                Ok((expression, rest))
            } else {
                Err(ParseErr::new(format!(
                    "no such table or operation `{}`",
                    name
                )))
            }
        }
        ")" => Err(ParseErr::new("unexpected `)`".to_string())),
        _ => {
            let element = parse_atom(
                token,
                &registry.element_tables.name_to_constant,
                &metadata.name_to_element_variable,
                parameters,
            )?;
            Ok((element, rest))
        }
    }
}

fn parse_operation<'a>(
    name: &str,
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
) -> Result<Option<(ElementExpression, &'a [String])>, ParseErr> {
    let op = match name {
        "+" => Some(NumericOperator::Add),
        "-" => Some(NumericOperator::Subtract),
        "*" => Some(NumericOperator::Multiply),
        "/" => Some(NumericOperator::Divide),
        "min" => Some(NumericOperator::Min),
        "max" => Some(NumericOperator::Max),
        _ => None,
    };
    if let Some(op) = op {
        let (x, rest) = parse_expression(tokens, metadata, registry, parameters)?;
        let (y, rest) = parse_expression(rest, metadata, registry, parameters)?;
        let rest = util::parse_closing(rest)?;
        Ok(Some((
            ElementExpression::NumericOperation(op, Box::new(x), Box::new(y)),
            rest,
        )))
    } else {
        Ok(None)
    }
}

fn parse_atom(
    token: &str,
    name_to_constant: &FxHashMap<String, Element>,
    name_to_variable: &FxHashMap<String, usize>,
    parameters: &FxHashMap<String, usize>,
) -> Result<ElementExpression, ParseErr> {
    if let Some(v) = parameters.get(token) {
        Ok(ElementExpression::Constant(*v))
    } else if let Some(v) = name_to_constant.get(token) {
        Ok(ElementExpression::Constant(*v))
    } else if let Some(i) = name_to_variable.get(token) {
        Ok(ElementExpression::Variable(*i))
    } else {
        let v: Element = token.parse().map_err(|e| {
            ParseErr::new(format!("could not parse {} as a number: {:?}", token, e))
        })?;
        Ok(ElementExpression::Constant(v))
    }
}

fn parse_element_from_vector<'a>(
    name: &str,
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<Option<(ElementExpression, &'a [String])>, ParseErr> {
    match name {
        "last" => {
            let (vector, rest) = parse_vector_expression(tokens, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok(Some((ElementExpression::Last(Box::new(vector)), rest)))
        }
        "at" => {
            let (vector, rest) = parse_vector_expression(tokens, metadata, registry, parameters)?;
            let (i, rest) = parse_expression(rest, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok(Some((
                ElementExpression::At(Box::new(vector), Box::new(i)),
                rest,
            )))
        }
        _ => Ok(None),
    }
}

type TableExpressionResult<'a, T> = Option<(TableExpression<T>, &'a [String])>;

pub fn parse_table_expression<'a, T: Clone>(
    name: &str,
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, usize>,
    tables: &TableData<T>,
) -> Result<TableExpressionResult<'a, T>, ParseErr> {
    if let Some(i) = tables.name_to_table_1d.get(name) {
        let (x, rest) = parse_expression(tokens, metadata, registry, parameters)?;
        let rest = util::parse_closing(rest)?;
        Ok(Some((TableExpression::Table1D(*i, x), rest)))
    } else if let Some(i) = tables.name_to_table_2d.get(name) {
        let (x, rest) = parse_expression(tokens, metadata, registry, parameters)?;
        let (y, rest) = parse_expression(rest, metadata, registry, parameters)?;
        let rest = util::parse_closing(rest)?;
        Ok(Some((TableExpression::Table2D(*i, x, y), rest)))
    } else if let Some(i) = tables.name_to_table_3d.get(name) {
        let (x, rest) = parse_expression(tokens, metadata, registry, parameters)?;
        let (y, rest) = parse_expression(rest, metadata, registry, parameters)?;
        let (z, rest) = parse_expression(rest, metadata, registry, parameters)?;
        let rest = util::parse_closing(rest)?;
        Ok(Some((TableExpression::Table3D(*i, x, y, z), rest)))
    } else if let Some(i) = tables.name_to_table.get(name) {
        let result = parse_table(*i, tokens, metadata, registry, parameters)?;
        Ok(Some(result))
    } else {
        Ok(None)
    }
}

fn parse_table<'a, T: Clone>(
    i: usize,
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<(TableExpression<T>, &'a [String]), ParseErr> {
    let mut args = Vec::new();
    let mut xs = tokens;
    loop {
        let (next_token, rest) = xs
            .split_first()
            .ok_or_else(|| ParseErr::new("could not find closing `)`".to_string()))?;
        if next_token == ")" {
            return Ok((TableExpression::Table(i, args), rest));
        }
        let (expression, new_xs) = parse_expression(xs, metadata, registry, parameters)?;
        args.push(expression);
        xs = new_xs;
    }
}

pub fn parse_vector_expression<'a>(
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<(VectorExpression, &'a [String]), ParseErr> {
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
    match &token[..] {
        "(" => {
            let (name, rest) = rest
                .split_first()
                .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
            if let Some((expression, rest)) = parse_table_expression(
                name,
                rest,
                metadata,
                registry,
                parameters,
                &registry.vector_tables,
            )? {
                Ok((
                    VectorExpression::Reference(ReferenceExpression::Table(expression)),
                    rest,
                ))
            } else if let Some((expression, rest)) =
                parse_vector_operation(name, rest, metadata, registry, parameters)?
            {
                Ok((expression, rest))
            } else if name == "vector" {
                parse_vector_from(rest, metadata, registry, parameters)
            } else {
                Err(ParseErr::new(format!(
                    "no such table or operation `{}`",
                    name
                )))
            }
        }
        ")" => Err(ParseErr::new("unexpected `)`".to_string())),
        _ => {
            let set = parse_reference_atom(
                token,
                &registry.vector_tables.name_to_constant,
                &metadata.name_to_vector_variable,
            )?;
            Ok((VectorExpression::Reference(set), rest))
        }
    }
}

fn parse_vector_from<'a>(
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
) -> Result<(VectorExpression, &'a [String]), ParseErr> {
    if let Ok((set, rest)) = parse_set_expression(tokens, metadata, registry, parameters) {
        let rest = util::parse_closing(rest)?;
        Ok((VectorExpression::FromSet(Box::new(set)), rest))
    } else {
        let (vector, rest) = parse_element_vector(
            tokens,
            &registry.element_tables.name_to_constant,
            parameters,
        )?;
        Ok((
            VectorExpression::Reference(ReferenceExpression::Constant(vector)),
            rest,
        ))
    }
}

fn parse_vector_operation<'a, 'b, 'c>(
    name: &'a str,
    tokens: &'a [String],
    metadata: &'b StateMetadata,
    registry: &'b TableRegistry,
    parameters: &'c FxHashMap<String, usize>,
) -> Result<Option<(VectorExpression, &'a [String])>, ParseErr> {
    match name {
        "reverse" => {
            let (x, rest) = parse_vector_expression(tokens, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok(Some((VectorExpression::Reverse(Box::new(x)), rest)))
        }
        "indices" => {
            let (x, rest) = parse_vector_expression(tokens, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok(Some((VectorExpression::Indices(Box::new(x)), rest)))
        }
        "set" => {
            let (value, rest) = parse_expression(tokens, metadata, registry, parameters)?;
            let (vector, rest) = parse_vector_expression(rest, metadata, registry, parameters)?;
            let (i, rest) = parse_expression(rest, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok(Some((
                VectorExpression::Set(value, Box::new(vector), i),
                rest,
            )))
        }
        "push" => {
            let (x, rest) = parse_expression(tokens, metadata, registry, parameters)?;
            let (y, rest) = parse_vector_expression(rest, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok(Some((VectorExpression::Push(x, Box::new(y)), rest)))
        }
        "pop" => {
            let (x, rest) = parse_vector_expression(tokens, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok(Some((VectorExpression::Pop(Box::new(x)), rest)))
        }
        _ => Ok(None),
    }
}

pub fn parse_set_expression<'a, 'b, 'c>(
    tokens: &'a [String],
    metadata: &'b StateMetadata,
    registry: &'b TableRegistry,
    parameters: &'c FxHashMap<String, usize>,
) -> Result<(SetExpression, &'a [String]), ParseErr> {
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
    match &token[..] {
        "!" => {
            let (expression, rest) = parse_set_expression(rest, metadata, registry, parameters)?;
            Ok((SetExpression::Complement(Box::new(expression)), rest))
        }
        "(" => {
            let (name, rest) = rest
                .split_first()
                .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
            if let Some((expression, rest)) = parse_table_expression(
                name,
                rest,
                metadata,
                registry,
                parameters,
                &registry.set_tables,
            )? {
                Ok((
                    SetExpression::Reference(ReferenceExpression::Table(expression)),
                    rest,
                ))
            } else if let Some((expression, rest)) =
                parse_set_from(name, rest, metadata, registry, parameters)?
            {
                Ok((expression, rest))
            } else if let Some((expression, rest)) =
                parse_set_operation(name, rest, metadata, registry, parameters)?
            {
                Ok((expression, rest))
            } else {
                Err(ParseErr::new(format!(
                    "no such table, object, or operation `{}`",
                    name
                )))
            }
        }
        ")" => Err(ParseErr::new("unexpected `)`".to_string())),
        _ => {
            let set = parse_reference_atom(
                token,
                &registry.set_tables.name_to_constant,
                &metadata.name_to_set_variable,
            )?;
            Ok((SetExpression::Reference(set), rest))
        }
    }
}

fn parse_set_from<'a>(
    name: &str,
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
) -> Result<Option<(SetExpression, &'a [String])>, ParseErr> {
    if let Some(i) = metadata.name_to_object.get(name) {
        let capacity = metadata.object_numbers[*i];
        if let Ok((vector, rest)) = parse_vector_expression(tokens, metadata, registry, parameters)
        {
            let rest = util::parse_closing(rest)?;
            Ok(Some((
                SetExpression::FromVector(capacity, Box::new(vector)),
                rest,
            )))
        } else {
            let (vector, rest) = parse_element_vector(
                tokens,
                &registry.element_tables.name_to_constant,
                parameters,
            )?;
            let mut set = Set::with_capacity(capacity);
            vector.into_iter().for_each(|v| set.insert(v));
            Ok(Some((
                SetExpression::Reference(ReferenceExpression::Constant(set)),
                rest,
            )))
        }
    } else {
        Ok(None)
    }
}

fn parse_set_operation<'a>(
    name: &str,
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
) -> Result<Option<(SetExpression, &'a [String])>, ParseErr> {
    match name {
        "union" => {
            let (x, rest) = parse_set_expression(tokens, metadata, registry, parameters)?;
            let (y, rest) = parse_set_expression(rest, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok(Some((
                SetExpression::SetOperation(SetOperator::Union, Box::new(x), Box::new(y)),
                rest,
            )))
        }
        "difference" => {
            let (x, rest) = parse_set_expression(tokens, metadata, registry, parameters)?;
            let (y, rest) = parse_set_expression(rest, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok(Some((
                SetExpression::SetOperation(SetOperator::Difference, Box::new(x), Box::new(y)),
                rest,
            )))
        }
        "intersection" => {
            let (x, rest) = parse_set_expression(tokens, metadata, registry, parameters)?;
            let (y, rest) = parse_set_expression(rest, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok(Some((
                SetExpression::SetOperation(SetOperator::Intersection, Box::new(x), Box::new(y)),
                rest,
            )))
        }
        "add" => {
            let (x, rest) = parse_expression(tokens, metadata, registry, parameters)?;
            let (y, rest) = parse_set_expression(rest, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok(Some((
                SetExpression::SetElementOperation(SetElementOperator::Add, x, Box::new(y)),
                rest,
            )))
        }
        "remove" => {
            let (x, rest) = parse_expression(tokens, metadata, registry, parameters)?;
            let (y, rest) = parse_set_expression(rest, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok(Some((
                SetExpression::SetElementOperation(SetElementOperator::Remove, x, Box::new(y)),
                rest,
            )))
        }
        _ => Ok(None),
    }
}

fn parse_element_vector<'a>(
    tokens: &'a [String],
    name_to_constant: &FxHashMap<String, Element>,
    parameters: &FxHashMap<String, Element>,
) -> Result<(Vec<Element>, &'a [String]), ParseErr> {
    let mut result = Vec::new();
    let mut xs = tokens;
    loop {
        let (next_token, rest) = xs
            .split_first()
            .ok_or_else(|| ParseErr::new("could not find closing `)`".to_string()))?;
        if next_token == ")" {
            return Ok((result, rest));
        }
        let element = if let Some(v) = parameters.get(next_token) {
            *v
        } else if let Some(v) = name_to_constant.get(next_token) {
            *v
        } else {
            let v: Element = next_token.parse().map_err(|e| {
                ParseErr::new(format!(
                    "could not parse {} as a number: {:?}",
                    next_token, e
                ))
            })?;
            v
        };
        result.push(element);
        xs = rest;
    }
}

fn parse_reference_atom<T: Clone>(
    token: &str,
    name_to_constant: &FxHashMap<String, T>,
    name_to_variable: &FxHashMap<String, usize>,
) -> Result<ReferenceExpression<T>, ParseErr> {
    if let Some(v) = name_to_constant.get(token) {
        Ok(ReferenceExpression::Constant(v.clone()))
    } else if let Some(i) = name_to_variable.get(token) {
        Ok(ReferenceExpression::Variable(*i))
    } else {
        Err(ParseErr::new(format!(
            "no such constant or variable `{}`",
            token
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::*;

    fn generate_metadata() -> StateMetadata {
        let object_names = vec![String::from("something")];
        let object_numbers = vec![3];
        let mut name_to_object = FxHashMap::default();
        name_to_object.insert(String::from("something"), 0);

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
        let set_variable_to_object = vec![0, 0, 0, 0];

        let vector_variable_names = vec![
            String::from("v0"),
            String::from("v1"),
            String::from("v2"),
            String::from("v3"),
        ];
        let mut name_to_vector_variable = FxHashMap::default();
        name_to_vector_variable.insert(String::from("v0"), 0);
        name_to_vector_variable.insert(String::from("v1"), 1);
        name_to_vector_variable.insert(String::from("v2"), 2);
        name_to_vector_variable.insert(String::from("v3"), 3);
        let vector_variable_to_object = vec![0, 0, 0, 0];

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
        let element_variable_to_object = vec![0, 0, 0, 0];

        let integer_variable_names = vec![
            String::from("n0"),
            String::from("n1"),
            String::from("n2"),
            String::from("n3"),
        ];
        let mut name_to_integer_variable = FxHashMap::default();
        name_to_integer_variable.insert(String::from("n0"), 0);
        name_to_integer_variable.insert(String::from("n1"), 1);
        name_to_integer_variable.insert(String::from("n2"), 2);
        name_to_integer_variable.insert(String::from("n3"), 3);

        StateMetadata {
            object_names,
            name_to_object,
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
            ..Default::default()
        }
    }

    fn generate_registry() -> TableRegistry {
        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("et0"), 1);

        let tables_1d = vec![Table1D::new(vec![1, 0])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("et1"), 0);

        let tables_2d = vec![Table2D::new(vec![vec![1, 0]])];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("et2"), 0);

        let tables_3d = vec![Table3D::new(vec![vec![vec![1, 0]]])];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("et3"), 0);

        let mut map = FxHashMap::default();
        let key = vec![0, 0, 0, 0];
        map.insert(key, 1);
        let key = vec![0, 0, 0, 1];
        map.insert(key, 0);
        let tables = vec![Table::new(map, 0)];
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("et4"), 0);

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

        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("st0"), set.clone());
        let default = Set::with_capacity(3);
        let tables_1d = vec![Table1D::new(vec![set, default.clone(), default])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("st1"), 0);

        let set_tables = TableData {
            name_to_constant,
            tables_1d,
            name_to_table_1d,
            ..Default::default()
        };

        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("vt0"), vec![0, 1]);
        let tables_1d = vec![Table1D::new(vec![vec![0, 1]])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("vt1"), 0);

        let vector_tables = TableData {
            name_to_constant,
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

    fn generate_parameters() -> FxHashMap<String, usize> {
        let mut parameters = FxHashMap::default();
        parameters.insert(String::from("param"), 0);
        parameters
    }

    #[test]
    fn parse_expression_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["s1", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["v1", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "top", "v1", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_atom_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["et0", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, ElementExpression::Constant(1));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["e1", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, ElementExpression::Variable(1));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["11", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, ElementExpression::Constant(11));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["param", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, ElementExpression::Constant(0));
        assert_eq!(rest, &tokens[1..]);
    }

    #[test]
    fn parse_atom_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["e4", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_element_table_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "et1", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ElementExpression::Table(Box::new(TableExpression::Table1D(
                0,
                ElementExpression::Constant(0)
            )))
        );
        assert_eq!(rest, &tokens[4..]);
    }

    #[test]
    fn parse_operation_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "+", "0", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ElementExpression::NumericOperation(
                NumericOperator::Add,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "-", "0", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ElementExpression::NumericOperation(
                NumericOperator::Subtract,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "*", "0", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ElementExpression::NumericOperation(
                NumericOperator::Multiply,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "/", "0", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ElementExpression::NumericOperation(
                NumericOperator::Divide,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "max", "0", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ElementExpression::NumericOperation(
                NumericOperator::Max,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "min", "0", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ElementExpression::NumericOperation(
                NumericOperator::Min,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            )
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_operation_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "+", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "+", "0", "1", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_last_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "last", "v0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ElementExpression::Last(Box::new(VectorExpression::Reference(
                ReferenceExpression::Variable(0)
            )))
        );
        assert_eq!(rest, &tokens[4..]);
    }

    #[test]
    fn parse_at_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "at", "v0", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ElementExpression::At(
                Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                    0
                ))),
                Box::new(ElementExpression::Constant(1))
            )
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_table_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["0", ")", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_table_expression(
            "et5",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.element_tables,
        );
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_none());

        let tokens: Vec<String> = ["0", ")", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_table_expression(
            "et1",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.element_tables,
        );
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            TableExpression::Table1D(0, ElementExpression::Constant(0))
        );
        assert_eq!(rest, &tokens[2..]);

        let tokens: Vec<String> = ["0", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_table_expression(
            "et2",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.element_tables,
        );
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            TableExpression::Table2D(
                0,
                ElementExpression::Constant(0),
                ElementExpression::Variable(0)
            )
        );
        assert_eq!(rest, &tokens[3..]);

        let tokens: Vec<String> = ["0", "e0", "et0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_table_expression(
            "et3",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.element_tables,
        );
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            TableExpression::Table3D(
                0,
                ElementExpression::Constant(0),
                ElementExpression::Variable(0),
                ElementExpression::Constant(1),
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["0", "e0", "et0", "param", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_table_expression(
            "et4",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.element_tables,
        );
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            TableExpression::Table(
                0,
                vec![
                    ElementExpression::Constant(0),
                    ElementExpression::Variable(0),
                    ElementExpression::Constant(1),
                    ElementExpression::Constant(0),
                ]
            )
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_table_error() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["0"].iter().map(|x| x.to_string()).collect();
        let result = parse_table_expression(
            "et1",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.element_tables,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = [")"].iter().map(|x| x.to_string()).collect();
        let result = parse_table_expression(
            "et1",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.element_tables,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["0", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_table_expression(
            "et1",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.element_tables,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["0", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_table_expression(
            "et2",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.element_tables,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["0", "0", "0", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_table_expression(
            "et2",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.element_tables,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["0", "0", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_table_expression(
            "et3",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.element_tables,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["0", "0", "0", "0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_table_expression(
            "et3",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.element_tables,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_vector_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [")", "(", "vector", "0", "1", "et0", "param", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_vector_atom_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["v0", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            VectorExpression::Reference(ReferenceExpression::Variable(0))
        );
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["vt0", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1]))
        );
        assert_eq!(rest, &tokens[1..]);
    }

    #[test]
    fn parse_vector_atom_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["vv0", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_vector_reverse_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "reverse", "v0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            VectorExpression::Reverse(Box::new(VectorExpression::Reference(
                ReferenceExpression::Variable(0)
            )))
        );
        assert_eq!(rest, &tokens[4..]);
    }

    #[test]
    fn parse_vector_reverse_err_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "reverse", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_vector_indices_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "indices", "v0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            VectorExpression::Indices(Box::new(VectorExpression::Reference(
                ReferenceExpression::Variable(0)
            )))
        );
        assert_eq!(rest, &tokens[4..]);
    }

    #[test]
    fn parse_vector_indices_err_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "indices", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_vector_set_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "set", "0", "v0", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            VectorExpression::Set(
                ElementExpression::Constant(0),
                Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                    0
                ))),
                ElementExpression::Constant(0),
            )
        );
        assert_eq!(rest, &tokens[6..]);
    }

    #[test]
    fn parse_vector_set_err_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "set", "v0", "0", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_vector_push_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "push", "0", "v0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            VectorExpression::Push(
                ElementExpression::Constant(0),
                Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                    0
                ))),
            )
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_vector_push_err_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "push", "v0", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_vector_pop_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "pop", "v0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            VectorExpression::Pop(Box::new(VectorExpression::Reference(
                ReferenceExpression::Variable(0)
            )))
        );
        assert_eq!(rest, &tokens[4..]);
    }

    #[test]
    fn parse_vector_pop_err_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "pop", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_vector_constant_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "vector", "0", "1", "et0", "param", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1, 1, 0]))
        );
        assert_eq!(rest, &tokens[7..]);
    }

    #[test]
    fn parse_vector_constant_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "vector", "e0", "1", "et0", "param", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(", "vector", "(", "et1", "0", ")", "1", "et0", "param", ")", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_vector_table_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "vt1", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            VectorExpression::Reference(ReferenceExpression::Table(TableExpression::Table1D(
                0,
                ElementExpression::Variable(0)
            )))
        );
        assert_eq!(rest, &tokens[4..]);
    }

    #[test]
    fn parse_vector_table_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "st1", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_vector_from_set_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "vector", "s0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            VectorExpression::FromSet(Box::new(SetExpression::Reference(
                ReferenceExpression::Variable(0)
            )))
        );
        assert_eq!(rest, &tokens[4..]);
    }

    #[test]
    fn parse_vector_from_set_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "vector", "v0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_set_expression_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["e1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_set_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["11", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_set_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_set_atom_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["s1", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            SetExpression::Reference(ReferenceExpression::Variable(1))
        );
        assert_eq!(rest, &tokens[1..]);
    }

    #[test]
    fn parse_set_atom_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["e4", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_set_complemnt_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["!", "s2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            SetExpression::Complement(Box::new(SetExpression::Reference(
                ReferenceExpression::Variable(2)
            )))
        );
        assert_eq!(rest, &tokens[2..]);
    }

    #[test]
    fn parse_set_complenent_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["!", "e2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["!", "n2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn pare_set_operation_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "union", "s2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            SetExpression::SetOperation(
                SetOperator::Union,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(2))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1)))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "difference", "s2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            SetExpression::SetOperation(
                SetOperator::Difference,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(2))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1)))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "intersection", "s2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            SetExpression::SetOperation(
                SetOperator::Intersection,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(2))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1)))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "add", "e1", "s2", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            SetExpression::SetElementOperation(
                SetElementOperator::Add,
                ElementExpression::Variable(1),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(2))),
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "remove", "1", "s2", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            SetExpression::SetElementOperation(
                SetElementOperator::Remove,
                ElementExpression::Constant(1),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(2))),
            )
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn pare_set_operation_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "add", "n1", "s2", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "remove", "n1", "s2", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "intersection", "s2", "e1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "/", "s2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_set_constant_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "something", "param", "et0", "2", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        set.insert(2);
        assert_eq!(
            expression,
            SetExpression::Reference(ReferenceExpression::Constant(set))
        );
        assert_eq!(rest, &tokens[6..]);
    }

    #[test]
    fn parse_set_constant_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "soemthing", "param", "et0", "2", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "something", "param", "e0", "2", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_set_table_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "st1", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            SetExpression::Reference(ReferenceExpression::Table(TableExpression::Table1D(
                0,
                ElementExpression::Variable(0)
            )))
        );
        assert_eq!(rest, &tokens[4..]);
    }

    #[test]
    fn parse_set_table_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "vt1", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_set_from_set_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "something", "v0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            SetExpression::FromVector(
                3,
                Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                    0
                )))
            )
        );
        assert_eq!(rest, &tokens[4..]);
    }

    #[test]
    fn parse_set_from_vectorerr() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "somtehing", "v0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "something", "s0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }
}
