use super::argument_parser::{parse_argument, parse_multiple_arguments};
use super::condition_parser;
use super::util::ParseErr;
use super::util::{self, get_next_token_and_rest};
use dypdl::expression::{
    BinaryOperator, ElementExpression, ReferenceExpression, SetElementOperator, SetExpression,
    SetOperator, SetReduceExpression, SetReduceOperator, TableExpression, VectorExpression,
};
use dypdl::variable_type::{Element, Set};
use dypdl::{StateFunctions, StateMetadata, TableData, TableRegistry};
use rustc_hash::FxHashMap;

pub fn parse_expression<'a>(
    tokens: &'a [String],
    metadata: &StateMetadata,
    functions: &StateFunctions,
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
                functions,
                registry,
                parameters,
                &registry.element_tables,
            )? {
                Ok((ElementExpression::Table(Box::new(expression)), rest))
            } else if name == "if" {
                let (condition, rest) = condition_parser::parse_expression(
                    rest, metadata, functions, registry, parameters,
                )?;
                let (x, rest) = parse_expression(rest, metadata, functions, registry, parameters)?;
                let (y, rest) = parse_expression(rest, metadata, functions, registry, parameters)?;
                let rest = util::parse_closing(rest)?;
                Ok((
                    ElementExpression::If(Box::new(condition), Box::new(x), Box::new(y)),
                    rest,
                ))
            } else if let Some((expression, rest)) =
                parse_operation(name, rest, metadata, functions, registry, parameters)?
            {
                Ok((expression, rest))
            } else if let Some((expression, rest)) =
                parse_element_from_vector(name, rest, metadata, functions, registry, parameters)?
            {
                Ok((expression, rest))
            } else if let Some((expression, rest)) =
                parse_parameterized_state_function(name, rest, functions, parameters)?
            {
                Ok((expression, rest))
            } else {
                Err(ParseErr::new(format!(
                    "no such table, state function, or operation `{}`",
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
                &metadata.name_to_element_resource_variable,
                parameters,
                functions,
            )?;
            Ok((element, rest))
        }
    }
}

fn parse_operation<'a>(
    name: &str,
    tokens: &'a [String],
    metadata: &StateMetadata,
    functions: &StateFunctions,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
) -> Result<Option<(ElementExpression, &'a [String])>, ParseErr> {
    let op = match name {
        "+" => Some(BinaryOperator::Add),
        "-" => Some(BinaryOperator::Sub),
        "*" => Some(BinaryOperator::Mul),
        "/" => Some(BinaryOperator::Div),
        "min" => Some(BinaryOperator::Min),
        "max" => Some(BinaryOperator::Max),
        _ => None,
    };
    if let Some(op) = op {
        let (x, rest) = parse_expression(tokens, metadata, functions, registry, parameters)?;
        let (y, rest) = parse_expression(rest, metadata, functions, registry, parameters)?;
        let rest = util::parse_closing(rest)?;
        Ok(Some((
            ElementExpression::BinaryOperation(op, Box::new(x), Box::new(y)),
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
    name_to_resource_variable: &FxHashMap<String, usize>,
    parameters: &FxHashMap<String, usize>,
    functions: &StateFunctions,
) -> Result<ElementExpression, ParseErr> {
    if let Some(v) = parameters.get(token) {
        Ok(ElementExpression::Constant(*v))
    } else if let Some(v) = name_to_constant.get(token) {
        Ok(ElementExpression::Constant(*v))
    } else if let Some(i) = name_to_variable.get(token) {
        Ok(ElementExpression::Variable(*i))
    } else if let Some(i) = name_to_resource_variable.get(token) {
        Ok(ElementExpression::ResourceVariable(*i))
    } else if let Ok(expression) = functions.get_element_function(token) {
        Ok(expression)
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
    functions: &StateFunctions,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<Option<(ElementExpression, &'a [String])>, ParseErr> {
    match name {
        "last" => {
            let (vector, rest) =
                parse_vector_expression(tokens, metadata, functions, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok(Some((ElementExpression::Last(Box::new(vector)), rest)))
        }
        "at" => {
            let (vector, rest) =
                parse_vector_expression(tokens, metadata, functions, registry, parameters)?;
            let (i, rest) = parse_expression(rest, metadata, functions, registry, parameters)?;
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
    functions: &StateFunctions,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, usize>,
    tables: &TableData<T>,
) -> Result<TableExpressionResult<'a, T>, ParseErr> {
    if let Some(i) = tables.name_to_table_1d.get(name) {
        let (x, rest) = parse_expression(tokens, metadata, functions, registry, parameters)?;
        let rest = util::parse_closing(rest)?;
        Ok(Some((TableExpression::Table1D(*i, x), rest)))
    } else if let Some(i) = tables.name_to_table_2d.get(name) {
        let (x, rest) = parse_expression(tokens, metadata, functions, registry, parameters)?;
        let (y, rest) = parse_expression(rest, metadata, functions, registry, parameters)?;
        let rest = util::parse_closing(rest)?;
        Ok(Some((TableExpression::Table2D(*i, x, y), rest)))
    } else if let Some(i) = tables.name_to_table_3d.get(name) {
        let (x, rest) = parse_expression(tokens, metadata, functions, registry, parameters)?;
        let (y, rest) = parse_expression(rest, metadata, functions, registry, parameters)?;
        let (z, rest) = parse_expression(rest, metadata, functions, registry, parameters)?;
        let rest = util::parse_closing(rest)?;
        Ok(Some((TableExpression::Table3D(*i, x, y, z), rest)))
    } else if let Some(i) = tables.name_to_table.get(name) {
        let result = parse_table(*i, tokens, metadata, functions, registry, parameters)?;
        Ok(Some(result))
    } else {
        Ok(None)
    }
}

fn parse_table<'a, T: Clone>(
    i: usize,
    tokens: &'a [String],
    metadata: &StateMetadata,
    functions: &StateFunctions,
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
        let (expression, new_xs) = parse_expression(xs, metadata, functions, registry, parameters)?;
        args.push(expression);
        xs = new_xs;
    }
}

fn parse_parameterized_state_function<'a>(
    name: &str,
    tokens: &'a [String],
    functions: &StateFunctions,
    parameters: &FxHashMap<String, usize>,
) -> Result<Option<(ElementExpression, &'a [String])>, ParseErr> {
    let (name, rest) = util::parse_parameterized_state_function_name(name, tokens, parameters)?;

    functions
        .get_element_function(&name)
        .map(|expression| Ok(Some((expression, rest))))
        .unwrap_or_else(|_| Ok(None))
}

pub fn parse_vector_expression<'a>(
    tokens: &'a [String],
    metadata: &StateMetadata,
    functions: &StateFunctions,
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
                functions,
                registry,
                parameters,
                &registry.vector_tables,
            )? {
                Ok((
                    VectorExpression::Reference(ReferenceExpression::Table(expression)),
                    rest,
                ))
            } else if name == "if" {
                let (condition, rest) = condition_parser::parse_expression(
                    rest, metadata, functions, registry, parameters,
                )?;
                let (x, rest) =
                    parse_vector_expression(rest, metadata, functions, registry, parameters)?;
                let (y, rest) =
                    parse_vector_expression(rest, metadata, functions, registry, parameters)?;
                let rest = util::parse_closing(rest)?;
                Ok((
                    VectorExpression::If(Box::new(condition), Box::new(x), Box::new(y)),
                    rest,
                ))
            } else if let Some((expression, rest)) =
                parse_vector_operation(name, rest, metadata, functions, registry, parameters)?
            {
                Ok((expression, rest))
            } else if name == "vector" {
                parse_vector_from(rest, metadata, functions, registry, parameters)
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
    functions: &StateFunctions,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
) -> Result<(VectorExpression, &'a [String]), ParseErr> {
    if let Ok((set, rest)) = parse_set_expression(tokens, metadata, functions, registry, parameters)
    {
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

fn parse_vector_operation<'a, 'b>(
    name: &'a str,
    tokens: &'a [String],
    metadata: &'b StateMetadata,
    functions: &'b StateFunctions,
    registry: &'b TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<Option<(VectorExpression, &'a [String])>, ParseErr> {
    match name {
        "reverse" => {
            let (x, rest) =
                parse_vector_expression(tokens, metadata, functions, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok(Some((VectorExpression::Reverse(Box::new(x)), rest)))
        }
        "indices" => {
            let (x, rest) =
                parse_vector_expression(tokens, metadata, functions, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok(Some((VectorExpression::Indices(Box::new(x)), rest)))
        }
        "set" => {
            let (value, rest) =
                parse_expression(tokens, metadata, functions, registry, parameters)?;
            let (vector, rest) =
                parse_vector_expression(rest, metadata, functions, registry, parameters)?;
            let (i, rest) = parse_expression(rest, metadata, functions, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok(Some((
                VectorExpression::Set(value, Box::new(vector), i),
                rest,
            )))
        }
        "push" => {
            let (x, rest) = parse_expression(tokens, metadata, functions, registry, parameters)?;
            let (y, rest) =
                parse_vector_expression(rest, metadata, functions, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok(Some((VectorExpression::Push(x, Box::new(y)), rest)))
        }
        "pop" => {
            let (x, rest) =
                parse_vector_expression(tokens, metadata, functions, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok(Some((VectorExpression::Pop(Box::new(x)), rest)))
        }
        _ => Ok(None),
    }
}

fn parse_set_reduce_expression<'a, 'b>(
    op_name: &str,
    tokens: &'a [String],
    metadata: &'b StateMetadata,
    functions: &'b StateFunctions,
    registry: &'b TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<Option<(SetExpression, &'a [String])>, ParseErr> {
    let op = match op_name {
        "union" => SetReduceOperator::Union,
        "intersection" => SetReduceOperator::Intersection,
        "disjunctive_union" => SetReduceOperator::SymmetricDifference,
        _ => return Ok(None),
    };
    let (table_name, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
    if let Some(i) = registry.set_tables.name_to_table_1d.get(table_name) {
        let (x, rest) = parse_argument(rest, metadata, functions, registry, parameters)?;
        let rest = util::parse_closing(rest)?;
        let capacity = registry.set_tables.tables_1d[*i].capacity_of_set();
        Ok(Some((
            SetExpression::Reduce(SetReduceExpression::Table1D(op, capacity, *i, Box::new(x))),
            rest,
        )))
    } else if let Some(i) = registry.set_tables.name_to_table_2d.get(table_name) {
        let (x, rest) = parse_argument(rest, metadata, functions, registry, parameters)?;
        let (y, rest) = parse_argument(rest, metadata, functions, registry, parameters)?;
        let rest = util::parse_closing(rest)?;
        let capacity = registry.set_tables.tables_2d[*i].capacity_of_set();
        Ok(Some((
            SetExpression::Reduce(SetReduceExpression::Table2D(
                op,
                capacity,
                *i,
                Box::new(x),
                Box::new(y),
            )),
            rest,
        )))
    } else if let Some(i) = registry.set_tables.name_to_table_3d.get(table_name) {
        let (x, rest) = parse_argument(rest, metadata, functions, registry, parameters)?;
        let (y, rest) = parse_argument(rest, metadata, functions, registry, parameters)?;
        let (z, rest) = parse_argument(rest, metadata, functions, registry, parameters)?;
        let rest = util::parse_closing(rest)?;
        let capacity = registry.set_tables.tables_3d[*i].capacity_of_set();
        Ok(Some((
            SetExpression::Reduce(SetReduceExpression::Table3D(
                op,
                capacity,
                *i,
                Box::new(x),
                Box::new(y),
                Box::new(z),
            )),
            rest,
        )))
    } else if let Some(i) = registry.set_tables.name_to_table.get(table_name) {
        let (args, rest) =
            parse_multiple_arguments(rest, metadata, functions, registry, parameters)?;
        let capacity = registry.set_tables.tables[*i].capacity_of_set();
        Ok(Some((
            SetExpression::Reduce(SetReduceExpression::Table(op, capacity, *i, args)),
            rest,
        )))
    } else {
        Ok(None)
    }
}

pub fn parse_set_expression<'a, 'b>(
    tokens: &'a [String],
    metadata: &'b StateMetadata,
    functions: &'b StateFunctions,
    registry: &'b TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<(SetExpression, &'a [String]), ParseErr> {
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
    match &token[..] {
        "~" => {
            let (expression, rest) =
                parse_set_expression(rest, metadata, functions, registry, parameters)?;
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
                functions,
                registry,
                parameters,
                &registry.set_tables,
            )? {
                Ok((
                    SetExpression::Reference(ReferenceExpression::Table(expression)),
                    rest,
                ))
            } else if let Some((expression, rest)) =
                parse_set_reduce_expression(name, rest, metadata, functions, registry, parameters)?
            {
                Ok((expression, rest))
            } else if let Some((expression, rest)) =
                parse_set_from(name, rest, metadata, functions, registry, parameters)?
            {
                Ok((expression, rest))
            } else if name == "if" {
                let (condition, rest) = condition_parser::parse_expression(
                    rest, metadata, functions, registry, parameters,
                )?;
                let (x, rest) =
                    parse_set_expression(rest, metadata, functions, registry, parameters)?;
                let (y, rest) =
                    parse_set_expression(rest, metadata, functions, registry, parameters)?;
                let rest = util::parse_closing(rest)?;
                Ok((
                    SetExpression::If(Box::new(condition), Box::new(x), Box::new(y)),
                    rest,
                ))
            } else if let Some((expression, rest)) =
                parse_set_operation(name, rest, metadata, functions, registry, parameters)?
            {
                Ok((expression, rest))
            } else if let Some((expression, rest)) =
                parse_parameterized_set_state_function(name, rest, functions, parameters)?
            {
                Ok((expression, rest))
            } else {
                Err(ParseErr::new(format!(
                    "no such table, state function, object, or operation `{}`",
                    name
                )))
            }
        }
        // parse the set constants, where a set of 1, 2, and 3 with a total size of 10 is represented by
        // the string '{1 2 3 : 10}'
        "{" => {
            let (mut item, mut rest) = get_next_token_and_rest(rest)?;
            let mut all_elements = Vec::<usize>::new();
            while item != ":" {
                if let Ok(value) = item.parse::<usize>() {
                    all_elements.push(value);
                } else {
                    return Err(ParseErr::new(
                        "could not parse the element token in set constant".to_string(),
                    ));
                }
                (item, rest) = get_next_token_and_rest(rest)?;
            }

            (item, rest) = get_next_token_and_rest(rest)?;

            if let Ok(size) = item.parse::<usize>() {
                let (closing, rest) = get_next_token_and_rest(rest)?;
                if closing == "}" {
                    let mut set = Set::with_capacity(size);
                    for element in &all_elements {
                        if *element >= size {
                            return Err(ParseErr::new("set element out of range".to_string()));
                        }
                    }
                    set.extend(all_elements);
                    Ok((
                        SetExpression::Reference(ReferenceExpression::Constant(set)),
                        rest,
                    ))
                } else {
                    Err(ParseErr::new(
                        "wrong closing symbol in set constant".to_string(),
                    ))
                }
            } else {
                Err(ParseErr::new(
                    "could not parse the size token in set constant".to_string(),
                ))
            }
        }
        ")" => Err(ParseErr::new("unexpected `)`".to_string())),
        _ => {
            if let Ok(set) = functions.get_set_function(token) {
                Ok((set, rest))
            } else {
                let set = parse_reference_atom(
                    token,
                    &registry.set_tables.name_to_constant,
                    &metadata.name_to_set_variable,
                )?;
                Ok((SetExpression::Reference(set), rest))
            }
        }
    }
}

fn parse_set_from<'a>(
    name: &str,
    tokens: &'a [String],
    metadata: &StateMetadata,
    functions: &StateFunctions,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
) -> Result<Option<(SetExpression, &'a [String])>, ParseErr> {
    if let Some(i) = metadata.name_to_object_type.get(name) {
        let capacity = metadata.object_numbers[*i];
        if let Ok((vector, rest)) =
            parse_vector_expression(tokens, metadata, functions, registry, parameters)
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

pub fn parse_parameterized_set_state_function<'a>(
    name: &str,
    tokens: &'a [String],
    functions: &StateFunctions,
    parameters: &FxHashMap<String, usize>,
) -> Result<Option<(SetExpression, &'a [String])>, ParseErr> {
    let (name, rest) = util::parse_parameterized_state_function_name(name, tokens, parameters)?;

    functions
        .get_set_function(&name)
        .map(|expression| Ok(Some((expression, rest))))
        .unwrap_or_else(|_| Ok(None))
}

fn parse_set_operation<'a>(
    name: &str,
    tokens: &'a [String],
    metadata: &StateMetadata,
    functions: &StateFunctions,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
) -> Result<Option<(SetExpression, &'a [String])>, ParseErr> {
    match name {
        "union" => {
            let (x, rest) =
                parse_set_expression(tokens, metadata, functions, registry, parameters)?;
            let (y, rest) = parse_set_expression(rest, metadata, functions, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok(Some((
                SetExpression::SetOperation(SetOperator::Union, Box::new(x), Box::new(y)),
                rest,
            )))
        }
        "difference" => {
            let (x, rest) =
                parse_set_expression(tokens, metadata, functions, registry, parameters)?;
            let (y, rest) = parse_set_expression(rest, metadata, functions, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok(Some((
                SetExpression::SetOperation(SetOperator::Difference, Box::new(x), Box::new(y)),
                rest,
            )))
        }
        "intersection" => {
            let (x, rest) =
                parse_set_expression(tokens, metadata, functions, registry, parameters)?;
            let (y, rest) = parse_set_expression(rest, metadata, functions, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok(Some((
                SetExpression::SetOperation(SetOperator::Intersection, Box::new(x), Box::new(y)),
                rest,
            )))
        }
        "add" => {
            let (x, rest) = parse_expression(tokens, metadata, functions, registry, parameters)?;
            let (y, rest) = parse_set_expression(rest, metadata, functions, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok(Some((
                SetExpression::SetElementOperation(SetElementOperator::Add, x, Box::new(y)),
                rest,
            )))
        }
        "remove" => {
            let (x, rest) = parse_expression(tokens, metadata, functions, registry, parameters)?;
            let (y, rest) = parse_set_expression(rest, metadata, functions, registry, parameters)?;
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
    use dypdl::expression::*;

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
        let element_resource_variable_to_object = vec![0, 0, 0, 0];

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
            element_resource_variable_names,
            name_to_element_resource_variable,
            element_resource_variable_to_object,
            element_less_is_better: vec![false, false, true, false],
            ..Default::default()
        }
    }

    fn generate_registry() -> TableRegistry {
        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("et0"), 1);

        let tables_1d = vec![dypdl::Table1D::new(vec![1, 0])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("et1"), 0);

        let tables_2d = vec![dypdl::Table2D::new(vec![vec![1, 0]])];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("et2"), 0);

        let tables_3d = vec![dypdl::Table3D::new(vec![vec![vec![1, 0]]])];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("et3"), 0);

        let mut map = FxHashMap::default();
        let key = vec![0, 0, 0, 0];
        map.insert(key, 1);
        let key = vec![0, 0, 0, 1];
        map.insert(key, 0);
        let tables = vec![dypdl::Table::new(map, 0)];
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
        let tables_1d = vec![dypdl::Table1D::new(vec![
            set.clone(),
            default.clone(),
            default.clone(),
        ])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("st1"), 0);
        let tables_2d = vec![dypdl::Table2D::new(vec![vec![
            set.clone(),
            default.clone(),
            default.clone(),
        ]])];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("st2"), 0);
        let tables_3d = vec![dypdl::Table3D::new(vec![vec![vec![
            set.clone(),
            default.clone(),
            default,
        ]]])];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("st3"), 0);
        let tables = vec![dypdl::Table::new(
            FxHashMap::default(),
            Set::with_capacity(3),
        )];
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("st4"), 0);

        let set_tables = TableData {
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

        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("vt0"), vec![0, 1]);
        let tables_1d = vec![dypdl::Table1D::new(vec![vec![0, 1]])];
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
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["s1", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["v1", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "top", "v1", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_atom_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["et0", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, ElementExpression::Constant(1));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["e1", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, ElementExpression::Variable(1));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["er1", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, ElementExpression::ResourceVariable(1));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["11", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, ElementExpression::Constant(11));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["param", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, ElementExpression::Constant(0));
        assert_eq!(rest, &tokens[1..]);
    }

    #[test]
    fn parse_atom_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["e4", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_element_state_function_ok() {
        let metadata = StateMetadata::default();
        let registry = TableRegistry::default();
        let parameters = FxHashMap::default();

        let mut functions = StateFunctions::default();
        let result = functions.add_element_function("sf", ElementExpression::Constant(0));
        assert!(result.is_ok());
        let expected = result.unwrap();

        let tokens: Vec<String> = ["sf", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, expected);
        assert_eq!(rest, &tokens[1..]);
    }

    #[test]
    fn parse_parameterized_element_state_function_ok() {
        let metadata = StateMetadata::default();
        let registry = TableRegistry::default();
        let parameters = FxHashMap::from_iter(vec![("a".to_string(), 0), ("b".to_string(), 2)]);

        let mut functions = StateFunctions::default();
        let result = functions.add_element_function("sf_0_1_2", ElementExpression::Constant(0));
        assert!(result.is_ok());
        let expected = result.unwrap();

        let tokens: Vec<_> = ["(", "sf", "a", "1", "b", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, expected);
        assert_eq!(rest, &tokens[6..]);
    }

    #[test]
    fn parse_parameterized_element_state_function_non_exist_err() {
        let metadata = StateMetadata::default();
        let registry = TableRegistry::default();
        let parameters = FxHashMap::from_iter(vec![("a".to_string(), 0), ("b".to_string(), 2)]);

        let functions = StateFunctions::default();

        let tokens: Vec<_> = ["(", "sf", "a", "1", "b", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_parameterized_element_state_function_no_closing_err() {
        let metadata = StateMetadata::default();
        let registry = TableRegistry::default();
        let parameters = FxHashMap::from_iter(vec![("a".to_string(), 0), ("b".to_string(), 2)]);

        let mut functions = StateFunctions::default();
        let result = functions.add_element_function("sf_0_1_2", ElementExpression::Constant(0));
        assert!(result.is_ok());

        let tokens: Vec<_> = ["(", "sf", "a", "1", "b"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_element_table_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "et1", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
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
    fn parse_if_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "if", "true", "0", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ElementExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            )
        );
        assert_eq!(rest, &tokens[6..]);
    }

    #[test]
    fn parse_if_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "if", "0", "0", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "if", "0", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "if", "true", "true", "0", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_operation_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "+", "0", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ElementExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "-", "0", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ElementExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "*", "0", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ElementExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "/", "0", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ElementExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "max", "0", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ElementExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "min", "0", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ElementExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ElementExpression::Constant(0)),
                Box::new(ElementExpression::Constant(1))
            )
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_operation_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "+", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "+", "0", "1", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_last_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "last", "v0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
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
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "at", "v0", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
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
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["0", ")", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_table_expression(
            "et5",
            &tokens,
            &metadata,
            &functions,
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
            &functions,
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
            &functions,
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
            &functions,
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
            &functions,
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
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["0"].iter().map(|x| x.to_string()).collect();
        let result = parse_table_expression(
            "et1",
            &tokens,
            &metadata,
            &functions,
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
            &functions,
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
            &functions,
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
            &functions,
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
            &functions,
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
            &functions,
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
            &functions,
            &registry,
            &parameters,
            &registry.element_tables,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_vector_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [")", "(", "vector", "0", "1", "et0", "param", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_vector_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_vector_atom_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["v0", ")"].iter().map(|x| x.to_string()).collect();
        let result =
            parse_vector_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            VectorExpression::Reference(ReferenceExpression::Variable(0))
        );
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["vt0", ")"].iter().map(|x| x.to_string()).collect();
        let result =
            parse_vector_expression(&tokens, &metadata, &functions, &registry, &parameters);
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
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["vv0", ")"].iter().map(|x| x.to_string()).collect();
        let result =
            parse_vector_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_vector_if_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "if", "true", "v0", "v1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_vector_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            VectorExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                    0
                ))),
                Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                    1
                ))),
            )
        );
        assert_eq!(rest, &tokens[6..]);
    }

    #[test]
    fn parse_vector_if_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "if", "v0", "v1", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_vector_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "if", "v0", "v1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_vector_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "if", "true", "true", "v0", "v1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_vector_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_vector_reverse_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "reverse", "v0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_vector_expression(&tokens, &metadata, &functions, &registry, &parameters);
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
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "reverse", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_vector_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_vector_indices_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "indices", "v0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_vector_expression(&tokens, &metadata, &functions, &registry, &parameters);
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
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "indices", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_vector_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_vector_set_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "set", "0", "v0", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_vector_expression(&tokens, &metadata, &functions, &registry, &parameters);
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
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "set", "v0", "0", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_vector_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_vector_push_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "push", "0", "v0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_vector_expression(&tokens, &metadata, &functions, &registry, &parameters);
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
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "push", "v0", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_vector_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_vector_pop_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "pop", "v0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_vector_expression(&tokens, &metadata, &functions, &registry, &parameters);
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
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "pop", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_vector_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_vector_constant_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "vector", "0", "1", "et0", "param", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_vector_expression(&tokens, &metadata, &functions, &registry, &parameters);
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
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "vector", "e0", "1", "et0", "param", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_vector_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(", "vector", "(", "et1", "0", ")", "1", "et0", "param", ")", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result =
            parse_vector_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_vector_table_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "vt1", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_vector_expression(&tokens, &metadata, &functions, &registry, &parameters);
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
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "st1", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_vector_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_vector_from_set_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "vector", "s0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_vector_expression(&tokens, &metadata, &functions, &registry, &parameters);
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
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "vector", "v0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_vector_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_set_expression_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["e1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["11", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_set_atom_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["s1", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
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
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["e4", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_set_function_ok() {
        let registry = TableRegistry::default();
        let parameters = FxHashMap::default();

        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 4);
        assert!(result.is_ok());
        let object = result.unwrap();
        let result = metadata.add_set_variable("v", object);
        assert!(result.is_ok());
        let v = result.unwrap();

        let mut functions = StateFunctions::default();
        let result = functions.add_set_function("f", v.add(1));
        assert!(result.is_ok());
        let expected = result.unwrap();

        let tokens: Vec<String> = ["f", ")", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, expected);
        assert_eq!(rest, &tokens[1..]);
    }

    #[test]
    fn parse_parameterized_set_state_function_ok() {
        let registry = TableRegistry::default();
        let parameters = FxHashMap::from_iter(vec![("a".to_string(), 0), ("b".to_string(), 2)]);

        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 4);
        assert!(result.is_ok());
        let object = result.unwrap();
        let result = metadata.add_set_variable("v", object);
        assert!(result.is_ok());
        let v = result.unwrap();

        let mut functions = StateFunctions::default();
        let result = functions.add_set_function("f_0_1_2", v.add(1));
        assert!(result.is_ok());
        let expected = result.unwrap();

        let tokens: Vec<_> = ["(", "f", "a", "1", "b", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, expected);
        assert_eq!(rest, &tokens[6..]);
    }

    #[test]
    fn parse_parameterized_set_state_function_non_exist_err() {
        let metadata = StateMetadata::default();
        let registry = TableRegistry::default();
        let parameters = FxHashMap::from_iter(vec![("a".to_string(), 0), ("b".to_string(), 2)]);

        let functions = StateFunctions::default();

        let tokens: Vec<_> = ["(", "f", "a", "1", "b", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_parameterized_set_state_function_no_closing_err() {
        let registry = TableRegistry::default();
        let parameters = FxHashMap::from_iter(vec![("a".to_string(), 0), ("b".to_string(), 2)]);

        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 4);
        assert!(result.is_ok());
        let object = result.unwrap();
        let result = metadata.add_set_variable("v", object);
        assert!(result.is_ok());
        let v = result.unwrap();

        let mut functions = StateFunctions::default();
        let result = functions.add_set_function("f_0_1_2", v.add(1));
        assert!(result.is_ok());

        let tokens: Vec<_> = ["(", "f", "a", "1", "b"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_set_complement_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["~", "s2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
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
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["~", "e2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["~", "n2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_set_if_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "if", "true", "s0", "s1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            SetExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
            )
        );
        assert_eq!(rest, &tokens[6..]);
    }

    #[test]
    fn parse_set_if_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "if", "s0", "s1", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "if", "s0", "s1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "if", "true", "true", "s0", "s1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn pare_set_operation_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "union", "s2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
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
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
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
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
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
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
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
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
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
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "add", "n1", "s2", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "remove", "n1", "s2", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "intersection", "s2", "e1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "/", "s2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_set_constant_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "something", "param", "et0", "2", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
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
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "soemthing", "param", "et0", "2", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "something", "param", "e0", "2", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_set_table_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "st1", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
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
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "vt1", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_set_reduce_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "union"].iter().map(|x| x.to_string()).collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_set_reduce_union_1d_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "union", "st1", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            SetExpression::Reduce(SetReduceExpression::Table1D(
                SetReduceOperator::Union,
                3,
                0,
                Box::new(ArgumentExpression::Element(ElementExpression::Variable(0)))
            ))
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_set_reduce_intersection_1d_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "intersection", "st1", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            SetExpression::Reduce(SetReduceExpression::Table1D(
                SetReduceOperator::Intersection,
                3,
                0,
                Box::new(ArgumentExpression::Element(ElementExpression::Variable(0)))
            ))
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_set_reduce_disjunctive_union_1d_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "disjunctive_union", "st1", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            SetExpression::Reduce(SetReduceExpression::Table1D(
                SetReduceOperator::SymmetricDifference,
                3,
                0,
                Box::new(ArgumentExpression::Element(ElementExpression::Variable(0)))
            ))
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_set_reduce_1d_x_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "disjunctive_union", "st1", "e4", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_set_reduce_1d_no_closing_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "disjunctive_union", "st1", "e0", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_set_reduce_union_2d_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "union", "st2", "e0", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            SetExpression::Reduce(SetReduceExpression::Table2D(
                SetReduceOperator::Union,
                3,
                0,
                Box::new(ArgumentExpression::Element(ElementExpression::Variable(0))),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
            ))
        );
        assert_eq!(rest, &tokens[6..]);
    }

    #[test]
    fn parse_set_reduce_intersection_2d_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "intersection", "st2", "e0", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            SetExpression::Reduce(SetReduceExpression::Table2D(
                SetReduceOperator::Intersection,
                3,
                0,
                Box::new(ArgumentExpression::Element(ElementExpression::Variable(0))),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
            ))
        );
        assert_eq!(rest, &tokens[6..]);
    }

    #[test]
    fn parse_set_reduce_disjunctive_union_2d_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "disjunctive_union", "st2", "e0", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            SetExpression::Reduce(SetReduceExpression::Table2D(
                SetReduceOperator::SymmetricDifference,
                3,
                0,
                Box::new(ArgumentExpression::Element(ElementExpression::Variable(0))),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
            ))
        );
        assert_eq!(rest, &tokens[6..]);
    }

    #[test]
    fn parse_set_reduce_2d_x_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "disjunctive_union", "st2", "e4", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_set_reduce_2d_y_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "disjunctive_union", "st2", "e0", "e4", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_set_reduce_2d_no_closing_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "disjunctive_union", "st1", "e0", "0", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_set_reduce_union_3d_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "union", "st3", "e0", "0", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            SetExpression::Reduce(SetReduceExpression::Table3D(
                SetReduceOperator::Union,
                3,
                0,
                Box::new(ArgumentExpression::Element(ElementExpression::Variable(0))),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
            ))
        );
        assert_eq!(rest, &tokens[7..]);
    }

    #[test]
    fn parse_set_reduce_intersection_3d_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "intersection", "st3", "e0", "0", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            SetExpression::Reduce(SetReduceExpression::Table3D(
                SetReduceOperator::Intersection,
                3,
                0,
                Box::new(ArgumentExpression::Element(ElementExpression::Variable(0))),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
            ))
        );
        assert_eq!(rest, &tokens[7..]);
    }

    #[test]
    fn parse_set_reduce_disjunctive_union_3d_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "disjunctive_union", "st3", "e0", "0", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            SetExpression::Reduce(SetReduceExpression::Table3D(
                SetReduceOperator::SymmetricDifference,
                3,
                0,
                Box::new(ArgumentExpression::Element(ElementExpression::Variable(0))),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
                Box::new(ArgumentExpression::Element(ElementExpression::Constant(0))),
            ))
        );
        assert_eq!(rest, &tokens[7..]);
    }

    #[test]
    fn parse_set_reduce_3d_x_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "disjunctive_union", "st3", "e4", "0", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_set_reduce_3d_y_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "disjunctive_union", "st3", "e0", "e4", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_set_reduce_2d_z_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "disjunctive_union", "st3", "e0", "0", "e4", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_set_reduce_3d_no_closing_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "disjunctive_union",
            "st1",
            "e0",
            "0",
            "0",
            "0",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_set_reduce_union_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "union", "st4", "e0", "0", "0", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            SetExpression::Reduce(SetReduceExpression::Table(
                SetReduceOperator::Union,
                3,
                0,
                vec![
                    ArgumentExpression::Element(ElementExpression::Variable(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                ]
            ))
        );
        assert_eq!(rest, &tokens[8..]);
    }

    #[test]
    fn parse_set_reduce_intersection_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "intersection", "st4", "e0", "0", "0", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            SetExpression::Reduce(SetReduceExpression::Table(
                SetReduceOperator::Intersection,
                3,
                0,
                vec![
                    ArgumentExpression::Element(ElementExpression::Variable(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                ]
            ))
        );
        assert_eq!(rest, &tokens[8..]);
    }

    #[test]
    fn parse_set_reduce_disjunctive_union_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "disjunctive_union",
            "st4",
            "e0",
            "0",
            "0",
            "0",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            SetExpression::Reduce(SetReduceExpression::Table(
                SetReduceOperator::SymmetricDifference,
                3,
                0,
                vec![
                    ArgumentExpression::Element(ElementExpression::Variable(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                ]
            ))
        );
        assert_eq!(rest, &tokens[8..]);
    }

    #[test]
    fn parse_set_reduce_indices_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "disjunctive_union",
            "st4",
            "e4",
            "0",
            "0",
            "0",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_set_from_set_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "something", "v0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
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
    fn parse_set_from_vector_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "somtehing", "v0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "something", "s0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_set_from_constant_ok() {
        let metadata = StateMetadata::default();
        let functions = StateFunctions::default();
        let registry = TableRegistry::default();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["{", "1", "2", ":", "6", "}", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        let mut set = Set::with_capacity(6);
        set.extend(1..3);
        assert_eq!(
            expression,
            SetExpression::Reference(ReferenceExpression::Constant(set))
        );
        assert_eq!(rest, &tokens[6..]);
    }

    #[test]
    fn parse_set_from_constant_err() {
        let metadata = StateMetadata::default();
        let functions = StateFunctions::default();
        let registry = TableRegistry::default();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["{", "1", "2", "6", "}", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["{", "1", "2", ":", "6", "10", "}", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["{", "1", "2", ":", "6", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["{", "1", "6", ":", "6", "}"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }
}
