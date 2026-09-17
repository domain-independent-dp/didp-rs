use super::condition_parser;
use super::element_parser;
use super::integer_parser;
use super::numeric_table_parser;
use super::util;
use super::util::{ModelData, ParseErr};
use dypdl::expression::{
    BinaryOperator, CastOperator, Condition, ContinuousBinaryOperator, ContinuousExpression,
    ContinuousUnaryOperator, IntegerExpression, ReduceOperator, SetExpression, UnaryOperator,
};
use dypdl::variable_type::Continuous;
#[cfg(test)]
use dypdl::LocalVariableData;
use dypdl::{StateFunctions, StateMetadata, TableRegistry};
use rustc_hash::FxHashMap;
use std::str;

type FractionalKnapsackItemExpressions<'a> = (
    Vec<(usize, ContinuousExpression, ContinuousExpression)>,
    &'a [String],
);

pub fn parse_expression<'a>(
    tokens: &'a [String],
    model_data: &mut ModelData,
    local_variables: &FxHashMap<String, usize>,
) -> Result<(ContinuousExpression, &'a [String]), ParseErr> {
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::new(String::from("could not get token")))?;
    match &token[..] {
        "(" => {
            let (name, rest) = rest
                .split_first()
                .ok_or_else(|| ParseErr::new(String::from("could not get token")))?;
            if let Some((expression, rest)) = numeric_table_parser::parse_expression(
                name,
                rest,
                model_data,
                local_variables,
                &model_data.registry.continuous_tables,
            )? {
                Ok((ContinuousExpression::Table(Box::new(expression)), rest))
            } else if let Some((expression, rest)) = numeric_table_parser::parse_expression(
                name,
                rest,
                model_data,
                local_variables,
                &model_data.registry.integer_tables,
            )? {
                Ok((
                    ContinuousExpression::FromInteger(Box::new(IntegerExpression::Table(
                        Box::new(expression),
                    ))),
                    rest,
                ))
            } else if name == "fractional_knapsack" {
                parse_fractional_knapsack(rest, model_data, local_variables)
            } else if name == "minimum_spanning_tree" {
                parse_minimum_spanning_tree(rest, model_data, local_variables)
            } else if name == "if" {
                let (condition, rest) =
                    condition_parser::parse_expression(rest, model_data, local_variables)?;
                let (x, rest) = parse_expression(rest, model_data, local_variables)?;
                let (y, rest) = parse_expression(rest, model_data, local_variables)?;
                let rest = util::parse_closing(rest)?;
                Ok((
                    ContinuousExpression::If(Box::new(condition), Box::new(x), Box::new(y)),
                    rest,
                ))
            } else if name == "reduce" {
                parse_reduce(rest, model_data, local_variables)
            } else if name == "continuous" {
                parse_from_integer(rest, model_data, local_variables)
            } else if let Ok(result) = util::try_parse(model_data, |model_data| {
                parse_round(name, rest, model_data, local_variables)
            }) {
                Ok(result)
            } else if let Some((expression, rest)) = parse_parameterized_state_function(
                name,
                rest,
                model_data.functions,
                model_data.parameters,
            )? {
                Ok((expression, rest))
            } else {
                let (x, rest) = parse_expression(rest, model_data, local_variables)?;
                let (expression, rest) =
                    if let Ok(expression) = parse_unary_operation(name, x.clone()) {
                        (expression, rest)
                    } else {
                        let (y, rest) = parse_expression(rest, model_data, local_variables)?;
                        (parse_binary_operation(name, x, y)?, rest)
                    };
                let rest = util::parse_closing(rest)?;
                Ok((expression, rest))
            }
        }
        ")" => Err(ParseErr::new("unexpected `)`".to_string())),
        "|" => parse_cardinality(rest, model_data, local_variables),
        _ => {
            let expression = parse_continuous_atom(
                token,
                model_data.metadata,
                model_data.functions,
                model_data.registry,
            )?;
            Ok((expression, rest))
        }
    }
}

enum NumericTable1D {
    Integer(usize),
    Continuous(usize),
}

fn parse_numeric_table_1d_name(
    name: &str,
    registry: &TableRegistry,
) -> Result<NumericTable1D, ParseErr> {
    if let Some(i) = registry.continuous_tables.name_to_table_1d.get(name) {
        Ok(NumericTable1D::Continuous(*i))
    } else if let Some(i) = registry.integer_tables.name_to_table_1d.get(name) {
        Ok(NumericTable1D::Integer(*i))
    } else {
        Err(ParseErr::new(format!("no such 1D numeric table `{name}`")))
    }
}

fn numeric_table_1d_len(table: &NumericTable1D, registry: &TableRegistry) -> usize {
    match table {
        NumericTable1D::Integer(i) => registry.integer_tables.tables_1d[*i].0.len(),
        NumericTable1D::Continuous(i) => registry.continuous_tables.tables_1d[*i].0.len(),
    }
}

fn check_fractional_knapsack_table_lengths(
    values: &NumericTable1D,
    weights: &NumericTable1D,
    registry: &TableRegistry,
) -> Result<(), ParseErr> {
    let values_len = numeric_table_1d_len(values, registry);
    let weights_len = numeric_table_1d_len(weights, registry);

    if values_len != weights_len {
        return Err(ParseErr::new(format!(
            "fractional knapsack value table length {values_len} != weight table length {weights_len}",
        )));
    }

    Ok(())
}

fn parse_fractional_knapsack<'a>(
    tokens: &'a [String],
    model_data: &mut ModelData,
    local_variables: &FxHashMap<String, usize>,
) -> Result<(ContinuousExpression, &'a [String]), ParseErr> {
    let (items, rest) = element_parser::parse_set_expression(tokens, model_data, local_variables)?;
    let (capacity, rest) = parse_expression(rest, model_data, local_variables)?;
    let (token, rest_after_token) = util::get_next_token_and_rest(rest)?;

    if token == "(" {
        let (item_expressions, rest) =
            parse_fractional_knapsack_items(rest, model_data, local_variables)?;
        let rest = util::parse_closing(rest)?;
        Ok((
            ContinuousExpression::FractionalKnapsack(
                Box::new(items),
                Box::new(capacity),
                item_expressions,
            ),
            rest,
        ))
    } else {
        let values = parse_numeric_table_1d_name(token, model_data.registry)?;
        let (weights, rest) = util::get_next_token_and_rest(rest_after_token)?;
        let weights = parse_numeric_table_1d_name(weights, model_data.registry)?;
        let rest = util::parse_closing(rest)?;

        check_fractional_knapsack_table_lengths(&values, &weights, model_data.registry)?;

        let expression = match (values, weights) {
            (NumericTable1D::Integer(values), NumericTable1D::Integer(weights)) => {
                ContinuousExpression::FractionalKnapsackIntegerTable(
                    Box::new(items),
                    Box::new(capacity),
                    values,
                    weights,
                )
            }
            (NumericTable1D::Continuous(values), NumericTable1D::Continuous(weights)) => {
                ContinuousExpression::FractionalKnapsackContinuousTable(
                    Box::new(items),
                    Box::new(capacity),
                    values,
                    weights,
                )
            }
            (NumericTable1D::Integer(values), NumericTable1D::Continuous(weights)) => {
                ContinuousExpression::FractionalKnapsackIntegerValueContinuousWeightTable(
                    Box::new(items),
                    Box::new(capacity),
                    values,
                    weights,
                )
            }
            (NumericTable1D::Continuous(values), NumericTable1D::Integer(weights)) => {
                ContinuousExpression::FractionalKnapsackContinuousValueIntegerWeightTable(
                    Box::new(items),
                    Box::new(capacity),
                    values,
                    weights,
                )
            }
        };

        Ok((expression, rest))
    }
}

fn parse_fractional_knapsack_items<'a>(
    tokens: &'a [String],
    model_data: &mut ModelData,
    local_variables: &FxHashMap<String, usize>,
) -> Result<FractionalKnapsackItemExpressions<'a>, ParseErr> {
    let (token, mut rest) = util::get_next_token_and_rest(tokens)?;
    if token != "(" {
        return Err(ParseErr::new(format!(
            "unexpected token: `{token}`, expected `(`",
        )));
    }

    let mut items = Vec::new();
    loop {
        let (token, next) = util::get_next_token_and_rest(rest)?;
        if token == ")" {
            return Ok((items, next));
        }
        if token != "(" {
            return Err(ParseErr::new(format!(
                "unexpected token: `{token}`, expected item tuple",
            )));
        }

        let (item, next) = util::get_next_token_and_rest(next)?;
        let item = item
            .parse()
            .map_err(|e| ParseErr::new(format!("could not parse `{item}` as an item id: {e:?}")))?;
        let (value, next) = parse_expression(next, model_data, local_variables)?;
        let (weight, next) = parse_expression(next, model_data, local_variables)?;
        let next = util::parse_closing(next)?;
        items.push((item, value, weight));
        rest = next;
    }
}

fn parse_reduce_operator(token: &str) -> Result<ReduceOperator, ParseErr> {
    match token {
        "sum" => Ok(ReduceOperator::Sum),
        "product" => Ok(ReduceOperator::Product),
        "max" => Ok(ReduceOperator::Max),
        "min" => Ok(ReduceOperator::Min),
        _ => Err(ParseErr::new(format!("no such reduce operator `{token}`"))),
    }
}

// `(reduce sum|product|max|min <name> <set expression> <body expression>)` reduces
// `<body expression>` over the elements of `<set expression>`. `<name>` is bound as a fresh
// local variable, in scope only while parsing `<body expression>`; the set expression itself
// is parsed without it.
fn parse_reduce<'a>(
    tokens: &'a [String],
    model_data: &mut ModelData,
    local_variables: &FxHashMap<String, usize>,
) -> Result<(ContinuousExpression, &'a [String]), ParseErr> {
    let (op, rest) = util::get_next_token_and_rest(tokens)?;
    let op = parse_reduce_operator(op)?;
    let (name, rest) = util::get_next_token_and_rest(rest)?;
    let (set, rest) = element_parser::parse_set_expression(rest, model_data, local_variables)?;
    let (id, local_variables) = util::bind_local_variable(model_data, name, local_variables);
    let (body, rest) = parse_expression(rest, model_data, &local_variables)?;
    let rest = util::parse_closing(rest)?;
    Ok((
        ContinuousExpression::Reduce(op, Box::new(set), id, Box::new(body)),
        rest,
    ))
}

fn parse_minimum_spanning_tree<'a>(
    tokens: &'a [String],
    model_data: &mut ModelData,
    local_variables: &FxHashMap<String, usize>,
) -> Result<(ContinuousExpression, &'a [String]), ParseErr> {
    let (nodes, rest) = element_parser::parse_set_expression(tokens, model_data, local_variables)?;
    let (token, rest_after_token) = util::get_next_token_and_rest(rest)?;

    if token == "(" {
        return parse_minimum_spanning_tree_edges(
            nodes,
            rest_after_token,
            model_data,
            local_variables,
        );
    }

    if let Some(edge_weights) = model_data
        .registry
        .continuous_tables
        .name_to_table_2d
        .get(token)
    {
        let edge_weights = *edge_weights;
        let (token, rest) = util::get_next_token_and_rest(rest_after_token)?;

        if token == ")" {
            Ok((
                ContinuousExpression::MinimumSpanningTree(Box::new(nodes), edge_weights),
                rest,
            ))
        } else {
            let connectivity = *model_data
                .registry
                .bool_tables
                .name_to_table_2d
                .get(token)
                .ok_or_else(|| ParseErr::new(format!("no such 2D boolean table `{token}`")))?;
            let rest = util::parse_closing(rest)?;
            Ok((
                ContinuousExpression::MinimumSpanningTreeWithConnectivity(
                    Box::new(nodes),
                    edge_weights,
                    connectivity,
                ),
                rest,
            ))
        }
    } else {
        let edge_weights = *model_data
            .registry
            .integer_tables
            .name_to_table_2d
            .get(token)
            .ok_or_else(|| ParseErr::new(format!("no such 2D numeric table `{token}`")))?;
        let (token, rest) = util::get_next_token_and_rest(rest_after_token)?;

        if token == ")" {
            Ok((
                ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::MinimumSpanningTree(Box::new(nodes), edge_weights),
                )),
                rest,
            ))
        } else {
            let connectivity = *model_data
                .registry
                .bool_tables
                .name_to_table_2d
                .get(token)
                .ok_or_else(|| ParseErr::new(format!("no such 2D boolean table `{token}`")))?;
            let rest = util::parse_closing(rest)?;
            Ok((
                ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::MinimumSpanningTreeWithConnectivity(
                        Box::new(nodes),
                        edge_weights,
                        connectivity,
                    ),
                )),
                rest,
            ))
        }
    }
}

// Each edge is `(i j weight)` or, if a connectivity expression is given for the first edge,
// `(i j weight connectivity)` for every edge. Weight and connectivity may be arbitrary
// expressions; whether they are constant is resolved later by `simplify`, which is also
// responsible for sorting the edges by weight rather than trusting the input order.
// `tokens` starts right after the opening `(` of the edge list, already consumed by the caller.
fn parse_minimum_spanning_tree_edges<'a>(
    nodes: SetExpression,
    tokens: &'a [String],
    model_data: &mut ModelData,
    local_variables: &FxHashMap<String, usize>,
) -> Result<(ContinuousExpression, &'a [String]), ParseErr> {
    let (token, next) = util::get_next_token_and_rest(tokens)?;
    if token == ")" {
        let rest = util::parse_closing(next)?;
        return Ok((
            ContinuousExpression::MinimumSpanningTreeWithEdges(Box::new(nodes), Vec::new()),
            rest,
        ));
    }
    if token != "(" {
        return Err(ParseErr::new(format!(
            "unexpected token: `{token}`, expected an edge tuple",
        )));
    }

    let (i, j, weight, condition, rest) = parse_edge(next, model_data, local_variables)?;

    if let Some(condition) = condition {
        let mut edges = vec![(i, j, weight, condition)];
        let mut rest = rest;

        loop {
            let (token, next) = util::get_next_token_and_rest(rest)?;
            if token == ")" {
                let rest = util::parse_closing(next)?;
                return Ok((
                    ContinuousExpression::MinimumSpanningTreeWithEdgesAndConnectivity(
                        Box::new(nodes),
                        edges,
                    ),
                    rest,
                ));
            }
            if token != "(" {
                return Err(ParseErr::new(format!(
                    "unexpected token: `{token}`, expected an edge tuple",
                )));
            }

            let (i, j, weight, condition, next) = parse_edge(next, model_data, local_variables)?;
            let condition = condition.ok_or_else(|| {
                ParseErr::new(String::from(
                    "expected a connectivity expression for an edge in a minimum spanning tree edge list",
                ))
            })?;
            edges.push((i, j, weight, condition));
            rest = next;
        }
    } else {
        let mut edges = vec![(i, j, weight)];
        let mut rest = rest;

        loop {
            let (token, next) = util::get_next_token_and_rest(rest)?;
            if token == ")" {
                let rest = util::parse_closing(next)?;
                return Ok((
                    ContinuousExpression::MinimumSpanningTreeWithEdges(Box::new(nodes), edges),
                    rest,
                ));
            }
            if token != "(" {
                return Err(ParseErr::new(format!(
                    "unexpected token: `{token}`, expected an edge tuple",
                )));
            }

            let (i, j, weight, condition, next) = parse_edge(next, model_data, local_variables)?;
            if condition.is_some() {
                return Err(ParseErr::new(String::from(
                    "unexpected connectivity expression for an edge in a minimum spanning tree edge list",
                )));
            }
            edges.push((i, j, weight));
            rest = next;
        }
    }
}

#[allow(clippy::type_complexity)]
fn parse_edge<'a>(
    tokens: &'a [String],
    model_data: &mut ModelData,
    local_variables: &FxHashMap<String, usize>,
) -> Result<
    (
        usize,
        usize,
        ContinuousExpression,
        Option<Condition>,
        &'a [String],
    ),
    ParseErr,
> {
    let (i, rest) = util::get_next_token_and_rest(tokens)?;
    let i = i
        .parse()
        .map_err(|e| ParseErr::new(format!("could not parse `{i}` as an edge endpoint: {e:?}")))?;
    let (j, rest) = util::get_next_token_and_rest(rest)?;
    let j = j
        .parse()
        .map_err(|e| ParseErr::new(format!("could not parse `{j}` as an edge endpoint: {e:?}")))?;
    let (weight, rest) = parse_expression(rest, model_data, local_variables)?;
    let (token, next) = util::get_next_token_and_rest(rest)?;

    if token == ")" {
        Ok((i, j, weight, None, next))
    } else {
        let (condition, rest) =
            condition_parser::parse_expression(rest, model_data, local_variables)?;
        let rest = util::parse_closing(rest)?;
        Ok((i, j, weight, Some(condition), rest))
    }
}

fn parse_parameterized_state_function<'a>(
    name: &str,
    tokens: &'a [String],
    functions: &StateFunctions,
    parameters: &FxHashMap<String, usize>,
) -> Result<Option<(ContinuousExpression, &'a [String])>, ParseErr> {
    let (name, rest) = util::parse_parameterized_state_function_name(name, tokens, parameters)?;

    if let Ok(expression) = functions.get_continuous_function(&name) {
        Ok(Some((expression, rest)))
    } else if let Ok(expression) = functions.get_integer_function(&name) {
        Ok(Some((
            ContinuousExpression::FromInteger(Box::new(expression)),
            rest,
        )))
    } else {
        Ok(None)
    }
}

fn parse_unary_operation(
    name: &str,
    x: ContinuousExpression,
) -> Result<ContinuousExpression, ParseErr> {
    match name {
        "abs" => Ok(ContinuousExpression::UnaryOperation(
            UnaryOperator::Abs,
            Box::new(x),
        )),
        "sqrt" => Ok(ContinuousExpression::ContinuousUnaryOperation(
            ContinuousUnaryOperator::Sqrt,
            Box::new(x),
        )),
        _ => Err(ParseErr::new(format!("no such unary operator `{name}`"))),
    }
}

fn parse_from_integer<'a>(
    tokens: &'a [String],
    model_data: &mut ModelData,
    local_variables: &FxHashMap<String, usize>,
) -> Result<(ContinuousExpression, &'a [String]), ParseErr> {
    let (expression, rest) = integer_parser::parse_expression(tokens, model_data, local_variables)?;
    let rest = util::parse_closing(rest)?;
    Ok((
        ContinuousExpression::FromInteger(Box::new(expression)),
        rest,
    ))
}

fn parse_round<'a>(
    name: &str,
    tokens: &'a [String],
    model_data: &mut ModelData,
    local_variables: &FxHashMap<String, usize>,
) -> Result<(ContinuousExpression, &'a [String]), ParseErr> {
    let op = match name {
        "ceil" => CastOperator::Ceil,
        "floor" => CastOperator::Floor,
        "round" => CastOperator::Round,
        "trunc" => CastOperator::Trunc,
        _ => return Err(ParseErr::new(format!("no such unary operator `{name}`"))),
    };
    let (expression, rest) = parse_expression(tokens, model_data, local_variables)?;
    let rest = util::parse_closing(rest)?;
    Ok((ContinuousExpression::Round(op, Box::new(expression)), rest))
}

fn parse_binary_operation(
    name: &str,
    x: ContinuousExpression,
    y: ContinuousExpression,
) -> Result<ContinuousExpression, ParseErr> {
    match name {
        "+" => Ok(ContinuousExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(x),
            Box::new(y),
        )),
        "-" => Ok(ContinuousExpression::BinaryOperation(
            BinaryOperator::Sub,
            Box::new(x),
            Box::new(y),
        )),
        "*" => Ok(ContinuousExpression::BinaryOperation(
            BinaryOperator::Mul,
            Box::new(x),
            Box::new(y),
        )),
        "/" => Ok(ContinuousExpression::BinaryOperation(
            BinaryOperator::Div,
            Box::new(x),
            Box::new(y),
        )),
        "%" => Ok(ContinuousExpression::BinaryOperation(
            BinaryOperator::Rem,
            Box::new(x),
            Box::new(y),
        )),
        "min" => Ok(ContinuousExpression::BinaryOperation(
            BinaryOperator::Min,
            Box::new(x),
            Box::new(y),
        )),
        "max" => Ok(ContinuousExpression::BinaryOperation(
            BinaryOperator::Max,
            Box::new(x),
            Box::new(y),
        )),
        "pow" => Ok(ContinuousExpression::ContinuousBinaryOperation(
            ContinuousBinaryOperator::Pow,
            Box::new(x),
            Box::new(y),
        )),
        "log" => Ok(ContinuousExpression::ContinuousBinaryOperation(
            ContinuousBinaryOperator::Log,
            Box::new(x),
            Box::new(y),
        )),
        _ => Err(ParseErr::new(format!("no such operator `{name}`"))),
    }
}

fn parse_cardinality<'a>(
    tokens: &'a [String],
    model_data: &mut ModelData,
    local_variables: &FxHashMap<String, usize>,
) -> Result<(ContinuousExpression, &'a [String]), ParseErr> {
    let (expression, rest) =
        element_parser::parse_set_expression(tokens, model_data, local_variables)?;
    let (token, rest) = rest
        .split_first()
        .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
    if token != "|" {
        return Err(ParseErr::new(format!(
            "unexpected token: `{token}`, expected `|`",
        )));
    }
    Ok((ContinuousExpression::Cardinality(expression), rest))
}

fn parse_continuous_atom(
    token: &str,
    metadata: &StateMetadata,
    functions: &StateFunctions,
    registry: &TableRegistry,
) -> Result<ContinuousExpression, ParseErr> {
    if let Some(v) = registry.continuous_tables.name_to_constant.get(token) {
        Ok(ContinuousExpression::Constant(*v))
    } else if let Some(i) = metadata.name_to_continuous_variable.get(token) {
        Ok(ContinuousExpression::Variable(*i))
    } else if let Some(i) = metadata.name_to_continuous_resource_variable.get(token) {
        Ok(ContinuousExpression::ResourceVariable(*i))
    } else if let Some(v) = registry.integer_tables.name_to_constant.get(token) {
        Ok(ContinuousExpression::Constant(*v as Continuous))
    } else if let Some(i) = metadata.name_to_integer_variable.get(token) {
        Ok(ContinuousExpression::FromInteger(Box::new(
            IntegerExpression::Variable(*i),
        )))
    } else if let Some(i) = metadata.name_to_integer_resource_variable.get(token) {
        Ok(ContinuousExpression::FromInteger(Box::new(
            IntegerExpression::ResourceVariable(*i),
        )))
    } else if let Ok(expression) = functions.get_continuous_function(token) {
        Ok(expression)
    } else if let Ok(expression) = functions.get_integer_function(token) {
        Ok(ContinuousExpression::FromInteger(Box::new(expression)))
    } else if token == "cost" {
        Ok(ContinuousExpression::Cost)
    } else {
        let n: Continuous = token.parse().map_err(|e| {
            ParseErr::new(format!(
                "could not parse {token} as a continuous atom: {e:?}",
            ))
        })?;
        Ok(ContinuousExpression::Constant(n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::expression::{
        ArgumentExpression, ElementExpression, NumericTableExpression, ReduceOperator,
        ReferenceExpression, SetExpression,
    };
    use dypdl::*;

    fn generate_metadata() -> StateMetadata {
        let object_names = vec![String::from("something")];
        let object_numbers = vec![10];
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
            ..Default::default()
        }
    }

    fn generate_parameters() -> FxHashMap<String, Element> {
        let mut parameters = FxHashMap::default();
        parameters.insert(String::from("param"), 0);
        parameters
    }

    fn generate_registry() -> TableRegistry {
        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("f0"), 0);

        let tables_1d = vec![Table1D::new(Vec::new())];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("f1"), 0);

        let tables_2d = vec![Table2D::new(Vec::new())];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("f2"), 0);

        let tables_3d = vec![Table3D::new(Vec::new())];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("f3"), 0);

        let tables = vec![Table::new(FxHashMap::default(), 0)];
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("f4"), 0);

        let integer_tables = dypdl::TableData {
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
        name_to_constant.insert(String::from("cf0"), 0.0);

        let tables_1d = vec![Table1D::new(Vec::new())];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("cf1"), 0);

        let tables_2d = vec![Table2D::new(Vec::new())];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("cf2"), 0);

        let tables_3d = vec![Table3D::new(Vec::new())];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("cf3"), 0);

        let tables = vec![Table::new(FxHashMap::default(), 0.0)];
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("cf4"), 0);

        let continuous_tables = dypdl::TableData {
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

        let tables_2d = vec![Table2D::new(Vec::new())];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("b2"), 0);
        let bool_tables = dypdl::TableData {
            tables_2d,
            name_to_table_2d,
            ..Default::default()
        };

        TableRegistry {
            integer_tables,
            continuous_tables,
            bool_tables,
            ..Default::default()
        }
    }

    #[test]
    fn parse_continuous_atom_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["cost", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, ContinuousExpression::Cost);
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["cf0", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, ContinuousExpression::Constant(0.0));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["c1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, ContinuousExpression::Variable(1));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["cr1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, ContinuousExpression::ResourceVariable(1));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["f0", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, ContinuousExpression::Constant(0.0));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["i1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::FromInteger(Box::new(IntegerExpression::Variable(1)))
        );
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["ir1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::FromInteger(Box::new(IntegerExpression::ResourceVariable(1)))
        );
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["11.5", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, ContinuousExpression::Constant(11.5));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["11", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, ContinuousExpression::Constant(11.0));
        assert_eq!(rest, &tokens[1..]);
    }

    #[test]
    fn parse_continuous_atom_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["e0", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_state_function_ok() {
        let metadata = StateMetadata::default();
        let registry = TableRegistry::default();
        let parameters = FxHashMap::default();

        let mut functions = StateFunctions::default();
        let result = functions.add_continuous_function("sf", ContinuousExpression::Constant(1.0));
        assert!(result.is_ok());
        let expected = result.unwrap();

        let tokens: Vec<String> = ["sf", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, expected);
        assert_eq!(rest, &tokens[1..]);
    }

    #[test]
    fn parse_continuous_table_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "sum", "cf4", "0", "e0", "s0", "0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::Table(Box::new(NumericTableExpression::TableReduce(
                ReduceOperator::Sum,
                0,
                vec![
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Variable(0)),
                    ArgumentExpression::Set(SetExpression::Reference(
                        ReferenceExpression::Variable(0)
                    )),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                ]
            )))
        );
        assert_eq!(rest, &tokens[8..]);
    }

    #[test]
    fn parse_fractional_knapsack_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "fractional_knapsack",
            "s0",
            "c0",
            "f1",
            "cf1",
            ")",
            "c0",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &StateFunctions::default(),
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::FractionalKnapsackIntegerValueContinuousWeightTable(
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                Box::new(ContinuousExpression::Variable(0)),
                0,
                0,
            )
        );
        assert_eq!(rest, &tokens[7..]);

        let mut functions = StateFunctions::default();
        let state_function = functions
            .add_continuous_function("sf_0", ContinuousExpression::Variable(0))
            .unwrap();
        let tokens: Vec<String> = [
            "(",
            "fractional_knapsack",
            "s0",
            "c0",
            "(",
            "(",
            "0",
            "(",
            "sf",
            "param",
            ")",
            "1",
            ")",
            "(",
            "1",
            "2.5",
            "c1",
            ")",
            ")",
            ")",
            "c0",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::FractionalKnapsack(
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                Box::new(ContinuousExpression::Variable(0)),
                vec![
                    (0, state_function, ContinuousExpression::Constant(1.0)),
                    (
                        1,
                        ContinuousExpression::Constant(2.5),
                        ContinuousExpression::Variable(1),
                    ),
                ],
            )
        );
        assert_eq!(rest, &tokens[20..]);
    }

    #[test]
    fn parse_reduce_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(", "reduce", "sum", "x", "s0", "(", "cf1", "x", ")", ")", "1", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::Reduce(
                ReduceOperator::Sum,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                0,
                Box::new(ContinuousExpression::Table(Box::new(
                    NumericTableExpression::Table1D(0, ElementExpression::LocalVariable(0))
                ))),
            )
        );
        assert_eq!(rest, &tokens[10..]);
    }

    #[test]
    fn parse_reduce_nested_ok() {
        // A nested reduce must be able to see the outer binder (`x`, id 0) while parsing its
        // own body, and its own binder (`y`) must get a distinct id (1), not collide with `x`:
        // `(reduce sum x s0 (reduce sum y s1 (cf2 x y)))`.
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(", "reduce", "sum", "x", "s0", "(", "reduce", "sum", "y", "s1", "(", "cf2", "x", "y",
            ")", ")", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::Reduce(
                ReduceOperator::Sum,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                0,
                Box::new(ContinuousExpression::Reduce(
                    ReduceOperator::Sum,
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
                    1,
                    Box::new(ContinuousExpression::Table(Box::new(
                        NumericTableExpression::Table2D(
                            0,
                            ElementExpression::LocalVariable(0),
                            ElementExpression::LocalVariable(1),
                        )
                    ))),
                )),
            )
        );
        assert!(rest.is_empty());
    }

    #[test]
    fn parse_minimum_spanning_tree_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "minimum_spanning_tree",
            "s0",
            "cf2",
            "b2",
            ")",
            "c0",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &StateFunctions::default(),
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::MinimumSpanningTreeWithConnectivity(
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                0,
                0,
            )
        );
        assert_eq!(rest, &tokens[6..]);

        let tokens: Vec<String> = [
            "(",
            "minimum_spanning_tree",
            "s0",
            "f2",
            "b2",
            ")",
            "c0",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &StateFunctions::default(),
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::FromInteger(Box::new(
                IntegerExpression::MinimumSpanningTreeWithConnectivity(
                    Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                    0,
                    0,
                )
            ))
        );
        assert_eq!(rest, &tokens[6..]);

        let tokens: Vec<String> = [
            "(",
            "minimum_spanning_tree",
            "s0",
            "(",
            "(",
            "1",
            "2",
            "3",
            ")",
            "(",
            "0",
            "1",
            "2.5",
            ")",
            ")",
            ")",
            "c0",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &StateFunctions::default(),
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::MinimumSpanningTreeWithEdges(
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                vec![
                    (1, 2, ContinuousExpression::Constant(3.0)),
                    (0, 1, ContinuousExpression::Constant(2.5)),
                ],
            )
        );
        assert_eq!(rest, &tokens[16..]);
        // Sorting is deferred to `simplify`, which does not trust the edges to already be
        // given in sorted order.
        assert_eq!(
            expression.simplify(&registry),
            ContinuousExpression::MinimumSpanningTreeWithSortedEdges(
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                vec![(0, 1, 2.5), (1, 2, 3.0)],
            )
        );
    }

    #[test]
    fn parse_minimum_spanning_tree_edges_with_connectivity_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "minimum_spanning_tree",
            "s0",
            "(",
            "(",
            "1",
            "2",
            "3",
            "false",
            ")",
            "(",
            "0",
            "1",
            "2.5",
            "true",
            ")",
            ")",
            ")",
            "c0",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &StateFunctions::default(),
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::MinimumSpanningTreeWithEdgesAndConnectivity(
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                vec![
                    (
                        1,
                        2,
                        ContinuousExpression::Constant(3.0),
                        Condition::Constant(false)
                    ),
                    (
                        0,
                        1,
                        ContinuousExpression::Constant(2.5),
                        Condition::Constant(true)
                    ),
                ],
            )
        );
        assert_eq!(rest, &tokens[18..]);
        assert_eq!(
            expression.simplify(&registry),
            ContinuousExpression::MinimumSpanningTreeWithSortedEdges(
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                vec![(0, 1, 2.5)],
            )
        );
    }

    #[test]
    fn parse_minimum_spanning_tree_edges_empty_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "minimum_spanning_tree", "s0", "(", ")", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &StateFunctions::default(),
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::MinimumSpanningTreeWithEdges(
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                Vec::new(),
            )
        );
        assert_eq!(rest, &tokens[6..]);
    }

    #[test]
    fn parse_minimum_spanning_tree_edges_inconsistent_arity_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "minimum_spanning_tree",
            "s0",
            "(",
            "(",
            "0",
            "1",
            "2.5",
            ")",
            "(",
            "1",
            "2",
            "3",
            "true",
            ")",
            ")",
            ")",
            "c0",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &StateFunctions::default(),
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_parameterized_continuous_state_function_ok() {
        let metadata = StateMetadata::default();
        let registry = TableRegistry::default();
        let parameters = FxHashMap::from_iter(vec![("a".to_string(), 0), ("b".to_string(), 2)]);

        let mut functions = StateFunctions::default();
        let result =
            functions.add_continuous_function("sf_0_1_2", ContinuousExpression::Constant(0.0));
        assert!(result.is_ok());
        let expected = result.unwrap();

        let tokens: Vec<_> = ["(", "sf", "a", "1", "b", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, expected);
        assert_eq!(rest, &tokens[6..]);
    }

    #[test]
    fn parse_parameterized_integer_state_function_as_continuous_ok() {
        let metadata = StateMetadata::default();
        let registry = TableRegistry::default();
        let parameters = FxHashMap::from_iter(vec![("a".to_string(), 0), ("b".to_string(), 2)]);

        let mut functions = StateFunctions::default();
        let result = functions.add_integer_function("sf_0_1_2", IntegerExpression::Constant(0));
        assert!(result.is_ok());
        let expected = result.unwrap();

        let tokens: Vec<_> = ["(", "sf", "a", "1", "b", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::FromInteger(Box::new(expected))
        );
        assert_eq!(rest, &tokens[6..]);
    }

    #[test]
    fn parse_parameterized_continuous_state_function_non_exist_err() {
        let metadata = StateMetadata::default();
        let registry = TableRegistry::default();
        let parameters = FxHashMap::from_iter(vec![("a".to_string(), 0), ("b".to_string(), 2)]);

        let functions = StateFunctions::default();

        let tokens: Vec<_> = ["(", "sf", "a", "1", "b", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_parameterized_continuous_state_function_no_closing_err() {
        let metadata = StateMetadata::default();
        let registry = TableRegistry::default();
        let parameters = FxHashMap::from_iter(vec![("a".to_string(), 0), ("b".to_string(), 2)]);

        let mut functions = StateFunctions::default();
        let result =
            functions.add_continuous_function("sf_0_1_2", ContinuousExpression::Constant(0.0));
        assert!(result.is_ok());

        let tokens: Vec<_> = ["(", "sf", "a", "1", "b"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_table_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "cf4", "0.0", "e0", "s0", "0", "c0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_from_integer_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "continuous", "1", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::FromInteger(Box::new(IntegerExpression::Constant(1)),)
        );
        assert_eq!(rest, &tokens[4..]);
    }

    #[test]
    fn parse_from_integer_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "continuous", "1.5", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_unary_operator_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "abs", "-4.5", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(ContinuousExpression::Constant(-4.5)),
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["(", "sqrt", "4.5", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::ContinuousUnaryOperation(
                ContinuousUnaryOperator::Sqrt,
                Box::new(ContinuousExpression::Constant(4.5)),
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["(", "floor", "4.5", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::Round(
                CastOperator::Floor,
                Box::new(ContinuousExpression::Constant(4.5)),
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["(", "ceil", "4.5", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::Round(
                CastOperator::Ceil,
                Box::new(ContinuousExpression::Constant(4.5)),
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["(", "round", "4.5", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::Round(
                CastOperator::Round,
                Box::new(ContinuousExpression::Constant(4.5)),
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["(", "trunc", "4.5", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::Round(
                CastOperator::Trunc,
                Box::new(ContinuousExpression::Constant(4.5)),
            )
        );
        assert_eq!(rest, &tokens[4..]);
    }

    #[test]
    fn parse_continuous_unary_operator_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "exp", "3.0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_if_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "if", "true", "0.0", "c0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[6..]);
    }

    #[test]
    fn parse_continuous_if_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "if", "true", "0.0", "0.0", "c0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "if", "0.0", "c0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "if", "0.0", "0.0", "c0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_binary_operation_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "+", "0.0", "c0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "-", "0.0", "c0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "*", "0.0", "c0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "/", "0.0", "c0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "%", "0.0", "c0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "min", "0.0", "c0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "max", "0.0", "c0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "pow", "2.0", "c0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Constant(2.0)),
                Box::new(ContinuousExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "log", "2.0", "c0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::Constant(2.0)),
                Box::new(ContinuousExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_continuous_binary_operation_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens = Vec::new();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "+", "0.0", "c0", "c1", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "+", "0.0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "^", "0.0", "c0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_cardinality_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["|", "s2", "|", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::Cardinality(SetExpression::Reference(
                ReferenceExpression::Variable(2)
            ))
        );
        assert_eq!(rest, &tokens[3..]);
    }

    #[test]
    fn parse_continuous_cardinality_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["|", "e2", "|", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["|", "s2", "s0", "|", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_length_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "length", "s0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_last_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "last", "(", "vector", "0", "1", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_at_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "at", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "at", "(", "continuous-vector", "0", "1", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "at", "(", "vector", "0", "1", ")", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_reduce_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "reduce-sum", "(", "vector", "0", "1", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(",
            "reduce-product",
            "(",
            "vector",
            "0",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "reduce-max", "(", "vector", "0", "1", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "reduce-min", "(", "vector", "0", "1", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(",
            "reduce-null",
            "(",
            "continuous-vector",
            "0",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(
            &tokens,
            &mut ModelData {
                metadata: &metadata,
                functions: &functions,
                registry: &registry,
                parameters: &parameters,
                local_variable_data: &mut LocalVariableData::default(),
            },
            &FxHashMap::default(),
        );
        assert!(result.is_err());
    }
}
