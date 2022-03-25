use super::element_parser;
use super::reference_parser;
use super::util;
use super::util::ParseErr;
use crate::expression::{ReferenceExpression, VectorExpression};
use crate::state::StateMetadata;
use crate::table_registry::TableRegistry;
use std::collections::HashMap;

pub fn parse_expression<'a, 'b, 'c>(
    tokens: &'a [String],
    metadata: &'b StateMetadata,
    registry: &'b TableRegistry,
    parameters: &'c HashMap<String, usize>,
) -> Result<(VectorExpression, &'a [String]), ParseErr> {
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
    match &token[..] {
        "(" => {
            let (name, rest) = rest
                .split_first()
                .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
            if let Some((expression, rest)) = element_parser::parse_table_expression(
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
            } else {
                let (expression, rest) =
                    parse_operation(name, rest, metadata, registry, parameters)?;
                Ok((expression, rest))
            }
        }
        ")" => Err(ParseErr::new("unexpected `)`".to_string())),
        _ => {
            let set = reference_parser::parse_atom(
                token,
                &registry.vector_tables.name_to_constant,
                &metadata.name_to_vector_variable,
            )?;
            Ok((VectorExpression::Reference(set), rest))
        }
    }
}

fn parse_operation<'a, 'b, 'c>(
    name: &'a str,
    tokens: &'a [String],
    metadata: &'b StateMetadata,
    registry: &'b TableRegistry,
    parameters: &'c HashMap<String, usize>,
) -> Result<(VectorExpression, &'a [String]), ParseErr> {
    match name {
        "indices" => {
            let (x, rest) = parse_expression(tokens, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok((VectorExpression::Indices(Box::new(x)), rest))
        }
        "push" => {
            let (x, rest) =
                element_parser::parse_expression(tokens, metadata, registry, parameters)?;
            let (y, rest) = parse_expression(rest, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok((VectorExpression::Push(x, Box::new(y)), rest))
        }
        "pop" => {
            let (x, rest) = parse_expression(tokens, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok((VectorExpression::Pop(Box::new(x)), rest))
        }
        op => Err(ParseErr::new(format!("no such operator `{}`", op))),
    }
}
