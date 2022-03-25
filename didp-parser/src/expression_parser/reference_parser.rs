use super::util::ParseErr;
use crate::expression::ReferenceExpression;
use std::collections::HashMap;

pub fn parse_atom<T: Clone>(
    token: &str,
    name_to_constant: &HashMap<String, T>,
    name_to_variable: &HashMap<String, usize>,
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
