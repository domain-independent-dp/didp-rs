use dypdl::expression;
use dypdl::variable_type::{Continuous, Element, Integer, ToVariableString};
use dypdl::CostExpression;
use dypdl::Set;
use dypdl::{StateMetadata, TableRegistry};
use rustc_hash::FxHashMap;

pub trait ToYamlString {
    fn to_yaml_string(
        &self,
        state_data: &StateMetadata,
        table_registry: &TableRegistry,
    ) -> Result<String, &'static str>;
}

impl ToYamlString for expression::UnaryOperator {
    fn to_yaml_string(&self, _: &StateMetadata, _: &TableRegistry) -> Result<String, &'static str> {
        match self {
            Self::Abs => Ok("abs".to_owned()),
            Self::Neg => Ok("neg".to_owned()),
        }
    }
}

impl ToYamlString for expression::ContinuousUnaryOperator {
    fn to_yaml_string(&self, _: &StateMetadata, _: &TableRegistry) -> Result<String, &'static str> {
        match self {
            Self::Sqrt => Ok("sqrt".to_owned()),
        }
    }
}

impl ToYamlString for expression::BinaryOperator {
    fn to_yaml_string(&self, _: &StateMetadata, _: &TableRegistry) -> Result<String, &'static str> {
        match self {
            Self::Add => Ok("+".to_owned()),
            Self::Sub => Ok("-".to_owned()),
            Self::Mul => Ok("*".to_owned()),
            Self::Div => Ok("/".to_owned()),
            Self::Rem => Ok("%".to_owned()),
            Self::Max => Ok("max".to_owned()),
            Self::Min => Ok("min".to_owned()),
        }
    }
}

impl ToYamlString for expression::ContinuousBinaryOperator {
    fn to_yaml_string(&self, _: &StateMetadata, _: &TableRegistry) -> Result<String, &'static str> {
        match self {
            Self::Log => Ok("log".to_owned()),
            Self::Pow => Ok("pow".to_owned()),
        }
    }
}

impl ToYamlString for expression::SetOperator {
    fn to_yaml_string(&self, _: &StateMetadata, _: &TableRegistry) -> Result<String, &'static str> {
        match self {
            Self::Union => Ok("union".to_owned()),
            Self::Difference => Ok("difference".to_owned()),
            Self::Intersection => Ok("intersection".to_owned()),
        }
    }
}

impl ToYamlString for expression::SetElementOperator {
    fn to_yaml_string(&self, _: &StateMetadata, _: &TableRegistry) -> Result<String, &'static str> {
        match self {
            Self::Add => Ok("add".to_owned()),
            Self::Remove => Ok("remove".to_owned()),
        }
    }
}

impl ToYamlString for expression::SetReduceOperator {
    fn to_yaml_string(&self, _: &StateMetadata, _: &TableRegistry) -> Result<String, &'static str> {
        match self {
            Self::Union => Ok("union".to_owned()),
            Self::Intersection => Ok("intersection".to_owned()),
            Self::SymmetricDifference => Ok("disjunctive_union".to_owned()),
        }
    }
}

impl ToYamlString for expression::ReduceOperator {
    fn to_yaml_string(&self, _: &StateMetadata, _: &TableRegistry) -> Result<String, &'static str> {
        match self {
            Self::Sum => Ok("sum".to_owned()),
            Self::Product => Ok("product".to_owned()),
            Self::Min => Ok("min".to_owned()),
            Self::Max => Ok("max".to_owned()),
        }
    }
}

impl ToYamlString for expression::CastOperator {
    fn to_yaml_string(&self, _: &StateMetadata, _: &TableRegistry) -> Result<String, &'static str> {
        match self {
            Self::Ceil => Ok("ceil".to_owned()),
            Self::Floor => Ok("floor".to_owned()),
            Self::Round => Ok("round".to_owned()),
            Self::Trunc => Ok("trunc".to_owned()),
        }
    }
}

impl ToYamlString for expression::ComparisonOperator {
    fn to_yaml_string(&self, _: &StateMetadata, _: &TableRegistry) -> Result<String, &'static str> {
        match self {
            Self::Eq => Ok("=".to_owned()),
            Self::Ne => Ok("!=".to_owned()),
            Self::Ge => Ok(">=".to_owned()),
            Self::Gt => Ok(">".to_owned()),
            Self::Le => Ok("<=".to_owned()),
            Self::Lt => Ok("<".to_owned()),
        }
    }
}

impl ToYamlString for expression::Condition {
    fn to_yaml_string(
        &self,
        state_data: &StateMetadata,
        table_registry: &TableRegistry,
    ) -> Result<String, &'static str> {
        match self {
            Self::Constant(b) => Ok(b.to_string()),
            Self::Not(c) => Ok(format!(
                "(not {c_str})",
                c_str = c.to_yaml_string(state_data, table_registry)?
            )),
            Self::And(c1, c2) => Ok(format!(
                "(and {c1_str} {c2_str})",
                c1_str = c1.to_yaml_string(state_data, table_registry)?,
                c2_str = c2.to_yaml_string(state_data, table_registry)?
            )),
            Self::Or(c1, c2) => Ok(format!(
                "(or {c1_str} {c2_str})",
                c1_str = c1.to_yaml_string(state_data, table_registry)?,
                c2_str = c2.to_yaml_string(state_data, table_registry)?
            )),
            Self::ComparisonE(op, eexp1, eexp2) => Ok(format!(
                "({op_str} {eexp1_str} {eexp2_str})",
                op_str = op.to_yaml_string(state_data, table_registry)?,
                eexp1_str = eexp1.to_yaml_string(state_data, table_registry)?,
                eexp2_str = eexp2.to_yaml_string(state_data, table_registry)?
            )),
            Self::ComparisonC(op, cexp1, cexp2) => Ok(format!(
                "({op_str} {cexp1_str} {cexp2_str})",
                op_str = op.to_yaml_string(state_data, table_registry)?,
                cexp1_str = cexp1.to_yaml_string(state_data, table_registry)?,
                cexp2_str = cexp2.to_yaml_string(state_data, table_registry)?
            )),
            Self::ComparisonI(op, iexp1, iexp2) => Ok(format!(
                "({op_str} {iexp1_str} {iexp2_str})",
                op_str = op.to_yaml_string(state_data, table_registry)?,
                iexp1_str = iexp1.to_yaml_string(state_data, table_registry)?,
                iexp2_str = iexp2.to_yaml_string(state_data, table_registry)?
            )),
            Self::Set(scond) => scond.to_yaml_string(state_data, table_registry),
            Self::Table(texp) => texp.to_yaml_string(state_data, table_registry),
        }
    }
}

impl ToYamlString for expression::SetCondition {
    fn to_yaml_string(
        &self,
        state_data: &StateMetadata,
        table_registry: &TableRegistry,
    ) -> Result<String, &'static str> {
        match self {
            Self::Constant(b) => Ok(b.to_string()),
            Self::IsEqual(sexp1, sexp2) => Ok(format!(
                "(= {sexp1_str} {sexp2_str})",
                sexp1_str = sexp1.to_yaml_string(state_data, table_registry)?,
                sexp2_str = sexp2.to_yaml_string(state_data, table_registry)?
            )),
            Self::IsNotEqual(sexp1, sexp2) => Ok(format!(
                "(!= {sexp1_str} {sexp2_str})",
                sexp1_str = sexp1.to_yaml_string(state_data, table_registry)?,
                sexp2_str = sexp2.to_yaml_string(state_data, table_registry)?
            )),
            Self::IsIn(eexp, sexp) => Ok(format!(
                "(is_in {eexp_str} {sexp_str})",
                eexp_str = eexp.to_yaml_string(state_data, table_registry)?,
                sexp_str = sexp.to_yaml_string(state_data, table_registry)?
            )),
            Self::IsSubset(sexp1, sexp2) => Ok(format!(
                "(is_subset {sexp1_str} {sexp2_str})",
                sexp1_str = sexp1.to_yaml_string(state_data, table_registry)?,
                sexp2_str = sexp2.to_yaml_string(state_data, table_registry)?
            )),
            Self::IsEmpty(sexp) => Ok(format!(
                "(is_empty {sexp_str})",
                sexp_str = sexp.to_yaml_string(state_data, table_registry)?
            )),
        }
    }
}

impl ToYamlString for expression::IntegerExpression {
    fn to_yaml_string(
        &self,
        state_data: &StateMetadata,
        table_registry: &TableRegistry,
    ) -> Result<String, &'static str> {
        match self {
            Self::Constant(value) => Ok(value.to_string()),
            Self::Variable(index) => Ok(state_data.integer_variable_names[*index].clone()),
            Self::ResourceVariable(index) => {
                Ok(state_data.integer_resource_variable_names[*index].clone())
            }
            Self::Cost => Ok("cost".to_owned()),
            Self::UnaryOperation(op, iexp) => Ok(format!(
                "({op_str} {iexp_str})",
                op_str = op.to_yaml_string(state_data, table_registry)?,
                iexp_str = iexp.to_yaml_string(state_data, table_registry)?
            )),
            Self::BinaryOperation(op, iexp1, iexp2) => Ok(format!(
                "({op_str} {iexp1_str} {iexp2_str})",
                op_str = op.to_yaml_string(state_data, table_registry)?,
                iexp1_str = iexp1.to_yaml_string(state_data, table_registry)?,
                iexp2_str = iexp2.to_yaml_string(state_data, table_registry)?
            )),
            Self::Cardinality(sexp) => Ok(format!(
                "|{sexp_str}|",
                sexp_str = sexp.to_yaml_string(state_data, table_registry)?
            )),
            Self::Table(texp) => texp.to_yaml_string(state_data, table_registry),
            Self::If(cond, iexp1, iexp2) => Ok(format!(
                "(if {cond_str} {iexp1_str} {iexp2_str})",
                cond_str = cond.to_yaml_string(state_data, table_registry)?,
                iexp1_str = iexp1.to_yaml_string(state_data, table_registry)?,
                iexp2_str = iexp2.to_yaml_string(state_data, table_registry)?
            )),
            Self::FromContinuous(op, cexp) => Ok(format!(
                "({op_str} {cexp_str})",
                op_str = op.to_yaml_string(state_data, table_registry)?,
                cexp_str = cexp.to_yaml_string(state_data, table_registry)?
            )),
            Self::Length(_) => Err("Current version doesn't support vector expressions"),
            Self::Last(_) => Err("Current version doesn't support vector expressions"),
            Self::At(_, _) => Err("Current version doesn't support vector expressions"),
            Self::Reduce(_, _) => Err("Current version doesn't support vector expressions"),
        }
    }
}

impl ToYamlString for expression::ContinuousExpression {
    fn to_yaml_string(
        &self,
        state_data: &StateMetadata,
        table_registry: &TableRegistry,
    ) -> Result<String, &'static str> {
        match self {
            Self::Constant(value) => Ok(value.to_string()),
            Self::Variable(index) => Ok(state_data.continuous_variable_names[*index].clone()),
            Self::ResourceVariable(index) => {
                Ok(state_data.continuous_resource_variable_names[*index].clone())
            }
            Self::Cost => Ok("cost".to_owned()),
            Self::UnaryOperation(op, cexp) => Ok(format!(
                "({op_str} {cexp_str})",
                op_str = op.to_yaml_string(state_data, table_registry)?,
                cexp_str = cexp.to_yaml_string(state_data, table_registry)?
            )),
            Self::ContinuousUnaryOperation(op, cexp) => Ok(format!(
                "({op_str} {cexp_str})",
                op_str = op.to_yaml_string(state_data, table_registry)?,
                cexp_str = cexp.to_yaml_string(state_data, table_registry)?
            )),
            Self::Round(op, cexp) => Ok(format!(
                "({op_str} {cexp_str})",
                op_str = op.to_yaml_string(state_data, table_registry)?,
                cexp_str = cexp.to_yaml_string(state_data, table_registry)?
            )),
            Self::BinaryOperation(op, cexp1, cexp2) => Ok(format!(
                "({op_str} {cexp1_str} {cexp2_str})",
                op_str = op.to_yaml_string(state_data, table_registry)?,
                cexp1_str = cexp1.to_yaml_string(state_data, table_registry)?,
                cexp2_str = cexp2.to_yaml_string(state_data, table_registry)?
            )),
            Self::ContinuousBinaryOperation(op, cexp1, cexp2) => Ok(format!(
                "({op_str} {cexp1_str} {cexp2_str})",
                op_str = op.to_yaml_string(state_data, table_registry)?,
                cexp1_str = cexp1.to_yaml_string(state_data, table_registry)?,
                cexp2_str = cexp2.to_yaml_string(state_data, table_registry)?
            )),
            Self::Cardinality(sexp) => Ok(format!(
                "|{sexp_str}|",
                sexp_str = sexp.to_yaml_string(state_data, table_registry)?
            )),
            Self::Table(texp) => texp.to_yaml_string(state_data, table_registry),
            Self::If(cond, cexp1, cexp2) => Ok(format!(
                "(if {cond_str} {cexp1_str} {cexp2_str})",
                cond_str = cond.to_yaml_string(state_data, table_registry)?,
                cexp1_str = cexp1.to_yaml_string(state_data, table_registry)?,
                cexp2_str = cexp2.to_yaml_string(state_data, table_registry)?
            )),
            Self::FromInteger(iexp) => iexp.to_yaml_string(state_data, table_registry),
            Self::Length(_) => Err("Current version doesn't support vector expressions"),
            Self::Last(_) => Err("Current version doesn't support vector expressions"),
            Self::At(_, _) => Err("Current version doesn't support vector expressions"),
            Self::Reduce(_, _) => Err("Current version doesn't support vector expressions"),
        }
    }
}

impl ToYamlString for CostExpression {
    fn to_yaml_string(
        &self,
        state_data: &StateMetadata,
        table_registry: &TableRegistry,
    ) -> Result<String, &'static str> {
        match self {
            Self::Integer(expr) => expr.to_yaml_string(state_data, table_registry),
            Self::Continuous(expr) => expr.to_yaml_string(state_data, table_registry),
        }
    }
}

impl ToYamlString for expression::ElementExpression {
    fn to_yaml_string(
        &self,
        state_data: &StateMetadata,
        table_registry: &TableRegistry,
    ) -> Result<String, &'static str> {
        match self {
            Self::Constant(value) => Ok(value.to_string()),
            Self::Variable(index) => Ok(state_data.element_variable_names[*index].clone()),
            Self::ResourceVariable(index) => {
                Ok(state_data.element_resource_variable_names[*index].clone())
            }
            Self::BinaryOperation(op, eexp1, eexp2) => Ok(format!(
                "({op_str} {eexp1_str} {eexp2_str})",
                op_str = op.to_yaml_string(state_data, table_registry)?,
                eexp1_str = eexp1.to_yaml_string(state_data, table_registry)?,
                eexp2_str = eexp2.to_yaml_string(state_data, table_registry)?
            )),
            Self::Table(texp) => texp.to_yaml_string(state_data, table_registry),
            Self::If(cond, eexp1, eexp2) => Ok(format!(
                "(if {cond_str} {eexp1_str} {eexp2_str})",
                cond_str = cond.to_yaml_string(state_data, table_registry)?,
                eexp1_str = eexp1.to_yaml_string(state_data, table_registry)?,
                eexp2_str = eexp2.to_yaml_string(state_data, table_registry)?
            )),
            Self::Last(_) => Err("Current version doesn't support vector expressions"),
            Self::At(_, _) => Err("Current version doesn't support vector expressions"),
        }
    }
}

impl ToYamlString for expression::SetExpression {
    fn to_yaml_string(
        &self,
        state_data: &StateMetadata,
        table_registry: &TableRegistry,
    ) -> Result<String, &'static str> {
        match self {
            Self::Reference(refer) => refer.to_yaml_string(state_data, table_registry),
            Self::Complement(sexp) => Ok(format!(
                "~{sexp_str}",
                sexp_str = sexp.to_yaml_string(state_data, table_registry)?
            )),
            Self::SetOperation(op, sexp1, sexp2) => Ok(format!(
                "({op_str} {sexp1_str} {sexp2_str})",
                op_str = op.to_yaml_string(state_data, table_registry)?,
                sexp1_str = sexp1.to_yaml_string(state_data, table_registry)?,
                sexp2_str = sexp2.to_yaml_string(state_data, table_registry)?
            )),
            Self::SetElementOperation(op, eexp, sexp) => Ok(format!(
                "({op_str} {eexp_str} {sexp_str})",
                op_str = op.to_yaml_string(state_data, table_registry)?,
                eexp_str = eexp.to_yaml_string(state_data, table_registry)?,
                sexp_str = sexp.to_yaml_string(state_data, table_registry)?
            )),
            Self::Reduce(srexp) => srexp.to_yaml_string(state_data, table_registry),
            Self::If(cond, sexp1, sexp2) => Ok(format!(
                "(if {cond_str} {sexp1_str} {sexp2_str})",
                cond_str = cond.to_yaml_string(state_data, table_registry)?,
                sexp1_str = sexp1.to_yaml_string(state_data, table_registry)?,
                sexp2_str = sexp2.to_yaml_string(state_data, table_registry)?
            )),
            Self::FromVector(_, _) => Err("Current version doesn't support vector expressions"),
        }
    }
}

fn find_key_for_value(
    map: &FxHashMap<String, usize>,
    value: &usize,
) -> Result<String, &'static str> {
    if let Some(name) = map.iter().find_map(|(key, val)| {
        if val == value {
            Some(key.clone())
        } else {
            None
        }
    }) {
        Ok(name)
    } else {
        Err("Unfound table index.")
    }
}

impl ToYamlString for expression::SetReduceExpression {
    fn to_yaml_string(
        &self,
        state_data: &StateMetadata,
        table_registry: &TableRegistry,
    ) -> Result<String, &'static str> {
        match self {
            Self::Constant(fbset) => Ok(fbset.to_variable_string()),
            Self::Table1D(op, _, index, argexp) => Ok(format!(
                "({op_str} {table_str} {argexp_str})",
                op_str = op.to_yaml_string(state_data, table_registry)?,
                table_str = find_key_for_value(&table_registry.set_tables.name_to_table_1d, index)?,
                argexp_str = argexp.to_yaml_string(state_data, table_registry)?
            )),
            Self::Table2D(op, _, index, argexp1, argexp2) => Ok(format!(
                "({op_str} {table_str} {argexp1_str} {argexp2_str})",
                op_str = op.to_yaml_string(state_data, table_registry)?,
                table_str = find_key_for_value(&table_registry.set_tables.name_to_table_2d, index)?,
                argexp1_str = argexp1.to_yaml_string(state_data, table_registry)?,
                argexp2_str = argexp2.to_yaml_string(state_data, table_registry)?
            )),
            Self::Table3D(op, _, index, argexp1, argexp2, argexp3) => Ok(format!(
                "({op_str} {table_str} {argexp1_str} {argexp2_str} {argexp3_str})",
                op_str = op.to_yaml_string(state_data, table_registry)?,
                table_str = find_key_for_value(&table_registry.set_tables.name_to_table_3d, index)?,
                argexp1_str = argexp1.to_yaml_string(state_data, table_registry)?,
                argexp2_str = argexp2.to_yaml_string(state_data, table_registry)?,
                argexp3_str = argexp3.to_yaml_string(state_data, table_registry)?
            )),
            Self::Table(op, _, index, argexp_vec) => Ok(format!(
                "({op_str} {table_str} {argexp_vec_str})",
                op_str = op.to_yaml_string(state_data, table_registry)?,
                table_str = find_key_for_value(&table_registry.set_tables.name_to_table, index)?,
                argexp_vec_str = argexp_vec
                    .iter()
                    .map(|argexp| argexp.to_yaml_string(state_data, table_registry).unwrap())
                    .collect::<Vec<String>>()
                    .join(" ")
            )),
        }
    }
}

impl ToYamlString for expression::ReferenceExpression<Set> {
    fn to_yaml_string(
        &self,
        state_data: &StateMetadata,
        table_registry: &TableRegistry,
    ) -> Result<String, &'static str> {
        match self {
            Self::Constant(set) => Ok(set.to_variable_string()),
            Self::Variable(index) => Ok(state_data.set_variable_names[*index].clone()),
            Self::Table(texp) => texp.to_yaml_string(state_data, table_registry),
        }
    }
}

impl ToYamlString for expression::ArgumentExpression {
    fn to_yaml_string(
        &self,
        state_data: &StateMetadata,
        table_registry: &TableRegistry,
    ) -> Result<String, &'static str> {
        match self {
            Self::Set(sexp) => sexp.to_yaml_string(state_data, table_registry),
            Self::Element(eexp) => eexp.to_yaml_string(state_data, table_registry),
            Self::Vector(_) => Err("Current version doesn't support vector expressions"),
        }
    }
}

macro_rules! define_table_exp_to_yaml {
    ($t:ty, $($field:ident).+) => {
        impl ToYamlString for expression::TableExpression<$t>{
            fn to_yaml_string(&self, state_data: &StateMetadata, table_registry: &TableRegistry) -> Result<String, &'static str> {
                let tables_in_model = &table_registry . $( $field ).+;
                match self{
                    Self::Constant(_) => Err("There should be no table presence as constant in the model."),
                    Self::Table1D(index, eexp) =>
                        Ok(format!("({table_str} {eexp_str})",
                            table_str=find_key_for_value(&tables_in_model.name_to_table_1d, index)?,
                            eexp_str=eexp.to_yaml_string(state_data, table_registry)?)),
                    Self::Table2D(index, eexp1, eexp2) =>
                        Ok(format!("({table_str} {eexp1_str} {eexp2_str})",
                            table_str=find_key_for_value(&tables_in_model.name_to_table_2d, index)?,
                            eexp1_str=eexp1.to_yaml_string(state_data, table_registry)?,
                            eexp2_str=eexp2.to_yaml_string(state_data, table_registry)?)),
                    Self::Table3D(index, eexp1, eexp2, eexp3) =>
                        Ok(format!("({table_str} {eexp1_str} {eexp2_str} {eexp3_str})",
                            table_str=find_key_for_value(&tables_in_model.name_to_table_3d, index)?,
                            eexp1_str=eexp1.to_yaml_string(state_data, table_registry)?,
                            eexp2_str=eexp2.to_yaml_string(state_data, table_registry)?,
                            eexp3_str=eexp3.to_yaml_string(state_data, table_registry)?)),
                    Self::Table(index, argexp_vec) =>
                        Ok(format!("({table_str} {argexp_vec_str})",
                            table_str=find_key_for_value(&tables_in_model.name_to_table, index)?,
                            argexp_vec_str=argexp_vec.iter()
                                        .map(|argexp| argexp.to_yaml_string(state_data, table_registry).unwrap())
                                        .collect::<Vec<String>>()
                                        .join(" "))),
                }
            }
        }
    };
}

define_table_exp_to_yaml!(Set, set_tables);
define_table_exp_to_yaml!(Element, element_tables);
define_table_exp_to_yaml!(Integer, integer_tables);
define_table_exp_to_yaml!(Continuous, continuous_tables);
define_table_exp_to_yaml!(bool, bool_tables);

macro_rules! define_numeric_table_exp_to_yaml {
    ($t:ty, $($field:ident).+) => {
        impl ToYamlString for expression::NumericTableExpression<$t>{
            fn to_yaml_string(&self, state_data: &StateMetadata, table_registry: &TableRegistry) -> Result<String, &'static str> {
                let tables_in_model = &table_registry . $( $field ).+;
                match self{
                    Self::Constant(value) => Ok(value.to_string()),
                    Self::Table(index, argexp_vec) =>
                        Ok(format!("({table_str} {argexp_vec_str})",
                            table_str=find_key_for_value(&tables_in_model.name_to_table, index)?,
                            argexp_vec_str=argexp_vec.iter()
                                        .map(|argexp| argexp.to_yaml_string(state_data, table_registry).unwrap())
                                        .collect::<Vec<String>>()
                                        .join(" "))),
                    Self::TableReduce(op, index, argexp_vec) =>
                        Ok(format!("({op_str} {table_str} {argexp_vec_str})",
                            op_str=op.to_yaml_string(state_data, table_registry)?,
                            table_str=find_key_for_value(&tables_in_model.name_to_table, index)?,
                            argexp_vec_str=argexp_vec.iter()
                                        .map(|argexp| argexp.to_yaml_string(state_data, table_registry).unwrap())
                                        .collect::<Vec<String>>()
                                        .join(" "))),

                    Self::Table1D(index, eexp) =>
                        Ok(format!("({table_str} {eexp_str})",
                            table_str=find_key_for_value(&tables_in_model.name_to_table_1d, index)?,
                            eexp_str=eexp.to_yaml_string(state_data, table_registry)?)),
                    Self::Table1DReduce(op, index, eexp) =>
                        Ok(format!("({op_str} {table_str} {eexp_str})",
                            op_str=op.to_yaml_string(state_data, table_registry)?,
                            table_str=find_key_for_value(&tables_in_model.name_to_table_1d, index)?,
                            eexp_str=eexp.to_yaml_string(state_data, table_registry)?)),

                    Self::Table2D(index, eexp1, eexp2) =>
                        Ok(format!("({table_str} {eexp1_str} {eexp2_str})",
                            table_str=find_key_for_value(&tables_in_model.name_to_table_2d, index)?,
                            eexp1_str=eexp1.to_yaml_string(state_data, table_registry)?,
                            eexp2_str=eexp2.to_yaml_string(state_data, table_registry)?)),
                    Self::Table2DReduce(op, index, eexp1, eexp2) =>
                        Ok(format!("({op_str} {table_str} {eexp1_str} {eexp2_str})",
                            op_str=op.to_yaml_string(state_data, table_registry)?,
                            table_str=find_key_for_value(&tables_in_model.name_to_table_2d, index)?,
                            eexp1_str=eexp1.to_yaml_string(state_data, table_registry)?,
                            eexp2_str=eexp2.to_yaml_string(state_data, table_registry)?)),
                    Self::Table2DReduceX(op, index, sexp, eexp) =>
                        Ok(format!("({op_str} {table_str} {sexp_str} {eexp_str})",
                            op_str=op.to_yaml_string(state_data, table_registry)?,
                            table_str=find_key_for_value(&tables_in_model.name_to_table_2d, index)?,
                            sexp_str=sexp.to_yaml_string(state_data, table_registry)?,
                            eexp_str=eexp.to_yaml_string(state_data, table_registry)?)),
                    Self::Table2DReduceY(op, index, eexp, sexp) =>
                        Ok(format!("({op_str} {table_str} {eexp_str} {sexp_str})",
                            op_str=op.to_yaml_string(state_data, table_registry)?,
                            table_str=find_key_for_value(&tables_in_model.name_to_table_2d, index)?,
                            eexp_str=eexp.to_yaml_string(state_data, table_registry)?,
                            sexp_str=sexp.to_yaml_string(state_data, table_registry)?)),

                    Self::Table3D(index, eexp1, eexp2, eexp3) =>
                        Ok(format!("({table_str} {eexp1_str} {eexp2_str} {eexp3_str})",
                            table_str=find_key_for_value(&tables_in_model.name_to_table_3d, index)?,
                            eexp1_str=eexp1.to_yaml_string(state_data, table_registry)?,
                            eexp2_str=eexp2.to_yaml_string(state_data, table_registry)?,
                            eexp3_str=eexp3.to_yaml_string(state_data, table_registry)?)),
                    Self::Table3DReduce(op, index, eexp1, eexp2, eexp3) =>
                        Ok(format!("({op_str} {table_str} {eexp1_str} {eexp2_str} {eexp3_str})",
                            op_str=op.to_yaml_string(state_data, table_registry)?,
                            table_str=find_key_for_value(&tables_in_model.name_to_table_3d, index)?,
                            eexp1_str=eexp1.to_yaml_string(state_data, table_registry)?,
                            eexp2_str=eexp2.to_yaml_string(state_data, table_registry)?,
                            eexp3_str=eexp3.to_yaml_string(state_data, table_registry)?)),
                    _ => Err("Current version doesn't support vector expressions"),
                }
            }
        }
    };
}

define_numeric_table_exp_to_yaml!(Integer, integer_tables);
define_numeric_table_exp_to_yaml!(Continuous, continuous_tables);

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;
    use crate::dypdl_parser::expression_parser::*;
    use dypdl::{Table, Table1D, Table2D, Table3D, TableData};

    use rustc_hash::FxHashMap;

    fn generate_metadata() -> dypdl::StateMetadata {
        let object_names = vec!["object".to_string()];
        let object_numbers = vec![10];
        let mut name_to_object = FxHashMap::default();
        name_to_object.insert("object".to_string(), 0);

        let set_variable_names = vec![
            "s0".to_string(),
            "s1".to_string(),
            "s2".to_string(),
            "s3".to_string(),
        ];
        let mut name_to_set_variable = FxHashMap::default();
        name_to_set_variable.insert("s0".to_string(), 0);
        name_to_set_variable.insert("s1".to_string(), 1);
        name_to_set_variable.insert("s2".to_string(), 2);
        name_to_set_variable.insert("s3".to_string(), 3);
        let set_variable_to_object = vec![0, 0, 0, 0];

        let vector_variable_names = vec![
            "p0".to_string(),
            "p1".to_string(),
            "p2".to_string(),
            "p3".to_string(),
        ];
        let mut name_to_vector_variable = FxHashMap::default();
        name_to_vector_variable.insert("p0".to_string(), 0);
        name_to_vector_variable.insert("p1".to_string(), 1);
        name_to_vector_variable.insert("p2".to_string(), 2);
        name_to_vector_variable.insert("p3".to_string(), 3);
        let vector_variable_to_object = vec![0, 0, 0, 0];

        let element_variable_names = vec![
            "e0".to_string(),
            "e1".to_string(),
            "e2".to_string(),
            "e3".to_string(),
        ];
        let mut name_to_element_variable = FxHashMap::default();
        name_to_element_variable.insert("e0".to_string(), 0);
        name_to_element_variable.insert("e1".to_string(), 1);
        name_to_element_variable.insert("e2".to_string(), 2);
        name_to_element_variable.insert("e3".to_string(), 3);
        let element_variable_to_object = vec![0, 0, 0, 0];

        let integer_variable_names = vec![
            "n0".to_string(),
            "n1".to_string(),
            "n2".to_string(),
            "n3".to_string(),
        ];
        let mut name_to_integer_variable = FxHashMap::default();
        name_to_integer_variable.insert("n0".to_string(), 0);
        name_to_integer_variable.insert("n1".to_string(), 1);
        name_to_integer_variable.insert("n2".to_string(), 2);
        name_to_integer_variable.insert("n3".to_string(), 3);

        let integer_resource_variable_names = vec![
            "r0".to_string(),
            "r1".to_string(),
            "r2".to_string(),
            "r3".to_string(),
        ];
        let mut name_to_integer_resource_variable = FxHashMap::default();
        name_to_integer_resource_variable.insert("r0".to_string(), 0);
        name_to_integer_resource_variable.insert("r1".to_string(), 1);
        name_to_integer_resource_variable.insert("r2".to_string(), 2);
        name_to_integer_resource_variable.insert("r3".to_string(), 3);

        let continuous_variable_names = vec![
            "c0".to_string(),
            "c1".to_string(),
            "c2".to_string(),
            "c3".to_string(),
        ];
        let mut name_to_continuous_variable = FxHashMap::default();
        name_to_continuous_variable.insert("c0".to_string(), 0);
        name_to_continuous_variable.insert("c1".to_string(), 1);
        name_to_continuous_variable.insert("c2".to_string(), 2);
        name_to_continuous_variable.insert("c3".to_string(), 3);

        dypdl::StateMetadata {
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
            integer_resource_variable_names,
            name_to_integer_resource_variable,
            integer_less_is_better: vec![false, false, true, false],
            continuous_variable_names,
            name_to_continuous_variable,
            ..Default::default()
        }
    }

    fn generate_registry() -> dypdl::TableRegistry {
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

        dypdl::TableRegistry {
            integer_tables: dypdl::TableData {
                name_to_constant,
                tables_1d,
                name_to_table_1d,
                tables_2d,
                name_to_table_2d,
                tables_3d,
                name_to_table_3d,
                tables,
                name_to_table,
            },
            ..Default::default()
        }
    }

    fn generate_parameters() -> FxHashMap<String, usize> {
        let mut parameters = FxHashMap::default();
        parameters.insert("param".to_string(), 0);
        parameters
    }

    #[test]
    fn unary_operator_to_yaml_string() {
        let op = expression::UnaryOperator::Abs;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "abs".to_owned()
        );

        let op = expression::UnaryOperator::Neg;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "neg".to_owned()
        );
    }

    #[test]
    fn continuous_unary_operator_to_yaml_string() {
        let op = expression::ContinuousUnaryOperator::Sqrt;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "sqrt".to_owned()
        );
    }

    #[test]
    fn binary_operator_to_yaml_string() {
        let op = expression::BinaryOperator::Add;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "+".to_owned()
        );

        let op = expression::BinaryOperator::Sub;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "-".to_owned()
        );

        let op = expression::BinaryOperator::Mul;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "*".to_owned()
        );

        let op = expression::BinaryOperator::Div;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "/".to_owned()
        );

        let op = expression::BinaryOperator::Rem;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "%".to_owned()
        );

        let op = expression::BinaryOperator::Max;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "max".to_owned()
        );

        let op = expression::BinaryOperator::Min;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "min".to_owned()
        );
    }

    #[test]
    fn continuous_binary_operator_to_yaml_string() {
        let op = expression::ContinuousBinaryOperator::Pow;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "pow".to_owned()
        );

        let op = expression::ContinuousBinaryOperator::Log;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "log".to_owned()
        );
    }

    #[test]
    fn set_operator_to_yaml_string() {
        let op = expression::SetOperator::Union;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "union".to_owned()
        );

        let op = expression::SetOperator::Difference;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "difference".to_owned()
        );

        let op = expression::SetOperator::Intersection;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "intersection".to_owned()
        );
    }

    #[test]
    fn set_element_operator_to_yaml_string() {
        let op = expression::SetElementOperator::Add;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "add".to_owned()
        );

        let op = expression::SetElementOperator::Remove;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "remove".to_owned()
        );
    }

    #[test]
    fn set_reduce_operator_to_yaml_string() {
        let op = expression::SetReduceOperator::Union;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "union".to_owned()
        );

        let op = expression::SetReduceOperator::Intersection;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "intersection".to_owned()
        );

        let op = expression::SetReduceOperator::SymmetricDifference;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "disjunctive_union".to_owned()
        );
    }

    #[test]
    fn reduce_operator_to_yaml_string() {
        let op = expression::ReduceOperator::Sum;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "sum".to_owned()
        );

        let op = expression::ReduceOperator::Product;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "product".to_owned()
        );

        let op = expression::ReduceOperator::Max;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "max".to_owned()
        );

        let op = expression::ReduceOperator::Min;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "min".to_owned()
        );
    }

    #[test]
    fn cast_operator_to_yaml_string() {
        let op = expression::CastOperator::Floor;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "floor".to_owned()
        );

        let op = expression::CastOperator::Ceil;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "ceil".to_owned()
        );

        let op = expression::CastOperator::Round;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "round".to_owned()
        );

        let op = expression::CastOperator::Trunc;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "trunc".to_owned()
        );
    }

    #[test]
    fn comparison_operator_to_yaml_string() {
        let op = expression::ComparisonOperator::Eq;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "=".to_owned()
        );

        let op = expression::ComparisonOperator::Ne;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "!=".to_owned()
        );

        let op = expression::ComparisonOperator::Ge;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            ">=".to_owned()
        );

        let op = expression::ComparisonOperator::Gt;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            ">".to_owned()
        );

        let op = expression::ComparisonOperator::Le;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "<=".to_owned()
        );

        let op = expression::ComparisonOperator::Lt;
        assert_eq!(
            op.to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "<".to_owned()
        );
    }

    #[test]
    fn condition_to_yaml_string() {
        let true_base_condition = expression::Condition::Constant(true);
        assert_eq!(
            true_base_condition
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "true".to_owned()
        );

        let condition = expression::Condition::Not(true_base_condition.clone().into());
        assert_eq!(
            condition
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "(not true)".to_owned()
        );

        let condition = expression::Condition::And(
            true_base_condition.clone().into(),
            true_base_condition.clone().into(),
        );
        assert_eq!(
            condition
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "(and true true)".to_owned()
        );

        let condition = expression::Condition::Or(
            true_base_condition.clone().into(),
            true_base_condition.clone().into(),
        );
        assert_eq!(
            condition
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "(or true true)".to_owned()
        );

        let condition = expression::Condition::ComparisonE(
            expression::ComparisonOperator::Eq,
            expression::ElementExpression::Constant(0).into(),
            expression::ElementExpression::Constant(0).into(),
        );
        assert_eq!(
            condition
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "(= 0 0)".to_owned()
        );

        let condition = expression::Condition::ComparisonI(
            expression::ComparisonOperator::Eq,
            expression::IntegerExpression::Constant(0).into(),
            expression::IntegerExpression::Constant(0).into(),
        );
        assert_eq!(
            condition
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "(= 0 0)".to_owned()
        );

        let condition = expression::Condition::ComparisonC(
            expression::ComparisonOperator::Eq,
            expression::ContinuousExpression::Constant(0.0).into(),
            expression::ContinuousExpression::Constant(0.0).into(),
        );
        assert_eq!(
            condition
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "(= 0 0)".to_owned()
        );
    }

    #[test]
    fn set_condition_to_yaml_string() {
        let true_base_condition = expression::SetCondition::Constant(true);
        assert_eq!(
            true_base_condition
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "true".to_owned()
        );

        let condition = expression::SetCondition::IsEqual(
            expression::SetExpression::Reduce(expression::SetReduceExpression::Constant(
                Set::default(),
            )),
            expression::SetExpression::Reduce(expression::SetReduceExpression::Constant(
                Set::default(),
            )),
        );
        assert_eq!(
            condition
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "(= { : 0} { : 0})".to_owned()
        );

        let condition = expression::SetCondition::IsNotEqual(
            expression::SetExpression::Reduce(expression::SetReduceExpression::Constant(
                Set::default(),
            )),
            expression::SetExpression::Reduce(expression::SetReduceExpression::Constant(
                Set::default(),
            )),
        );
        assert_eq!(
            condition
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "(!= { : 0} { : 0})".to_owned()
        );

        let condition = expression::SetCondition::IsIn(
            expression::ElementExpression::Constant(0),
            expression::SetExpression::Reduce(expression::SetReduceExpression::Constant(
                Set::default(),
            )),
        );
        assert_eq!(
            condition
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "(is_in 0 { : 0})".to_owned()
        );

        let condition = expression::SetCondition::IsSubset(
            expression::SetExpression::Reduce(expression::SetReduceExpression::Constant(
                Set::default(),
            )),
            expression::SetExpression::Reduce(expression::SetReduceExpression::Constant(
                Set::default(),
            )),
        );
        assert_eq!(
            condition
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "(is_subset { : 0} { : 0})".to_owned()
        );

        let condition = expression::SetCondition::IsEmpty(expression::SetExpression::Reduce(
            expression::SetReduceExpression::Constant(Set::default()),
        ));
        assert_eq!(
            condition
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "(is_empty { : 0})".to_owned()
        );
    }

    #[test]
    fn integer_expression_to_yaml_string() {
        let base_integer_expression = expression::IntegerExpression::Constant(0);
        assert_eq!(
            base_integer_expression
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "0".to_owned()
        );

        let integer_expression = expression::IntegerExpression::Variable(0);
        assert_eq!(
            integer_expression
                .to_yaml_string(
                    &StateMetadata {
                        integer_variable_names: vec!["i0".to_owned()],
                        ..Default::default()
                    },
                    &TableRegistry::default()
                )
                .unwrap(),
            "i0".to_owned()
        );

        let integer_expression = expression::IntegerExpression::ResourceVariable(0);
        assert_eq!(
            integer_expression
                .to_yaml_string(
                    &StateMetadata {
                        integer_resource_variable_names: vec!["ir0".to_owned()],
                        ..Default::default()
                    },
                    &TableRegistry::default()
                )
                .unwrap(),
            "ir0".to_owned()
        );

        let integer_expression = expression::IntegerExpression::Cost;
        assert_eq!(
            integer_expression
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "cost".to_owned()
        );

        let integer_expression = expression::IntegerExpression::UnaryOperation(
            expression::UnaryOperator::Abs,
            base_integer_expression.clone().into(),
        );
        assert_eq!(
            integer_expression
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "(abs 0)".to_owned()
        );

        let integer_expression = expression::IntegerExpression::BinaryOperation(
            expression::BinaryOperator::Add,
            base_integer_expression.clone().into(),
            base_integer_expression.clone().into(),
        );
        assert_eq!(
            integer_expression
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "(+ 0 0)".to_owned()
        );

        let integer_expression =
            expression::IntegerExpression::Cardinality(expression::SetExpression::Reduce(
                expression::SetReduceExpression::Constant(Set::default()),
            ));
        assert_eq!(
            integer_expression
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "|{ : 0}|".to_owned()
        );

        let integer_expression = expression::IntegerExpression::If(
            expression::Condition::Constant(true).into(),
            base_integer_expression.clone().into(),
            base_integer_expression.clone().into(),
        );
        assert_eq!(
            integer_expression
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "(if true 0 0)".to_owned()
        );

        let integer_expression = expression::IntegerExpression::FromContinuous(
            expression::CastOperator::Ceil,
            expression::ContinuousExpression::Constant(3.3).into(),
        );
        assert_eq!(
            integer_expression
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "(ceil 3.3)".to_owned()
        );
    }

    #[test]
    fn continuous_expression_to_yaml_string() {
        let base_continuous_expression = expression::ContinuousExpression::Constant(0.1);
        assert_eq!(
            base_continuous_expression
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "0.1".to_owned()
        );

        let continuous_expression = expression::ContinuousExpression::Variable(0);
        assert_eq!(
            continuous_expression
                .to_yaml_string(
                    &StateMetadata {
                        continuous_variable_names: vec!["c0".to_owned()],
                        ..Default::default()
                    },
                    &TableRegistry::default()
                )
                .unwrap(),
            "c0".to_owned()
        );

        let continuous_expression = expression::ContinuousExpression::ResourceVariable(0);
        assert_eq!(
            continuous_expression
                .to_yaml_string(
                    &StateMetadata {
                        continuous_resource_variable_names: vec!["cr0".to_owned()],
                        ..Default::default()
                    },
                    &TableRegistry::default()
                )
                .unwrap(),
            "cr0".to_owned()
        );

        let continuous_expression = expression::ContinuousExpression::UnaryOperation(
            expression::UnaryOperator::Abs,
            base_continuous_expression.clone().into(),
        );
        assert_eq!(
            continuous_expression
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "(abs 0.1)".to_owned()
        );

        let continuous_expression = expression::ContinuousExpression::ContinuousUnaryOperation(
            expression::ContinuousUnaryOperator::Sqrt,
            base_continuous_expression.clone().into(),
        );
        assert_eq!(
            continuous_expression
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "(sqrt 0.1)".to_owned()
        );

        let continuous_expression: dypdl::prelude::ContinuousExpression =
            expression::ContinuousExpression::Round(
                expression::CastOperator::Ceil,
                base_continuous_expression.clone().into(),
            );
        assert_eq!(
            continuous_expression
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "(ceil 0.1)".to_owned()
        );

        let continuous_expression = expression::ContinuousExpression::BinaryOperation(
            expression::BinaryOperator::Add,
            base_continuous_expression.clone().into(),
            base_continuous_expression.clone().into(),
        );
        assert_eq!(
            continuous_expression
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "(+ 0.1 0.1)".to_owned()
        );

        let continuous_expression = expression::ContinuousExpression::ContinuousBinaryOperation(
            expression::ContinuousBinaryOperator::Pow,
            base_continuous_expression.clone().into(),
            base_continuous_expression.clone().into(),
        );
        assert_eq!(
            continuous_expression
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "(pow 0.1 0.1)".to_owned()
        );

        let continuous_expression =
            expression::ContinuousExpression::Cardinality(expression::SetExpression::Reduce(
                expression::SetReduceExpression::Constant(Set::default()),
            ));
        assert_eq!(
            continuous_expression
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "|{ : 0}|".to_owned()
        );

        let continuous_expression = expression::ContinuousExpression::If(
            expression::Condition::Constant(true).into(),
            base_continuous_expression.clone().into(),
            base_continuous_expression.clone().into(),
        );
        assert_eq!(
            continuous_expression
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "(if true 0.1 0.1)".to_owned()
        );
    }

    #[test]
    fn cost_expression_to_yaml_string() {
        let cost_expression = CostExpression::Integer(expression::IntegerExpression::Constant(0));
        assert_eq!(
            cost_expression
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "0".to_owned()
        );

        let cost_expression =
            CostExpression::Continuous(expression::ContinuousExpression::Constant(0.1));
        assert_eq!(
            cost_expression
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "0.1".to_owned()
        );
    }

    #[test]
    fn element_expression_to_yaml_string() {
        let base_element_expression = expression::ElementExpression::Constant(0);
        assert_eq!(
            base_element_expression
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "0".to_owned()
        );

        let element_expression = expression::ElementExpression::Variable(0);
        assert_eq!(
            element_expression
                .to_yaml_string(
                    &StateMetadata {
                        element_variable_names: vec!["e0".to_owned()],
                        ..Default::default()
                    },
                    &TableRegistry::default()
                )
                .unwrap(),
            "e0".to_owned()
        );

        let element_expression = expression::ElementExpression::ResourceVariable(0);
        assert_eq!(
            element_expression
                .to_yaml_string(
                    &StateMetadata {
                        element_resource_variable_names: vec!["er0".to_owned()],
                        ..Default::default()
                    },
                    &TableRegistry::default()
                )
                .unwrap(),
            "er0".to_owned()
        );

        let element_expression = expression::ElementExpression::BinaryOperation(
            expression::BinaryOperator::Add,
            base_element_expression.clone().into(),
            base_element_expression.clone().into(),
        );
        assert_eq!(
            element_expression
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "(+ 0 0)".to_owned()
        );

        let element_expression = expression::ElementExpression::If(
            expression::Condition::Constant(true).into(),
            base_element_expression.clone().into(),
            base_element_expression.clone().into(),
        );
        assert_eq!(
            element_expression
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "(if true 0 0)".to_owned()
        );
    }

    #[test]
    fn set_expression_to_yaml_string() {
        let base_set_expression = expression::SetExpression::Reference(
            expression::ReferenceExpression::Constant(Set::new()),
        );
        assert_eq!(
            base_set_expression
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "{ : 0}".to_owned()
        );

        let set_expression =
            expression::SetExpression::Complement(base_set_expression.clone().into());
        assert_eq!(
            set_expression
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "~{ : 0}".to_owned()
        );

        let set_expression = expression::SetExpression::SetOperation(
            expression::SetOperator::Difference,
            base_set_expression.clone().into(),
            base_set_expression.clone().into(),
        );
        assert_eq!(
            set_expression
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "(difference { : 0} { : 0})".to_owned()
        );

        let set_expression = expression::SetExpression::SetElementOperation(
            expression::SetElementOperator::Add,
            expression::ElementExpression::Constant(0),
            base_set_expression.clone().into(),
        );
        assert_eq!(
            set_expression
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "(add 0 { : 0})".to_owned()
        );

        let set_expression = expression::SetExpression::Reduce(
            expression::SetReduceExpression::Constant(Set::new()),
        );
        assert_eq!(
            set_expression
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "{ : 0}".to_owned()
        );

        let set_expression = expression::SetExpression::If(
            expression::Condition::Constant(true).into(),
            base_set_expression.clone().into(),
            base_set_expression.clone().into(),
        );
        assert_eq!(
            set_expression
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "(if true { : 0} { : 0})".to_owned()
        );
    }

    #[test]
    fn set_reduce_expression_to_yaml_string() {
        let mut table_names = FxHashMap::default();
        table_names.insert("st0".to_owned(), 0);
        let base_set_reduce_expression = expression::SetReduceExpression::Constant(Set::default());
        assert_eq!(
            base_set_reduce_expression
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "{ : 0}".to_owned()
        );

        let set_reduce_expression = expression::SetReduceExpression::Table1D(
            expression::SetReduceOperator::Intersection,
            0,
            0,
            expression::ArgumentExpression::Element(expression::ElementExpression::Constant(0))
                .into(),
        );
        assert_eq!(
            set_reduce_expression
                .to_yaml_string(
                    &StateMetadata::default(),
                    &TableRegistry {
                        set_tables: TableData {
                            name_to_table_1d: table_names.clone(),
                            ..Default::default()
                        },
                        ..Default::default()
                    }
                )
                .unwrap(),
            "(intersection st0 0)".to_owned()
        );

        let set_reduce_expression = expression::SetReduceExpression::Table2D(
            expression::SetReduceOperator::Intersection,
            0,
            0,
            expression::ArgumentExpression::Element(expression::ElementExpression::Constant(0))
                .into(),
            expression::ArgumentExpression::Element(expression::ElementExpression::Constant(0))
                .into(),
        );
        assert_eq!(
            set_reduce_expression
                .to_yaml_string(
                    &StateMetadata::default(),
                    &TableRegistry {
                        set_tables: TableData {
                            name_to_table_2d: table_names.clone(),
                            ..Default::default()
                        },
                        ..Default::default()
                    }
                )
                .unwrap(),
            "(intersection st0 0 0)".to_owned()
        );

        let set_reduce_expression = expression::SetReduceExpression::Table3D(
            expression::SetReduceOperator::Intersection,
            0,
            0,
            expression::ArgumentExpression::Element(expression::ElementExpression::Constant(0))
                .into(),
            expression::ArgumentExpression::Element(expression::ElementExpression::Constant(0))
                .into(),
            expression::ArgumentExpression::Element(expression::ElementExpression::Constant(0))
                .into(),
        );
        assert_eq!(
            set_reduce_expression
                .to_yaml_string(
                    &StateMetadata::default(),
                    &TableRegistry {
                        set_tables: TableData {
                            name_to_table_3d: table_names.clone(),
                            ..Default::default()
                        },
                        ..Default::default()
                    }
                )
                .unwrap(),
            "(intersection st0 0 0 0)".to_owned()
        );

        let set_reduce_expression = expression::SetReduceExpression::Table(
            expression::SetReduceOperator::Intersection,
            0,
            0,
            vec![
                expression::ArgumentExpression::Element(expression::ElementExpression::Constant(0)),
                expression::ArgumentExpression::Element(expression::ElementExpression::Constant(0)),
                expression::ArgumentExpression::Element(expression::ElementExpression::Constant(0)),
                expression::ArgumentExpression::Element(expression::ElementExpression::Constant(0)),
            ],
        );
        assert_eq!(
            set_reduce_expression
                .to_yaml_string(
                    &StateMetadata::default(),
                    &TableRegistry {
                        set_tables: TableData {
                            name_to_table: table_names.clone(),
                            ..Default::default()
                        },
                        ..Default::default()
                    }
                )
                .unwrap(),
            "(intersection st0 0 0 0 0)".to_owned()
        );
    }

    #[test]
    fn reference_expression_to_yaml_string() {
        let mut table_names = FxHashMap::default();
        table_names.insert("st0".to_owned(), 0);

        let reference_expression = expression::ReferenceExpression::Constant(Set::default());
        assert_eq!(
            reference_expression
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "{ : 0}".to_owned()
        );

        let reference_expression = expression::ReferenceExpression::Variable(0);
        assert_eq!(
            reference_expression
                .to_yaml_string(
                    &StateMetadata {
                        set_variable_names: vec!["s0".to_owned()],
                        ..Default::default()
                    },
                    &TableRegistry::default()
                )
                .unwrap(),
            "s0".to_owned()
        );

        let reference_expression = expression::ReferenceExpression::Table(
            expression::TableExpression::Table1D(0, expression::ElementExpression::Constant(0)),
        );
        assert_eq!(
            reference_expression
                .to_yaml_string(
                    &StateMetadata::default(),
                    &TableRegistry {
                        set_tables: TableData {
                            name_to_table_1d: table_names,
                            ..Default::default()
                        },
                        ..Default::default()
                    }
                )
                .unwrap(),
            "(st0 0)".to_owned()
        );
    }

    #[test]
    fn argument_expression_to_yaml_string() {
        let argument_expression =
            expression::ArgumentExpression::Set(expression::SetExpression::Reference(
                expression::ReferenceExpression::Constant(Set::default()),
            ));
        assert_eq!(
            argument_expression
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "{ : 0}".to_owned()
        );

        let argument_expression =
            expression::ArgumentExpression::Element(expression::ElementExpression::Constant(0));
        assert_eq!(
            argument_expression
                .to_yaml_string(&StateMetadata::default(), &TableRegistry::default())
                .unwrap(),
            "0".to_owned()
        );
    }

    macro_rules! make_table_expression_tests {
        ($ok_test_name:ident, $err_test_name:ident, $field_name:ident, $element_type:ty) => {
            #[test]
            fn $ok_test_name() {
                let tables_1d = vec![Table1D::new(Vec::new())];
                let mut name_to_table_1d = FxHashMap::default();
                name_to_table_1d.insert(String::from("f1"), 0);

                let tables_2d = vec![Table2D::new(Vec::new())];
                let mut name_to_table_2d = FxHashMap::default();
                name_to_table_2d.insert(String::from("f2"), 0);

                let tables_3d = vec![Table3D::new(Vec::new())];
                let mut name_to_table_3d = FxHashMap::default();
                name_to_table_3d.insert(String::from("f3"), 0);

                let tables = vec![Table::new(FxHashMap::default(), <$element_type>::default())];
                let mut name_to_table = FxHashMap::default();
                name_to_table.insert(String::from("f4"), 0);

                let table_registry = dypdl::TableRegistry {
                    $field_name: dypdl::TableData {
                        tables_1d,
                        name_to_table_1d,
                        tables_2d,
                        name_to_table_2d,
                        tables_3d,
                        name_to_table_3d,
                        tables,
                        name_to_table,
                        ..Default::default()
                    },
                    ..Default::default()
                };

                let table_expression = expression::TableExpression::<$element_type>::Table1D(
                    0,
                    expression::ElementExpression::Constant(0),
                );
                let result =
                    table_expression.to_yaml_string(&StateMetadata::default(), &table_registry);
                assert!(result.is_ok());
                assert_eq!(result.unwrap(), "(f1 0)".to_owned());

                let table_expression = expression::TableExpression::<$element_type>::Table2D(
                    0,
                    expression::ElementExpression::Constant(0),
                    expression::ElementExpression::Constant(0),
                );
                let result =
                    table_expression.to_yaml_string(&StateMetadata::default(), &table_registry);
                assert!(result.is_ok());
                assert_eq!(result.unwrap(), "(f2 0 0)".to_owned());

                let table_expression = expression::TableExpression::<$element_type>::Table3D(
                    0,
                    expression::ElementExpression::Constant(0),
                    expression::ElementExpression::Constant(0),
                    expression::ElementExpression::Constant(0),
                );
                let result =
                    table_expression.to_yaml_string(&StateMetadata::default(), &table_registry);
                assert!(result.is_ok());
                assert_eq!(result.unwrap(), "(f3 0 0 0)".to_owned());

                let table_expression = expression::TableExpression::<$element_type>::Table(
                    0,
                    vec![
                        expression::ElementExpression::Constant(0),
                        expression::ElementExpression::Constant(0),
                        expression::ElementExpression::Constant(0),
                        expression::ElementExpression::Constant(0),
                    ],
                );
                let result =
                    table_expression.to_yaml_string(&StateMetadata::default(), &table_registry);
                assert!(result.is_ok());
                assert_eq!(result.unwrap(), "(f4 0 0 0 0)".to_owned());
            }

            #[test]
            fn $err_test_name() {
                let table_expression = expression::TableExpression::<$element_type>::Constant(
                    <$element_type>::default(),
                );
                let result = table_expression
                    .to_yaml_string(&StateMetadata::default(), &TableRegistry::default());
                assert!(result.is_err());
            }
        };
    }

    make_table_expression_tests!(
        integer_table_expression_to_yaml_string_ok,
        integer_table_expression_to_yaml_string_err,
        integer_tables,
        Integer
    );
    make_table_expression_tests!(
        continuous_table_expression_to_yaml_string_ok,
        continuous_table_expression_to_yaml_string_err,
        continuous_tables,
        Continuous
    );
    make_table_expression_tests!(
        element_table_expression_to_yaml_string_ok,
        element_table_expression_to_yaml_string_err,
        element_tables,
        Element
    );
    make_table_expression_tests!(
        set_table_expression_to_yaml_string_ok,
        set_table_expression_to_yaml_string_err,
        set_tables,
        Set
    );

    macro_rules! make_numeric_table_expression_tests {
        ($ok_test_name:ident, $field_name:ident, $element_type:ty) => {
            #[test]
            fn $ok_test_name() {
                let tables_1d = vec![Table1D::new(Vec::new())];
                let mut name_to_table_1d = FxHashMap::default();
                name_to_table_1d.insert(String::from("f1"), 0);

                let tables_2d = vec![Table2D::new(Vec::new())];
                let mut name_to_table_2d = FxHashMap::default();
                name_to_table_2d.insert(String::from("f2"), 0);

                let tables_3d = vec![Table3D::new(Vec::new())];
                let mut name_to_table_3d = FxHashMap::default();
                name_to_table_3d.insert(String::from("f3"), 0);

                let tables = vec![Table::new(FxHashMap::default(), <$element_type>::default())];
                let mut name_to_table = FxHashMap::default();
                name_to_table.insert(String::from("f4"), 0);

                let table_registry = dypdl::TableRegistry {
                    $field_name: dypdl::TableData {
                        tables_1d,
                        name_to_table_1d,
                        tables_2d,
                        name_to_table_2d,
                        tables_3d,
                        name_to_table_3d,
                        tables,
                        name_to_table,
                        ..Default::default()
                    },
                    ..Default::default()
                };

                let table_expression =
                    expression::NumericTableExpression::<$element_type>::Constant(
                        0 as $element_type,
                    );
                let result =
                    table_expression.to_yaml_string(&StateMetadata::default(), &table_registry);
                assert!(result.is_ok());
                assert_eq!(result.unwrap(), "0".to_owned());

                let table_expression = expression::NumericTableExpression::<$element_type>::Table1D(
                    0,
                    expression::ElementExpression::Constant(0),
                );
                let result =
                    table_expression.to_yaml_string(&StateMetadata::default(), &table_registry);
                assert!(result.is_ok());
                assert_eq!(result.unwrap(), "(f1 0)".to_owned());

                let table_expression =
                    expression::NumericTableExpression::<$element_type>::Table1DReduce(
                        expression::ReduceOperator::Sum,
                        0,
                        expression::SetExpression::Reference(
                            expression::ReferenceExpression::Constant(Set::default()),
                        ),
                    );
                let result =
                    table_expression.to_yaml_string(&StateMetadata::default(), &table_registry);
                assert!(result.is_ok());
                assert_eq!(result.unwrap(), "(sum f1 { : 0})".to_owned());

                let table_expression = expression::NumericTableExpression::<$element_type>::Table2D(
                    0,
                    expression::ElementExpression::Constant(0),
                    expression::ElementExpression::Constant(0),
                );
                let result =
                    table_expression.to_yaml_string(&StateMetadata::default(), &table_registry);
                assert!(result.is_ok());
                assert_eq!(result.unwrap(), "(f2 0 0)".to_owned());

                let table_expression =
                    expression::NumericTableExpression::<$element_type>::Table2DReduce(
                        expression::ReduceOperator::Sum,
                        0,
                        expression::SetExpression::Reference(
                            expression::ReferenceExpression::Constant(Set::default()),
                        ),
                        expression::SetExpression::Reference(
                            expression::ReferenceExpression::Constant(Set::default()),
                        ),
                    );
                let result =
                    table_expression.to_yaml_string(&StateMetadata::default(), &table_registry);
                assert!(result.is_ok());
                assert_eq!(result.unwrap(), "(sum f2 { : 0} { : 0})".to_owned());

                let table_expression =
                    expression::NumericTableExpression::<$element_type>::Table2DReduceX(
                        expression::ReduceOperator::Sum,
                        0,
                        expression::SetExpression::Reference(
                            expression::ReferenceExpression::Constant(Set::default()),
                        ),
                        expression::ElementExpression::Constant(0),
                    );
                let result =
                    table_expression.to_yaml_string(&StateMetadata::default(), &table_registry);
                assert!(result.is_ok());
                assert_eq!(result.unwrap(), "(sum f2 { : 0} 0)".to_owned());

                let table_expression =
                    expression::NumericTableExpression::<$element_type>::Table2DReduceY(
                        expression::ReduceOperator::Sum,
                        0,
                        expression::ElementExpression::Constant(0),
                        expression::SetExpression::Reference(
                            expression::ReferenceExpression::Constant(Set::default()),
                        ),
                    );
                let result =
                    table_expression.to_yaml_string(&StateMetadata::default(), &table_registry);
                assert!(result.is_ok());
                assert_eq!(result.unwrap(), "(sum f2 0 { : 0})".to_owned());

                let table_expression = expression::NumericTableExpression::<$element_type>::Table3D(
                    0,
                    expression::ElementExpression::Constant(0),
                    expression::ElementExpression::Constant(0),
                    expression::ElementExpression::Constant(0),
                );
                let result =
                    table_expression.to_yaml_string(&StateMetadata::default(), &table_registry);
                assert!(result.is_ok());
                assert_eq!(result.unwrap(), "(f3 0 0 0)".to_owned());

                let table_expression =
                    expression::NumericTableExpression::<$element_type>::Table3DReduce(
                        expression::ReduceOperator::Sum,
                        0,
                        expression::ArgumentExpression::Element(
                            expression::ElementExpression::Constant(0),
                        ),
                        expression::ArgumentExpression::Element(
                            expression::ElementExpression::Constant(0),
                        ),
                        expression::ArgumentExpression::Element(
                            expression::ElementExpression::Constant(0),
                        ),
                    );
                let result =
                    table_expression.to_yaml_string(&StateMetadata::default(), &table_registry);
                assert!(result.is_ok());
                assert_eq!(result.unwrap(), "(sum f3 0 0 0)".to_owned());

                let table_expression = expression::NumericTableExpression::<$element_type>::Table(
                    0,
                    vec![
                        expression::ElementExpression::Constant(0),
                        expression::ElementExpression::Constant(0),
                        expression::ElementExpression::Constant(0),
                        expression::ElementExpression::Constant(0),
                    ],
                );
                let result =
                    table_expression.to_yaml_string(&StateMetadata::default(), &table_registry);
                assert!(result.is_ok());
                assert_eq!(result.unwrap(), "(f4 0 0 0 0)".to_owned());

                let table_expression =
                    expression::NumericTableExpression::<$element_type>::TableReduce(
                        expression::ReduceOperator::Sum,
                        0,
                        vec![
                            expression::ArgumentExpression::Element(
                                expression::ElementExpression::Constant(0),
                            ),
                            expression::ArgumentExpression::Element(
                                expression::ElementExpression::Constant(0),
                            ),
                            expression::ArgumentExpression::Element(
                                expression::ElementExpression::Constant(0),
                            ),
                            expression::ArgumentExpression::Element(
                                expression::ElementExpression::Constant(0),
                            ),
                        ],
                    );
                let result =
                    table_expression.to_yaml_string(&StateMetadata::default(), &table_registry);
                assert!(result.is_ok());
                assert_eq!(result.unwrap(), "(sum f4 0 0 0 0)".to_owned());
            }
        };
    }

    make_numeric_table_expression_tests!(
        integer_numeric_table_expression_to_yaml_string,
        integer_tables,
        Integer
    );

    make_numeric_table_expression_tests!(
        continuous_numeric_table_expression_to_yaml_string,
        continuous_tables,
        Continuous
    );

    #[test]
    fn long_set_expression_to_yaml_string() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let text = "(union (intersection s0 (difference s2 (add 2 s3))) (remove 1 s1))".to_string();
        let result = parse_set(text.clone(), &metadata, &registry, &parameters);
        assert!(result.is_ok());

        let expr_string = result.unwrap().to_yaml_string(&metadata, &registry);
        assert!(expr_string.is_ok());
        assert_eq!(expr_string.unwrap(), text);
    }

    #[test]
    fn long_integer_expression_to_yaml_string() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let text = "(abs (+ n0 (neg n1)))".to_string();
        let result = parse_integer(text.clone(), &metadata, &registry, &parameters);
        assert!(result.is_ok());

        let expr_string = result.unwrap().to_yaml_string(&metadata, &registry);
        assert!(expr_string.is_ok());
        assert_eq!(expr_string.unwrap(), text);
    }

    #[test]
    fn long_continuous_expression_to_yaml_string() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let text = "(sqrt (pow (* c0 c1) 0.4))".to_string();
        let result = parse_continuous(text.clone(), &metadata, &registry, &parameters);
        assert!(result.is_ok());

        let expr_string = result.unwrap().to_yaml_string(&metadata, &registry);
        assert!(expr_string.is_ok());
        assert_eq!(expr_string.unwrap(), text);
    }

    #[test]
    fn long_element_expression_to_yaml_string() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let text = "(max (+ e0 e1) (- e2 e3))".to_string();
        let result = parse_element(text.clone(), &metadata, &registry, &parameters);
        assert!(result.is_ok());

        let expr_string = result.unwrap().to_yaml_string(&metadata, &registry);
        assert!(expr_string.is_ok());
        assert_eq!(expr_string.unwrap(), text);
    }

    #[test]
    fn long_condition_to_yaml_string() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let text = "(not (and (and (and true (is_subset s0 s1)) (is_empty s0)) (or (< 1 n1) (is_in 2 s0))))"
            .to_string();
        let result = parse_condition(text.clone(), &metadata, &registry, &parameters);
        assert!(result.is_ok());

        let expr_string = result.unwrap().to_yaml_string(&metadata, &registry);
        assert!(expr_string.is_ok());
        assert_eq!(expr_string.unwrap(), text);
    }

    #[test]
    /// This test shows the behavior that if the user writes a float constant that is actually an integer value,
    /// the yaml string representation of the parsed expression will convert it to integer. This is due to the
    /// implementation of the string representation of f64 objects, that integral values will have an integeral
    /// string representation (without decimal).
    ///
    /// This difference does not affect the correctness of the model.
    fn expression_with_float_to_yaml_string_changed() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let text = "(- (ceil (sum f1 s1)) (if (>= n0 (/ c1 3.0)) 1 0))".to_string();
        let test_text = "(- (ceil (sum f1 s1)) (if (>= n0 (/ c1 3)) 1 0))".to_string();

        let result = parse_integer(text.clone(), &metadata, &registry, &parameters);
        assert!(result.is_ok());

        let expr_string = result.unwrap().to_yaml_string(&metadata, &registry);
        assert!(expr_string.is_ok());
        assert_eq!(expr_string.unwrap(), test_text);
    }

    #[test]
    fn expression_with_float_to_yaml_string_unchanged() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let text = "(- (ceil (sum f1 s1)) (if (>= n0 (/ c1 3.5)) 1 0))".to_string();

        let result = parse_integer(text.clone(), &metadata, &registry, &parameters);
        assert!(result.is_ok());

        let expr_string = result.unwrap().to_yaml_string(&metadata, &registry);
        assert!(expr_string.is_ok());
        assert_eq!(expr_string.unwrap(), text);
    }
}
