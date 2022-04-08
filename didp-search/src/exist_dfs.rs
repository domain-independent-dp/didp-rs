use crate::hashable_state;
use crate::solver;
use crate::successor_generator;
use didp_parser::variable;
use rustc_hash::FxHashMap;
use std::fmt;
use std::rc::Rc;
use std::str;

#[derive(Default)]
pub struct IterativeForwardExistDfs<T: variable::Numeric> {
    primal_bound: Option<T>,
    capacity: Option<usize>,
}

impl<T: variable::Numeric + fmt::Display> solver::Solver<T> for IterativeForwardExistDfs<T> {
    #[inline]
    fn set_primal_bound(&mut self, bound: Option<T>) {
        self.primal_bound = bound;
    }

    #[inline]
    fn solve(&mut self, model: &didp_parser::Model<T>) -> solver::Solution<T> {
        forward_iterative_exist_dfs(model, self.primal_bound, self.capacity)
    }
}

impl<T: variable::Numeric> IterativeForwardExistDfs<T> {
    pub fn new(config: &yaml_rust::Yaml) -> Result<IterativeForwardExistDfs<T>, solver::ConfigErr>
    where
        <T as str::FromStr>::Err: fmt::Debug,
    {
        let map = match config {
            yaml_rust::Yaml::Hash(map) => map,
            yaml_rust::Yaml::Null => return Ok(IterativeForwardExistDfs::default()),
            _ => {
                return Err(solver::ConfigErr::new(format!(
                    "expected Hash, but found `{:?}`",
                    config
                )))
            }
        };
        let capacity = match map.get(&yaml_rust::Yaml::from_str("capacity")) {
            Some(yaml_rust::Yaml::Integer(value)) => Some(*value as usize),
            None => Some(1000000),
            value => {
                return Err(solver::ConfigErr::new(format!(
                    "expected Integer, but found `{:?}`",
                    value
                )))
            }
        };
        let primal_bound = match map.get(&yaml_rust::Yaml::from_str("primal_bound")) {
            Some(yaml_rust::Yaml::Integer(value)) => {
                Some(T::from_integer(*value as variable::Integer))
            }
            Some(yaml_rust::Yaml::Real(value)) => Some(value.parse().map_err(|e| {
                solver::ConfigErr::new(format!("could not parse {} as a number: {:?}", value, e))
            })?),
            None => None,
            value => {
                return Err(solver::ConfigErr::new(format!(
                    "expected Integer, but found `{:?}`",
                    value
                )))
            }
        };
        Ok(IterativeForwardExistDfs {
            capacity,
            primal_bound,
        })
    }
}

pub fn forward_iterative_exist_dfs<T: variable::Numeric + fmt::Display>(
    model: &didp_parser::Model<T>,
    mut primal_bound: Option<T>,
    capacity: Option<usize>,
) -> solver::Solution<T> {
    let mut nodes = 0;
    let generator = successor_generator::SuccessorGenerator::new(model, false);
    let mut prob = FxHashMap::default();
    if let Some(capacity) = capacity {
        prob.reserve(capacity);
    };
    let mut incumbent = Vec::new();
    while let Some((cost, transitions)) = exist_dfs(
        hashable_state::HashableState::new(&model.target),
        T::zero(),
        model,
        &generator,
        &mut prob,
        primal_bound,
        &mut nodes,
    ) {
        println!("New primal bound: {}, expanded: {}", cost, nodes);
        primal_bound = Some(cost);
        incumbent = transitions;
    }
    println!("Expanded: {}", nodes);
    if let Some(cost) = primal_bound {
        incumbent.reverse();
        Some((cost, incumbent))
    } else {
        None
    }
}

pub fn exist_dfs<T: variable::Numeric>(
    state: hashable_state::HashableState,
    cost: T,
    model: &didp_parser::Model<T>,
    generator: &successor_generator::SuccessorGenerator<T>,
    prob: &mut FxHashMap<hashable_state::HashableState, T>,
    primal_bound: Option<T>,
    nodes: &mut u32,
) -> Option<(T, Vec<Rc<didp_parser::Transition<T>>>)> {
    *nodes += 1;
    if model.get_base_cost(&state).is_some() {
        return Some((cost, Vec::new()));
    }
    if let Some(other_cost) = prob.get(&state) {
        if cost >= *other_cost {
            return None;
        } else {
            prob.remove(&state);
        }
    }
    for transition in generator.applicable_transitions(&state) {
        let cost = transition.eval_cost(cost, &state, &model.table_registry);
        if let Some(bound) = primal_bound {
            match model.reduce_function {
                didp_parser::ReduceFunction::Min if cost >= bound => continue,
                didp_parser::ReduceFunction::Max if cost <= bound => continue,
                _ => {}
            }
        }
        let successor = transition.apply(&state, &model.table_registry);
        if model.check_constraints(&successor) {
            let result = exist_dfs(successor, cost, model, generator, prob, primal_bound, nodes);
            if let Some((cost, mut transitions)) = result {
                transitions.push(transition);
                return Some((cost, transitions));
            }
        }
    }
    prob.insert(state, cost);
    None
}
