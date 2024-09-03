//! A module for successor generators.

use super::{successor_generator::SuccessorGenerator, transition::TransitionWithId};
use core::ops::Deref;
use dypdl::{GroundedCondition, StateFunctionCache, Transition, TransitionInterface};
use std::{mem, rc::Rc};

#[derive(Debug, PartialEq, Clone)]
struct TarjanTemporal {
    lowlink: Vec<usize>,
    index: Vec<usize>,
    stack: Vec<usize>,
    in_stack: Vec<bool>,
}

impl TarjanTemporal {
    fn new(n: usize) -> Self {
        TarjanTemporal {
            lowlink: vec![0; n],
            index: vec![0; n],
            stack: Vec::new(),
            in_stack: vec![false; n],
        }
    }

    fn reset(&mut self) {
        self.lowlink.iter_mut().for_each(|e| *e = 0);
        self.index.iter_mut().for_each(|e| *e = 0);
        self.stack.clear();
        self.in_stack.iter_mut().for_each(|e| *e = false);
    }
}

fn tarjan_dfs(
    adjacent_list: &[Vec<usize>],
    current: usize,
    time: &mut usize,
    tmp: &mut TarjanTemporal,
    node_to_scc_root: &mut Vec<usize>,
) {
    *time += 1;
    tmp.index[current] = *time;
    tmp.lowlink[current] = *time;
    tmp.stack.push(current);
    tmp.in_stack[current] = true;

    for &next in adjacent_list[current].iter() {
        if tmp.index[next] == 0 {
            tarjan_dfs(adjacent_list, next, time, tmp, node_to_scc_root);
            tmp.lowlink[current] = tmp.lowlink[current].min(tmp.lowlink[next]);
        } else if tmp.in_stack[next] {
            tmp.lowlink[current] = tmp.lowlink[current].min(tmp.index[next]);
        }
    }

    if tmp.lowlink[current] == tmp.index[current] {
        while let Some(top) = tmp.stack.pop() {
            tmp.in_stack[top] = false;
            node_to_scc_root[top] = current;

            if top == current {
                break;
            }
        }
    }
}

fn tarjan(
    adjacent_list: &[Vec<usize>],
    tmp: &mut TarjanTemporal,
    node_to_scc_root: &mut Vec<usize>,
) {
    tmp.reset();
    let n = adjacent_list.len();
    node_to_scc_root.resize(n, 0);
    node_to_scc_root
        .iter_mut()
        .enumerate()
        .for_each(|(i, e)| *e = i);
    let mut time = 0;

    for i in 0..n {
        if tmp.index[i] == 0 {
            tarjan_dfs(adjacent_list, i, &mut time, tmp, node_to_scc_root);
        }
    }
}

fn check_dominance<T, U>(
    adjacent_list: &[Vec<usize>],
    applicable_transitions: &[U],
    node_to_scc_root: &[usize],
    is_applicable: &mut [bool],
) where
    T: TransitionInterface,
    U: Deref<Target = TransitionWithId<T>>,
{
    for t in applicable_transitions {
        let i = t.id;

        if i != node_to_scc_root[i] {
            is_applicable[i] = false;
        }

        for &next in adjacent_list[i].iter() {
            if node_to_scc_root[i] != node_to_scc_root[next] {
                is_applicable[next] = false;
            }
        }
    }
}

/// Generator of applicable transitions with dominance.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, PartialEq, Clone)]
pub struct SuccessorGeneratorWithDominance<
    T = Transition,
    U = Rc<TransitionWithId<T>>,
    R = Rc<dypdl::Model>,
> where
    T: TransitionInterface,
    U: Deref<Target = TransitionWithId<T>> + Clone,
    R: Deref<Target = dypdl::Model>,
{
    /// Successor generator
    pub generator: SuccessorGenerator<TransitionWithId<T>, U, R>,
    /// Transition dominance
    dominance_map: Vec<Vec<(usize, Option<GroundedCondition>)>>,
    /// Temporal data structure to store applicable transitions.
    applicable_transitions: Vec<U>,
    /// Temporal data structure to check whether a transition is applicable.
    is_applicable: Vec<bool>,
    /// Temporal data structure to store active transition dominance rules.
    adjacent_list: Vec<Vec<usize>>,
    /// Temporal data structure for Tarjan's algorithm.
    tarjan_temporal: TarjanTemporal,
    /// Temporal data structure to map nodes to strongly connected components.
    node_to_scc_root: Vec<usize>,
}

impl<T, U, R> SuccessorGeneratorWithDominance<T, U, R>
where
    T: TransitionInterface,
    U: Deref<Target = TransitionWithId<T>> + Clone,
    R: Deref<Target = dypdl::Model>,
{
    pub fn get_model(&self) -> &R {
        &self.generator.model
    }

    /// Returns a vector of applicable transitions.
    ///
    /// `result` is used as a buffer to avoid memory allocation.
    pub fn generate_applicable_transitions<S: dypdl::StateInterface>(
        &self,
        state: &S,
        function_cache: &mut StateFunctionCache,
        result: &mut Vec<U>,
    ) {
        self.generator
            .generate_applicable_transitions(state, function_cache, result)
    }

    fn extract_active_edges<S: dypdl::StateInterface>(
        &mut self,
        state: &S,
        function_cache: &mut StateFunctionCache,
    ) {
        self.is_applicable.clear();
        self.is_applicable.resize(self.dominance_map.len(), false);

        for t in self.applicable_transitions.iter() {
            self.is_applicable[t.id] = true;
        }

        self.adjacent_list
            .resize(self.dominance_map.len(), Vec::new());
        self.adjacent_list.iter_mut().for_each(|e| e.clear());

        for t in self.applicable_transitions.iter() {
            for (next, c) in self.dominance_map[t.id].iter() {
                if self.is_applicable[*next]
                    && c.as_ref().map_or(true, |c| {
                        c.is_satisfied(
                            state,
                            function_cache,
                            &self.generator.model.state_functions,
                            &self.generator.model.table_registry,
                        )
                    })
                {
                    self.adjacent_list[t.id].push(*next);
                }
            }
        }
    }

    pub fn generate_applicable_transitions_with_dominance<S: dypdl::StateInterface>(
        &mut self,
        state: &S,
        function_cache: &mut StateFunctionCache,
        result: &mut Vec<U>,
    ) {
        self.generator.generate_applicable_transitions(
            state,
            function_cache,
            &mut self.applicable_transitions,
        );

        // We do not need to check dominance if there is at most one applicable transition
        // or there is no dominance.
        if self.applicable_transitions.len() <= 1
            || self.get_model().transition_dominance.is_empty()
        {
            mem::swap(result, &mut self.applicable_transitions);

            return;
        }

        self.extract_active_edges(state, function_cache);
        tarjan(
            &self.adjacent_list,
            &mut self.tarjan_temporal,
            &mut self.node_to_scc_root,
        );
        check_dominance(
            &self.adjacent_list,
            &self.applicable_transitions,
            &self.node_to_scc_root,
            &mut self.is_applicable,
        );

        result.clear();

        for t in self.applicable_transitions.drain(..) {
            if self.is_applicable[t.id] {
                result.push(t);
            }
        }
    }
}

impl<T, U, R> From<SuccessorGenerator<TransitionWithId<T>, U, R>>
    for SuccessorGeneratorWithDominance<T, U, R>
where
    T: TransitionInterface,
    U: Deref<Target = TransitionWithId<T>> + Clone,
    R: Deref<Target = dypdl::Model>,
{
    fn from(generator: SuccessorGenerator<TransitionWithId<T>, U, R>) -> Self {
        let transition_dominance: Vec<_> = generator
            .model
            .transition_dominance
            .iter()
            .filter_map(|d| {
                if d.backward != generator.backward {
                    None
                } else {
                    Some((d.dominating, d.dominated, &d.condition))
                }
            })
            .collect();

        let mut dominance_map = vec![Vec::new(); generator.transitions.len()];

        transition_dominance
            .into_iter()
            .for_each(|(t1_id, t2_id, c)| {
                assert!(dominance_map.len() > t1_id);
                dominance_map[t1_id].push((t2_id, c.clone()));
            });

        let n_transitions = generator.transitions.len();
        let tarjan_temporal = TarjanTemporal::new(n_transitions);

        SuccessorGeneratorWithDominance {
            generator,
            dominance_map,
            applicable_transitions: Vec::new(),
            adjacent_list: Vec::new(),
            tarjan_temporal,
            is_applicable: vec![false; n_transitions],
            node_to_scc_root: vec![0; n_transitions],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::expression::*;
    use dypdl::prelude::*;
    use std::rc::Rc;

    #[test]
    fn generate_applicable_transitions() {
        let mut model = Model::default();

        let result = model.add_integer_variable("v", 0);
        assert!(result.is_ok());
        let v = result.unwrap();

        let mut transition1 = Transition::new("t1");
        transition1.add_precondition(Condition::comparison_i(
            ComparisonOperator::Ge,
            v,
            IntegerExpression::Constant(1),
        ));
        let result = model.add_forward_transition(transition1.clone());
        assert!(result.is_ok());
        let id1 = result.unwrap();

        let mut transition2 = Transition::new("t2");
        transition2.add_precondition(Condition::comparison_i(
            ComparisonOperator::Ge,
            v,
            IntegerExpression::Constant(2),
        ));
        let result = model.add_forward_transition(transition2.clone());
        assert!(result.is_ok());
        let id2 = result.unwrap();

        let mut transition3 = Transition::new("t3");
        transition3.add_precondition(Condition::comparison_i(
            ComparisonOperator::Ge,
            v,
            IntegerExpression::Constant(3),
        ));
        let result = model.add_forward_transition(transition3.clone());
        assert!(result.is_ok());
        let id3 = result.unwrap();

        let result = model.add_transition_dominance_with_condition(
            &id1,
            &id2,
            Condition::comparison_i(ComparisonOperator::Ge, v, IntegerExpression::Constant(2)),
        );
        assert!(result.is_ok());

        let result = model.add_transition_dominance_with_condition(
            &id2,
            &id3,
            Condition::comparison_i(ComparisonOperator::Ge, v, IntegerExpression::Constant(3)),
        );
        assert!(result.is_ok());

        let mut function_cache = StateFunctionCache::new(&model.state_functions);

        let generator = SuccessorGenerator::<TransitionWithId>::from_model(Rc::new(model), false);
        let mut generator = SuccessorGeneratorWithDominance::<Transition>::from(generator);

        let mut state = generator.get_model().target.clone();
        state.signature_variables.integer_variables[v.id()] = 2;

        let mut result = Vec::default();
        generator.generate_applicable_transitions(&state, &mut function_cache, &mut result);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].id, id1.id);
        assert_eq!(result[0].forced, id1.forced);
        assert_eq!(result[0].transition, transition1);
        assert_eq!(result[1].id, id2.id);
        assert_eq!(result[1].forced, id2.forced);
        assert_eq!(result[1].transition, transition2);

        generator.generate_applicable_transitions_with_dominance(
            &state,
            &mut function_cache,
            &mut result,
        );
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, id1.id);
        assert_eq!(result[0].forced, id1.forced);
        assert_eq!(result[0].transition, transition1);

        state.signature_variables.integer_variables[v.id()] = 3;

        generator.generate_applicable_transitions_with_dominance(
            &state,
            &mut function_cache,
            &mut result,
        );
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, id1.id);
        assert_eq!(result[0].forced, id1.forced);
        assert_eq!(result[0].transition, transition1);
    }

    #[test]
    fn generate_applicable_transition_with_complicated_dominance() {
        // >=2   >=1   <=1   >=1   >=2
        // t1 -> t2 -> t3 -> t4 -> t1, but t1 and t3 are not simultaneously applicable.
        //
        // >=2     >=2     >=2
        // t5 ---> t6 ---> t8
        //     |        |
        //     --> t7 --
        //         >=2
        //
        // >=3   >=3    >=3    >=1
        // t9 -> t10 -> t11 -> t2
        //        |      |
        //        |       ---> t12
        //        |            >=3
        //         ---> t13
        //              >=3

        let mut model = Model::default();

        let result = model.add_integer_variable("v", 0);
        assert!(result.is_ok());
        let v = result.unwrap();

        let mut transition1 = Transition::new("t1");
        transition1.add_precondition(Condition::comparison_i(
            ComparisonOperator::Ge,
            v,
            IntegerExpression::Constant(2),
        ));
        let result = model.add_forward_transition(transition1.clone());
        assert!(result.is_ok());
        let id1 = result.unwrap();

        let mut transition2 = Transition::new("t2");
        transition2.add_precondition(Condition::comparison_i(
            ComparisonOperator::Ge,
            v,
            IntegerExpression::Constant(1),
        ));
        let result = model.add_forward_transition(transition2.clone());
        assert!(result.is_ok());
        let id2 = result.unwrap();

        let mut transition3 = Transition::new("t3");
        transition3.add_precondition(Condition::comparison_i(
            ComparisonOperator::Le,
            v,
            IntegerExpression::Constant(1),
        ));
        let result = model.add_forward_transition(transition3.clone());
        assert!(result.is_ok());
        let id3 = result.unwrap();

        let mut transition4 = Transition::new("t4");
        transition4.add_precondition(Condition::comparison_i(
            ComparisonOperator::Ge,
            v,
            IntegerExpression::Constant(1),
        ));
        let result = model.add_forward_transition(transition4.clone());
        assert!(result.is_ok());
        let id4 = result.unwrap();

        let mut transition5 = Transition::new("t5");
        transition5.add_precondition(Condition::comparison_i(
            ComparisonOperator::Ge,
            v,
            IntegerExpression::Constant(2),
        ));
        let result = model.add_forward_transition(transition5.clone());
        assert!(result.is_ok());
        let id5 = result.unwrap();

        let mut transition6 = Transition::new("t6");
        transition6.add_precondition(Condition::comparison_i(
            ComparisonOperator::Ge,
            v,
            IntegerExpression::Constant(2),
        ));
        let result = model.add_forward_transition(transition6.clone());
        assert!(result.is_ok());
        let id6 = result.unwrap();

        let mut transition7 = Transition::new("t7");
        transition7.add_precondition(Condition::comparison_i(
            ComparisonOperator::Ge,
            v,
            IntegerExpression::Constant(2),
        ));
        let result = model.add_forward_transition(transition7.clone());
        assert!(result.is_ok());
        let id7 = result.unwrap();

        let mut transition8 = Transition::new("t8");
        transition8.add_precondition(Condition::comparison_i(
            ComparisonOperator::Ge,
            v,
            IntegerExpression::Constant(2),
        ));
        let result = model.add_forward_transition(transition8.clone());
        assert!(result.is_ok());
        let id8 = result.unwrap();

        let mut transition9 = Transition::new("t9");
        transition9.add_precondition(Condition::comparison_i(
            ComparisonOperator::Ge,
            v,
            IntegerExpression::Constant(3),
        ));
        let result = model.add_forward_transition(transition9.clone());
        assert!(result.is_ok());
        let id9 = result.unwrap();

        let mut transition10 = Transition::new("t9");
        transition10.add_precondition(Condition::comparison_i(
            ComparisonOperator::Ge,
            v,
            IntegerExpression::Constant(3),
        ));
        let result = model.add_forward_transition(transition10.clone());
        assert!(result.is_ok());
        let id10 = result.unwrap();

        let mut transition11 = Transition::new("t11");
        transition11.add_precondition(Condition::comparison_i(
            ComparisonOperator::Ge,
            v,
            IntegerExpression::Constant(3),
        ));
        let result = model.add_forward_transition(transition11.clone());
        assert!(result.is_ok());
        let id11 = result.unwrap();

        let mut transition12 = Transition::new("t12");
        transition12.add_precondition(Condition::comparison_i(
            ComparisonOperator::Ge,
            v,
            IntegerExpression::Constant(3),
        ));
        let result = model.add_forward_transition(transition9.clone());
        assert!(result.is_ok());
        let id12 = result.unwrap();

        let mut transition13 = Transition::new("t13");
        transition13.add_precondition(Condition::comparison_i(
            ComparisonOperator::Ge,
            v,
            IntegerExpression::Constant(3),
        ));
        let result = model.add_forward_transition(transition13.clone());
        assert!(result.is_ok());
        let id13 = result.unwrap();

        let result = model.add_transition_dominance_with_condition(
            &id1,
            &id2,
            Condition::comparison_i(ComparisonOperator::Ge, v, IntegerExpression::Constant(1)),
        );
        assert!(result.is_ok());

        let result = model.add_transition_dominance(&id2, &id3);
        assert!(result.is_ok());

        let result = model.add_transition_dominance(&id3, &id4);
        assert!(result.is_ok());

        let result = model.add_transition_dominance(&id4, &id1);
        assert!(result.is_ok());

        let result = model.add_transition_dominance(&id5, &id6);
        assert!(result.is_ok());

        let result = model.add_transition_dominance(&id6, &id8);
        assert!(result.is_ok());

        let result = model.add_transition_dominance_with_condition(
            &id5,
            &id7,
            Condition::comparison_i(ComparisonOperator::Ge, v, IntegerExpression::Constant(1)),
        );
        assert!(result.is_ok());

        let result = model.add_transition_dominance_with_condition(
            &id7,
            &id8,
            Condition::comparison_i(ComparisonOperator::Ge, v, IntegerExpression::Constant(2)),
        );
        assert!(result.is_ok());

        let result = model.add_transition_dominance(&id9, &id10);
        assert!(result.is_ok());

        let result = model.add_transition_dominance(&id10, &id11);
        assert!(result.is_ok());

        let result = model.add_transition_dominance(&id11, &id12);
        assert!(result.is_ok());

        let result = model.add_transition_dominance(&id11, &id2);
        assert!(result.is_ok());

        let result = model.add_transition_dominance(&id10, &id13);
        assert!(result.is_ok());

        let mut function_cache = StateFunctionCache::new(&model.state_functions);
        let generator = SuccessorGenerator::<TransitionWithId>::from_model(Rc::new(model), false);
        let mut generator = SuccessorGeneratorWithDominance::<Transition>::from(generator);
        let mut state = generator.get_model().target.clone();
        let mut result = Vec::default();

        state.signature_variables.integer_variables[v.id()] = 0;
        generator.generate_applicable_transitions_with_dominance(
            &state,
            &mut function_cache,
            &mut result,
        );
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, id3.id);
        assert_eq!(result[0].forced, id3.forced);
        assert_eq!(result[0].transition, transition3);

        state.signature_variables.integer_variables[v.id()] = 1;
        generator.generate_applicable_transitions_with_dominance(
            &state,
            &mut function_cache,
            &mut result,
        );
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, id2.id);
        assert_eq!(result[0].forced, id2.forced);
        assert_eq!(result[0].transition, transition2);

        state.signature_variables.integer_variables[v.id()] = 2;
        generator.generate_applicable_transitions_with_dominance(
            &state,
            &mut function_cache,
            &mut result,
        );
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].id, id4.id);
        assert_eq!(result[0].forced, id4.forced);
        assert_eq!(result[0].transition, transition4);
        assert_eq!(result[1].id, id5.id);
        assert_eq!(result[1].forced, id5.forced);
        assert_eq!(result[1].transition, transition5);

        state.signature_variables.integer_variables[v.id()] = 3;
        generator.generate_applicable_transitions_with_dominance(
            &state,
            &mut function_cache,
            &mut result,
        );
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].id, id4.id);
        assert_eq!(result[0].forced, id4.forced);
        assert_eq!(result[0].transition, transition4);
        assert_eq!(result[1].id, id5.id);
        assert_eq!(result[1].forced, id5.forced);
        assert_eq!(result[1].transition, transition5);
        assert_eq!(result[2].id, id9.id);
        assert_eq!(result[2].forced, id9.forced);
        assert_eq!(result[2].transition, transition9);
    }

    #[test]
    fn break_cycle() {
        // >=2            >=4    >= 3 (->)
        // t1 -> t2 -> t3 --> t4 <-> t5
        // ^              |
        // |______________|
        //        >= 1
        //
        // t6 -> t7

        let mut model = Model::default();

        let result = model.add_integer_variable("v", 0);
        assert!(result.is_ok());
        let v = result.unwrap();

        let mut transition1 = Transition::new("t1");
        transition1.add_precondition(Condition::comparison_i(
            ComparisonOperator::Ge,
            v,
            IntegerExpression::Constant(2),
        ));
        let result = model.add_forward_transition(transition1.clone());
        assert!(result.is_ok());
        let id1 = result.unwrap();

        let transition2 = Transition::new("t2");
        let result = model.add_forward_transition(transition2.clone());
        assert!(result.is_ok());
        let id2 = result.unwrap();

        let transition3 = Transition::new("t3");
        let result = model.add_forward_transition(transition3.clone());
        assert!(result.is_ok());
        let id3 = result.unwrap();

        let transition4 = Transition::new("t4");
        let result = model.add_forward_transition(transition4.clone());
        assert!(result.is_ok());
        let id4 = result.unwrap();

        let transition5 = Transition::new("t5");
        let result = model.add_forward_transition(transition5.clone());
        assert!(result.is_ok());
        let id5 = result.unwrap();

        let transition6 = Transition::new("t6");
        let result = model.add_forward_transition(transition6.clone());
        assert!(result.is_ok());
        let id6 = result.unwrap();

        let transition7 = Transition::new("t7");
        let result = model.add_forward_transition(transition7.clone());
        assert!(result.is_ok());
        let id7 = result.unwrap();

        let result = model.add_transition_dominance(&id1, &id2);
        assert!(result.is_ok());

        let result = model.add_transition_dominance(&id2, &id3);
        assert!(result.is_ok());

        let result = model.add_transition_dominance_with_condition(
            &id3,
            &id1,
            Condition::comparison_i(ComparisonOperator::Ge, v, IntegerExpression::Constant(2)),
        );
        assert!(result.is_ok());

        let result = model.add_transition_dominance_with_condition(
            &id3,
            &id4,
            Condition::comparison_i(ComparisonOperator::Ge, v, IntegerExpression::Constant(4)),
        );
        assert!(result.is_ok());

        let result = model.add_transition_dominance_with_condition(
            &id4,
            &id5,
            Condition::comparison_i(ComparisonOperator::Ge, v, IntegerExpression::Constant(3)),
        );
        assert!(result.is_ok());

        let result = model.add_transition_dominance(&id5, &id4);
        assert!(result.is_ok());

        let result = model.add_transition_dominance(&id6, &id7);
        assert!(result.is_ok());

        let mut function_cache = StateFunctionCache::new(&model.state_functions);
        let generator = SuccessorGenerator::<TransitionWithId>::from_model(Rc::new(model), false);
        let mut generator = SuccessorGeneratorWithDominance::<Transition>::from(generator);
        let mut state = generator.get_model().target.clone();

        state.signature_variables.integer_variables[v.id()] = 0;
        let mut result = Vec::default();
        generator.generate_applicable_transitions_with_dominance(
            &state,
            &mut function_cache,
            &mut result,
        );
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].id, id2.id);
        assert_eq!(result[0].forced, id2.forced);
        assert_eq!(result[0].transition, transition2);
        assert_eq!(result[1].id, id5.id);
        assert_eq!(result[1].forced, id5.forced);
        assert_eq!(result[1].transition, transition5);
        assert_eq!(result[2].id, id6.id);
        assert_eq!(result[2].forced, id6.forced);
        assert_eq!(result[2].transition, transition6);

        state.signature_variables.integer_variables[v.id()] = 1;
        let mut result = Vec::default();
        generator.generate_applicable_transitions_with_dominance(
            &state,
            &mut function_cache,
            &mut result,
        );
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].id, id2.id);
        assert_eq!(result[0].forced, id2.forced);
        assert_eq!(result[0].transition, transition2);
        assert_eq!(result[1].id, id5.id);
        assert_eq!(result[1].forced, id5.forced);
        assert_eq!(result[1].transition, transition5);
        assert_eq!(result[2].id, id6.id);
        assert_eq!(result[2].forced, id6.forced);
        assert_eq!(result[2].transition, transition6);

        state.signature_variables.integer_variables[v.id()] = 2;
        let mut result = Vec::default();
        generator.generate_applicable_transitions_with_dominance(
            &state,
            &mut function_cache,
            &mut result,
        );
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].id, id1.id);
        assert_eq!(result[0].forced, id1.forced);
        assert_eq!(result[0].transition, transition1);
        assert_eq!(result[1].id, id5.id);
        assert_eq!(result[1].forced, id5.forced);
        assert_eq!(result[1].transition, transition5);
        assert_eq!(result[2].id, id6.id);
        assert_eq!(result[2].forced, id6.forced);
        assert_eq!(result[2].transition, transition6);

        state.signature_variables.integer_variables[v.id()] = 3;
        let mut result = Vec::default();
        generator.generate_applicable_transitions_with_dominance(
            &state,
            &mut function_cache,
            &mut result,
        );
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].id, id1.id);
        assert_eq!(result[0].forced, id1.forced);
        assert_eq!(result[0].transition, transition1);
        assert_eq!(result[1].id, id4.id);
        assert_eq!(result[1].forced, id4.forced);
        assert_eq!(result[1].transition, transition4);
        assert_eq!(result[2].id, id6.id);
        assert_eq!(result[2].forced, id6.forced);
        assert_eq!(result[2].transition, transition6);

        state.signature_variables.integer_variables[v.id()] = 4;
        let mut result = Vec::default();
        generator.generate_applicable_transitions_with_dominance(
            &state,
            &mut function_cache,
            &mut result,
        );
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].id, id1.id);
        assert_eq!(result[0].forced, id1.forced);
        assert_eq!(result[0].transition, transition1);
        assert_eq!(result[1].id, id6.id);
        assert_eq!(result[1].forced, id6.forced);
        assert_eq!(result[1].transition, transition6);
    }
}
