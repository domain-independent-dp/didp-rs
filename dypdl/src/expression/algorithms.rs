use crate::variable_type::Set;
use ordered_float::OrderedFloat;
use std::ops::Add;

/// Sorts `(value, weight)` pairs by increasing weight-to-value ratio.
pub fn sort_fractional_knapsack_items<I>(items: I) -> Vec<(f64, f64)>
where
    I: IntoIterator<Item = (f64, f64)>,
{
    let mut sorted_items = items.into_iter().collect::<Vec<_>>();
    sorted_items.sort_by_key(|&(value, weight)| OrderedFloat(weight / value));
    sorted_items
}

/// Sorts `(item index, value, weight)` tuples by increasing weight-to-value ratio.
pub fn sort_fractional_knapsack_items_with_indices<I>(items: I) -> Vec<(usize, f64, f64)>
where
    I: IntoIterator<Item = (usize, f64, f64)>,
{
    let mut sorted_items = items.into_iter().collect::<Vec<_>>();
    sorted_items.sort_by_key(|&(_, value, weight)| OrderedFloat(weight / value));
    sorted_items
}

/// Computes the fractional knapsack value from `(value, weight)` pairs.
pub fn compute_fractional_knapsack<I>(capacity: f64, items: I) -> f64
where
    I: Iterator<Item = (f64, f64)>,
{
    let sorted_items = sort_fractional_knapsack_items(items);

    compute_fractional_knapsack_sorted(capacity, sorted_items.into_iter())
}

/// Computes the fractional knapsack value from pairs sorted by increasing weight-to-value ratio.
/// The first item that exceeds the remaining capacity is taken fractionally.
pub fn compute_fractional_knapsack_sorted<I>(capacity: f64, sorted_items: I) -> f64
where
    I: Iterator<Item = (f64, f64)>,
{
    let mut total_value = 0.0;
    let mut remaining_capacity = capacity;

    for (value, weight) in sorted_items {
        if remaining_capacity <= 0.0 {
            break;
        }

        if weight <= remaining_capacity {
            total_value += value;
            remaining_capacity -= weight;
        } else {
            total_value += value * (remaining_capacity / weight);
            remaining_capacity = 0.0;
        }
    }

    total_value
}

/// Computes a minimum spanning tree over `nodes`, using callbacks for edge weights and connectivity.
/// Each pair of nodes uses the cheaper available direction, ordered by `key`.
///
/// # Panics
///
/// Panics if the selected nodes cannot be connected.
pub fn compute_minimum_spanning_tree_with_connectivity<T, K, F, C, G>(
    nodes: &Set,
    edge_weight: F,
    connected: C,
    key: G,
) -> T
where
    T: Add<Output = T> + Copy + Default,
    K: Ord,
    F: FnMut(usize, usize) -> T,
    C: FnMut(usize, usize) -> bool,
    G: FnMut(T) -> K,
{
    let sorted_edges =
        sort_minimum_spanning_tree_edges_by_callback(nodes, edge_weight, connected, key);

    kruskal(nodes.count_ones(..), nodes.len(), sorted_edges)
}

fn sort_minimum_spanning_tree_edges_by_callback<T, K, F, C, G>(
    nodes: &Set,
    mut edge_weight: F,
    mut connected: C,
    mut key: G,
) -> Vec<(usize, usize, T)>
where
    T: Copy,
    K: Ord,
    F: FnMut(usize, usize) -> T,
    C: FnMut(usize, usize) -> bool,
    G: FnMut(T) -> K,
{
    let node_list = nodes.ones().collect::<Vec<_>>();
    let mut edges = Vec::new();

    for (index, &i) in node_list.iter().enumerate() {
        for &j in &node_list[index + 1..] {
            let forward = connected(i, j).then(|| edge_weight(i, j));
            let backward = connected(j, i).then(|| edge_weight(j, i));

            if let Some((weight, _)) = min_available_edge(forward, backward, &mut key) {
                edges.push((i, j, weight));
            }
        }
    }

    sort_minimum_spanning_tree_edges(edges, key)
}

/// Sorts `(source, target, weight)` edges by the key computed from each weight.
pub fn sort_minimum_spanning_tree_edges<T, K, I, G>(edges: I, mut key: G) -> Vec<(usize, usize, T)>
where
    T: Copy,
    K: Ord,
    I: IntoIterator<Item = (usize, usize, T)>,
    G: FnMut(T) -> K,
{
    let mut edges = edges.into_iter().collect::<Vec<_>>();
    edges.sort_by_key(|&(_, _, weight)| key(weight));
    edges
}

/// Builds sorted undirected edges from a square weight matrix and a connectivity predicate.
/// Each pair of nodes uses the cheaper available direction, ordered by `key`.
pub fn sort_minimum_spanning_tree_edges_with_connectivity<T, K, F, G>(
    matrix: &[Vec<T>],
    mut connected: F,
    mut key: G,
) -> Vec<(usize, usize, T)>
where
    T: Copy,
    K: Ord,
    F: FnMut(usize, usize) -> bool,
    G: FnMut(T) -> K,
{
    let n_edges = matrix.len().saturating_mul(matrix.len().saturating_sub(1)) / 2;
    let mut edges = Vec::with_capacity(n_edges);

    for (i, row) in matrix.iter().enumerate() {
        for (j, reverse_row) in matrix.iter().enumerate().skip(i + 1) {
            let forward = connected(i, j).then_some(row[j]);
            let backward = connected(j, i).then_some(reverse_row[i]);

            if let Some((weight, _)) = min_available_edge(forward, backward, &mut key) {
                edges.push((i, j, weight));
            }
        }
    }

    sort_minimum_spanning_tree_edges(edges, key)
}

/// Computes a minimum spanning tree using edges sorted by increasing weight.
/// Edges with an endpoint outside `nodes` are ignored; `T::default()` must be zero.
///
/// # Panics
///
/// Panics if the selected nodes cannot be connected.
pub fn compute_minimum_spanning_tree_from_sorted_edges<T, I>(nodes: &Set, sorted_edges: I) -> T
where
    T: Add<Output = T> + Copy + Default,
    I: IntoIterator<Item = (usize, usize, T)>,
{
    let n_nodes = nodes.count_ones(..);
    let sorted_edges = sorted_edges
        .into_iter()
        .filter(|(i, j, _)| nodes.contains(*i) && nodes.contains(*j));

    kruskal(n_nodes, nodes.len(), sorted_edges)
}

fn kruskal<T, I>(n_nodes: usize, union_find_size: usize, sorted_edges: I) -> T
where
    T: Add<Output = T> + Copy + Default,
    I: IntoIterator<Item = (usize, usize, T)>,
{
    debug_assert!(
        n_nodes <= union_find_size,
        "#nodes must be <= union-find size in a minimum spanning tree expression",
    );

    if n_nodes <= 1 {
        return T::default();
    }

    let mut parent = (0..union_find_size).collect::<Vec<_>>();
    let mut rank = vec![0; union_find_size];
    let mut total = T::default();
    let mut selected = 0;

    for (i, j, weight) in sorted_edges {
        if union(&mut parent, &mut rank, i, j) {
            total = total + weight;
            selected += 1;

            if selected == n_nodes - 1 {
                break;
            }
        }
    }

    assert_eq!(
        selected,
        n_nodes - 1,
        "the node set is disconnected under the given connectivity relation, \
         so no spanning tree exists for a minimum spanning tree expression",
    );

    total
}

fn min_available_edge<T, K, G>(
    forward: Option<T>,
    backward: Option<T>,
    key: &mut G,
) -> Option<(T, K)>
where
    T: Copy,
    K: Ord,
    G: FnMut(T) -> K,
{
    match (forward, backward) {
        (Some(forward), Some(backward)) => {
            let forward_key = key(forward);
            let backward_key = key(backward);

            if forward_key <= backward_key {
                Some((forward, forward_key))
            } else {
                Some((backward, backward_key))
            }
        }
        (Some(forward), None) => {
            let forward_key = key(forward);
            Some((forward, forward_key))
        }
        (None, Some(backward)) => {
            let backward_key = key(backward);
            Some((backward, backward_key))
        }
        (None, None) => None,
    }
}

fn find(parent: &mut [usize], x: usize) -> usize {
    if parent[x] != x {
        parent[x] = find(parent, parent[x]);
    }

    parent[x]
}

fn union(parent: &mut [usize], rank: &mut [usize], x: usize, y: usize) -> bool {
    let x = find(parent, x);
    let y = find(parent, y);

    if x == y {
        return false;
    }

    if rank[x] < rank[y] {
        parent[x] = y;
    } else {
        parent[y] = x;

        if rank[x] == rank[y] {
            rank[x] += 1;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_fractional_knapsack() {
        let items = vec![(100.0, 20.0), (60.0, 10.0), (120.0, 30.0)];
        let capacity = 50.0;
        let result = compute_fractional_knapsack(capacity, items.into_iter());
        assert_eq!(result, 240.0);
    }

    #[test]
    fn test_compute_fractional_knapsack_sorted() {
        let items = vec![(60.0, 10.0), (100.0, 20.0), (120.0, 30.0)];
        let capacity = 50.0;
        let result = compute_fractional_knapsack_sorted(capacity, items.into_iter());
        assert_eq!(result, 240.0);
    }

    #[test]
    fn test_compute_fractional_knapsack_negative_capacity() {
        let items = vec![(60.0, 10.0), (100.0, 20.0), (120.0, 30.0)];
        let capacity = -5.0;
        let result = compute_fractional_knapsack(capacity, items.into_iter());
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_compute_minimum_spanning_tree_with_connectivity() {
        let mut nodes = Set::with_capacity(4);
        nodes.insert(0);
        nodes.insert(1);
        nodes.insert(2);
        nodes.insert(3);
        let weights = [[0, 1, 4, 3], [1, 0, 2, 5], [4, 2, 0, 6], [3, 5, 6, 0]];
        let result = compute_minimum_spanning_tree_with_connectivity(
            &nodes,
            |i, j| weights[i][j],
            |_, _| true,
            |weight| weight,
        );
        assert_eq!(result, 6);
    }

    #[test]
    #[should_panic(expected = "the node set is disconnected")]
    fn test_compute_minimum_spanning_tree_disconnected_panics() {
        let mut nodes = Set::with_capacity(2);
        nodes.insert(0);
        nodes.insert(1);

        compute_minimum_spanning_tree_with_connectivity(
            &nodes,
            |_, _| 1,
            |_, _| false,
            |weight| weight,
        );
    }

    #[test]
    fn test_compute_minimum_spanning_tree_with_connectivity_and_asymmetric_weights() {
        let mut nodes = Set::with_capacity(3);
        nodes.insert(0);
        nodes.insert(1);
        nodes.insert(2);
        let weights = [[0, 10, 4], [1, 0, 7], [5, 2, 0]];
        let result = compute_minimum_spanning_tree_with_connectivity(
            &nodes,
            |i, j| weights[i][j],
            |_, _| true,
            |weight| weight,
        );
        assert_eq!(result, 3);
    }

    #[test]
    fn test_sort_minimum_spanning_tree_edges_with_connectivity() {
        let weights = vec![
            vec![0, 1, 4, 3],
            vec![1, 0, 2, 5],
            vec![4, 2, 0, 6],
            vec![3, 5, 6, 0],
        ];
        let result = sort_minimum_spanning_tree_edges_with_connectivity(
            &weights,
            |_, _| true,
            |weight| weight,
        );
        assert_eq!(
            result,
            vec![
                (0, 1, 1),
                (1, 2, 2),
                (0, 3, 3),
                (0, 2, 4),
                (1, 3, 5),
                (2, 3, 6),
            ]
        );
    }

    #[test]
    fn test_sort_minimum_spanning_tree_edges_with_connectivity_and_asymmetric_weights() {
        let weights = vec![vec![0, 10, 4], vec![1, 0, 7], vec![5, 2, 0]];
        let result = sort_minimum_spanning_tree_edges_with_connectivity(
            &weights,
            |_, _| true,
            |weight| weight,
        );
        assert_eq!(result, vec![(0, 1, 1), (1, 2, 2), (0, 2, 4)]);
    }

    #[test]
    fn test_compute_minimum_spanning_tree_with_connectivity_and_connected_callback() {
        let mut nodes = Set::with_capacity(4);
        nodes.insert(0);
        nodes.insert(1);
        nodes.insert(2);
        nodes.insert(3);
        let weights = [[0, 1, 4, 3], [1, 0, 2, 5], [4, 2, 0, 6], [3, 5, 6, 0]];
        let connected = [
            [true, true, true, false],
            [true, true, true, true],
            [true, true, true, true],
            [false, true, true, true],
        ];
        let result = compute_minimum_spanning_tree_with_connectivity(
            &nodes,
            |i, j| weights[i][j],
            |i, j| connected[i][j],
            |weight| weight,
        );
        assert_eq!(result, 8);
    }

    #[test]
    fn test_compute_minimum_spanning_tree_with_asymmetric_callbacks() {
        let mut nodes = Set::with_capacity(3);
        nodes.insert(0);
        nodes.insert(1);
        nodes.insert(2);
        let weights = [[0, 10, 1], [100, 0, 2], [100, 100, 0]];
        let connected = [
            [true, true, true],
            [false, true, true],
            [false, false, true],
        ];
        let result = compute_minimum_spanning_tree_with_connectivity(
            &nodes,
            |i, j| weights[i][j],
            |i, j| connected[i][j],
            |weight| weight,
        );
        assert_eq!(result, 3);
    }

    #[test]
    fn test_compute_minimum_spanning_tree_with_reverse_connectivity() {
        let mut nodes = Set::with_capacity(3);
        nodes.insert(0);
        nodes.insert(1);
        nodes.insert(2);
        let weights = [[0, 10, 5], [1, 0, 7], [4, 2, 0]];
        let connected = [
            [true, false, true],
            [true, true, false],
            [false, true, true],
        ];
        let result = compute_minimum_spanning_tree_with_connectivity(
            &nodes,
            |i, j| weights[i][j],
            |i, j| connected[i][j],
            |weight| weight,
        );
        assert_eq!(result, 3);
    }

    #[test]
    fn test_sort_minimum_spanning_tree_edges_with_connectivity_and_connected_callback() {
        let weights = vec![
            vec![0, 1, 4, 3],
            vec![1, 0, 2, 5],
            vec![4, 2, 0, 6],
            vec![3, 5, 6, 0],
        ];
        let connected = [
            [true, true, true, false],
            [true, true, true, true],
            [true, true, true, true],
            [false, true, true, true],
        ];
        let result = sort_minimum_spanning_tree_edges_with_connectivity(
            &weights,
            |i, j| connected[i][j],
            |weight| weight,
        );
        assert_eq!(
            result,
            vec![(0, 1, 1), (1, 2, 2), (0, 2, 4), (1, 3, 5), (2, 3, 6)]
        );
    }

    #[test]
    fn test_sort_minimum_spanning_tree_edges_with_reverse_connectivity() {
        let weights = vec![vec![0, 10, 5], vec![1, 0, 7], vec![4, 2, 0]];
        let connected = [
            [true, false, true],
            [true, true, false],
            [false, true, true],
        ];
        let result = sort_minimum_spanning_tree_edges_with_connectivity(
            &weights,
            |i, j| connected[i][j],
            |weight| weight,
        );
        assert_eq!(result, vec![(0, 1, 1), (1, 2, 2), (0, 2, 5)]);
    }

    #[test]
    fn test_compute_minimum_spanning_tree_from_sorted_edges() {
        let mut nodes = Set::with_capacity(4);
        nodes.insert(0);
        nodes.insert(1);
        nodes.insert(3);
        let sorted_edges = vec![
            (0, 1, 1),
            (1, 2, 2),
            (0, 3, 3),
            (0, 2, 4),
            (1, 3, 5),
            (2, 3, 6),
        ];
        let result = compute_minimum_spanning_tree_from_sorted_edges(&nodes, sorted_edges);
        assert_eq!(result, 4);
    }

    #[test]
    fn test_compute_minimum_spanning_tree_with_connectivity_continuous() {
        let mut nodes = Set::with_capacity(4);
        nodes.insert(0);
        nodes.insert(1);
        nodes.insert(2);
        nodes.insert(3);
        let weights = [
            [0.0, 1.5, 4.0, 3.0],
            [1.5, 0.0, 2.0, 5.0],
            [4.0, 2.0, 0.0, 6.0],
            [3.0, 5.0, 6.0, 0.0],
        ];
        let result = compute_minimum_spanning_tree_with_connectivity(
            &nodes,
            |i, j| weights[i][j],
            |_, _| true,
            OrderedFloat,
        );
        assert_eq!(result, 6.5);
    }

    #[test]
    fn test_compute_minimum_spanning_tree_empty() {
        let nodes = Set::with_capacity(4);
        let result = compute_minimum_spanning_tree_with_connectivity(
            &nodes,
            |_, _| 1,
            |_, _| true,
            |weight| weight,
        );
        assert_eq!(result, 0);
    }
}
