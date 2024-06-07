use crate::variable_type::{Element, Set};
use approx::{AbsDiffEq, RelativeEq};
use rustc_hash::FxHashMap;

pub trait HasShape {
    /// Returns the size of the Table.
    fn shape(&self) -> Vec<usize>;
}

/// 1D table of constants.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, PartialEq, Clone)]
pub struct Table1D<T>(pub Vec<T>);

impl<T> Table1D<T> {
    /// Returns a new table.
    #[inline]
    pub fn new(vector: Vec<T>) -> Table1D<T> {
        Table1D(vector)
    }

    /// Returns a constant.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bound.
    #[inline]
    pub fn get(&self, x: Element) -> &T {
        &self.0[x]
    }

    /// Sets a constant.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bound.
    #[inline]
    pub fn set(&mut self, x: Element, v: T) {
        self.0[x] = v;
    }

    /// Updates the table.
    #[inline]
    pub fn update(&mut self, vector: Vec<T>) {
        self.0 = vector
    }
}

impl<T: Copy> Table1D<T> {
    #[inline]
    /// Returns a constant.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bound.
    pub fn eval(&self, x: Element) -> T {
        self.0[x]
    }
}

impl Table1D<Set> {
    #[inline]
    /// Return the capacity of a set constant.
    ///
    /// # Panics
    ///
    /// Panics if the table is empty.
    pub fn capacity_of_set(&self) -> usize {
        self.0[0].len()
    }
}

impl<T: AbsDiffEq> AbsDiffEq for Table1D<T>
where
    T::Epsilon: Copy,
{
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> T::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: T::Epsilon) -> bool {
        self.0
            .iter()
            .zip(other.0.iter())
            .all(|(x, y)| x.abs_diff_eq(y, epsilon))
    }
}

impl<T> HasShape for Table1D<T> {
    fn shape(&self) -> Vec<usize> {
        vec![self.0.len()]
    }
}

impl<T: RelativeEq> RelativeEq for Table1D<T>
where
    T::Epsilon: Copy,
{
    fn default_max_relative() -> T::Epsilon {
        T::default_max_relative()
    }

    fn relative_eq(&self, other: &Self, epsilon: T::Epsilon, max_relative: T::Epsilon) -> bool {
        self.0
            .iter()
            .zip(other.0.iter())
            .all(|(x, y)| x.relative_eq(y, epsilon, max_relative))
    }
}

/// 2D table of constants.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, PartialEq, Clone)]
pub struct Table2D<T>(pub Vec<Vec<T>>);

impl<T> Table2D<T> {
    /// Returns a new table.
    #[inline]
    pub fn new(vector: Vec<Vec<T>>) -> Table2D<T> {
        Table2D(vector)
    }

    /// Returns a constant.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bound.
    #[inline]
    pub fn get(&self, x: Element, y: Element) -> &T {
        &self.0[x][y]
    }

    /// Sets a constant.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bound.
    #[inline]
    pub fn set(&mut self, x: Element, y: Element, v: T) {
        self.0[x][y] = v;
    }

    /// Updates the table.
    #[inline]
    pub fn update(&mut self, vector: Vec<Vec<T>>) {
        self.0 = vector
    }
}

impl<T: Copy> Table2D<T> {
    /// Returns a constant.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bound.
    #[inline]
    pub fn eval(&self, x: Element, y: Element) -> T {
        self.0[x][y]
    }
}

impl Table2D<Set> {
    #[inline]
    /// Return the capacity of a set constant.
    ///
    /// # Panics
    ///
    /// Panics if the table is empty.
    pub fn capacity_of_set(&self) -> usize {
        self.0[0][0].len()
    }
}

impl<T: AbsDiffEq> AbsDiffEq for Table2D<T>
where
    T::Epsilon: Copy,
{
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> T::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: T::Epsilon) -> bool {
        self.0.iter().zip(other.0.iter()).all(|(x, y)| {
            x.iter()
                .zip(y.iter())
                .all(|(x, y)| x.abs_diff_eq(y, epsilon))
        })
    }
}

impl<T> HasShape for Table2D<T> {
    fn shape(&self) -> Vec<usize> {
        if self.0.is_empty() {
            vec![0, 0]
        } else {
            vec![self.0.len(), self.0[0].len()]
        }
    }
}

impl<T: RelativeEq> RelativeEq for Table2D<T>
where
    T::Epsilon: Copy,
{
    fn default_max_relative() -> T::Epsilon {
        T::default_max_relative()
    }

    fn relative_eq(&self, other: &Self, epsilon: T::Epsilon, max_relative: T::Epsilon) -> bool {
        self.0.iter().zip(other.0.iter()).all(|(x, y)| {
            x.iter()
                .zip(y.iter())
                .all(|(x, y)| x.relative_eq(y, epsilon, max_relative))
        })
    }
}

/// 3D table of constants.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, PartialEq, Clone)]
pub struct Table3D<T>(pub Vec<Vec<Vec<T>>>);

impl<T> Table3D<T> {
    /// Returns a new table.
    #[inline]
    pub fn new(vector: Vec<Vec<Vec<T>>>) -> Table3D<T> {
        Table3D(vector)
    }

    /// Returns a constant.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bound.
    #[inline]
    pub fn get(&self, x: Element, y: Element, z: Element) -> &T {
        &self.0[x][y][z]
    }

    /// Sets a constant.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bound.
    #[inline]
    pub fn set(&mut self, x: Element, y: Element, z: Element, v: T) {
        self.0[x][y][z] = v;
    }

    /// Updates the table.
    #[inline]
    pub fn update(&mut self, vector: Vec<Vec<Vec<T>>>) {
        self.0 = vector
    }
}

impl<T: Copy> Table3D<T> {
    /// Returns a constant.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bound.
    #[inline]
    pub fn eval(&self, x: Element, y: Element, z: Element) -> T {
        self.0[x][y][z]
    }
}

impl Table3D<Set> {
    #[inline]
    /// Return the capacity of a set constant.
    ///
    /// # Panics
    ///
    /// Panics if the table is empty.
    pub fn capacity_of_set(&self) -> usize {
        self.0[0][0][0].len()
    }
}

impl<T: AbsDiffEq> AbsDiffEq for Table3D<T>
where
    T::Epsilon: Copy,
{
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> T::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: T::Epsilon) -> bool {
        self.0.iter().zip(other.0.iter()).all(|(x, y)| {
            x.iter().zip(y.iter()).all(|(x, y)| {
                x.iter()
                    .zip(y.iter())
                    .all(|(x, y)| x.abs_diff_eq(y, epsilon))
            })
        })
    }
}

impl<T> HasShape for Table3D<T> {
    fn shape(&self) -> Vec<usize> {
        if self.0.is_empty() || self.0[0].is_empty() {
            vec![0, 0, 0]
        } else {
            vec![self.0.len(), self.0[0].len(), self.0[0][0].len()]
        }
    }
}

impl<T: RelativeEq> RelativeEq for Table3D<T>
where
    T::Epsilon: Copy,
{
    fn default_max_relative() -> T::Epsilon {
        T::default_max_relative()
    }

    fn relative_eq(&self, other: &Self, epsilon: T::Epsilon, max_relative: T::Epsilon) -> bool {
        self.0.iter().zip(other.0.iter()).all(|(x, y)| {
            x.iter().zip(y.iter()).all(|(x, y)| {
                x.iter()
                    .zip(y.iter())
                    .all(|(x, y)| x.relative_eq(y, epsilon, max_relative))
            })
        })
    }
}

/// Table of constants.
///
/// A constant is indexed by a sequence of elements, and the dimension is not fixed.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, PartialEq, Clone)]
pub struct Table<T> {
    pub map: FxHashMap<Vec<Element>, T>,
    pub default: T,
}

impl<T> Table<T> {
    /// Returns a new table.
    #[inline]
    pub fn new(map: FxHashMap<Vec<Element>, T>, default: T) -> Table<T> {
        Table { map, default }
    }

    /// Returns a constant.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bound.
    pub fn get(&self, args: &[Element]) -> &T {
        match self.map.get(args) {
            Some(value) => value,
            None => &self.default,
        }
    }

    /// Sets a constant.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bound.
    #[inline]
    pub fn set(&mut self, key: Vec<Element>, v: T) {
        self.map.insert(key, v);
    }

    /// Sets the default value.
    #[inline]
    pub fn set_default(&mut self, default: T) {
        self.default = default
    }

    /// Updates the table.
    #[inline]
    pub fn update(&mut self, map: FxHashMap<Vec<Element>, T>, default: T) {
        self.map = map;
        self.default = default;
    }
}

impl<T: Copy> Table<T> {
    /// Returns a constant.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bound.
    pub fn eval(&self, args: &[Element]) -> T {
        match self.map.get(args) {
            Some(value) => *value,
            None => self.default,
        }
    }
}

impl Table<Set> {
    #[inline]
    /// Return the capacity of a set constant.
    ///
    /// # Panics
    ///
    /// Panics if the table is empty.
    pub fn capacity_of_set(&self) -> usize {
        self.default.len()
    }
}

impl<T: AbsDiffEq> AbsDiffEq for Table<T>
where
    T::Epsilon: Copy,
{
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> T::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: T::Epsilon) -> bool {
        if self.map.len() != other.map.len() || self.default.abs_diff_eq(&other.default, epsilon) {
            return false;
        }
        for (key, x) in &self.map {
            match other.map.get(key) {
                Some(y) if x.abs_diff_eq(y, epsilon) => {}
                _ => return false,
            }
        }
        true
    }
}

impl<T: RelativeEq> RelativeEq for Table<T>
where
    T::Epsilon: Copy,
{
    fn default_max_relative() -> T::Epsilon {
        T::default_max_relative()
    }

    fn relative_eq(&self, other: &Self, epsilon: T::Epsilon, max_relative: T::Epsilon) -> bool {
        if self.map.len() != other.map.len()
            || !self
                .default
                .relative_eq(&other.default, epsilon, max_relative)
        {
            return false;
        }
        for (key, x) in &self.map {
            match other.map.get(key) {
                Some(y) if x.relative_eq(y, epsilon, max_relative) => {}
                _ => return false,
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variable_type::*;
    use approx::{assert_relative_eq, assert_relative_ne};

    #[test]
    fn table_1d_get() {
        let f = Table1D::new(vec![10, 20, 30]);
        assert_eq!(*f.get(0), 10);
        assert_eq!(*f.get(1), 20);
        assert_eq!(*f.get(2), 30);
    }

    #[test]
    fn table_1d_set() {
        let mut f = Table1D::new(vec![10, 20, 30]);
        f.set(0, 0);
        assert_eq!(*f.get(0), 0);
    }

    #[test]
    fn table_1d_update() {
        let mut f = Table1D::new(vec![0, 0, 0]);
        f.update(vec![10, 20, 30]);
        assert_eq!(*f.get(0), 10);
        assert_eq!(*f.get(1), 20);
        assert_eq!(*f.get(2), 30);
    }

    #[test]
    fn table_1d_eval() {
        let f = Table1D::new(vec![10, 20, 30]);
        assert_eq!(f.eval(0), 10);
        assert_eq!(f.eval(1), 20);
        assert_eq!(f.eval(2), 30);
    }

    #[test]
    fn table_1d_capacity_of_set() {
        let f = Table1D::new(vec![Set::with_capacity(3)]);
        assert_eq!(f.capacity_of_set(), 3);
    }

    #[test]
    fn table_1d_shape() {
        let table = Table1D::new(vec![10.0, 20.0, 30.0]);
        assert_eq!(table.shape(), vec![3]);
    }

    #[test]
    fn table_1d_relative_eq() {
        let t1 = Table1D::new(vec![10.0, 20.0, 30.0]);
        let t2 = Table1D::new(vec![10.0, 20.0, 30.0]);
        assert_relative_eq!(t1, t2);
        let t2 = Table1D::new(vec![10.0, 20.0, 31.0]);
        assert_relative_ne!(t1, t2);
    }

    #[test]
    fn table_2d_get() {
        let f = Table2D::new(vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]]);
        assert_eq!(*f.get(0, 1), 20);
    }

    #[test]
    fn table_2d_set() {
        let mut f = Table2D::new(vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]]);
        f.set(0, 1, 0);
        assert_eq!(*f.get(0, 1), 0);
    }

    #[test]
    fn table_2d_update() {
        let mut f = Table2D::new(vec![vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0]]);
        f.update(vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]]);
        assert_eq!(*f.get(0, 1), 20);
    }

    #[test]
    fn table_2d_eval() {
        let f = Table2D::new(vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]]);
        assert_eq!(f.eval(0, 1), 20);
    }

    #[test]
    fn table_2d_capacity_of_set() {
        let f = Table2D::new(vec![vec![Set::with_capacity(3)]]);
        assert_eq!(f.capacity_of_set(), 3);
    }

    #[test]
    fn table_2d_shape() {
        let table = Table2D::new(vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]]);
        assert_eq!(table.shape(), vec![3, 3]);
    }

    #[test]
    fn table_2d_relative_eq() {
        let t1 = Table2D::new(vec![vec![10.0], vec![20.0]]);
        let t2 = Table2D::new(vec![vec![10.0], vec![20.0]]);
        assert_relative_eq!(t1, t2);
        let t2 = Table2D::new(vec![vec![10.0], vec![21.0]]);
        assert_relative_ne!(t1, t2);
    }

    #[test]
    fn table_3d_get() {
        let f = Table3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        assert_eq!(*f.get(0, 1, 2), 60);
    }

    #[test]
    fn table_3d_set() {
        let mut f = Table3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        f.set(0, 1, 2, 0);
        assert_eq!(*f.get(0, 1, 2), 0);
    }

    #[test]
    fn table_3d_update() {
        let mut f = Table3D::new(vec![vec![vec![]]]);
        f.update(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        assert_eq!(*f.get(0, 1, 2), 60);
    }

    #[test]
    fn table_3d_eval() {
        let f = Table3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        assert_eq!(f.eval(0, 1, 2), 60);
    }

    #[test]
    fn table_3d_capacity_of_set() {
        let f = Table3D::new(vec![vec![vec![Set::with_capacity(3)]]]);
        assert_eq!(f.capacity_of_set(), 3);
    }

    #[test]
    fn table_3d_shape() {
        let table = Table3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        assert_eq!(table.shape(), vec![3, 3, 3]);
    }

    #[test]
    fn table_3d_relative_eq() {
        let t1 = Table3D::new(vec![vec![vec![10.0]], vec![vec![20.0]]]);
        let t2 = Table3D::new(vec![vec![vec![10.0]], vec![vec![20.0]]]);
        assert_relative_eq!(t1, t2);
        let t2 = Table3D::new(vec![vec![vec![10.0]], vec![vec![21.0]]]);
        assert_relative_ne!(t1, t2);
    }

    #[test]
    fn table_get() {
        let mut map = FxHashMap::<Vec<Element>, Integer>::default();
        let key = vec![0, 1, 0, 0];
        map.insert(key, 100);
        let key = vec![0, 1, 0, 1];
        map.insert(key, 200);
        let key = vec![0, 1, 2, 0];
        map.insert(key, 300);
        let key = vec![0, 1, 2, 1];
        map.insert(key, 400);
        let f = Table::new(map, 0);
        assert_eq!(*f.get(&[0, 1, 0, 0]), 100);
        assert_eq!(*f.get(&[0, 1, 0, 1]), 200);
        assert_eq!(*f.get(&[0, 1, 2, 0]), 300);
        assert_eq!(*f.get(&[0, 1, 2, 1]), 400);
        assert_eq!(*f.get(&[0, 1, 2, 2]), 0);
    }

    #[test]
    fn table_set() {
        let mut map = FxHashMap::<Vec<Element>, Integer>::default();
        let key = vec![0, 1, 0, 0];
        map.insert(key, 100);
        let key = vec![0, 1, 0, 1];
        map.insert(key, 200);
        let key = vec![0, 1, 2, 0];
        map.insert(key, 300);
        let key = vec![0, 1, 2, 1];
        map.insert(key, 400);
        let mut f = Table::new(map, 0);
        f.set(vec![0, 0, 0, 0], 1);
        assert_eq!(*f.get(&[0, 0, 0, 0]), 1);
        assert_eq!(*f.get(&[0, 1, 0, 0]), 100);
        assert_eq!(*f.get(&[0, 1, 0, 1]), 200);
        assert_eq!(*f.get(&[0, 1, 2, 0]), 300);
        assert_eq!(*f.get(&[0, 1, 2, 1]), 400);
        assert_eq!(*f.get(&[0, 1, 2, 2]), 0);
    }

    #[test]
    fn table_set_default() {
        let mut map = FxHashMap::<Vec<Element>, Integer>::default();
        let key = vec![0, 1, 0, 0];
        map.insert(key, 100);
        let key = vec![0, 1, 0, 1];
        map.insert(key, 200);
        let key = vec![0, 1, 2, 0];
        map.insert(key, 300);
        let key = vec![0, 1, 2, 1];
        map.insert(key, 400);
        let mut f = Table::new(map, 0);
        f.set_default(1);
        assert_eq!(*f.get(&[0, 1, 0, 0]), 100);
        assert_eq!(*f.get(&[0, 1, 0, 1]), 200);
        assert_eq!(*f.get(&[0, 1, 2, 0]), 300);
        assert_eq!(*f.get(&[0, 1, 2, 1]), 400);
        assert_eq!(*f.get(&[0, 1, 2, 2]), 1);
    }

    #[test]
    fn table_update() {
        let mut f = Table::new(FxHashMap::default(), 1);
        let mut map = FxHashMap::<Vec<Element>, Integer>::default();
        let key = vec![0, 1, 0, 0];
        map.insert(key, 100);
        let key = vec![0, 1, 0, 1];
        map.insert(key, 200);
        let key = vec![0, 1, 2, 0];
        map.insert(key, 300);
        let key = vec![0, 1, 2, 1];
        map.insert(key, 400);
        f.update(map, 0);
        assert_eq!(*f.get(&[0, 1, 0, 0]), 100);
        assert_eq!(*f.get(&[0, 1, 0, 1]), 200);
        assert_eq!(*f.get(&[0, 1, 2, 0]), 300);
        assert_eq!(*f.get(&[0, 1, 2, 1]), 400);
        assert_eq!(*f.get(&[0, 1, 2, 2]), 0);
    }

    #[test]
    fn table_eval() {
        let mut map = FxHashMap::<Vec<Element>, Integer>::default();
        let key = vec![0, 1, 0, 0];
        map.insert(key, 100);
        let key = vec![0, 1, 0, 1];
        map.insert(key, 200);
        let key = vec![0, 1, 2, 0];
        map.insert(key, 300);
        let key = vec![0, 1, 2, 1];
        map.insert(key, 400);
        let f = Table::new(map, 0);
        assert_eq!(f.eval(&[0, 1, 0, 0]), 100);
        assert_eq!(f.eval(&[0, 1, 0, 1]), 200);
        assert_eq!(f.eval(&[0, 1, 2, 0]), 300);
        assert_eq!(f.eval(&[0, 1, 2, 1]), 400);
        assert_eq!(f.eval(&[0, 1, 2, 2]), 0);
    }

    #[test]
    fn table_capacity_of_set() {
        let f = Table::new(FxHashMap::default(), Set::with_capacity(3));
        assert_eq!(f.capacity_of_set(), 3);
    }

    #[test]
    fn table_relative_eq() {
        let mut map = FxHashMap::<Vec<Element>, Continuous>::default();
        map.insert(vec![0, 1, 0, 0], 10.0);
        let t1 = Table::new(map, 0.0);
        let mut map = FxHashMap::<Vec<Element>, Continuous>::default();
        map.insert(vec![0, 1, 0, 0], 10.0);
        let t2 = Table::new(map, 0.0);
        assert_relative_eq!(t1, t2);
        let mut map = FxHashMap::<Vec<Element>, Continuous>::default();
        map.insert(vec![0, 1, 0, 0], 11.0);
        let t2 = Table::new(map, 0.0);
        assert_relative_ne!(t1, t2);
        let mut map = FxHashMap::<Vec<Element>, Continuous>::default();
        map.insert(vec![0, 0, 0, 0], 10.0);
        let t2 = Table::new(map, 0.0);
        assert_relative_ne!(t1, t2);
        let mut map = FxHashMap::<Vec<Element>, Continuous>::default();
        map.insert(vec![0, 1, 0, 0], 10.0);
        let t2 = Table::new(map, 1.0);
        assert_relative_ne!(t1, t2);
    }
}
