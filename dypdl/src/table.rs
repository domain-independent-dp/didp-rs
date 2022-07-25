use crate::variable_type::{Element, Numeric};
use approx::{AbsDiffEq, RelativeEq};
use rustc_hash::FxHashMap;

/// 1D table of constants.
#[derive(Debug, PartialEq, Clone)]
pub struct Table1D<T>(Vec<T>);

impl<T> Table1D<T> {
    #[inline]
    pub fn new(vector: Vec<T>) -> Table1D<T> {
        Table1D(vector)
    }

    /// Returns a constant.
    ///
    /// # Panics
    ///
    /// the index is out of bound.
    #[inline]
    pub fn get(&self, x: Element) -> &T {
        &self.0[x]
    }

    /// Sets a constant.
    ///
    /// # Panics
    ///
    /// the index is out of bound.
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
    /// the index is out of bound.
    pub fn eval(&self, x: Element) -> T {
        self.0[x]
    }
}

impl<T: Numeric> Table1D<T> {
    #[inline]
    /// Returns the sum of constants.
    ///
    /// # Panics
    ///
    /// an index is out of bound.
    pub fn sum<I>(&self, x: I) -> T
    where
        I: Iterator<Item = Element>,
    {
        x.map(|x| self.eval(x)).sum()
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
#[derive(Debug, PartialEq, Clone)]
pub struct Table2D<T>(Vec<Vec<T>>);

impl<T> Table2D<T> {
    #[inline]
    pub fn new(vector: Vec<Vec<T>>) -> Table2D<T> {
        Table2D(vector)
    }

    /// Returns a constant.
    ///
    /// # Panics
    ///
    /// the index is out of bound.
    #[inline]
    pub fn get(&self, x: Element, y: Element) -> &T {
        &self.0[x][y]
    }

    /// Sets a constant.
    ///
    /// # Panics
    ///
    /// the index is out of bound.
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
    /// the index is out of bound.
    #[inline]
    pub fn eval(&self, x: Element, y: Element) -> T {
        self.0[x][y]
    }
}

impl<T: Numeric> Table2D<T> {
    /// Returns the sum of elements.
    ///
    /// # Panics
    ///
    /// an index is out of bound.
    #[inline]
    pub fn sum<I, J>(&self, x: I, y: J) -> T
    where
        I: Iterator<Item = Element>,
        J: Iterator<Item = Element> + Clone,
    {
        x.map(|x| self.sum_y(x, y.clone())).sum()
    }

    /// Returns the sum of elements.
    ///
    /// # Panics
    ///
    /// an index is out of bound.
    #[inline]
    pub fn sum_x<I>(&self, x: I, y: Element) -> T
    where
        I: Iterator<Item = Element>,
    {
        x.map(|x| self.eval(x, y)).sum()
    }

    /// Returns the sum of elements.
    ///
    /// # Panics
    ///
    /// an index is out of bound.
    #[inline]
    pub fn sum_y<I>(&self, x: Element, y: I) -> T
    where
        I: Iterator<Item = Element>,
    {
        y.map(|y| self.eval(x, y)).sum()
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
#[derive(Debug, PartialEq, Clone)]
pub struct Table3D<T>(Vec<Vec<Vec<T>>>);

impl<T> Table3D<T> {
    #[inline]
    pub fn new(vector: Vec<Vec<Vec<T>>>) -> Table3D<T> {
        Table3D(vector)
    }

    /// Returns a constant.
    ///
    /// # Panics
    ///
    /// the index is out of bound.
    #[inline]
    pub fn get(&self, x: Element, y: Element, z: Element) -> &T {
        &self.0[x][y][z]
    }

    /// Sets a constant.
    ///
    /// # Panics
    ///
    /// the index is out of bound.
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
    /// the index is out of bound.
    #[inline]
    pub fn eval(&self, x: Element, y: Element, z: Element) -> T {
        self.0[x][y][z]
    }
}

impl<T: Numeric> Table3D<T> {
    /// Returns the sum of constants.
    ///
    /// # Panics
    ///
    /// an index is out of bound.
    #[inline]
    pub fn sum<I, J, K>(&self, x: I, y: J, z: K) -> T
    where
        I: Iterator<Item = Element>,
        J: Iterator<Item = Element> + Clone,
        K: Iterator<Item = Element> + Clone,
    {
        x.map(|x| self.sum_yz(x, y.clone(), z.clone())).sum()
    }

    /// Returns the sum of constants.
    ///
    /// # Panics
    ///
    /// an index is out of bound.
    #[inline]
    pub fn sum_x<I>(&self, x: I, y: Element, z: Element) -> T
    where
        I: Iterator<Item = Element>,
    {
        x.map(|x| self.eval(x, y, z)).sum()
    }

    /// Returns the sum of constants.
    ///
    /// # Panics
    ///
    /// an index is out of bound.
    #[inline]
    pub fn sum_y<I>(&self, x: Element, y: I, z: Element) -> T
    where
        I: Iterator<Item = Element>,
    {
        y.map(|y| self.eval(x, y, z)).sum()
    }

    /// Returns the sum of constants.
    ///
    /// # Panics
    ///
    /// an index is out of bound.
    #[inline]
    pub fn sum_z<I>(&self, x: Element, y: Element, z: I) -> T
    where
        I: Iterator<Item = Element>,
    {
        z.map(|z| self.eval(x, y, z)).sum()
    }

    /// Returns the sum of constants.
    ///
    /// # Panics
    ///
    /// an index is out of bound.
    #[inline]
    pub fn sum_xy<I, J>(&self, x: I, y: J, z: Element) -> T
    where
        I: Iterator<Item = Element>,
        J: Iterator<Item = Element> + Clone,
    {
        x.map(|x| self.sum_y(x, y.clone(), z)).sum()
    }

    /// Returns the sum of constants.
    ///
    /// # Panics
    ///
    /// an index is out of bound.
    #[inline]
    pub fn sum_xz<I, J>(&self, x: I, y: Element, z: J) -> T
    where
        I: Iterator<Item = Element>,
        J: Iterator<Item = Element> + Clone,
    {
        x.map(|x| self.sum_z(x, y, z.clone())).sum()
    }

    /// Returns the sum of constants.
    ///
    /// # Panics
    ///
    /// an index is out of bound.
    #[inline]
    pub fn sum_yz<I, J>(&self, x: Element, y: I, z: J) -> T
    where
        I: Iterator<Item = Element>,
        J: Iterator<Item = Element> + Clone,
    {
        y.map(|y| self.sum_z(x, y, z.clone())).sum()
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
#[derive(Debug, PartialEq, Clone)]
pub struct Table<T> {
    map: FxHashMap<Vec<Element>, T>,
    default: T,
}

impl<T> Table<T> {
    #[inline]
    pub fn new(map: FxHashMap<Vec<Element>, T>, default: T) -> Table<T> {
        Table { map, default }
    }

    /// Returns a constant.
    ///
    /// # Panics
    ///
    /// the index is out of bound.
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
    /// the index is out of bound.
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
    /// the index is out of bound.
    pub fn eval(&self, args: &[Element]) -> T {
        match self.map.get(args) {
            Some(value) => *value,
            None => self.default,
        }
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
    fn table_1d_sum_eval() {
        let f = Table1D::new(vec![10, 20, 30]);
        let x = vec![0, 2];
        assert_eq!(f.sum(x.into_iter()), 40);
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
    fn table_2d_sum_eval() {
        let f = Table2D::new(vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]]);
        let x = vec![0, 2];
        let y = vec![0, 1];
        assert_eq!(f.sum(x.into_iter(), y.into_iter()), 180);
    }

    #[test]
    fn table_2d_sum_x_eval() {
        let f = Table2D::new(vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]]);
        let x = vec![0, 2];
        assert_eq!(f.sum_x(x.into_iter(), 0), 80);
    }

    #[test]
    fn table_2d_sum_y_eval() {
        let f = Table2D::new(vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]]);
        let y = vec![0, 2];
        assert_eq!(f.sum_y(0, y.into_iter()), 40);
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
    fn table_3d_sum_eval() {
        let f = Table3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let x = vec![0, 2];
        let y = vec![0, 1];
        let z = vec![0, 1];
        assert_eq!(f.sum(x.into_iter(), y.into_iter(), z.into_iter()), 240);
    }

    #[test]
    fn table_3d_sum_x_eval() {
        let f = Table3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let x = vec![0, 2];
        assert_eq!(f.sum_x(x.into_iter(), 1, 2), 120);
    }

    #[test]
    fn table_3d_sum_y_eval() {
        let f = Table3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let y = vec![0, 2];
        assert_eq!(f.sum_y(1, y.into_iter(), 2), 120);
    }

    #[test]
    fn table_3d_sum_z_eval() {
        let f = Table3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let z = vec![0, 2];
        assert_eq!(f.sum_z(1, 2, z.into_iter()), 160);
    }

    #[test]
    fn table_3d_sum_xy_eval() {
        let f = Table3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let x = vec![0, 2];
        let y = vec![0, 1];
        assert_eq!(f.sum_xy(x.into_iter(), y.into_iter(), 2), 180);
    }

    #[test]
    fn table_3d_sum_xz_eval() {
        let f = Table3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let x = vec![0, 2];
        let z = vec![0, 1];
        assert_eq!(f.sum_xz(x.into_iter(), 2, z.into_iter()), 300);
    }

    #[test]
    fn table_3d_sum_yz_eval() {
        let f = Table3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let y = vec![0, 2];
        let z = vec![0, 1];
        assert_eq!(f.sum_yz(2, y.into_iter(), z.into_iter()), 180);
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