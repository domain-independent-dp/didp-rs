use crate::variable;
use approx::{AbsDiffEq, RelativeEq};
use std::collections;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Table1D<T: Copy>(Vec<T>);

impl<T: Copy> Table1D<T> {
    pub fn new(vector: Vec<T>) -> Table1D<T> {
        Table1D(vector)
    }

    pub fn eval(&self, x: variable::Element) -> T {
        self.0[x]
    }
}

impl<T: variable::Numeric> Table1D<T> {
    pub fn sum(&self, x: &variable::Set) -> T {
        x.ones().map(|x| self.eval(x)).sum()
    }

    pub fn sum_slice(&self, x: &[variable::Element]) -> T {
        x.iter().map(|x| self.eval(*x)).sum()
    }
}

impl<T: Copy + AbsDiffEq> AbsDiffEq for Table1D<T>
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

impl<T: Copy + RelativeEq> RelativeEq for Table1D<T>
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

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Table2D<T: Copy>(Vec<Vec<T>>);

impl<T: Copy> Table2D<T> {
    pub fn new(vector: Vec<Vec<T>>) -> Table2D<T> {
        Table2D(vector)
    }

    pub fn eval(&self, x: variable::Element, y: variable::Element) -> T {
        self.0[x][y]
    }
}

impl<T: variable::Numeric> Table2D<T> {
    pub fn sum(&self, x: &variable::Set, y: &variable::Set) -> T {
        x.ones().map(|x| self.sum_y(x, y)).sum()
    }

    pub fn sum_x(&self, x: &variable::Set, y: variable::Element) -> T {
        x.ones().map(|x| self.eval(x, y)).sum()
    }

    pub fn sum_y(&self, x: variable::Element, y: &variable::Set) -> T {
        y.ones().map(|y| self.eval(x, y)).sum()
    }
}

impl<T: Copy + AbsDiffEq> AbsDiffEq for Table2D<T>
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

impl<T: Copy + RelativeEq> RelativeEq for Table2D<T>
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

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Table3D<T: Copy>(Vec<Vec<Vec<T>>>);

impl<T: Copy> Table3D<T> {
    pub fn new(vector: Vec<Vec<Vec<T>>>) -> Table3D<T> {
        Table3D(vector)
    }

    pub fn eval(&self, x: variable::Element, y: variable::Element, z: variable::Element) -> T {
        self.0[x][y][z]
    }
}

impl<T: variable::Numeric> Table3D<T> {
    pub fn sum(&self, x: &variable::Set, y: &variable::Set, z: &variable::Set) -> T {
        x.ones().map(|x| self.sum_yz(x, y, z)).sum()
    }

    pub fn sum_x(&self, x: &variable::Set, y: variable::Element, z: variable::Element) -> T {
        x.ones().map(|x| self.eval(x, y, z)).sum()
    }

    pub fn sum_y(&self, x: variable::Element, y: &variable::Set, z: variable::Element) -> T {
        y.ones().map(|y| self.eval(x, y, z)).sum()
    }

    pub fn sum_z(&self, x: variable::Element, y: variable::Element, z: &variable::Set) -> T {
        z.ones().map(|z| self.eval(x, y, z)).sum()
    }

    pub fn sum_xy(&self, x: &variable::Set, y: &variable::Set, z: variable::Element) -> T {
        x.ones().map(|x| self.sum_y(x, y, z)).sum()
    }

    pub fn sum_xz(&self, x: &variable::Set, y: variable::Element, z: &variable::Set) -> T {
        x.ones().map(|x| self.sum_z(x, y, z)).sum()
    }

    pub fn sum_yz(&self, x: variable::Element, y: &variable::Set, z: &variable::Set) -> T {
        y.ones().map(|y| self.sum_z(x, y, z)).sum()
    }
}

impl<T: Copy + AbsDiffEq> AbsDiffEq for Table3D<T>
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

impl<T: Copy + RelativeEq> RelativeEq for Table3D<T>
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

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Table<T: Copy> {
    map: collections::HashMap<Vec<variable::Element>, T>,
    default: T,
}

impl<T: Copy> Table<T> {
    pub fn new(map: collections::HashMap<Vec<variable::Element>, T>, default: T) -> Table<T> {
        Table { map, default }
    }

    pub fn eval(&self, args: &[variable::Element]) -> T {
        match self.map.get(args) {
            Some(value) => *value,
            None => self.default,
        }
    }
}

impl<T: Copy + AbsDiffEq> AbsDiffEq for Table<T>
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

impl<T: Copy + RelativeEq> RelativeEq for Table<T>
where
    T::Epsilon: Copy,
{
    fn default_max_relative() -> T::Epsilon {
        T::default_max_relative()
    }

    fn relative_eq(&self, other: &Self, epsilon: T::Epsilon, max_relative: T::Epsilon) -> bool {
        if self.map.len() != other.map.len()
            || self
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

    #[test]
    fn function_1d_eval() {
        let f = Table1D::new(vec![10, 20, 30]);
        assert_eq!(f.eval(0), 10);
        assert_eq!(f.eval(1), 20);
        assert_eq!(f.eval(2), 30);
    }

    #[test]
    fn function_1d_sum_eval() {
        let f = Table1D::new(vec![10, 20, 30]);
        let mut x = variable::Set::with_capacity(3);
        x.insert(0);
        x.insert(2);
        assert_eq!(f.sum(&x), 40);
    }

    #[test]
    fn function_2d_eval() {
        let f = Table2D::new(vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]]);
        assert_eq!(f.eval(0, 1), 20);
    }

    #[test]
    fn function_2d_sum_eval() {
        let f = Table2D::new(vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]]);
        let mut x = variable::Set::with_capacity(3);
        x.insert(0);
        x.insert(2);
        let mut y = variable::Set::with_capacity(3);
        y.insert(0);
        y.insert(1);
        assert_eq!(f.sum(&x, &y), 180);
    }

    #[test]
    fn function_2d_sum_x_eval() {
        let f = Table2D::new(vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]]);
        let mut x = variable::Set::with_capacity(3);
        x.insert(0);
        x.insert(2);
        assert_eq!(f.sum_x(&x, 0), 80);
    }

    #[test]
    fn function_2d_sum_y_eval() {
        let f = Table2D::new(vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]]);
        let mut y = variable::Set::with_capacity(3);
        y.insert(0);
        y.insert(2);
        assert_eq!(f.sum_y(0, &y), 40);
    }

    #[test]
    fn function_3d_eval() {
        let f = Table3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        assert_eq!(f.eval(0, 1, 2), 60);
    }

    #[test]
    fn function_3d_sum_eval() {
        let f = Table3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let mut x = variable::Set::with_capacity(3);
        x.insert(0);
        x.insert(2);
        let mut y = variable::Set::with_capacity(3);
        y.insert(0);
        y.insert(1);
        let mut z = variable::Set::with_capacity(3);
        z.insert(0);
        z.insert(1);
        assert_eq!(f.sum(&x, &y, &z), 240);
    }

    #[test]
    fn function_3d_sum_x_eval() {
        let f = Table3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let mut x = variable::Set::with_capacity(3);
        x.insert(0);
        x.insert(2);
        assert_eq!(f.sum_x(&x, 1, 2), 120);
    }

    #[test]
    fn function_3d_sum_y_eval() {
        let f = Table3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let mut y = variable::Set::with_capacity(3);
        y.insert(0);
        y.insert(2);
        assert_eq!(f.sum_y(1, &y, 2), 120);
    }

    #[test]
    fn function_3d_sum_z_eval() {
        let f = Table3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let mut z = variable::Set::with_capacity(3);
        z.insert(0);
        z.insert(2);
        assert_eq!(f.sum_z(1, 2, &z), 160);
    }

    #[test]
    fn function_3d_sum_xy_eval() {
        let f = Table3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let mut x = variable::Set::with_capacity(3);
        x.insert(0);
        x.insert(2);
        let mut y = variable::Set::with_capacity(3);
        y.insert(0);
        y.insert(1);
        assert_eq!(f.sum_xy(&x, &y, 2), 180);
    }

    #[test]
    fn function_3d_sum_xz_eval() {
        let f = Table3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let mut x = variable::Set::with_capacity(3);
        x.insert(0);
        x.insert(2);
        let mut z = variable::Set::with_capacity(3);
        z.insert(0);
        z.insert(1);
        assert_eq!(f.sum_xz(&x, 2, &z), 300);
    }

    #[test]
    fn function_3d_sum_yz_eval() {
        let f = Table3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let mut y = variable::Set::with_capacity(3);
        y.insert(0);
        y.insert(2);
        let mut z = variable::Set::with_capacity(3);
        z.insert(0);
        z.insert(1);
        assert_eq!(f.sum_yz(2, &y, &z), 180);
    }

    #[test]
    fn function_eval() {
        let mut map = collections::HashMap::<Vec<variable::Element>, variable::Integer>::new();
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
}
