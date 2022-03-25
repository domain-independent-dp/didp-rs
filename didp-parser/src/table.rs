use crate::variable::{Element, Numeric, Set};
use approx::{AbsDiffEq, RelativeEq};
use std::collections;

#[derive(Debug, PartialEq, Clone)]
pub struct Table1D<T>(Vec<T>);

impl<T> Table1D<T> {
    pub fn new(vector: Vec<T>) -> Table1D<T> {
        Table1D(vector)
    }

    pub fn get(&self, x: Element) -> &T {
        &self.0[x]
    }
}

impl<T: Copy> Table1D<T> {
    pub fn eval(&self, x: Element) -> T {
        self.0[x]
    }
}

impl<T: Numeric> Table1D<T> {
    pub fn sum(&self, x: &Set) -> T {
        x.ones().map(|x| self.eval(x)).sum()
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

#[derive(Debug, PartialEq, Clone)]
pub struct Table2D<T>(Vec<Vec<T>>);

impl<T> Table2D<T> {
    pub fn new(vector: Vec<Vec<T>>) -> Table2D<T> {
        Table2D(vector)
    }

    pub fn get(&self, x: Element, y: Element) -> &T {
        &self.0[x][y]
    }
}

impl<T: Copy> Table2D<T> {
    pub fn eval(&self, x: Element, y: Element) -> T {
        self.0[x][y]
    }
}

impl<T: Numeric> Table2D<T> {
    pub fn sum(&self, x: &Set, y: &Set) -> T {
        x.ones().map(|x| self.sum_y(x, y)).sum()
    }

    pub fn sum_x(&self, x: &Set, y: Element) -> T {
        x.ones().map(|x| self.eval(x, y)).sum()
    }

    pub fn sum_y(&self, x: Element, y: &Set) -> T {
        y.ones().map(|y| self.eval(x, y)).sum()
    }

    pub fn zip_sum(&self, x: &[Element], y: &[Element]) -> T {
        x.iter().zip(y.iter()).map(|(x, y)| self.eval(*x, *y)).sum()
    }

    pub fn vector_sum_x(&self, x: &[Element], y: Element) -> T {
        x.iter().map(|x| self.eval(*x, y)).sum()
    }

    pub fn vector_sum_y(&self, x: Element, y: &[Element]) -> T {
        y.iter().map(|y| self.eval(x, *y)).sum()
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

#[derive(Debug, PartialEq, Clone)]
pub struct Table3D<T>(Vec<Vec<Vec<T>>>);

impl<T> Table3D<T> {
    pub fn new(vector: Vec<Vec<Vec<T>>>) -> Table3D<T> {
        Table3D(vector)
    }

    pub fn get(&self, x: Element, y: Element, z: Element) -> &T {
        &self.0[x][y][z]
    }
}

impl<T: Copy> Table3D<T> {
    pub fn eval(&self, x: Element, y: Element, z: Element) -> T {
        self.0[x][y][z]
    }
}

impl<T: Numeric> Table3D<T> {
    pub fn sum(&self, x: &Set, y: &Set, z: &Set) -> T {
        x.ones().map(|x| self.sum_yz(x, y, z)).sum()
    }

    pub fn sum_x(&self, x: &Set, y: Element, z: Element) -> T {
        x.ones().map(|x| self.eval(x, y, z)).sum()
    }

    pub fn sum_y(&self, x: Element, y: &Set, z: Element) -> T {
        y.ones().map(|y| self.eval(x, y, z)).sum()
    }

    pub fn sum_z(&self, x: Element, y: Element, z: &Set) -> T {
        z.ones().map(|z| self.eval(x, y, z)).sum()
    }

    pub fn sum_xy(&self, x: &Set, y: &Set, z: Element) -> T {
        x.ones().map(|x| self.sum_y(x, y, z)).sum()
    }

    pub fn sum_xz(&self, x: &Set, y: Element, z: &Set) -> T {
        x.ones().map(|x| self.sum_z(x, y, z)).sum()
    }

    pub fn sum_yz(&self, x: Element, y: &Set, z: &Set) -> T {
        y.ones().map(|y| self.sum_z(x, y, z)).sum()
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

#[derive(Debug, PartialEq, Clone)]
pub struct Table<T> {
    map: collections::HashMap<Vec<Element>, T>,
    default: T,
}

impl<T> Table<T> {
    pub fn new(map: collections::HashMap<Vec<Element>, T>, default: T) -> Table<T> {
        Table { map, default }
    }

    pub fn get(&self, args: &[Element]) -> &T {
        match self.map.get(args) {
            Some(value) => value,
            None => &self.default,
        }
    }
}

impl<T: Copy> Table<T> {
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
    use crate::variable::*;
    use approx::{assert_relative_eq, assert_relative_ne};

    #[test]
    fn table_1d_get() {
        let f = Table1D::new(vec![10, 20, 30]);
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
        let mut x = Set::with_capacity(3);
        x.insert(0);
        x.insert(2);
        assert_eq!(f.sum(&x), 40);
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
    fn table_2d_eval() {
        let f = Table2D::new(vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]]);
        assert_eq!(f.eval(0, 1), 20);
    }

    #[test]
    fn table_2d_sum_eval() {
        let f = Table2D::new(vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]]);
        let mut x = Set::with_capacity(3);
        x.insert(0);
        x.insert(2);
        let mut y = Set::with_capacity(3);
        y.insert(0);
        y.insert(1);
        assert_eq!(f.sum(&x, &y), 180);
    }

    #[test]
    fn table_2d_sum_x_eval() {
        let f = Table2D::new(vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]]);
        let mut x = Set::with_capacity(3);
        x.insert(0);
        x.insert(2);
        assert_eq!(f.sum_x(&x, 0), 80);
    }

    #[test]
    fn table_2d_sum_y_eval() {
        let f = Table2D::new(vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]]);
        let mut y = Set::with_capacity(3);
        y.insert(0);
        y.insert(2);
        assert_eq!(f.sum_y(0, &y), 40);
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
        let mut x = Set::with_capacity(3);
        x.insert(0);
        x.insert(2);
        let mut y = Set::with_capacity(3);
        y.insert(0);
        y.insert(1);
        let mut z = Set::with_capacity(3);
        z.insert(0);
        z.insert(1);
        assert_eq!(f.sum(&x, &y, &z), 240);
    }

    #[test]
    fn table_3d_sum_x_eval() {
        let f = Table3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let mut x = Set::with_capacity(3);
        x.insert(0);
        x.insert(2);
        assert_eq!(f.sum_x(&x, 1, 2), 120);
    }

    #[test]
    fn table_3d_sum_y_eval() {
        let f = Table3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let mut y = Set::with_capacity(3);
        y.insert(0);
        y.insert(2);
        assert_eq!(f.sum_y(1, &y, 2), 120);
    }

    #[test]
    fn table_3d_sum_z_eval() {
        let f = Table3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let mut z = Set::with_capacity(3);
        z.insert(0);
        z.insert(2);
        assert_eq!(f.sum_z(1, 2, &z), 160);
    }

    #[test]
    fn table_3d_sum_xy_eval() {
        let f = Table3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let mut x = Set::with_capacity(3);
        x.insert(0);
        x.insert(2);
        let mut y = Set::with_capacity(3);
        y.insert(0);
        y.insert(1);
        assert_eq!(f.sum_xy(&x, &y, 2), 180);
    }

    #[test]
    fn table_3d_sum_xz_eval() {
        let f = Table3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let mut x = Set::with_capacity(3);
        x.insert(0);
        x.insert(2);
        let mut z = Set::with_capacity(3);
        z.insert(0);
        z.insert(1);
        assert_eq!(f.sum_xz(&x, 2, &z), 300);
    }

    #[test]
    fn table_3d_sum_yz_eval() {
        let f = Table3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let mut y = Set::with_capacity(3);
        y.insert(0);
        y.insert(2);
        let mut z = Set::with_capacity(3);
        z.insert(0);
        z.insert(1);
        assert_eq!(f.sum_yz(2, &y, &z), 180);
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
        let mut map = collections::HashMap::<Vec<Element>, Integer>::new();
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
    fn table_eval() {
        let mut map = collections::HashMap::<Vec<Element>, Integer>::new();
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
        let mut map = collections::HashMap::<Vec<Element>, Continuous>::new();
        map.insert(vec![0, 1, 0, 0], 10.0);
        let t1 = Table::new(map, 0.0);
        let mut map = collections::HashMap::<Vec<Element>, Continuous>::new();
        map.insert(vec![0, 1, 0, 0], 10.0);
        let t2 = Table::new(map, 0.0);
        assert_relative_eq!(t1, t2);
        let mut map = collections::HashMap::<Vec<Element>, Continuous>::new();
        map.insert(vec![0, 1, 0, 0], 11.0);
        let t2 = Table::new(map, 0.0);
        assert_relative_ne!(t1, t2);
        let mut map = collections::HashMap::<Vec<Element>, Continuous>::new();
        map.insert(vec![0, 0, 0, 0], 10.0);
        let t2 = Table::new(map, 0.0);
        assert_relative_ne!(t1, t2);
        let mut map = collections::HashMap::<Vec<Element>, Continuous>::new();
        map.insert(vec![0, 1, 0, 0], 10.0);
        let t2 = Table::new(map, 1.0);
        assert_relative_ne!(t1, t2);
    }
}
