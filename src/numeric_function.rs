use crate::variable;
use std::collections;
use std::iter;

pub struct NumericFunction1D<T: variable::Numeric>(Vec<T>);

impl<T: variable::Numeric> NumericFunction1D<T> {
    pub fn new(vector: Vec<T>) -> NumericFunction1D<T> {
        NumericFunction1D(vector)
    }

    pub fn eval(&self, x: variable::ElementVariable) -> T {
        self.0[x]
    }

    pub fn sum(&self, x: &variable::SetVariable) -> T {
        x.ones().map(|x| self.eval(x)).sum()
    }
}

pub struct NumericFunction2D<T: variable::Numeric>(Vec<Vec<T>>);

impl<T: variable::Numeric> NumericFunction2D<T> {
    pub fn new(vector: Vec<Vec<T>>) -> NumericFunction2D<T> {
        NumericFunction2D(vector)
    }

    pub fn eval(&self, x: variable::ElementVariable, y: variable::ElementVariable) -> T {
        self.0[x][y]
    }

    pub fn sum(&self, x: &variable::SetVariable, y: &variable::SetVariable) -> T {
        x.ones()
            .map(|x| y.ones().map(|y| self.eval(x, y)).sum())
            .sum()
    }

    pub fn sum_x(&self, x: &variable::SetVariable, y: variable::ElementVariable) -> T {
        x.ones()
            .zip(iter::repeat(y))
            .map(|(x, y)| self.eval(x, y))
            .sum()
    }

    pub fn sum_y(&self, x: variable::ElementVariable, y: &variable::SetVariable) -> T {
        y.ones()
            .zip(iter::repeat(x))
            .map(|(y, x)| self.eval(x, y))
            .sum()
    }
}

pub struct NumericFunction3D<T: variable::Numeric>(Vec<Vec<Vec<T>>>);

impl<T: variable::Numeric> NumericFunction3D<T> {
    pub fn new(vector: Vec<Vec<Vec<T>>>) -> NumericFunction3D<T> {
        NumericFunction3D(vector)
    }

    pub fn eval(
        &self,
        x: variable::ElementVariable,
        y: variable::ElementVariable,
        z: variable::ElementVariable,
    ) -> T {
        self.0[x][y][z]
    }

    pub fn sum(
        &self,
        x: &variable::SetVariable,
        y: &variable::SetVariable,
        z: &variable::SetVariable,
    ) -> T {
        x.ones()
            .map(|x| {
                y.ones()
                    .map(|y| z.ones().map(|z| self.eval(x, y, z)).sum())
                    .sum()
            })
            .sum()
    }

    pub fn sum_x(
        &self,
        x: &variable::SetVariable,
        y: variable::ElementVariable,
        z: variable::ElementVariable,
    ) -> T {
        x.ones().map(|x| self.eval(x, y, z)).sum()
    }

    pub fn sum_y(
        &self,
        x: variable::ElementVariable,
        y: &variable::SetVariable,
        z: variable::ElementVariable,
    ) -> T {
        y.ones().map(|y| self.eval(x, y, z)).sum()
    }

    pub fn sum_z(
        &self,
        x: variable::ElementVariable,
        y: variable::ElementVariable,
        z: &variable::SetVariable,
    ) -> T {
        z.ones().map(|z| self.eval(x, y, z)).sum()
    }

    pub fn sum_xy(
        &self,
        x: &variable::SetVariable,
        y: &variable::SetVariable,
        z: variable::ElementVariable,
    ) -> T {
        x.ones()
            .map(|x| y.ones().map(|y| self.eval(x, y, z)).sum())
            .sum()
    }

    pub fn sum_xz(
        &self,
        x: &variable::SetVariable,
        y: variable::ElementVariable,
        z: &variable::SetVariable,
    ) -> T {
        x.ones()
            .map(|x| z.ones().map(|z| self.eval(x, y, z)).sum())
            .sum()
    }

    pub fn sum_yz(
        &self,
        x: variable::ElementVariable,
        y: &variable::SetVariable,
        z: &variable::SetVariable,
    ) -> T {
        y.ones()
            .map(|y| z.ones().map(|z| self.eval(x, y, z)).sum())
            .sum()
    }
}

pub struct NumericFunction<T: variable::Numeric>(
    collections::HashMap<Vec<variable::ElementVariable>, T>,
);

impl<T: variable::Numeric> NumericFunction<T> {
    pub fn new(map: collections::HashMap<Vec<variable::ElementVariable>, T>) -> NumericFunction<T> {
        NumericFunction(map)
    }

    pub fn eval(&self, args: &[variable::ElementVariable]) -> T {
        self.0[args]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn function_1d_eval() {
        let f = NumericFunction1D::new(vec![10, 20, 30]);
        assert_eq!(f.eval(0), 10);
        assert_eq!(f.eval(1), 20);
        assert_eq!(f.eval(2), 30);
    }

    #[test]
    fn function_1d_sum_eval() {
        let f = NumericFunction1D::new(vec![10, 20, 30]);
        let mut x = variable::SetVariable::with_capacity(3);
        x.insert(0);
        x.insert(2);
        assert_eq!(f.sum(&x), 40);
    }

    #[test]
    fn function_2d_eval() {
        let f = NumericFunction2D::new(vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]]);
        assert_eq!(f.eval(0, 1), 20);
    }

    #[test]
    fn function_2d_sum_eval() {
        let f = NumericFunction2D::new(vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]]);
        let mut x = variable::SetVariable::with_capacity(3);
        x.insert(0);
        x.insert(2);
        let mut y = variable::SetVariable::with_capacity(3);
        y.insert(0);
        y.insert(1);
        assert_eq!(f.sum(&x, &y), 180);
    }

    #[test]
    fn function_2d_sum_x_eval() {
        let f = NumericFunction2D::new(vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]]);
        let mut x = variable::SetVariable::with_capacity(3);
        x.insert(0);
        x.insert(2);
        assert_eq!(f.sum_x(&x, 0), 80);
    }

    #[test]
    fn function_2d_sum_y_eval() {
        let f = NumericFunction2D::new(vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]]);
        let mut y = variable::SetVariable::with_capacity(3);
        y.insert(0);
        y.insert(2);
        assert_eq!(f.sum_y(0, &y), 40);
    }

    #[test]
    fn function_3d_eval() {
        let f = NumericFunction3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        assert_eq!(f.eval(0, 1, 2), 60);
    }

    #[test]
    fn function_3d_sum_eval() {
        let f = NumericFunction3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let mut x = variable::SetVariable::with_capacity(3);
        x.insert(0);
        x.insert(2);
        let mut y = variable::SetVariable::with_capacity(3);
        y.insert(0);
        y.insert(1);
        let mut z = variable::SetVariable::with_capacity(3);
        z.insert(0);
        z.insert(1);
        assert_eq!(f.sum(&x, &y, &z), 240);
    }

    #[test]
    fn function_3d_sum_x_eval() {
        let f = NumericFunction3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let mut x = variable::SetVariable::with_capacity(3);
        x.insert(0);
        x.insert(2);
        assert_eq!(f.sum_x(&x, 1, 2), 120);
    }

    #[test]
    fn function_3d_sum_y_eval() {
        let f = NumericFunction3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let mut y = variable::SetVariable::with_capacity(3);
        y.insert(0);
        y.insert(2);
        assert_eq!(f.sum_y(1, &y, 2), 120);
    }

    #[test]
    fn function_3d_sum_z_eval() {
        let f = NumericFunction3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let mut z = variable::SetVariable::with_capacity(3);
        z.insert(0);
        z.insert(2);
        assert_eq!(f.sum_z(1, 2, &z), 160);
    }

    #[test]
    fn function_3d_sum_xy_eval() {
        let f = NumericFunction3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let mut x = variable::SetVariable::with_capacity(3);
        x.insert(0);
        x.insert(2);
        let mut y = variable::SetVariable::with_capacity(3);
        y.insert(0);
        y.insert(1);
        assert_eq!(f.sum_xy(&x, &y, 2), 180);
    }

    #[test]
    fn function_3d_sum_xz_eval() {
        let f = NumericFunction3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let mut x = variable::SetVariable::with_capacity(3);
        x.insert(0);
        x.insert(2);
        let mut z = variable::SetVariable::with_capacity(3);
        z.insert(0);
        z.insert(1);
        assert_eq!(f.sum_xz(&x, 2, &z), 300);
    }

    #[test]
    fn function_3d_sum_yz_eval() {
        let f = NumericFunction3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let mut y = variable::SetVariable::with_capacity(3);
        y.insert(0);
        y.insert(2);
        let mut z = variable::SetVariable::with_capacity(3);
        z.insert(0);
        z.insert(1);
        assert_eq!(f.sum_yz(2, &y, &z), 180);
    }

    #[test]
    fn function_eval() {
        let mut map =
            collections::HashMap::<Vec<variable::ElementVariable>, variable::IntegerVariable>::new(
            );
        let key = vec![0, 1, 0, 0];
        map.insert(key, 100);
        let key = vec![0, 1, 0, 1];
        map.insert(key, 200);
        let key = vec![0, 1, 2, 0];
        map.insert(key, 300);
        let key = vec![0, 1, 2, 1];
        map.insert(key, 400);
        let f = NumericFunction::new(map);
        assert_eq!(f.eval(&[0, 1, 0, 0]), 100);
        assert_eq!(f.eval(&[0, 1, 0, 1]), 200);
        assert_eq!(f.eval(&[0, 1, 2, 0]), 300);
        assert_eq!(f.eval(&[0, 1, 2, 1]), 400);
    }
}
