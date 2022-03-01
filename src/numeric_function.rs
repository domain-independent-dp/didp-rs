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
