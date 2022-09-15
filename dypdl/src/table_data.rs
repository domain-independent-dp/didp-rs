use crate::table;
use crate::util::ModelErr;
use crate::variable_type::Element;
use approx::{AbsDiffEq, RelativeEq};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::hash_map::Entry;
use std::marker::PhantomData;

macro_rules! define_table_handle {
    ($x:ident) => {
        /// A struct wrapping the id of a table.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $x<T>(usize, PhantomData<T>);

        impl<T> $x<T> {
            /// Returns the id
            pub fn id(&self) -> usize {
                self.0
            }
        }
    };
}

define_table_handle!(Table1DHandle);
define_table_handle!(Table2DHandle);
define_table_handle!(Table3DHandle);
define_table_handle!(TableHandle);

/// A trait for adding and updating tables of constants.
pub trait TableInterface<T> {
    /// Adds and returns a 1D table.
    ///
    /// # Errors
    ///
    /// if the name is already used.
    fn add_table_1d<U>(&mut self, name: U, v: Vec<T>) -> Result<Table1DHandle<T>, ModelErr>
    where
        String: From<U>;

    /// Set an item in a 1D table.
    ///
    /// # Errors
    ///
    /// if the table is not in the model.
    fn set_table_1d(&mut self, t: Table1DHandle<T>, x: Element, v: T) -> Result<(), ModelErr>;

    /// Update a 1D table.
    ///
    /// # Errors
    ///
    /// if the table is not in the model.
    fn update_table_1d(&mut self, t: Table1DHandle<T>, v: Vec<T>) -> Result<(), ModelErr>;

    /// Adds and returns a 2D table.
    ///
    /// # Errors
    ///
    /// if the name is already used.
    fn add_table_2d<U>(&mut self, name: U, v: Vec<Vec<T>>) -> Result<Table2DHandle<T>, ModelErr>
    where
        String: From<U>;

    /// Set an item in a 2D table.
    ///
    /// # Errors
    ///
    /// if the table is not in the model.
    fn set_table_2d(
        &mut self,
        t: Table2DHandle<T>,
        x: Element,
        y: Element,
        v: T,
    ) -> Result<(), ModelErr>;

    /// Update a 2D table.
    ///
    /// # Errors
    ///
    /// if the table is not in the model.
    fn update_table_2d(&mut self, t: Table2DHandle<T>, v: Vec<Vec<T>>) -> Result<(), ModelErr>;

    /// Adds and returns a 3D table.
    ///
    /// # Errors
    ///
    /// if the name is already used.
    fn add_table_3d<U>(
        &mut self,
        name: U,
        v: Vec<Vec<Vec<T>>>,
    ) -> Result<Table3DHandle<T>, ModelErr>
    where
        String: From<U>;

    /// Set an item in a 3D table.
    ///
    /// # Errors
    ///
    /// if the table is not in the model.
    fn set_table_3d(
        &mut self,
        t: Table3DHandle<T>,
        x: Element,
        y: Element,
        z: Element,
        v: T,
    ) -> Result<(), ModelErr>;

    /// Update a 3D table.
    ///
    /// # Errors
    ///
    /// if the table is not in the model.
    fn update_table_3d(&mut self, t: Table3DHandle<T>, v: Vec<Vec<Vec<T>>>)
        -> Result<(), ModelErr>;

    /// Adds and returns a 3D table.
    ///
    /// # Errors
    ///
    /// if the name is already used.
    fn add_table<U>(
        &mut self,
        name: U,
        map: FxHashMap<Vec<Element>, T>,
        default: T,
    ) -> Result<TableHandle<T>, ModelErr>
    where
        String: From<U>;

    /// Set an item in a table.
    ///
    /// # Errors
    ///
    /// if the table is not in the model.
    fn set_table(&mut self, t: TableHandle<T>, key: Vec<Element>, v: T) -> Result<(), ModelErr>;

    /// Set the default value of a table.
    ///
    /// # Errors
    ///
    /// if the table is not in the model.
    fn set_default(&mut self, t: TableHandle<T>, default: T) -> Result<(), ModelErr>;

    /// Update a table.
    ///
    /// # Errors
    ///
    /// if the table is not in the model.
    fn update_table(
        &mut self,
        t: TableHandle<T>,
        map: FxHashMap<Vec<Element>, T>,
        default: T,
    ) -> Result<(), ModelErr>;
}

/// Tables of constants havint a particular type.
#[derive(Debug, PartialEq, Clone, Default)]
pub struct TableData<T> {
    pub name_to_constant: FxHashMap<String, T>,
    pub tables_1d: Vec<table::Table1D<T>>,
    pub name_to_table_1d: FxHashMap<String, usize>,
    pub tables_2d: Vec<table::Table2D<T>>,
    pub name_to_table_2d: FxHashMap<String, usize>,
    pub tables_3d: Vec<table::Table3D<T>>,
    pub name_to_table_3d: FxHashMap<String, usize>,
    pub tables: Vec<table::Table<T>>,
    pub name_to_table: FxHashMap<String, usize>,
}

impl<T> TableInterface<T> for TableData<T> {
    fn add_table_1d<U>(&mut self, name: U, v: Vec<T>) -> Result<Table1DHandle<T>, ModelErr>
    where
        String: From<U>,
    {
        let name = String::from(name);
        if v.is_empty() {
            return Err(ModelErr::new(format!("1D table `{}` is empty", name)));
        }
        match self.name_to_table_1d.entry(name) {
            Entry::Vacant(e) => {
                let id = self.tables_1d.len();
                self.tables_1d.push(table::Table1D::new(v));
                e.insert(id);
                Ok(Table1DHandle(id, PhantomData::default()))
            }
            Entry::Occupied(e) => Err(ModelErr::new(format!(
                "1D table `{}` already exists",
                e.key()
            ))),
        }
    }

    fn set_table_1d(&mut self, t: Table1DHandle<T>, x: Element, v: T) -> Result<(), ModelErr> {
        self.check_table_1d(t.id())?;
        self.tables_1d[t.id()].set(x, v);
        Ok(())
    }

    fn update_table_1d(&mut self, t: Table1DHandle<T>, v: Vec<T>) -> Result<(), ModelErr> {
        self.check_table_1d(t.id())?;
        if v.is_empty() {
            return Err(ModelErr::new(format!(
                "1D table with id `{}` is updated to be empty",
                t.id()
            )));
        }
        self.tables_1d[t.id()].update(v);
        Ok(())
    }

    fn add_table_2d<U>(&mut self, name: U, v: Vec<Vec<T>>) -> Result<Table2DHandle<T>, ModelErr>
    where
        String: From<U>,
    {
        let name = String::from(name);
        if v.is_empty() || v[0].is_empty() {
            return Err(ModelErr::new(format!("2D table `{}` is empty", name)));
        }
        match self.name_to_table_2d.entry(name) {
            Entry::Vacant(e) => {
                let id = self.tables_2d.len();
                self.tables_2d.push(table::Table2D::new(v));
                e.insert(id);
                Ok(Table2DHandle(id, PhantomData::default()))
            }
            Entry::Occupied(e) => Err(ModelErr::new(format!(
                "2D table `{}` already exists",
                e.key()
            ))),
        }
    }

    fn set_table_2d(
        &mut self,
        t: Table2DHandle<T>,
        x: Element,
        y: Element,
        v: T,
    ) -> Result<(), ModelErr> {
        self.check_table_2d(t.id())?;
        self.tables_2d[t.id()].set(x, y, v);
        Ok(())
    }

    fn update_table_2d(&mut self, t: Table2DHandle<T>, v: Vec<Vec<T>>) -> Result<(), ModelErr> {
        self.check_table_2d(t.id())?;
        if v.is_empty() || v[0].is_empty() {
            return Err(ModelErr::new(format!(
                "2D table with id `{}` is updated to be empty",
                t.id()
            )));
        }
        self.tables_2d[t.id()].update(v);
        Ok(())
    }

    fn add_table_3d<U>(
        &mut self,
        name: U,
        v: Vec<Vec<Vec<T>>>,
    ) -> Result<Table3DHandle<T>, ModelErr>
    where
        String: From<U>,
    {
        let name = String::from(name);
        if v.is_empty() || v[0].is_empty() || v[0][0].is_empty() {
            return Err(ModelErr::new(format!("3D table `{}` is empty", name)));
        }
        match self.name_to_table_3d.entry(name) {
            Entry::Vacant(e) => {
                let id = self.tables_3d.len();
                self.tables_3d.push(table::Table3D::new(v));
                e.insert(id);
                Ok(Table3DHandle(id, PhantomData::default()))
            }
            Entry::Occupied(e) => Err(ModelErr::new(format!(
                "3D table `{}` already exists",
                e.key()
            ))),
        }
    }

    fn set_table_3d(
        &mut self,
        t: Table3DHandle<T>,
        x: Element,
        y: Element,
        z: Element,
        v: T,
    ) -> Result<(), ModelErr> {
        self.check_table_3d(t.id())?;
        self.tables_3d[t.id()].set(x, y, z, v);
        Ok(())
    }

    fn update_table_3d(
        &mut self,
        t: Table3DHandle<T>,
        v: Vec<Vec<Vec<T>>>,
    ) -> Result<(), ModelErr> {
        self.check_table_3d(t.id())?;
        if v.is_empty() || v[0].is_empty() || v[0][0].is_empty() {
            return Err(ModelErr::new(format!(
                "3D table with id `{}` is updated to be empty",
                t.id()
            )));
        }
        self.tables_3d[t.id()].update(v);
        Ok(())
    }

    fn add_table<U>(
        &mut self,
        name: U,
        map: FxHashMap<Vec<Element>, T>,
        default: T,
    ) -> Result<TableHandle<T>, ModelErr>
    where
        String: From<U>,
    {
        let name = String::from(name);
        match self.name_to_table.entry(name) {
            Entry::Vacant(e) => {
                let id = self.tables.len();
                self.tables.push(table::Table::new(map, default));
                e.insert(id);
                Ok(TableHandle(id, PhantomData::default()))
            }
            Entry::Occupied(e) => Err(ModelErr::new(format!("table `{}` already exists", e.key()))),
        }
    }

    fn set_table(&mut self, t: TableHandle<T>, key: Vec<Element>, v: T) -> Result<(), ModelErr> {
        self.check_table(t.id())?;
        self.tables[t.id()].set(key, v);
        Ok(())
    }

    fn set_default(&mut self, t: TableHandle<T>, default: T) -> Result<(), ModelErr> {
        self.check_table(t.id())?;
        self.tables[t.id()].set_default(default);
        Ok(())
    }

    fn update_table(
        &mut self,
        t: TableHandle<T>,
        map: FxHashMap<Vec<Element>, T>,
        default: T,
    ) -> Result<(), ModelErr> {
        self.check_table(t.id())?;
        self.tables[t.id()].update(map, default);
        Ok(())
    }
}

impl<T> TableData<T> {
    /// Returns the set of names used by constants and tables.
    pub fn get_name_set(&self) -> FxHashSet<String> {
        let mut name_set = FxHashSet::default();
        for name in self.name_to_constant.keys() {
            name_set.insert(name.clone());
        }
        for name in self.name_to_table_1d.keys() {
            name_set.insert(name.clone());
        }
        for name in self.name_to_table_2d.keys() {
            name_set.insert(name.clone());
        }
        for name in self.name_to_table_3d.keys() {
            name_set.insert(name.clone());
        }
        for name in self.name_to_table.keys() {
            name_set.insert(name.clone());
        }
        name_set
    }

    /// Checks if the id of a 1D table is valid.
    ///
    /// # Errors
    ///
    /// if the id is not used.
    pub fn check_table_1d(&self, id: usize) -> Result<(), ModelErr> {
        let n = self.tables_1d.len();
        if id >= n {
            Err(ModelErr::new(format!(
                "table 1d id {} >= #tables ({})",
                id, n
            )))
        } else {
            Ok(())
        }
    }

    /// Checks if the id of a 2D table is valid.
    ///
    /// # Errors
    ///
    /// if the id is not used.
    pub fn check_table_2d(&self, id: usize) -> Result<(), ModelErr> {
        let n = self.tables_2d.len();
        if id >= n {
            Err(ModelErr::new(format!(
                "table 2d id {} >= #tables ({})",
                id, n
            )))
        } else {
            Ok(())
        }
    }

    /// Checks if the id of a 3D table is valid.
    ///
    /// # Errors
    ///
    /// if the id is not used.
    pub fn check_table_3d(&self, id: usize) -> Result<(), ModelErr> {
        let n = self.tables_3d.len();
        if id >= n {
            Err(ModelErr::new(format!(
                "table 3d id {} >= #tables ({})",
                id, n
            )))
        } else {
            Ok(())
        }
    }

    /// Checks if the id of a table is valid.
    ///
    /// # Errors
    ///
    /// if the id is not used.
    pub fn check_table(&self, id: usize) -> Result<(), ModelErr> {
        let n = self.tables.len();
        if id >= n {
            Err(ModelErr::new(format!("table id {} >= #tables ({})", id, n)))
        } else {
            Ok(())
        }
    }
}

impl<T: AbsDiffEq> AbsDiffEq for TableData<T>
where
    T::Epsilon: Copy,
{
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> T::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: T::Epsilon) -> bool {
        if self.name_to_constant.len() != other.name_to_constant.len() {
            return false;
        }
        for (key, x) in &self.name_to_constant {
            match other.name_to_constant.get(key) {
                Some(y) if x.abs_diff_eq(y, epsilon) => {}
                _ => return false,
            }
        }
        self.name_to_table_1d == other.name_to_table_1d
            && self.name_to_table_2d == other.name_to_table_2d
            && self.name_to_table_3d == other.name_to_table_3d
            && self.name_to_table == other.name_to_table
            && self
                .tables_1d
                .iter()
                .zip(other.tables_1d.iter())
                .all(|(x, y)| x.abs_diff_eq(y, epsilon))
            && self
                .tables_2d
                .iter()
                .zip(other.tables_2d.iter())
                .all(|(x, y)| x.abs_diff_eq(y, epsilon))
            && self
                .tables_3d
                .iter()
                .zip(other.tables_3d.iter())
                .all(|(x, y)| x.abs_diff_eq(y, epsilon))
            && self
                .tables
                .iter()
                .zip(other.tables.iter())
                .all(|(x, y)| x.abs_diff_eq(y, epsilon))
    }
}

impl<T: RelativeEq> RelativeEq for TableData<T>
where
    T::Epsilon: Copy,
{
    fn default_max_relative() -> T::Epsilon {
        T::default_max_relative()
    }

    fn relative_eq(&self, other: &Self, epsilon: T::Epsilon, max_relative: T::Epsilon) -> bool {
        if self.name_to_constant.len() != other.name_to_constant.len() {
            return false;
        }
        for (key, x) in &self.name_to_constant {
            match other.name_to_constant.get(key) {
                Some(y) if x.relative_eq(y, epsilon, max_relative) => {}
                _ => return false,
            }
        }
        self.name_to_table_1d == other.name_to_table_1d
            && self.name_to_table_2d == other.name_to_table_2d
            && self.name_to_table_3d == other.name_to_table_3d
            && self.name_to_table == other.name_to_table
            && self
                .tables_1d
                .iter()
                .zip(other.tables_1d.iter())
                .all(|(x, y)| x.relative_eq(y, epsilon, max_relative))
            && self
                .tables_2d
                .iter()
                .zip(other.tables_2d.iter())
                .all(|(x, y)| x.relative_eq(y, epsilon, max_relative))
            && self
                .tables_3d
                .iter()
                .zip(other.tables_3d.iter())
                .all(|(x, y)| x.relative_eq(y, epsilon, max_relative))
            && self
                .tables
                .iter()
                .zip(other.tables.iter())
                .all(|(x, y)| x.relative_eq(y, epsilon, max_relative))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_relative_eq, assert_relative_ne};
    use table::*;

    #[test]
    fn add_table_1d_ok() {
        let mut table_data = TableData::default();
        let t = table_data.add_table_1d(String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        assert_eq!(table_data.tables_1d, vec![Table1D::new(vec![0, 1])]);
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("t1"), 0);
        assert_eq!(table_data.name_to_table_1d, name_to_table);
        let t = table_data.add_table_1d(String::from("t2"), vec![0, 2]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
        assert_eq!(
            table_data.tables_1d,
            vec![Table1D::new(vec![0, 1]), Table1D::new(vec![0, 2])]
        );
        name_to_table.insert(String::from("t2"), 1);
        assert_eq!(table_data.name_to_table_1d, name_to_table);
    }

    #[test]
    fn add_table_1d_err() {
        let mut table_data = TableData::default();
        let t = table_data.add_table_1d(String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());
        let t = table_data.add_table_1d(String::from("t1"), vec![0, 1]);
        assert!(t.is_err());
        assert_eq!(table_data.tables_1d, vec![Table1D::new(vec![0, 1])]);
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("t1"), 0);
        assert_eq!(table_data.name_to_table_1d, name_to_table);
    }

    #[test]
    fn add_table_1d_empty_err() {
        let mut table_data: TableData<Element> = TableData::default();
        let t = table_data.add_table_1d(String::from("t1"), vec![]);
        assert!(t.is_err());
        assert_eq!(table_data.tables_1d, vec![]);
        let name_to_table = FxHashMap::default();
        assert_eq!(table_data.name_to_table_1d, name_to_table);
    }

    #[test]
    fn set_table_1d_ok() {
        let mut table_data = TableData::default();
        let t = table_data.add_table_1d(String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = table_data.set_table_1d(t, 0, 1);
        assert!(result.is_ok());
        assert_eq!(table_data.tables_1d, vec![Table1D::new(vec![1, 1])]);
    }

    #[test]
    fn set_table_1d_err() {
        let mut table_data = TableData::default();
        let t = table_data.add_table_1d(String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());

        let mut table_data1 = TableData::default();
        let t = table_data1.add_table_1d(String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());
        let t = table_data1.add_table_1d(String::from("t2"), vec![0, 1]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = table_data.set_table_1d(t, 0, 1);
        assert!(result.is_err());
        assert_eq!(table_data.tables_1d, vec![Table1D::new(vec![0, 1])]);
    }

    #[test]
    fn update_table_1d_ok() {
        let mut table_data = TableData::default();
        let t = table_data.add_table_1d(String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = table_data.update_table_1d(t, vec![1, 1]);
        assert!(result.is_ok());
        assert_eq!(table_data.tables_1d, vec![Table1D::new(vec![1, 1])]);
    }

    #[test]
    fn update_table_1d_err() {
        let mut table_data = TableData::default();
        let t = table_data.add_table_1d(String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());

        let mut table_data1 = TableData::default();
        let t = table_data1.add_table_1d(String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());
        let t = table_data1.add_table_1d(String::from("t2"), vec![0, 1]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = table_data.update_table_1d(t, vec![1, 1]);
        assert!(result.is_err());
        assert_eq!(table_data.tables_1d, vec![Table1D::new(vec![0, 1])]);
    }

    #[test]
    fn update_table_1d_empty_err() {
        let mut table_data = TableData::default();
        let t = table_data.add_table_1d(String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = table_data.update_table_1d(t, vec![]);
        assert!(result.is_err());
        assert_eq!(table_data.tables_1d, vec![Table1D::new(vec![0, 1])]);
    }

    #[test]
    fn add_table_2d_ok() {
        let mut table_data = TableData::default();
        let t = table_data.add_table_2d(String::from("t1"), vec![vec![0, 1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        assert_eq!(table_data.tables_2d, vec![Table2D::new(vec![vec![0, 1]])]);
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("t1"), 0);
        assert_eq!(table_data.name_to_table_2d, name_to_table);
        let t = table_data.add_table_2d(String::from("t2"), vec![vec![0, 2]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
        assert_eq!(
            table_data.tables_2d,
            vec![
                Table2D::new(vec![vec![0, 1]]),
                Table2D::new(vec![vec![0, 2]])
            ]
        );
        name_to_table.insert(String::from("t2"), 1);
        assert_eq!(table_data.name_to_table_2d, name_to_table);
    }

    #[test]
    fn add_table_2d_err() {
        let mut table_data = TableData::default();
        let t = table_data.add_table_2d(String::from("t1"), vec![vec![0, 1]]);
        assert!(t.is_ok());
        let t = table_data.add_table_2d(String::from("t1"), vec![vec![0, 1]]);
        assert!(t.is_err());
        assert_eq!(table_data.tables_2d, vec![Table2D::new(vec![vec![0, 1]])]);
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("t1"), 0);
        assert_eq!(table_data.name_to_table_2d, name_to_table);
    }

    #[test]
    fn add_table_2d_empty_1d_err() {
        let mut table_data: TableData<Element> = TableData::default();
        let t = table_data.add_table_2d(String::from("t1"), vec![]);
        assert!(t.is_err());
        assert_eq!(table_data.tables_2d, vec![]);
        let name_to_table = FxHashMap::default();
        assert_eq!(table_data.name_to_table_2d, name_to_table);
    }

    #[test]
    fn add_table_2d_empty_2d_err() {
        let mut table_data: TableData<Element> = TableData::default();
        let t = table_data.add_table_2d(String::from("t1"), vec![vec![]]);
        assert!(t.is_err());
        assert_eq!(table_data.tables_2d, vec![]);
        let name_to_table = FxHashMap::default();
        assert_eq!(table_data.name_to_table_2d, name_to_table);
    }

    #[test]
    fn set_table_2d_ok() {
        let mut table_data = TableData::default();
        let t = table_data.add_table_2d(String::from("t1"), vec![vec![0, 1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = table_data.set_table_2d(t, 0, 0, 1);
        assert!(result.is_ok());
        assert_eq!(table_data.tables_2d, vec![Table2D::new(vec![vec![1, 1]])]);
    }

    #[test]
    fn set_table_2d_err() {
        let mut table_data = TableData::default();
        let t = table_data.add_table_2d(String::from("t1"), vec![vec![0, 1]]);
        assert!(t.is_ok());

        let mut table_data1 = TableData::default();
        let t = table_data1.add_table_2d(String::from("t1"), vec![vec![0, 1]]);
        assert!(t.is_ok());
        let t = table_data1.add_table_2d(String::from("t2"), vec![vec![0, 1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = table_data.set_table_2d(t, 0, 0, 1);
        assert!(result.is_err());
        assert_eq!(table_data.tables_2d, vec![Table2D::new(vec![vec![0, 1]])]);
    }

    #[test]
    fn update_table_2d_ok() {
        let mut table_data = TableData::default();
        let t = table_data.add_table_2d(String::from("t1"), vec![vec![0, 1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = table_data.update_table_2d(t, vec![vec![1, 1]]);
        assert!(result.is_ok());
        assert_eq!(table_data.tables_2d, vec![Table2D::new(vec![vec![1, 1]])]);
    }

    #[test]
    fn update_table_2d_err() {
        let mut table_data = TableData::default();
        let t = table_data.add_table_2d(String::from("t1"), vec![vec![0, 1]]);
        assert!(t.is_ok());

        let mut table_data1 = TableData::default();
        let t = table_data1.add_table_2d(String::from("t1"), vec![vec![0, 1]]);
        assert!(t.is_ok());
        let t = table_data1.add_table_2d(String::from("t2"), vec![vec![0, 1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = table_data.update_table_2d(t, vec![vec![1, 1]]);
        assert!(result.is_err());
        assert_eq!(table_data.tables_2d, vec![Table2D::new(vec![vec![0, 1]])]);
    }

    #[test]
    fn update_table_2d_empty_1d_ok() {
        let mut table_data = TableData::default();
        let t = table_data.add_table_2d(String::from("t1"), vec![vec![0, 1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = table_data.update_table_2d(t, vec![]);
        assert!(result.is_err());
        assert_eq!(table_data.tables_2d, vec![Table2D::new(vec![vec![0, 1]])]);
    }

    #[test]
    fn update_table_2d_empty_2d_ok() {
        let mut table_data = TableData::default();
        let t = table_data.add_table_2d(String::from("t1"), vec![vec![0, 1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = table_data.update_table_2d(t, vec![vec![]]);
        assert!(result.is_err());
        assert_eq!(table_data.tables_2d, vec![Table2D::new(vec![vec![0, 1]])]);
    }

    #[test]
    fn add_table_3d_ok() {
        let mut table_data = TableData::default();
        let t = table_data.add_table_3d(String::from("t1"), vec![vec![vec![0, 1]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        assert_eq!(
            table_data.tables_3d,
            vec![Table3D::new(vec![vec![vec![0, 1]]])]
        );
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("t1"), 0);
        assert_eq!(table_data.name_to_table_3d, name_to_table);
        let t = table_data.add_table_3d(String::from("t2"), vec![vec![vec![0, 2]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
        assert_eq!(
            table_data.tables_3d,
            vec![
                Table3D::new(vec![vec![vec![0, 1]]]),
                Table3D::new(vec![vec![vec![0, 2]]])
            ]
        );
        name_to_table.insert(String::from("t2"), 1);
        assert_eq!(table_data.name_to_table_3d, name_to_table);
    }

    #[test]
    fn add_table_3d_err() {
        let mut table_data = TableData::default();
        let t = table_data.add_table_3d(String::from("t1"), vec![vec![vec![0, 1]]]);
        assert!(t.is_ok());
        let t = table_data.add_table_3d(String::from("t1"), vec![vec![vec![0, 1]]]);
        assert!(t.is_err());
        assert_eq!(
            table_data.tables_3d,
            vec![Table3D::new(vec![vec![vec![0, 1]]])]
        );
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("t1"), 0);
        assert_eq!(table_data.name_to_table_3d, name_to_table);
    }

    #[test]
    fn add_table_3d_empty_1d_err() {
        let mut table_data: TableData<Element> = TableData::default();
        let t = table_data.add_table_3d(String::from("t1"), vec![]);
        assert!(t.is_err());
        assert_eq!(table_data.tables_3d, vec![]);
        let name_to_table = FxHashMap::default();
        assert_eq!(table_data.name_to_table_3d, name_to_table);
    }

    #[test]
    fn add_table_3d_empty_2d_err() {
        let mut table_data: TableData<Element> = TableData::default();
        let t = table_data.add_table_3d(String::from("t1"), vec![vec![]]);
        assert!(t.is_err());
        assert_eq!(table_data.tables_3d, vec![]);
        let name_to_table = FxHashMap::default();
        assert_eq!(table_data.name_to_table_3d, name_to_table);
    }

    #[test]
    fn add_table_3d_empty_3d_err() {
        let mut table_data: TableData<Element> = TableData::default();
        let t = table_data.add_table_3d(String::from("t1"), vec![vec![vec![]]]);
        assert!(t.is_err());
        assert_eq!(table_data.tables_3d, vec![]);
        let name_to_table = FxHashMap::default();
        assert_eq!(table_data.name_to_table_3d, name_to_table);
    }

    #[test]
    fn set_table_3d_ok() {
        let mut table_data = TableData::default();
        let t = table_data.add_table_3d(String::from("t1"), vec![vec![vec![0, 1]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = table_data.set_table_3d(t, 0, 0, 0, 1);
        assert!(result.is_ok());
        assert_eq!(
            table_data.tables_3d,
            vec![Table3D::new(vec![vec![vec![1, 1]]])]
        );
    }

    #[test]
    fn set_table_3d_err() {
        let mut table_data = TableData::default();
        let t = table_data.add_table_3d(String::from("t1"), vec![vec![vec![0, 1]]]);
        assert!(t.is_ok());

        let mut table_data1 = TableData::default();
        let t = table_data1.add_table_3d(String::from("t1"), vec![vec![vec![0, 1]]]);
        assert!(t.is_ok());
        let t = table_data1.add_table_3d(String::from("t2"), vec![vec![vec![0, 1]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = table_data.set_table_3d(t, 0, 0, 0, 1);
        assert!(result.is_err());
        assert_eq!(
            table_data.tables_3d,
            vec![Table3D::new(vec![vec![vec![0, 1]]])]
        );
    }

    #[test]
    fn update_table_3d_ok() {
        let mut table_data = TableData::default();
        let t = table_data.add_table_3d(String::from("t1"), vec![vec![vec![0, 1]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = table_data.update_table_3d(t, vec![vec![vec![1, 1]]]);
        assert!(result.is_ok());
        assert_eq!(
            table_data.tables_3d,
            vec![Table3D::new(vec![vec![vec![1, 1]]])]
        );
    }

    #[test]
    fn update_table_3d_err() {
        let mut table_data = TableData::default();
        let t = table_data.add_table_3d(String::from("t1"), vec![vec![vec![0, 1]]]);
        assert!(t.is_ok());

        let mut table_data1 = TableData::default();
        let t = table_data1.add_table_3d(String::from("t1"), vec![vec![vec![0, 1]]]);
        assert!(t.is_ok());
        let t = table_data1.add_table_3d(String::from("t2"), vec![vec![vec![0, 1]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = table_data.update_table_3d(t, vec![vec![vec![1, 1]]]);
        assert!(result.is_err());
        assert_eq!(
            table_data.tables_3d,
            vec![Table3D::new(vec![vec![vec![0, 1]]])]
        );
    }

    #[test]
    fn update_table_3d_empty_1d_err() {
        let mut table_data = TableData::default();
        let t = table_data.add_table_3d(String::from("t1"), vec![vec![vec![0, 1]]]);
        assert!(t.is_ok());
        let t = table_data.add_table_3d(String::from("t2"), vec![]);
        assert!(t.is_err());
        assert_eq!(
            table_data.tables_3d,
            vec![Table3D::new(vec![vec![vec![0, 1]]])]
        );
    }

    #[test]
    fn update_table_3d_empty_2d_err() {
        let mut table_data = TableData::default();
        let t = table_data.add_table_3d(String::from("t1"), vec![vec![vec![0, 1]]]);
        assert!(t.is_ok());
        let t = table_data.add_table_3d(String::from("t2"), vec![vec![]]);
        assert!(t.is_err());
        assert_eq!(
            table_data.tables_3d,
            vec![Table3D::new(vec![vec![vec![0, 1]]])]
        );
    }

    #[test]
    fn update_table_3d_empty_3d_err() {
        let mut table_data = TableData::default();
        let t = table_data.add_table_3d(String::from("t1"), vec![vec![vec![0, 1]]]);
        assert!(t.is_ok());
        let t = table_data.add_table_3d(String::from("t2"), vec![vec![vec![]]]);
        assert!(t.is_err());
        assert_eq!(
            table_data.tables_3d,
            vec![Table3D::new(vec![vec![vec![0, 1]]])]
        );
    }

    #[test]
    fn add_table_ok() {
        let mut table_data = TableData::default();
        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1);
        let t = table_data.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        assert_eq!(table_data.tables, vec![Table::new(map.clone(), 0)]);
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("t1"), 0);
        assert_eq!(table_data.name_to_table, name_to_table);
        let mut map2 = FxHashMap::default();
        map2.insert(vec![0, 0, 0, 1], 2);
        let t = table_data.add_table(String::from("t2"), map2.clone(), 1);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
        assert_eq!(
            table_data.tables,
            vec![Table::new(map, 0), Table::new(map2, 1)]
        );
        name_to_table.insert(String::from("t2"), 1);
        assert_eq!(table_data.name_to_table, name_to_table);
    }

    #[test]
    fn add_table_err() {
        let mut table_data = TableData::default();
        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1);
        let t = table_data.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());
        let t = table_data.add_table(String::from("t1"), map.clone(), 1);
        assert!(t.is_err());
        assert_eq!(table_data.tables, vec![Table::new(map, 0)]);
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("t1"), 0);
        assert_eq!(table_data.name_to_table, name_to_table);
    }

    #[test]
    fn set_table_ok() {
        let mut table_data = TableData::default();
        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1);
        let t = table_data.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = table_data.set_table(t, vec![0, 0, 0, 0], 1);
        assert!(result.is_ok());
        map.insert(vec![0, 0, 0, 0], 1);
        assert_eq!(table_data.tables, vec![Table::new(map, 0)]);
    }

    #[test]
    fn set_table_err() {
        let mut table_data = TableData::default();
        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1);
        let t = table_data.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());

        let mut table_data1 = TableData::default();
        let t = table_data1.add_table(String::from("t1"), map.clone(), 1);
        assert!(t.is_ok());
        let t = table_data1.add_table(String::from("t2"), map.clone(), 2);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = table_data.set_table(t, vec![0, 0, 0, 0], 1);
        assert!(result.is_err());
        assert_eq!(table_data.tables, vec![Table::new(map, 0)]);
    }

    #[test]
    fn set_default_ok() {
        let mut table_data = TableData::default();
        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1);
        let t = table_data.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = table_data.set_default(t, 1);
        assert!(result.is_ok());
        assert_eq!(table_data.tables, vec![Table::new(map, 1)]);
    }

    #[test]
    fn set_default_err() {
        let mut table_data = TableData::default();
        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1);
        let t = table_data.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());

        let mut table_data1 = TableData::default();
        let t = table_data1.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());
        let t = table_data1.add_table(String::from("t2"), map.clone(), 0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let result = table_data.set_default(t, 1);
        assert!(result.is_err());
        assert_eq!(table_data.tables, vec![Table::new(map, 0)]);
    }

    #[test]
    fn update_table_ok() {
        let mut table_data = TableData::default();
        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1);
        let t = table_data.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());
        let t = t.unwrap();
        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 2);
        let result = table_data.update_table(t, map.clone(), 1);
        assert!(result.is_ok());
        assert_eq!(table_data.tables, vec![Table::new(map, 1)]);
    }

    #[test]
    fn update_table_err() {
        let mut table_data = TableData::default();
        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1);
        let t = table_data.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());

        let mut table_data1 = TableData::default();
        let t = table_data1.add_table(String::from("t1"), map.clone(), 1);
        assert!(t.is_ok());
        let t = table_data1.add_table(String::from("t2"), map.clone(), 2);
        assert!(t.is_ok());
        let t = t.unwrap();
        let mut map2 = FxHashMap::default();
        map2.insert(vec![0, 0, 0, 1], 2);
        let result = table_data.update_table(t, map2, 3);
        assert!(result.is_err());
        assert_eq!(table_data.tables, vec![Table::new(map, 0)]);
    }

    #[test]
    fn get_name_set() {
        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("i0"), 0);

        let tables_1d = vec![table::Table1D::new(vec![10, 20, 30])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("i1"), 0);

        let tables_2d = vec![table::Table2D::new(vec![
            vec![10, 20, 30],
            vec![10, 10, 10],
            vec![10, 10, 10],
        ])];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("i2"), 0);

        let tables_3d = vec![table::Table3D::new(vec![
            vec![vec![10, 20, 30], vec![0, 0, 0], vec![0, 0, 0]],
            vec![vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0]],
            vec![vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0]],
        ])];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("i3"), 0);

        let mut map = FxHashMap::default();
        let key = vec![0, 1, 0, 0];
        map.insert(key, 100);
        let key = vec![0, 1, 0, 1];
        map.insert(key, 200);
        let key = vec![0, 1, 2, 0];
        map.insert(key, 300);
        let key = vec![0, 1, 2, 1];
        map.insert(key, 400);
        let tables = vec![table::Table::new(map, 0)];
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("i4"), 0);

        let integer_tables = TableData {
            name_to_constant,
            tables_1d,
            name_to_table_1d,
            tables_2d,
            name_to_table_2d,
            tables_3d,
            name_to_table_3d,
            tables,
            name_to_table,
        };
        let mut expected = FxHashSet::default();
        expected.insert(String::from("i0"));
        expected.insert(String::from("i1"));
        expected.insert(String::from("i2"));
        expected.insert(String::from("i3"));
        expected.insert(String::from("i4"));
        assert_eq!(integer_tables.get_name_set(), expected);
    }

    #[test]
    fn check_table_1d() {
        let mut table_data = TableData::default();
        let result = table_data.check_table_1d(0);
        assert!(result.is_err());
        let t = table_data.add_table_1d(String::from("t"), vec![0, 1]);
        assert!(t.is_ok());
        let result = table_data.check_table_1d(0);
        assert!(result.is_ok());
        let result = table_data.check_table_1d(1);
        assert!(result.is_err());
    }

    #[test]
    fn check_table_2d() {
        let mut table_data = TableData::default();
        let result = table_data.check_table_2d(0);
        assert!(result.is_err());
        let t = table_data.add_table_2d(String::from("t"), vec![vec![0, 1]]);
        assert!(t.is_ok());
        let result = table_data.check_table_2d(0);
        assert!(result.is_ok());
        let result = table_data.check_table_2d(1);
        assert!(result.is_err());
    }

    #[test]
    fn check_table_3d() {
        let mut table_data = TableData::default();
        let result = table_data.check_table_3d(0);
        assert!(result.is_err());
        let t = table_data.add_table_3d(String::from("t"), vec![vec![vec![0, 1]]]);
        assert!(t.is_ok());
        let result = table_data.check_table_3d(0);
        assert!(result.is_ok());
        let result = table_data.check_table_3d(1);
        assert!(result.is_err());
    }

    #[test]
    fn check_table() {
        let mut table_data = TableData::default();
        let result = table_data.check_table(0);
        assert!(result.is_err());
        let t = table_data.add_table(String::from("t"), FxHashMap::default(), 0);
        assert!(t.is_ok());
        let result = table_data.check_table(0);
        assert!(result.is_ok());
        let result = table_data.check_table(1);
        assert!(result.is_err());
    }

    #[test]
    fn relative_eq() {
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("t0"), 0);
        name_to_table_1d.insert(String::from("t1"), 1);
        let t1 = TableData {
            tables_1d: vec![
                table::Table1D::new(vec![1.0, 2.0]),
                table::Table1D::new(vec![2.0, 3.0]),
            ],
            name_to_table_1d: name_to_table_1d.clone(),
            ..Default::default()
        };
        let t2 = TableData {
            tables_1d: vec![
                table::Table1D::new(vec![1.0, 2.0]),
                table::Table1D::new(vec![2.0, 3.0]),
            ],
            name_to_table_1d: name_to_table_1d.clone(),
            ..Default::default()
        };
        assert_relative_eq!(t1, t2);
        let t2 = TableData {
            tables_1d: vec![
                table::Table1D::new(vec![1.0, 2.0]),
                table::Table1D::new(vec![3.0, 3.0]),
            ],
            name_to_table_1d,
            ..Default::default()
        };
        assert_relative_ne!(t1, t2);
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("t0"), 0);
        let t2 = TableData {
            tables_1d: vec![table::Table1D::new(vec![1.0, 2.0])],
            name_to_table_1d,
            ..Default::default()
        };
        assert_relative_ne!(t1, t2);
    }
}
