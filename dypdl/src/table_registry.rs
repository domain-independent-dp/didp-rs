use crate::table_data::{
    Table1DHandle, Table2DHandle, Table3DHandle, TableData, TableHandle, TableInterface,
};
use crate::util::ModelErr;
use crate::variable_type::{Continuous, Element, Integer, Set};
use rustc_hash::{FxHashMap, FxHashSet};

/// Tables of constants.
#[derive(Debug, PartialEq, Clone, Default)]
pub struct TableRegistry {
    /// Integer tables.
    pub integer_tables: TableData<Integer>,
    /// Continuous tables.
    pub continuous_tables: TableData<Continuous>,
    /// Set tables.
    pub set_tables: TableData<Set>,
    /// Element tables.
    pub element_tables: TableData<Element>,
    /// Bool tables.
    pub bool_tables: TableData<bool>,
}

macro_rules! impl_table_interface {
    ($T:ty, $tables:ident) => {
        impl TableInterface<$T> for TableRegistry {
            #[inline]
            fn add_table_1d<U>(
                &mut self,
                name: U,
                v: Vec<$T>,
            ) -> Result<Table1DHandle<$T>, ModelErr>
            where
                String: From<U>,
            {
                self.$tables.add_table_1d(name, v)
            }

            // #[inline]
            // fn set_table_1d(
            //     &mut self,
            //     t: Table1DHandle<$T>,
            //     x: Element,
            //     v: $T,
            // ) -> Result<(), ModelErr> {
            //     self.$tables.set_table_1d(t, x, v)
            // }

            // #[inline]
            // fn update_table_1d(
            //     &mut self,
            //     t: Table1DHandle<$T>,
            //     v: Vec<$T>,
            // ) -> Result<(), ModelErr> {
            //     self.$tables.update_table_1d(t, v)
            // }

            #[inline]
            fn add_table_2d<U>(
                &mut self,
                name: U,
                v: Vec<Vec<$T>>,
            ) -> Result<Table2DHandle<$T>, ModelErr>
            where
                String: From<U>,
            {
                self.$tables.add_table_2d(name, v)
            }

            // #[inline]
            // fn set_table_2d(
            //     &mut self,
            //     t: Table2DHandle<$T>,
            //     x: Element,
            //     y: Element,
            //     v: $T,
            // ) -> Result<(), ModelErr> {
            //     self.$tables.set_table_2d(t, x, y, v)
            // }

            // #[inline]
            // fn update_table_2d(
            //     &mut self,
            //     t: Table2DHandle<$T>,
            //     v: Vec<Vec<$T>>,
            // ) -> Result<(), ModelErr> {
            //     self.$tables.update_table_2d(t, v)
            // }

            #[inline]
            fn add_table_3d<U>(
                &mut self,
                name: U,
                v: Vec<Vec<Vec<$T>>>,
            ) -> Result<Table3DHandle<$T>, ModelErr>
            where
                String: From<U>,
            {
                self.$tables.add_table_3d(name, v)
            }

            // #[inline]
            // fn set_table_3d(
            //     &mut self,
            //     t: Table3DHandle<$T>,
            //     x: Element,
            //     y: Element,
            //     z: Element,
            //     v: $T,
            // ) -> Result<(), ModelErr> {
            //     self.$tables.set_table_3d(t, x, y, z, v)
            // }

            // #[inline]
            // fn update_table_3d(
            //     &mut self,
            //     t: Table3DHandle<$T>,
            //     v: Vec<Vec<Vec<$T>>>,
            // ) -> Result<(), ModelErr> {
            //     self.$tables.update_table_3d(t, v)
            // }

            #[inline]
            fn add_table<U>(
                &mut self,
                name: U,
                map: FxHashMap<Vec<Element>, $T>,
                default: $T,
            ) -> Result<TableHandle<$T>, ModelErr>
            where
                String: From<U>,
            {
                self.$tables.add_table(name, map, default)
            }

            // #[inline]
            // fn set_table(
            //     &mut self,
            //     t: TableHandle<$T>,
            //     key: Vec<Element>,
            //     v: $T,
            // ) -> Result<(), ModelErr> {
            //     self.$tables.set_table(t, key, v)
            // }

            // #[inline]
            // fn set_default(&mut self, t: TableHandle<$T>, default: $T) -> Result<(), ModelErr> {
            //     self.$tables.set_default(t, default)
            // }

            // #[inline]
            // fn update_table(
            //     &mut self,
            //     t: TableHandle<$T>,
            //     map: FxHashMap<Vec<Element>, $T>,
            //     default: $T,
            // ) -> Result<(), ModelErr> {
            //     self.$tables.update_table(t, map, default)
            // }
        }
    };
}

impl_table_interface!(Integer, integer_tables);
impl_table_interface!(Continuous, continuous_tables);
impl_table_interface!(Set, set_tables);
impl_table_interface!(Element, element_tables);
impl_table_interface!(bool, bool_tables);

impl TableRegistry {
    /// Returns the set of names used by constants and tables.
    pub fn get_name_set(&self) -> FxHashSet<String> {
        let mut name_set = FxHashSet::default();
        name_set.extend(self.integer_tables.get_name_set());
        name_set.extend(self.continuous_tables.get_name_set());
        name_set.extend(self.set_tables.get_name_set());
        name_set.extend(self.element_tables.get_name_set());
        name_set.extend(self.bool_tables.get_name_set());
        name_set
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_table_1d_ok() {
        let mut registry = TableRegistry::default();
        let t = registry.add_table_1d(String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let t = registry.add_table_1d(String::from("t2"), vec![0, 2]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
        let t = registry.add_table_1d(String::from("t1"), vec![0.0, 1.0]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let t = registry.add_table_1d(String::from("t2"), vec![0.0, 2.0]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
        let t = registry.add_table_1d(String::from("t1"), vec![true, false]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let t = registry.add_table_1d(String::from("t2"), vec![true, false]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
        let t = registry.add_table_1d(String::from("t1"), vec![Set::default()]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let t = registry.add_table_1d(String::from("t2"), vec![Set::default()]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
        let t: Result<Table1DHandle<Element>, _> =
            registry.add_table_1d(String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let t: Result<Table1DHandle<Element>, _> =
            registry.add_table_1d(String::from("t2"), vec![0, 2]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
    }

    #[test]
    fn add_table_1d_err() {
        let mut registry = TableRegistry::default();
        let t = registry.add_table_1d(String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());
        let t = registry.add_table_1d(String::from("t1"), vec![0, 1]);
        assert!(t.is_err());
        let t = registry.add_table_1d(String::from("t1"), vec![0.0, 1.0]);
        assert!(t.is_ok());
        let t = registry.add_table_1d(String::from("t1"), vec![0.0, 1.0]);
        assert!(t.is_err());
        let t = registry.add_table_1d(String::from("t1"), vec![true]);
        assert!(t.is_ok());
        let t = registry.add_table_1d(String::from("t1"), vec![true]);
        assert!(t.is_err());
        let t = registry.add_table_1d(String::from("t1"), vec![Set::default()]);
        assert!(t.is_ok());
        let t = registry.add_table_1d(String::from("t1"), vec![Set::default()]);
        assert!(t.is_err());
        let t: Result<Table1DHandle<Element>, _> =
            registry.add_table_1d(String::from("t1"), vec![0, 1]);
        assert!(t.is_ok());
        let t: Result<Table1DHandle<Element>, _> =
            registry.add_table_1d(String::from("t1"), vec![0, 2]);
        assert!(t.is_err());
    }

    // #[test]
    // fn set_table_1d_ok() {
    //     let mut registry = TableRegistry::default();
    //     let t = registry.add_table_1d(String::from("t1"), vec![0, 1]);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = registry.set_table_1d(t, 0, 1);
    //     assert!(result.is_ok());
    //     let t = registry.add_table_1d(String::from("t1"), vec![0.0, 1.0]);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = registry.set_table_1d(t, 0, 1.0);
    //     assert!(result.is_ok());
    //     let t = registry.add_table_1d(String::from("t1"), vec![true]);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = registry.set_table_1d(t, 0, false);
    //     assert!(result.is_ok());
    #[test]
    fn add_table_2d_ok() {
        let mut registry = TableRegistry::default();
        let t = registry.add_table_2d(String::from("t1"), vec![vec![0, 1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("t1"), 0);
        let t = registry.add_table_2d(String::from("t2"), vec![vec![0, 2]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
        let t = registry.add_table_2d(String::from("t1"), vec![vec![0.0, 1.0]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("t1"), 0);
        let t = registry.add_table_2d(String::from("t2"), vec![vec![0.0, 2.0]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
        let t = registry.add_table_2d(String::from("t1"), vec![vec![false]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("t1"), 0);
        let t = registry.add_table_2d(String::from("t2"), vec![vec![true]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
        let t = registry.add_table_2d(String::from("t1"), vec![vec![Set::default()]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("t1"), 0);
        let t = registry.add_table_2d(String::from("t2"), vec![vec![Set::default()]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
        let t: Result<Table2DHandle<Element>, _> =
            registry.add_table_2d(String::from("t1"), vec![vec![1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("t1"), 0);
        let t: Result<Table2DHandle<Element>, _> =
            registry.add_table_2d(String::from("t2"), vec![vec![1]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
    }

    #[test]
    fn add_table_2d_err() {
        let mut registry = TableRegistry::default();
        let t = registry.add_table_2d(String::from("t1"), vec![vec![0, 1]]);
        assert!(t.is_ok());
        let t = registry.add_table_2d(String::from("t1"), vec![vec![0, 1]]);
        assert!(t.is_err());
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("t1"), 0);
        let t = registry.add_table_2d(String::from("t1"), vec![vec![0.0, 1.0]]);
        assert!(t.is_ok());
        let t = registry.add_table_2d(String::from("t1"), vec![vec![0.0, 1.0]]);
        assert!(t.is_err());
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("t1"), 0);
        let t = registry.add_table_2d(String::from("t1"), vec![vec![false]]);
        assert!(t.is_ok());
        let t = registry.add_table_2d(String::from("t1"), vec![vec![true]]);
        assert!(t.is_err());
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("t1"), 0);
        let t = registry.add_table_2d(String::from("t1"), vec![vec![Set::default()]]);
        assert!(t.is_ok());
        let t = registry.add_table_2d(String::from("t1"), vec![vec![Set::default()]]);
        assert!(t.is_err());
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("t1"), 0);
        let t: Result<Table2DHandle<Element>, _> =
            registry.add_table_2d(String::from("t1"), vec![vec![1]]);
        assert!(t.is_ok());
        let t: Result<Table2DHandle<Element>, _> =
            registry.add_table_2d(String::from("t1"), vec![vec![1]]);
        assert!(t.is_err());
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("t1"), 0);
    }

    // #[test]
    // fn set_table_2d_ok() {
    //     let mut registry = TableRegistry::default();
    //     let t = registry.add_table_2d(String::from("t1"), vec![vec![0, 1]]);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = registry.set_table_2d(t, 0, 0, 1);
    //     assert!(result.is_ok());
    //     let t = registry.add_table_2d(String::from("t1"), vec![vec![0.0, 1.0]]);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = registry.set_table_2d(t, 0, 0, 1.0);
    //     assert!(result.is_ok());
    //     let t = registry.add_table_2d(String::from("t1"), vec![vec![false]]);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = registry.set_table_2d(t, 0, 0, true);
    //     assert!(result.is_ok());
    #[test]
    fn add_table_3d_ok() {
        let mut registry = TableRegistry::default();
        let t = registry.add_table_3d(String::from("t1"), vec![vec![vec![0, 1]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let t = registry.add_table_3d(String::from("t2"), vec![vec![vec![0, 2]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
        let t = registry.add_table_3d(String::from("t1"), vec![vec![vec![0.0, 1.0]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let t = registry.add_table_3d(String::from("t2"), vec![vec![vec![0.0, 2.0]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
        let t = registry.add_table_3d(String::from("t1"), vec![vec![vec![false]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let t = registry.add_table_3d(String::from("t2"), vec![vec![vec![true]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
        let t = registry.add_table_3d(String::from("t1"), vec![vec![vec![Set::default()]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let t = registry.add_table_3d(String::from("t2"), vec![vec![vec![Set::default()]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
        let t: Result<Table3DHandle<Element>, _> =
            registry.add_table_3d(String::from("t1"), vec![vec![vec![1]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let t: Result<Table3DHandle<Element>, _> =
            registry.add_table_3d(String::from("t2"), vec![vec![vec![1]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
    }

    #[test]
    fn add_table_3d_err() {
        let mut registry = TableRegistry::default();
        let t = registry.add_table_3d(String::from("t1"), vec![vec![vec![0, 1]]]);
        assert!(t.is_ok());
        let t = registry.add_table_3d(String::from("t1"), vec![vec![vec![0, 1]]]);
        assert!(t.is_err());
        let t = registry.add_table_3d(String::from("t1"), vec![vec![vec![0.0, 1.0]]]);
        assert!(t.is_ok());
        let t = registry.add_table_3d(String::from("t1"), vec![vec![vec![0.0, 1.0]]]);
        assert!(t.is_err());
        let t = registry.add_table_3d(String::from("t1"), vec![vec![vec![false]]]);
        assert!(t.is_ok());
        let t = registry.add_table_3d(String::from("t1"), vec![vec![vec![false]]]);
        assert!(t.is_err());
        let t = registry.add_table_3d(String::from("t1"), vec![vec![vec![Set::default()]]]);
        assert!(t.is_ok());
        let t = registry.add_table_3d(String::from("t1"), vec![vec![vec![Set::default()]]]);
        assert!(t.is_err());
        let t: Result<Table3DHandle<Element>, _> =
            registry.add_table_3d(String::from("t1"), vec![vec![vec![1]]]);
        assert!(t.is_ok());
        let t: Result<Table3DHandle<Element>, _> =
            registry.add_table_3d(String::from("t1"), vec![vec![vec![1]]]);
        assert!(t.is_err());
    }

    // #[test]
    // fn set_table_3d_ok() {
    //     let mut registry = TableRegistry::default();
    //     let t = registry.add_table_3d(String::from("t1"), vec![vec![vec![0, 1]]]);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = registry.set_table_3d(t, 0, 0, 0, 1);
    //     assert!(result.is_ok());
    //     let t = registry.add_table_3d(String::from("t1"), vec![vec![vec![0.0, 1.0]]]);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = registry.set_table_3d(t, 0, 0, 0, 1.0);
    //     assert!(result.is_ok());
    //     let t = registry.add_table_3d(String::from("t1"), vec![vec![vec![false]]]);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = registry.set_table_3d(t, 0, 0, 0, true);
    //     assert!(result.is_ok());
    #[test]
    fn add_table_ok() {
        let mut registry = TableRegistry::default();
        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1);
        let t = registry.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let mut map2 = FxHashMap::default();
        map2.insert(vec![0, 0, 0, 1], 2);
        let t = registry.add_table(String::from("t2"), map2.clone(), 1);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1.0);
        let t = registry.add_table(String::from("t1"), map.clone(), 0.0);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let mut map2 = FxHashMap::default();
        map2.insert(vec![0, 0, 0, 1], 2.0);
        let t = registry.add_table(String::from("t2"), map2.clone(), 1.0);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], true);
        let t = registry.add_table(String::from("t1"), map.clone(), false);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let mut map2 = FxHashMap::default();
        map2.insert(vec![0, 0, 0, 1], true);
        let t = registry.add_table(String::from("t2"), map2.clone(), false);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], Set::with_capacity(1));
        let t = registry.add_table(String::from("t1"), map.clone(), Set::default());
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let mut map2 = FxHashMap::default();
        map2.insert(vec![0, 0, 0, 1], Set::with_capacity(1));
        let t = registry.add_table(String::from("t2"), map2.clone(), Set::default());
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);

        let mut map: FxHashMap<_, Element> = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1);
        let t = registry.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 0);
        let mut map2: FxHashMap<_, Element> = FxHashMap::default();
        map2.insert(vec![0, 0, 0, 1], 2);
        let t = registry.add_table(String::from("t2"), map2.clone(), 0);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.id(), 1);
    }

    #[test]
    fn add_table_err() {
        let mut registry = TableRegistry::default();
        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1);
        let t = registry.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());
        let t = registry.add_table(String::from("t1"), map.clone(), 1);
        assert!(t.is_err());

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1.0);
        let t = registry.add_table(String::from("t1"), map.clone(), 0.0);
        assert!(t.is_ok());
        let t = registry.add_table(String::from("t1"), map.clone(), 1.0);
        assert!(t.is_err());

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], true);
        let t = registry.add_table(String::from("t1"), map.clone(), false);
        assert!(t.is_ok());
        let t = registry.add_table(String::from("t1"), map.clone(), false);
        assert!(t.is_err());

        let mut map: FxHashMap<_, Element> = FxHashMap::default();
        map.insert(vec![0, 0, 0, 1], 1);
        let t = registry.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_ok());
        let t = registry.add_table(String::from("t1"), map.clone(), 0);
        assert!(t.is_err());
    }

    // #[test]
    // fn set_table_ok() {
    //     let mut registry = TableRegistry::default();
    //     let mut map = FxHashMap::default();
    //     map.insert(vec![0, 0, 0, 1], 1);
    //     let t = registry.add_table(String::from("t1"), map.clone(), 0);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = registry.set_table(t, vec![0, 0, 0, 0], 1);
    //     assert!(result.is_ok());

    //     let mut map = FxHashMap::default();
    //     map.insert(vec![0, 0, 0, 1], 1.0);
    //     let t = registry.add_table(String::from("t1"), map.clone(), 0.0);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = registry.set_table(t, vec![0, 0, 0, 0], 1.0);
    //     assert!(result.is_ok());

    //     let mut map = FxHashMap::default();
    //     map.insert(vec![0, 0, 0, 1], true);
    //     let t = registry.add_table(String::from("t1"), map.clone(), false);
    //     assert!(t.is_ok());
    //     let t = t.unwrap();
    //     let result = registry.set_table(t, vec![0, 0, 0, 0], true);
    //     assert!(result.is_ok());

    //     let mut map = FxHashMap::default();
    //     map.insert(vec![0, 0, 0, 1], vec![1]);
}
