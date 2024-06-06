use dypdl::{Continuous, Element, Integer, Set};
use std::error::Error;
use yaml_rust::{yaml::Array, Yaml};

pub trait ToYaml {
    fn to_yaml(&self) -> Result<Yaml, Box<dyn Error>>;
}

impl ToYaml for Integer {
    fn to_yaml(&self) -> Result<Yaml, Box<dyn Error>> {
        Ok(Yaml::Integer(i64::from(*self)))
    }
}

impl ToYaml for Continuous {
    fn to_yaml(&self) -> Result<Yaml, Box<dyn Error>> {
        Ok(Yaml::Real(self.to_string()))
    }
}

impl ToYaml for Element {
    fn to_yaml(&self) -> Result<Yaml, Box<dyn Error>> {
        Ok(Yaml::Integer(i64::try_from(*self)?))
    }
}

impl ToYaml for Set {
    fn to_yaml(&self) -> Result<Yaml, Box<dyn Error>> {
        let mut array = Array::new();
        for i in self.ones() {
            array.push(Yaml::Integer(i64::try_from(i)?));
        }
        Ok(Yaml::Array(array))
    }
}

impl ToYaml for bool {
    fn to_yaml(&self) -> Result<Yaml, Box<dyn Error>> {
        Ok(Yaml::Boolean(*self))
    }
}

#[cfg(test)]
mod tests {
    use dypdl::Set;
    use yaml_rust::Yaml;

    use crate::dypdl_yaml_dumper::ToYaml;

    #[test]
    fn integer_to_yaml() {
        assert_eq!(Yaml::Integer(32), i32::to_yaml(&32).unwrap());
    }

    #[test]
    fn continuous_to_yaml() {
        assert_eq!(Yaml::Real("3.2".to_owned()), (3.2).to_yaml().unwrap());
    }

    #[test]
    fn bool_to_yaml() {
        assert_eq!(Yaml::Boolean(false), false.to_yaml().unwrap());
    }

    #[test]
    fn element_to_yaml_ok() {
        let result = usize::to_yaml(&3);
        assert!(result.is_ok());
        assert_eq!(Yaml::Integer(3), result.unwrap());
    }

    #[test]
    fn element_to_yaml_err() {
        assert!(((i64::MAX as usize) + 1).to_yaml().is_err());
    }

    #[test]
    fn set_to_yaml() {
        let mut s = Set::with_capacity(10);
        s.insert(1);
        s.insert(3);
        assert_eq!(
            Yaml::Array(vec![Yaml::Integer(1), Yaml::Integer(3)]),
            s.to_yaml().unwrap()
        );
    }
}
