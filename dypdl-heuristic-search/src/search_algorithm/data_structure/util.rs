use dypdl::variable_type::Numeric;
use dypdl::{Model, ReduceFunction};

/// Returns if the given value equals to or exceeds the given bound.
/// The comparison depends on if the model is for minimization or maximization.
pub fn exceed_bound<T: Numeric + PartialOrd>(model: &Model, value: T, bound: Option<T>) -> bool {
    bound.map_or(false, |bound| match model.reduce_function {
        ReduceFunction::Max => value <= bound,
        ReduceFunction::Min => value >= bound,
        _ => false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exceed_bound_none() {
        let model = Model::default();
        assert!(!exceed_bound(&model, 0, None));
    }

    #[test]
    fn exceed_bound_minimization() {
        let model = Model {
            reduce_function: ReduceFunction::Min,
            ..Default::default()
        };
        assert!(exceed_bound(&model, 0, Some(-1)));
    }

    #[test]
    fn not_exceed_bound_minimization() {
        let model = Model {
            reduce_function: ReduceFunction::Min,
            ..Default::default()
        };
        assert!(!exceed_bound(&model, 0, Some(1)));
    }

    #[test]
    fn exceed_bound_maximization() {
        let model = Model {
            reduce_function: ReduceFunction::Max,
            ..Default::default()
        };
        assert!(exceed_bound(&model, 0, Some(1)));
    }

    #[test]
    fn not_exceed_bound_maximization() {
        let model = Model {
            reduce_function: ReduceFunction::Max,
            ..Default::default()
        };
        assert!(!exceed_bound(&model, 0, Some(-1)));
    }
}
