use crate::util;
use dypdl::variable_type::Numeric;
use dypdl_heuristic_search::{Callback, Solution};
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Write;

pub fn dump_solution<T: Numeric + fmt::Display>(solution: &Solution<T>, file: &mut File) {
    let mut line = format!(
        "{}, {}, {}, {}, ",
        solution.time,
        solution.cost.unwrap(),
        solution.expanded,
        solution.generated
    );
    for transition in &solution.transitions {
        line += format!("{}\t", transition.get_full_name()).as_str();
    }
    line += "\n";
    file.write_all(line.as_bytes())
        .expect("Could not write to a file.");
}

pub fn get_callback<T: Numeric + fmt::Display>(
    map: &linked_hash_map::LinkedHashMap<yaml_rust::Yaml, yaml_rust::Yaml>,
) -> Result<Box<Callback<T>>, Box<dyn Error>> {
    let dump_to = match map.get(&yaml_rust::Yaml::from_str("dump_to")) {
        Some(value) => Some(util::get_string(value)?),
        None => None,
    };
    if let Some(dump_to) = dump_to {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(dump_to.as_str())?;
        Ok(Box::new(move |solution| dump_solution(solution, &mut file)))
    } else {
        Ok(Box::new(|_| {}))
    }
}
