#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HdSearchStatistics {
    pub expanded: Vec<usize>,
    pub generated: Vec<usize>,
    pub kept: Vec<usize>,
    pub sent: Vec<usize>,
}
