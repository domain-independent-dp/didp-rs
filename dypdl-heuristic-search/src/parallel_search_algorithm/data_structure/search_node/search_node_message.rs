/// Trait representing a message representing a search node.
pub trait SearchNodeMessage: Send + Sync {
    fn assign_thread(&self, threads: usize) -> usize;
}
