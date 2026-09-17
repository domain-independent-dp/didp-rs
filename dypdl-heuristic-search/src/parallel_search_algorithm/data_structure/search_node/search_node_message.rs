/// Trait representing a message representing a search node.
pub trait SearchNodeMessage: Send + Sync {
    /// Returns the destination thread index in `0..threads`.
    ///
    /// # Panics
    ///
    /// Panics if `threads` is zero.
    fn assign_thread(&self, threads: usize) -> usize;
}
