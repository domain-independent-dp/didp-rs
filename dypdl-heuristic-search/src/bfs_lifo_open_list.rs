use std::collections::BTreeMap;

pub struct BFSLIFOOpenList<T: Ord + Copy, U> {
    map: BTreeMap<T, Vec<U>>,
}

impl<T: Ord + Copy, U> Default for BFSLIFOOpenList<T, U> {
    fn default() -> Self {
        Self {
            map: BTreeMap::default(),
        }
    }
}

impl<T: Ord + Copy, U> BFSLIFOOpenList<T, U> {
    pub fn peek(&self) -> Option<&U> {
        self.map
            .first_key_value()
            .and_then(|key_value| key_value.1.last())
    }

    pub fn pop(&mut self) -> Option<U> {
        if let Some(mut entry) = self.map.first_entry() {
            let bucket = entry.get_mut();
            let node = bucket.pop();
            if bucket.is_empty() {
                entry.remove_entry();
            }
            node
        } else {
            None
        }
    }

    pub fn push(&mut self, key: T, node: U) {
        let bucket = self.map.entry(key).or_default();
        bucket.push(node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_pop() {
        let mut open = BFSLIFOOpenList::default();
        assert_eq!(open.peek(), None);
        open.push(4, -4);
        assert_eq!(open.peek(), Some(&-4));
        open.push(4, -5);
        assert_eq!(open.peek(), Some(&-5));
        open.push(2, -2);
        assert_eq!(open.peek(), Some(&-2));
        open.push(3, -3);
        assert_eq!(open.peek(), Some(&-2));
        open.push(1, -1);
        assert_eq!(open.peek(), Some(&-1));
        open.push(1, 0);
        assert_eq!(open.peek(), Some(&0));
        assert_eq!(open.pop(), Some(0));
        assert_eq!(open.peek(), Some(&-1));
        assert_eq!(open.pop(), Some(-1));
        assert_eq!(open.peek(), Some(&-2));
        assert_eq!(open.pop(), Some(-2));
        assert_eq!(open.peek(), Some(&-3));
        assert_eq!(open.pop(), Some(-3));
        assert_eq!(open.peek(), Some(&-5));
        assert_eq!(open.pop(), Some(-5));
        assert_eq!(open.peek(), Some(&-4));
        assert_eq!(open.pop(), Some(-4));
        assert_eq!(open.peek(), None);
    }
}
