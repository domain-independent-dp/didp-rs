use std::cmp::Reverse;
use std::collections;

pub struct PriorityQueue<T: Ord> {
    min_heap: collections::BinaryHeap<Reverse<T>>,
    max_heap: collections::BinaryHeap<T>,
    reverse: bool,
}

impl<T: Ord> PriorityQueue<T> {
    pub fn new(reverse: bool) -> PriorityQueue<T> {
        PriorityQueue {
            min_heap: collections::BinaryHeap::new(),
            max_heap: collections::BinaryHeap::new(),
            reverse,
        }
    }

    pub fn with_capacity(reverse: bool, capacity: usize) -> PriorityQueue<T> {
        if reverse {
            PriorityQueue {
                min_heap: collections::BinaryHeap::with_capacity(capacity),
                max_heap: collections::BinaryHeap::new(),
                reverse,
            }
        } else {
            PriorityQueue {
                min_heap: collections::BinaryHeap::new(),
                max_heap: collections::BinaryHeap::with_capacity(capacity),
                reverse,
            }
        }
    }

    pub fn capacity(&self) -> usize {
        if self.reverse {
            self.min_heap.capacity()
        } else {
            self.max_heap.capacity()
        }
    }

    pub fn push(&mut self, node: T) {
        if self.reverse {
            self.min_heap.push(Reverse(node))
        } else {
            self.max_heap.push(node)
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.reverse {
            self.min_heap.pop().map(|node| node.0)
        } else {
            self.max_heap.pop()
        }
    }

    pub fn is_empty(&self) -> bool {
        if self.reverse {
            self.min_heap.is_empty()
        } else {
            self.max_heap.is_empty()
        }
    }

    pub fn len(&self) -> usize {
        if self.reverse {
            self.min_heap.len()
        } else {
            self.max_heap.len()
        }
    }

    pub fn peek(&self) -> Option<&T> {
        if self.reverse {
            match self.min_heap.peek() {
                Some(&Reverse(ref value)) => Some(value),
                None => None,
            }
        } else {
            self.max_heap.peek()
        }
    }

    pub fn clear(&mut self) {
        if self.reverse {
            self.min_heap.clear()
        } else {
            self.max_heap.clear()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_test() {
        let queue = PriorityQueue::<i32>::new(true);
        assert!(queue.reverse);
        assert_eq!(queue.capacity(), 0);
        assert_eq!(queue.len(), 0);
        assert!(queue.is_empty());
    }

    #[test]
    fn with_capacity_test() {
        let queue = PriorityQueue::<i32>::with_capacity(false, 10);
        assert!(!queue.reverse);
        assert_eq!(queue.capacity(), 10);
        assert_eq!(queue.len(), 0);
        assert!(queue.is_empty());
    }

    #[test]
    fn push_pop_peek_test() {
        let mut queue = PriorityQueue::<i32>::new(true);
        queue.push(10);
        assert!(!queue.is_empty());
        assert_eq!(queue.len(), 1);
        queue.push(20);
        assert_eq!(queue.len(), 2);
        queue.push(30);
        assert_eq!(queue.len(), 3);
        assert_eq!(queue.peek(), Some(&10));
        assert_eq!(queue.pop(), Some(10));
        assert_eq!(queue.len(), 2);
        assert_eq!(queue.peek(), Some(&20));
        assert_eq!(queue.pop(), Some(20));
        assert_eq!(queue.len(), 1);
        assert_eq!(queue.peek(), Some(&30));
        assert_eq!(queue.pop(), Some(30));
        assert_eq!(queue.len(), 0);
        assert_eq!(queue.peek(), None);
        assert_eq!(queue.pop(), None);
        assert!(queue.is_empty());

        let mut queue = PriorityQueue::<i32>::new(false);
        queue.push(10);
        queue.push(20);
        queue.push(30);
        assert_eq!(queue.peek(), Some(&30));
        assert_eq!(queue.pop(), Some(30));
        assert_eq!(queue.peek(), Some(&20));
        assert_eq!(queue.pop(), Some(20));
        assert_eq!(queue.peek(), Some(&10));
        assert_eq!(queue.pop(), Some(10));
        assert_eq!(queue.peek(), None);
        assert_eq!(queue.pop(), None);
        assert!(queue.is_empty());
    }
}
