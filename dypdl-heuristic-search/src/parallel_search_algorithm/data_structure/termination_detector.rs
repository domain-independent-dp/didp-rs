use crossbeam_channel::{Receiver, Sender};
use std::cmp::max;

/// Distributed termination algorithm by Mattern (1987).
///
/// # References
///
/// Friedmann Mattern. "Algorithms for Distributed Termination Detection,"
/// Distributed Computing, vol. 2, pp. 161-175, 1987.
pub struct TerminationDetector {
    id: usize,
    clock: usize,
    tmax: usize,
    count: i32,
    tx: Sender<(usize, i32, bool, usize)>,
    rx: Receiver<(usize, i32, bool, usize)>,
}

impl TerminationDetector {
    /// Creates a new termination detector.
    pub fn new(
        id: usize,
        tx: Sender<(usize, i32, bool, usize)>,
        rx: Receiver<(usize, i32, bool, usize)>,
    ) -> Self {
        Self {
            id,
            clock: 0,
            tmax: 0,
            count: 0,
            tx,
            rx,
        }
    }

    /// Gets the timestamp to send.
    pub fn get_clock_to_send(&mut self) -> usize {
        self.count += 1;
        self.clock
    }

    /// Notifies that a message has been received.
    pub fn notify_received(&mut self, tstamp: usize) {
        self.tmax = max(tstamp, self.tmax);
        self.count -= 1;
    }

    /// Initiates the termination detection.
    pub fn initiate(&mut self) {
        self.clock += 1;
        self.tx
            .send((self.clock, self.count, false, self.id))
            .unwrap();
    }

    /// Checks if the termination detection is finished and forward the message.
    /// `local_invalid` is true if the local thread is not terminated.
    pub fn check_and_forward(&mut self, local_invalid: bool) -> Option<bool> {
        if let Ok((time, accu, invalid, init)) = self.rx.try_recv() {
            self.clock = max(time, self.clock);
            let invalid = invalid || local_invalid;

            if self.id == init {
                Some(accu == 0 && !invalid)
            } else {
                self.tx
                    .send((time, accu + self.count, invalid || self.tmax >= time, init))
                    .unwrap();
                Some(false)
            }
        } else {
            None
        }
    }

    /// Receives the termination detection check message.
    pub fn receive(&mut self) -> bool {
        let (time, accu, invalid, init) = self.rx.recv().unwrap();
        self.clock = max(time, self.clock);
        self.id == init && accu == 0 && !invalid
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam_channel::{bounded, unbounded};
    use std::thread;

    #[test]
    fn test_termination_success() {
        let (tx1, rx1) = unbounded();
        let (tx2, rx2) = unbounded();
        let (tx3, rx3) = unbounded();
        let mut detector1 = TerminationDetector::new(1, tx2, rx1);
        let mut detector2 = TerminationDetector::new(2, tx3, rx2);
        let mut detector3 = TerminationDetector::new(3, tx1, rx3);

        let (termination_tx2, termination_rx2) = bounded(0);
        let (termination_tx3, termination_rx3) = bounded(0);

        thread::scope(|s| {
            s.spawn(move || {
                detector1.get_clock_to_send();
                detector1.get_clock_to_send();
                detector1.get_clock_to_send();
                detector1.notify_received(0);
                detector1.initiate();

                loop {
                    if let Some(result) = detector1.check_and_forward(false) {
                        assert!(result);
                        break;
                    }
                }

                assert!(termination_tx2.send(()).is_ok());
                assert!(termination_tx3.send(()).is_ok());
            });

            s.spawn(move || {
                detector2.get_clock_to_send();
                detector2.get_clock_to_send();
                detector2.notify_received(0);

                while termination_rx2.try_recv().is_err() {
                    if let Some(result) = detector2.check_and_forward(false) {
                        assert!(!result);
                    }
                }
            });

            s.spawn(move || {
                detector3.get_clock_to_send();
                detector3.notify_received(0);
                detector3.notify_received(0);
                detector3.notify_received(0);
                detector3.notify_received(0);

                while termination_rx3.try_recv().is_err() {
                    if let Some(result) = detector3.check_and_forward(false) {
                        assert!(!result);
                    }
                }
            });
        });
    }

    #[test]
    fn test_termination_success_with_receive() {
        let (tx1, rx1) = unbounded();
        let (tx2, rx2) = unbounded();
        let (tx3, rx3) = unbounded();
        let mut detector1 = TerminationDetector::new(1, tx2, rx1);
        let mut detector2 = TerminationDetector::new(2, tx3, rx2);
        let mut detector3 = TerminationDetector::new(3, tx1, rx3);

        let (termination_tx2, termination_rx2) = bounded(0);
        let (termination_tx3, termination_rx3) = bounded(0);

        thread::scope(|s| {
            s.spawn(move || {
                detector1.get_clock_to_send();
                detector1.get_clock_to_send();
                detector1.get_clock_to_send();
                detector1.notify_received(0);
                detector1.initiate();
                assert!(detector1.receive());
                assert!(termination_tx2.send(()).is_ok());
                assert!(termination_tx3.send(()).is_ok());
            });

            s.spawn(move || {
                detector2.get_clock_to_send();
                detector2.get_clock_to_send();
                detector2.notify_received(0);

                while termination_rx2.try_recv().is_err() {
                    if let Some(result) = detector2.check_and_forward(false) {
                        assert!(!result);
                    }
                }
            });

            s.spawn(move || {
                detector3.get_clock_to_send();
                detector3.notify_received(0);
                detector3.notify_received(0);
                detector3.notify_received(0);
                detector3.notify_received(0);

                while termination_rx3.try_recv().is_err() {
                    if let Some(result) = detector3.check_and_forward(false) {
                        assert!(!result);
                    }
                }
            });
        });
    }

    #[test]
    fn test_termination_fail_due_to_count() {
        let (tx1, rx1) = unbounded();
        let (tx2, rx2) = unbounded();
        let (tx3, rx3) = unbounded();
        let mut detector1 = TerminationDetector::new(1, tx2, rx1);
        let mut detector2 = TerminationDetector::new(2, tx3, rx2);
        let mut detector3 = TerminationDetector::new(3, tx1, rx3);

        detector1.get_clock_to_send();
        detector1.get_clock_to_send();
        detector1.get_clock_to_send();

        detector2.get_clock_to_send();
        detector2.get_clock_to_send();

        detector3.get_clock_to_send();

        detector1.notify_received(0);

        detector2.notify_received(0);

        detector3.notify_received(0);
        detector3.notify_received(0);
        detector3.notify_received(0);

        detector1.initiate();
        assert_eq!(detector2.check_and_forward(false), Some(false));
        assert_eq!(detector3.check_and_forward(false), Some(false));
        assert!(!detector1.receive());
    }

    #[test]
    fn test_termination_fail_due_to_time() {
        let (tx1, rx1) = unbounded();
        let (tx2, rx2) = unbounded();
        let (tx3, rx3) = unbounded();
        let mut detector1 = TerminationDetector::new(1, tx2, rx1);
        let mut detector2 = TerminationDetector::new(2, tx3, rx2);
        let mut detector3 = TerminationDetector::new(3, tx1, rx3);

        detector1.get_clock_to_send();
        detector1.get_clock_to_send();

        detector2.get_clock_to_send();
        detector2.get_clock_to_send();

        detector3.get_clock_to_send();

        detector1.notify_received(0);

        detector2.notify_received(0);

        detector3.notify_received(0);
        detector3.notify_received(0);
        detector3.notify_received(0);

        detector1.initiate();
        assert_eq!(detector2.check_and_forward(false), Some(false));

        detector2.get_clock_to_send();
        detector3.notify_received(1);

        assert_eq!(detector3.check_and_forward(false), Some(false));
        assert!(!detector1.receive());
    }

    #[test]
    fn test_termination_fail_due_to_local() {
        let (tx1, rx1) = unbounded();
        let (tx2, rx2) = unbounded();
        let (tx3, rx3) = unbounded();
        let mut detector1 = TerminationDetector::new(1, tx2, rx1);
        let mut detector2 = TerminationDetector::new(2, tx3, rx2);
        let mut detector3 = TerminationDetector::new(3, tx1, rx3);

        detector1.get_clock_to_send();
        detector1.get_clock_to_send();

        detector2.get_clock_to_send();
        detector2.get_clock_to_send();

        detector3.get_clock_to_send();

        detector1.notify_received(0);

        detector2.notify_received(0);

        detector3.notify_received(0);
        detector3.notify_received(0);
        detector3.notify_received(0);

        detector1.initiate();
        assert_eq!(detector2.check_and_forward(false), Some(false));
        assert_eq!(detector3.check_and_forward(true), Some(false));
        assert!(!detector1.receive());
    }
}
