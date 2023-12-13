use super::TerminationDetector;
use crossbeam_channel::{Receiver, Sender};

/// Channel to send and receive nodes between threads with termination detection.
///
/// # References
///
/// Friedmann Mattern. "Algorithms for Distributed Termination Detection,"
/// Distributed Computing, vol. 2, pp 161-175, 1987
pub struct HdNodeChannel<N> {
    node_txs: Vec<Sender<(usize, N)>>,
    node_rx: Receiver<(usize, N)>,
    termination_detector: TerminationDetector,
}

impl<N> HdNodeChannel<N> {
    /// Creates a new channel.
    pub fn new(
        id: usize,
        node_txs: Vec<Sender<(usize, N)>>,
        node_rx: Receiver<(usize, N)>,
        termination_check_tx: Sender<(usize, i32, bool, usize)>,
        termination_check_rx: Receiver<(usize, i32, bool, usize)>,
    ) -> Self {
        let termination_detector =
            TerminationDetector::new(id, termination_check_tx, termination_check_rx);

        Self {
            node_txs,
            node_rx,
            termination_detector,
        }
    }

    /// Sends a node to a destination.
    pub fn send(&mut self, node: N, destination: usize) {
        self.node_txs[destination]
            .send((self.termination_detector.get_clock_to_send(), node))
            .unwrap();
    }

    /// Tries to receive a node.
    /// This method is non-blocking and returns `None` if no node is received.
    pub fn try_receive(&mut self) -> Option<N> {
        if let Ok((tstamp, node)) = self.node_rx.try_recv() {
            self.termination_detector.notify_received(tstamp);
            Some(node)
        } else {
            None
        }
    }

    /// Receives a node.
    /// This method is blocking.
    pub fn receive(&mut self) -> N {
        let (tstamp, node) = self.node_rx.recv().unwrap();
        self.termination_detector.notify_received(tstamp);
        node
    }

    /// Initiates the termination detection.
    pub fn initiate_termination(&mut self) {
        self.termination_detector.initiate();
    }

    /// Checks if the termination detection is finished and forward the message.
    pub fn termination_check_and_forward(&mut self, local_invalid: bool) -> Option<bool> {
        self.termination_detector.check_and_forward(local_invalid)
    }

    /// Receives the termination detection check message.
    pub fn receive_termination_check(&mut self) -> bool {
        self.termination_detector.receive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam_channel::{bounded, unbounded};
    use std::thread;

    #[test]
    fn test_send_receive() {
        let (node_tx1, node_rx1) = unbounded();
        let (termination_detection_tx1, termination_detection_rx1) = unbounded();

        let (node_tx2, node_rx2) = unbounded();
        let (termination_detection_tx2, termination_detection_rx2) = unbounded();

        let mut channel1 = HdNodeChannel::new(
            0,
            vec![node_tx1.clone(), node_tx2.clone()],
            node_rx1,
            termination_detection_tx2,
            termination_detection_rx1,
        );

        let mut channel2 = HdNodeChannel::new(
            1,
            vec![node_tx1, node_tx2],
            node_rx2,
            termination_detection_tx1,
            termination_detection_rx2,
        );

        channel1.send(42, 1);
        channel1.send(23, 1);
        assert_eq!(channel2.receive(), 42);
        assert_eq!(channel2.receive(), 23);
    }

    #[test]
    fn test_send_try_receive() {
        let (node_tx1, node_rx1) = unbounded();
        let (termination_detection_tx1, termination_detection_rx1) = unbounded();

        let (node_tx2, node_rx2) = unbounded();
        let (termination_detection_tx2, termination_detection_rx2) = unbounded();

        let mut channel1 = HdNodeChannel::new(
            0,
            vec![node_tx1.clone(), node_tx2.clone()],
            node_rx1,
            termination_detection_tx2,
            termination_detection_rx1,
        );

        let mut channel2 = HdNodeChannel::new(
            1,
            vec![node_tx1, node_tx2],
            node_rx2,
            termination_detection_tx1,
            termination_detection_rx2,
        );

        assert_eq!(channel1.try_receive(), None);
        assert_eq!(channel2.try_receive(), None);
        channel1.send(42, 1);
        channel1.send(23, 1);
        assert_eq!(channel1.try_receive(), None);
        assert_eq!(channel2.try_receive(), Some(42));
        assert_eq!(channel2.try_receive(), Some(23));
        assert_eq!(channel2.try_receive(), None);
    }

    #[test]
    fn test_termination_success() {
        let (node_tx1, node_rx1) = unbounded();
        let (termination_detection_tx1, termination_detection_rx1) = unbounded();

        let (node_tx2, node_rx2) = unbounded();
        let (termination_detection_tx2, termination_detection_rx2) = unbounded();

        let (node_tx3, node_rx3) = unbounded();
        let (termination_detection_tx3, termination_detection_rx3) = unbounded();

        let mut channel1 = HdNodeChannel::new(
            0,
            vec![node_tx1.clone(), node_tx2.clone(), node_tx3.clone()],
            node_rx1,
            termination_detection_tx2,
            termination_detection_rx1,
        );

        let mut channel2 = HdNodeChannel::new(
            1,
            vec![node_tx1.clone(), node_tx2.clone(), node_tx3.clone()],
            node_rx2,
            termination_detection_tx3,
            termination_detection_rx2,
        );

        let mut channel3 = HdNodeChannel::new(
            2,
            vec![node_tx1, node_tx2, node_tx3],
            node_rx3,
            termination_detection_tx1,
            termination_detection_rx3,
        );

        let (termination_tx2, termination_rx2) = bounded(0);
        let (termination_tx3, termination_rx3) = bounded(0);

        thread::scope(|s| {
            s.spawn(move || {
                channel1.send(0, 1);
                channel1.send(0, 2);
                channel1.send(0, 2);

                while channel1.try_receive().is_none() {}

                channel1.initiate_termination();

                loop {
                    if let Some(result) = channel1.termination_check_and_forward(false) {
                        assert!(result);
                        break;
                    }
                }

                assert!(termination_tx2.send(()).is_ok());
                assert!(termination_tx3.send(()).is_ok());
            });

            s.spawn(move || {
                channel2.send(0, 2);
                channel2.send(0, 2);

                while channel2.try_receive().is_none() {}

                while termination_rx2.try_recv().is_err() {
                    if let Some(result) = channel2.termination_check_and_forward(false) {
                        assert!(!result);
                    }
                }
            });

            s.spawn(move || {
                channel3.send(0, 0);

                while channel3.try_receive().is_none() {}
                while channel3.try_receive().is_none() {}
                while channel3.try_receive().is_none() {}
                while channel3.try_receive().is_none() {}

                while termination_rx3.try_recv().is_err() {
                    if let Some(result) = channel3.termination_check_and_forward(false) {
                        assert!(!result);
                    }
                }
            });
        });
    }

    #[test]
    fn test_termination_success_with_receive() {
        let (node_tx1, node_rx1) = unbounded();
        let (termination_detection_tx1, termination_detection_rx1) = unbounded();

        let (node_tx2, node_rx2) = unbounded();
        let (termination_detection_tx2, termination_detection_rx2) = unbounded();

        let (node_tx3, node_rx3) = unbounded();
        let (termination_detection_tx3, termination_detection_rx3) = unbounded();

        let mut channel1 = HdNodeChannel::new(
            0,
            vec![node_tx1.clone(), node_tx2.clone(), node_tx3.clone()],
            node_rx1,
            termination_detection_tx2,
            termination_detection_rx1,
        );

        let mut channel2 = HdNodeChannel::new(
            1,
            vec![node_tx1.clone(), node_tx2.clone(), node_tx3.clone()],
            node_rx2,
            termination_detection_tx3,
            termination_detection_rx2,
        );

        let mut channel3 = HdNodeChannel::new(
            2,
            vec![node_tx1, node_tx2, node_tx3],
            node_rx3,
            termination_detection_tx1,
            termination_detection_rx3,
        );

        let (termination_tx2, termination_rx2) = bounded(0);
        let (termination_tx3, termination_rx3) = bounded(0);

        thread::scope(|s| {
            s.spawn(move || {
                channel1.send(0, 1);
                channel1.send(0, 2);
                channel1.send(0, 2);

                while channel1.try_receive().is_none() {}

                channel1.initiate_termination();

                assert!(channel1.receive_termination_check());
                assert!(termination_tx2.send(()).is_ok());
                assert!(termination_tx3.send(()).is_ok());
            });

            s.spawn(move || {
                channel2.send(0, 2);
                channel2.send(0, 2);

                while channel2.try_receive().is_none() {}

                while termination_rx2.try_recv().is_err() {
                    if let Some(result) = channel2.termination_check_and_forward(false) {
                        assert!(!result);
                    }
                }
            });

            s.spawn(move || {
                channel3.send(0, 0);

                while channel3.try_receive().is_none() {}
                while channel3.try_receive().is_none() {}
                while channel3.try_receive().is_none() {}
                while channel3.try_receive().is_none() {}

                while termination_rx3.try_recv().is_err() {
                    if let Some(result) = channel3.termination_check_and_forward(false) {
                        assert!(!result);
                    }
                }
            });
        });
    }

    #[test]
    fn test_termination_fail_due_to_count() {
        let (node_tx1, node_rx1) = unbounded();
        let (termination_detection_tx1, termination_detection_rx1) = unbounded();

        let (node_tx2, node_rx2) = unbounded();
        let (termination_detection_tx2, termination_detection_rx2) = unbounded();

        let (node_tx3, node_rx3) = unbounded();
        let (termination_detection_tx3, termination_detection_rx3) = unbounded();

        let mut channel1 = HdNodeChannel::new(
            0,
            vec![node_tx1.clone(), node_tx2.clone(), node_tx3.clone()],
            node_rx1,
            termination_detection_tx2,
            termination_detection_rx1,
        );

        let mut channel2 = HdNodeChannel::new(
            1,
            vec![node_tx1.clone(), node_tx2.clone(), node_tx3.clone()],
            node_rx2,
            termination_detection_tx3,
            termination_detection_rx2,
        );

        let mut channel3 = HdNodeChannel::new(
            2,
            vec![node_tx1, node_tx2, node_tx3],
            node_rx3,
            termination_detection_tx1,
            termination_detection_rx3,
        );

        channel1.send(2, 1);
        channel1.send(3, 2);
        channel1.send(5, 2);

        channel2.send(7, 2);
        channel2.send(11, 2);

        channel3.send(13, 0);

        assert_eq!(channel1.try_receive(), Some(13));

        assert_eq!(channel2.try_receive(), Some(2));

        assert_eq!(channel3.try_receive(), Some(3));
        assert_eq!(channel3.try_receive(), Some(5));
        assert_eq!(channel3.try_receive(), Some(7));

        channel1.initiate_termination();
        assert_eq!(channel2.termination_check_and_forward(false), Some(false));
        assert_eq!(channel3.termination_check_and_forward(false), Some(false));
        assert!(!channel1.receive_termination_check());
    }

    #[test]
    fn test_termination_fail_due_to_time() {
        let (node_tx1, node_rx1) = unbounded();
        let (termination_detection_tx1, termination_detection_rx1) = unbounded();

        let (node_tx2, node_rx2) = unbounded();
        let (termination_detection_tx2, termination_detection_rx2) = unbounded();

        let (node_tx3, node_rx3) = unbounded();
        let (termination_detection_tx3, termination_detection_rx3) = unbounded();

        let mut channel1 = HdNodeChannel::new(
            0,
            vec![node_tx1.clone(), node_tx2.clone(), node_tx3.clone()],
            node_rx1,
            termination_detection_tx2,
            termination_detection_rx1,
        );

        let mut channel2 = HdNodeChannel::new(
            1,
            vec![node_tx1.clone(), node_tx2.clone(), node_tx3.clone()],
            node_rx2,
            termination_detection_tx3,
            termination_detection_rx2,
        );

        let mut channel3 = HdNodeChannel::new(
            2,
            vec![node_tx1, node_tx2, node_tx3],
            node_rx3,
            termination_detection_tx1,
            termination_detection_rx3,
        );

        channel1.send(2, 1);
        channel1.send(3, 2);

        channel2.send(5, 2);
        channel2.send(7, 2);

        channel3.send(11, 0);

        assert_eq!(channel1.try_receive(), Some(11));

        assert_eq!(channel2.try_receive(), Some(2));

        assert_eq!(channel3.try_receive(), Some(3));
        assert_eq!(channel3.try_receive(), Some(5));
        assert_eq!(channel3.try_receive(), Some(7));

        channel1.initiate_termination();
        assert_eq!(channel2.termination_check_and_forward(false), Some(false));

        channel2.send(13, 2);
        assert_eq!(channel3.try_receive(), Some(13));

        assert_eq!(channel3.termination_check_and_forward(false), Some(false));
        assert!(!channel1.receive_termination_check());
    }

    #[test]
    fn test_termination_fail_due_to_local() {
        let (node_tx1, node_rx1) = unbounded();
        let (termination_detection_tx1, termination_detection_rx1) = unbounded();

        let (node_tx2, node_rx2) = unbounded();
        let (termination_detection_tx2, termination_detection_rx2) = unbounded();

        let (node_tx3, node_rx3) = unbounded();
        let (termination_detection_tx3, termination_detection_rx3) = unbounded();

        let mut channel1 = HdNodeChannel::new(
            0,
            vec![node_tx1.clone(), node_tx2.clone(), node_tx3.clone()],
            node_rx1,
            termination_detection_tx2,
            termination_detection_rx1,
        );

        let mut channel2 = HdNodeChannel::new(
            1,
            vec![node_tx1.clone(), node_tx2.clone(), node_tx3.clone()],
            node_rx2,
            termination_detection_tx3,
            termination_detection_rx2,
        );

        let mut channel3 = HdNodeChannel::new(
            2,
            vec![node_tx1, node_tx2, node_tx3],
            node_rx3,
            termination_detection_tx1,
            termination_detection_rx3,
        );

        channel1.send(2, 1);
        channel1.send(3, 2);

        channel2.send(5, 2);
        channel2.send(7, 2);
        channel2.send(13, 2);

        channel3.send(11, 0);

        assert_eq!(channel1.try_receive(), Some(11));

        assert_eq!(channel2.try_receive(), Some(2));

        assert_eq!(channel3.try_receive(), Some(3));
        assert_eq!(channel3.try_receive(), Some(5));
        assert_eq!(channel3.try_receive(), Some(7));
        assert_eq!(channel3.try_receive(), Some(13));

        channel1.initiate_termination();
        assert_eq!(channel2.termination_check_and_forward(false), Some(false));
        assert_eq!(channel3.termination_check_and_forward(true), Some(false));
        assert!(!channel1.receive_termination_check());
    }
}
