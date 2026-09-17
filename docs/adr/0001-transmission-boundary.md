# Receiver information must cross the communication boundary

The study compares model split locations under a transmitter-owned-input system, not arbitrary noise injection into an otherwise co-located model. The receiver must not obtain clean encoder memory for free; decoder splits must transmit the memory needed by receiver layers as well as the split hidden state, and both streams count toward communication cost. Historical clean-memory runs remain indexed as diagnostic evidence and cannot establish performance under this system model.

The earlier alternative of labeling clean-memory decoder runs as another supported deployment was rejected by the owner on 2026-09-17. A global encoder-memory perturbation also changes transmitter-side decoder computation, so historical memory-coded runs require separate protocol labels rather than automatic equivalence claims.
