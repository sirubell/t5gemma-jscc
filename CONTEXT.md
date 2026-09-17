# Task-oriented split communication

This project studies how the location of a model partition affects task quality and communication cost. Historical experiment families retain their own system assumptions and evaluation protocols.

## Language

**Transmitter**: The side that owns the task input and computes the model prefix before transmission.

**Receiver**: The side that computes the remaining model from transmitted representations, without free access to input-derived information on the transmitter.

**Split location**: The boundary between the model computation assigned to the transmitter and receiver.
_Avoid_: Noise injection point, when referring to a complete communication boundary.

**Decoder memory**: Encoder representations used by decoder cross-attention. When the encoder remains on the transmitter, receiver access to this memory requires transmission too.

**Clean-memory bypass**: Receiver access to encoder representations that have not passed through the communication path. It violates the intended system model, even if the decoder hidden-state stream is noisy.

**No-noise condition**: Communication through the trained codec with channel noise disabled; compression and reconstruction remain present.
_Avoid_: Vanilla, uncompressed baseline.

**Vanilla baseline**: The frozen backbone with all communication codecs and channel perturbations bypassed.

**Bottleneck width**: The number of representation coordinates sent per token in a stream. It is not a bit rate; sample length and additional transmitted streams also affect communication cost.

**Selection set**: Examples used to choose a checkpoint or training settings, excluded from gradient-based training and distinct from final evaluation examples.

**Final evaluation set**: Examples reserved for reporting the selected model's performance. Repeatedly tuning against its scores makes it development evidence rather than an untouched test.

**Comparable cohort**: Experiments whose relevant data, system assumptions, training and evaluation protocols match, apart from the variable being compared.

**Documentary evidence**: A reported result in a ledger, report or presentation whose underlying raw artifact has not yet been verified.

**Verified artifact**: An inspected result or configuration with an identified location and recorded provenance. Verification of the file does not establish validity of the experiment.
