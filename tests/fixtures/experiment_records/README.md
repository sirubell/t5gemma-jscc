# Synthetic experiment record evidence

These files contain synthetic CPU-only scalar observations, never measurements of a
model or a scientific comparison. `build_fixture.py` regenerates the input bundle
in a fresh directory. It exercises both COCO and HellaSwag identities, local and
combined phases, six conditions, initialization/vanilla and held-out comparisons,
failed/missing observations, and first-use/reuse accounting. No tensor loader is
used. Rendered reports are local evidence generated from this bundle; their
inventories bind input bytes and analysis settings.
