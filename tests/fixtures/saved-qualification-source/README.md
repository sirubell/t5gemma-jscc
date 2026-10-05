# Saved qualification source bytes

These text fixtures are read as bytes and copied to an isolated retained source root for checksum validation. They are never imported or executed.

The reviewed guard and script come from Git commit `be7593a716992230b5fc0cf68d2f06b595f49bb0`, paths `jscc/sharing_qualification_guard.py` and `scripts/sharing_qualification.py`. Their SHA256 digests are recorded in the test package fixture receipt. The unreviewed alias guard is a negative fixture from the observed R5 alias, whose digest differs from the historical published guard. It must reject under the unchanged reader digest.

The contract fixture still mocks prepared-package loading to isolate later schema and binding checks. These tests do not execute historical qualification, establish archive loader equivalence, or authorize scientific work.
