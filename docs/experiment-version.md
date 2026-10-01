# Frozen experiment source version

This branch collects the released October 1 codec experiment source into a new local Git version. It starts from `d8886125fc65e82f318fd52c0ec2d24ef14a94c7` and incorporates the 111 portable source files from the frozen 114-file manifest `4cb992442f5b0eb93be247df22871f470722f9063117a5562c980cf175e08bda`. Three Ruff cache files are omitted. Copied production files retain their exact released bytes. The resulting Git commit is a new identity; it does not replace any existing deployment or historical run identity.

The codec source supports explicit one-Linear and Linear–GELU–Linear designs with no external normalization, LayerNorm on both boundaries, or the backbone RMSNorm on both boundaries. Historical residual designs and task defaults remain available. RMSNorm designs require explicit epsilon `1e-6`; two-Linear designs require hidden width 1152, GELU, zero residual blocks, no dropout and FiLM disabled. A design name is a configuration choice, not an adopted scientific winner.

The snapshot also retains shared-training modules already present in the frozen source. Their inclusion preserves the source package; it does not complete paused research work or establish full-weight shared-training acceptance. Host-specific controllers, prepared private inputs, model/cache paths, result records and scheduler approval remain separate local material. The current campaign packages continue to use their original identities.

For future experiments, use the tested commit SHA as the source pin. A branch name is a convenient label and can move; record and verify the full SHA before launching. Reuse one immutable source checkout for multiple configurations. Develop changes in a separate branch and worktree, then freeze a new version after validation.

Keep each run's configuration, input identity receipt and command with its outputs. A minimal host layout is:

```text
t5gemma-jscc/
  releases/<version>-<sha>/       # one source checkout for this version
  campaigns/<campaign>/          # frozen configs and input identities
  runs/<run-name>/               # resolved config, command and source SHA
    outputs/
    logs/
    checkpoints/
```

Existing host runtimes and model/data caches may be referenced by path and revision; they need not be copied for each run. Create the Python environment separately on each host. Spark's recorded CUDA runtime differs from the default lock installation and must be selected explicitly.

The standard `train.py` already creates timestamped run directories under `run.output_dir`. Use a fresh `runs/<run-name>/outputs` as that parent if retaining its normal naming. Preserve the full resolved configuration it writes. Relative output paths and `model_config` references resolve from the YAML location; configure the intended paths explicitly when storing YAML outside the source checkout.

A clean commit identifies tracked source bytes. Uncommitted edits or required untracked modules need separate byte identities and prevent calling that checkout the pinned version. Complete and validate those changes in development before adopting another experiment commit. No active checkout or campaign should be updated in place.
