# Mio research notes

- [Mio on Qwen 3.6](mio-qwen36-research.md) — working, non-final engineering
  report covering the slower vendored exact 27B path, matched upstream 27B
  DSpark/DFlash experiments, cap ablations, mixture routing, and fused cold-prefill R&D.
- [BMP-DFlash](bmp-dflash.md) — historical Qwen 3/3.5 multi-path experiment.
- [Prefill speedups](prefill-speedups.md) — historical prefix-cache and LM-head
  experiment.

The checked-in Qwen 3.6 schema-v1 JSON files describe their recorded commit.
Current matched files are preliminary working-tree artifacts and record a dirty
revision; rerun them from a clean checkpoint before a release claim. Historical
notes remain useful for hypotheses but must not be combined with current
ratios without a controlled paired rerun.
