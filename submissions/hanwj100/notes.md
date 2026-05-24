# hanwj100 Submission

Full AlphaFold2 monomer profile with competition-tuned training dynamics.

## Changes from minalphafold2_full baseline

- **Earlier finetune (step 7000 vs 8696)**: structural violation loss (clash penalty) is now active for 3000 steps instead of 1304 — directly improves MolProbity Clash score.
- **Reduced MSA weight (1.5 vs 2.0)**: rebalances gradient budget away from MSA reconstruction toward structural FAPE losses.
- **Sidechain FAPE upweight (0.55 vs 0.5)**: shifts slightly toward all-atom accuracy to improve lDDT-atom14.
- **n_cycles=4 (vs 3)**: one extra recycling pass refines the global fold at no parameter cost.
- **n_ensemble=4 at inference (vs 1)**: averages four forward passes for more stable predictions; training still uses n_ensemble=1 per AF2 spec.

## Model

- Architecture: full AlphaFold2 (`alphafold2.toml`), ~94.2M parameters
- Submodule: `third_party/minAlphaFold2` → `hanwj100/minAlphaFold2`
- Track: `limited` (10,000 steps, effective batch 2, crop 256, MSA depth 192)

## Competition compliance

- [x] No external data, pretrained weights, or network access
- [x] `run_batch(training=False)` does not use supervision labels
- [x] Output: `pred_atom14` shape `(B, L, 14, 3)` in Ångströms
- [x] Checkpoint under 500 MB
- [x] All weights bundled in checkpoint (no network access at eval)
- [x] Data paths absolute — works from any unzip location
