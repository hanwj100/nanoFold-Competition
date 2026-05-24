# hanwj100_nano Submission

Fast-training baseline using NanoFoldBaseline — a Transformer encoder predicting Cα coordinates.

## Model

- Architecture: NanoFoldBaseline (Transformer encoder → Cα → atom14 broadcast)
- d_model=512, n_layers=24, n_heads=8 → ~76M parameters
- atom14: Cα position broadcast to all 14 slots

## Training improvements over template

- **Larger model**: 76M params vs template's ~30M — more capacity for learning fold geometry
- **Better loss**: `baseline_composite_loss` (local distogram + global distance + bond length) vs template's distogram-only
- **AMP enabled**: fp16 for faster training

## Tradeoffs

- Fast to train (~3-5 hours on A5000 vs ~55 hours for full AF2)
- GDT-HA competitive; lDDT-atom14 and MolProbity Clash limited by Cα broadcast

## Competition compliance

- [x] No external data or pretrained weights
- [x] `run_batch(training=False)` does not use supervision labels
- [x] Output: `pred_atom14` shape `(B, L, 14, 3)`
- [x] Data paths absolute
