# PCAS-Prompt: Visual Semantic FSCIL Research Repository

The default implementation on `dev` is **VSMR** (Visual Semantic Decomposition and Prototype Reconstruction), documented in [README_VSMR.md](README_VSMR.md). The previous APL-PRSN, PCAS, LSP and Prompt-Slot-OT implementations remain selectable as explicit historical baselines and are not part of the VSMR computation graph.

## Run the default method

```bash
python train.py --method vsmr -project base -dataset cub200 -vit \
  -dataroot "$FSCIL_DATA_ROOT" --index_list_root "$FSCIL_INDEX_LIST_ROOT" \
  -gpu 0 -seed 1 -epochs_base 50 -num_workers 4 -test_batch_size 64
```

See [README_VSMR.md](README_VSMR.md) for the architecture, full objective, incremental protocol, diagnostics, experiment settings, and test commands.
