# Released inference weights

- `backbone.pt`: frozen backbone, probability/readout parameters and shared lookup/beam tables (9,035,391 bytes).
- `reconstruction.npz`: original reconstruction asset containing the four `gelu64` scale-specific heads (275,660 bytes). Other stored arrays remain solely to preserve the original asset hash.

Both files are ordinary Git blobs, not Git LFS pointers. Their SHA-256 values are recorded in `../ASSETS.json`. Run `python tools/verify_assets.py` from the repository root after cloning.

These are inference assets, not full optimizer/RNG training checkpoints. The backbone includes inherited ELiC weights; the original ETRI license and patent notice in `../licenses/ELiC.txt` remain applicable. See `../THIRD_PARTY.md` for attribution. No new exclusive ownership of inherited weights is claimed by this release.
