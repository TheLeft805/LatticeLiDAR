# Source attribution

This directory is an extraction of the maintained project, not the unmodified ELiC or RENO distribution.

- [ELiCv1](https://github.com/moolgom/ELiCv1): sparse-expert backbone, octant embeddings, occupancy hierarchy and inherited trained backbone. The original ETRI BSD license and patent notice are retained in `licenses/ELiC.txt`. Public module and class names were reorganized; the layer parameter correspondence is preserved in the research export record.
- [RENO](https://github.com/NJUVISION/RENO): inherited positive integer-CDF normalization convention. The original MIT notice is retained in `licenses/RENO.txt`.
- [TorchSparse](https://github.com/mit-han-lab/torchsparse): native implicit-GEMM source under `lidar_codec/csrc/sparse/`; its MIT license is retained there. The maintained wrapper launches on the current CUDA stream. Neighbor offset order and arithmetic are preserved.

The current attention/readout integration, beam features, bounded posterior ANS and captured execution orchestration come from the maintained research implementation. The existing project license is copied verbatim to `LICENSE`; third-party notices remain applicable to their respective portions. No upstream files or repositories were changed by this extraction.
