# LatticeLiDAR: Neural Compression for 3D LiDAR Point Clouds

## Introduction

This repository provides the PyTorch implementation and pre-trained weights of LatticeLiDAR for LiDAR point cloud geometry compression.

## Environment Setup

Linux, Python 3.10, PyTorch 2.2.1 and CUDA Toolkit 11.8 (including `nvcc` and a compatible C++ compiler).

```bash
git clone https://github.com/TheLeft805/LatticeLiDAR.git
cd LatticeLiDAR

conda create -n latticelidar python=3.10
conda activate latticelidar
pip install -r requirements.txt

export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
```

## Usage

The pre-trained weights are included in `./weights/` and loaded automatically. Input point clouds can be KITTI `.bin` files (float32 XYZ and reflectance) or `.npy` files containing an `N x 3` coordinate array. Coordinates are in meters; only geometry is compressed.

### Compression

Use `--quantization-mm` to adjust the compression rate. Supported values are `{8, 16, 32, 64}`, with a default of `16`.

```bash
python encode.py \
    --input data/000000.bin \
    --output outputs/frame.lgr \
    --quantization-mm 16
```

To compress a sequence, provide a directory of `.bin` files with numeric frame names:

```bash
python encode.py \
    --input data/11/velodyne \
    --output outputs/seq11.lsq \
    --sequence 11 \
    --quantization-mm 16
```

### Decompression

Use the same quantization value as during compression. Reconstructed point clouds can be saved as `.ply` or `.npy` files.

```bash
python decode.py \
    --input outputs/frame.lgr \
    --output outputs/frame.ply \
    --quantization-mm 16
```

For a sequence, specify the same sequence ID and an output directory:

```bash
python decode.py \
    --input outputs/seq11.lsq \
    --output outputs/seq11 \
    --sequence 11 \
    --quantization-mm 16
```

## Acknowledgements

This project builds on [ELiCv1](https://github.com/moolgom/ELiCv1), [RENO](https://github.com/NJUVISION/RENO) and [TorchSparse](https://github.com/mit-han-lab/torchsparse). See [THIRD_PARTY.md](THIRD_PARTY.md) for attribution and licenses.
