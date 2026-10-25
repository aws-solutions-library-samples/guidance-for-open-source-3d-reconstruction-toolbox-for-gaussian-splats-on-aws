# Faster-GS: Standalone EC2 Test Guide

This guide sets up [Faster-GS](https://github.com/nerficg-project/faster-gaussian-splatting) (CVPR'26, Hahlbohm et al.) on a GPU EC2 instance. Use it to compare Faster-GS against `splatfacto` / gsplat before adding it to the toolbox as the `faster-gs` model.

Faster-GS is a plugin "method" for the [NeRFICG](https://github.com/nerficg-project/nerficg) framework. It does not run on its own, so you install NeRFICG first and then clone Faster-GS into it.

---

## 1. License review (do this first)

| Component | Source | License | Commercial use |
|---|---|---|---|
| Faster-GS (method + `FasterGSCudaBackend` rasterizer) | nerficg-project/faster-gaussian-splatting | Apache-2.0 | Yes |
| NeRFICG framework | nerficg-project/nerficg | MIT | Yes |
| ICGui (optional GUI submodule) | nerficg-project/icgui | MIT | Yes |
| fused-ssim (DSSIM loss) | rahul-goel/fused-ssim | MIT | Yes |
| PPISP (optional photometric compensation) | nv-tlabs/ppisp | Apache-2.0 | Yes |
| simple-knn (`src/Thirdparty/SimpleKNN.py`) | camenduru/simple-knn (from graphdeco-inria 3DGS) | No LICENSE file. It is derived from the Inria Gaussian-Splatting License, which is non-commercial | NO |
| diff-gaussian-rasterization (only used by the `GaussianSplatting` method) | graphdeco-inria | Gaussian-Splatting License (non-commercial) | NO |

Two rules keep the setup commercially clean:

1. Set `_ALLOW_PROPRIETARY_LICENSED_CODE = False` in `src/Optim/knn_utils.py` before you run `install.py`. The default is `True`, which pulls in simple-knn for KNN scale initialization. With the flag off, NeRFICG uses a scikit-learn KNN instead. It is slightly slower at init and does not change training.
2. Install only the `FasterGS` method. Do not install or run `GaussianSplatting`.

Step 6 checks that neither non-commercial package ended up in the environment. Before you ship, run your own OSS review on the exact pinned commits. This table reflects the repos as of 2026-10-21.

Pinned commits used in this guide:

```
NERFICG_HASH=e458ac6e41810ad6d9405687c1e698fadd28dac5      # 2026-09-06
FASTERGS_HASH=c60ea269b7cf3ebda527d640b6fdd31462f91b41     # 2026-10-10
```

---

## 2. EC2 prerequisites

- Instance: g5 / g6 / g6e (A10G / L4 / L40S). For comparisons, use the same instance type you run splatfacto on.
- AMI: Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04/24.04). It ships with the driver and CUDA toolkits.
- Driver: 570 or newer, which is enough for CUDA 12.8. This test env does not need the 580+ driver that the toolbox's CUDA 13.2 container needs.
- Disk: 100 GB or more of gp3.
- Security group: SSH only. The NeRFICG GUI is a local desktop app (imgui), not a web server. Run headless on EC2.

```bash
nvidia-smi                      # confirm driver >= 570 and the GPU is visible
ls /usr/local/ | grep cuda      # confirm a cuda-12.8 toolkit exists (install it if missing)
gcc --version                   # gcc 11-13 works with CUDA 12.8
```

If CUDA 12.8 is missing:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8
```

Add CUDA to the environment (append to `~/.bashrc`):

```bash
export PATH="/usr/local/cuda-12.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH"
export CUDA_PATH="/usr/local/cuda-12.8/"
export CUDA_HOME="/usr/local/cuda-12.8/"
# Build only for your GPU arch (A10G=8.6, L4/L40S=8.9) to cut compile time
export TORCH_CUDA_ARCH_LIST="8.6;8.9"
```

Install Miniconda if `conda` isn't available:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)" && conda init bash
```

---

## 3. Install NeRFICG (headless)

Set the install root once (any path works; EFS is fine but compiles are slower). Later steps use `$NERFICG_ROOT`.

```bash
export NERFICG_ROOT=/mnt/efs/faster-gs/nerficg     # or ~/nerficg
echo "export NERFICG_ROOT=$NERFICG_ROOT" >> ~/.bashrc
mkdir -p "$(dirname $NERFICG_ROOT)" && cd "$(dirname $NERFICG_ROOT)"
git clone https://github.com/nerficg-project/nerficg.git --recursive "$NERFICG_ROOT"
cd "$NERFICG_ROOT"
git checkout e458ac6e41810ad6d9405687c1e698fadd28dac5
git submodule update --init --recursive

# Headless env: Python 3.11, torch 2.8.0+cu128, includes pycolmap for the Colmap loader
conda env create -f ./environments/py311_cu128_no_gui.yaml
conda activate nerficg
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# expect: 2.8.0+cu128 12.8 True

# Pin the CUDA 12.8 toolkit for extension builds. The Deep Learning AMI exports
# CUDA_HOME=/usr/local/cuda-11.8 from a profile script, and torch prefers CUDA_HOME,
# so builds fail with "detected CUDA version (11.8) mismatches ... (12.8)".
# Conda env vars apply at activation and override it.
conda env config vars set CUDA_HOME=/usr/local/cuda-12.8 CUDA_PATH=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST="8.6;8.9"
conda deactivate && conda activate nerficg
export PATH=/usr/local/cuda-12.8/bin:$PATH   # also keep this in ~/.bashrc (step 2)

which nvcc                                                                      # /usr/local/cuda-12.8/bin/nvcc
python -c "from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME)"  # /usr/local/cuda-12.8
```

---

## 4. Disable non-commercial code (before running install.py)

```bash
sed -i 's/^_ALLOW_PROPRIETARY_LICENSED_CODE = True/_ALLOW_PROPRIETARY_LICENSED_CODE = False/' src/Optim/knn_utils.py
grep -n "_ALLOW_PROPRIETARY_LICENSED_CODE =" src/Optim/knn_utils.py
# expect: _ALLOW_PROPRIETARY_LICENSED_CODE = False
```

---

## 5. Install Faster-GS

```bash
git clone https://github.com/nerficg-project/faster-gaussian-splatting.git src/Methods/FasterGS
git -C src/Methods/FasterGS checkout c60ea269b7cf3ebda527d640b6fdd31462f91b41

# Builds FasterGSCudaBackend, MortonEncoding, fused-ssim, PPISP
python ./scripts/install.py -m FasterGS
```

The upstream docs say install.py discovers extensions on the fly and sometimes stops early. If that happens, rerun the same command before you debug. If you're short on RAM, set `MAX_JOBS=4` to limit nvcc parallelism.

---

## 6. Verify the install and the license posture

```bash
pip list 2>/dev/null | grep -iE "fastergs|fused.ssim|ppisp|morton|simple.knn|diff.gaussian"
```

You should see FasterGSCudaBackend, fused-ssim, ppisp, and MortonEncoding. You should not see simple-knn or diff-gaussian-rasterization. If either one shows up, uninstall it and recheck step 4:

```bash
pip uninstall -y simple-knn diff-gaussian-rasterization
```

Import smoke test:

```bash
cd $NERFICG_ROOT/src
python -c "
import Optim.knn_utils as k; print('fused knn (must be False):', k._FUSED_KNN_AVAILABLE)
from Methods.FasterGS.FasterGSCudaBackend import diff_rasterize; print('FasterGS backend OK')
"
cd $NERFICG_ROOT
```

---

## 7. Prepare a dataset (reuse toolbox COLMAP output)

NeRFICG's `Colmap` loader expects this layout:

```
<scene>/
  images/          # RGB frames, names must match COLMAP image names
  sparse/0/        # cameras.bin, images.bin, points3D.bin (or .txt)
  sfm_masks/       # optional: <image_name>.png (loader treats these as INVERTED segmentation masks)
```

The loader accepts `SIMPLE_PINHOLE`, `PINHOLE`, `SIMPLE_RADIAL`, `RADIAL`, `OPENCV`. Fisheye models such as `OPENCV_FISHEYE` raise `DatasetError`. The Faster-GS rasterizer is pinhole-only, so it ignores the distortion terms. If you train on non-pinhole input, the log repeats `found distortion parameters that will be ignored by the rasterizer` and the run trains on distorted images. Always undistort to PINHOLE first:

```bash
colmap image_undistorter \
  --image_path  /data/<scene>/images \
  --input_path  /data/<scene>/sparse/0 \
  --output_path /data/<scene>_undist \
  --output_type COLMAP
mkdir -p /data/<scene>_undist/sparse/0 && mv /data/<scene>_undist/sparse/*.bin /data/<scene>_undist/sparse/0/
```

To test the pipeline end to end first, you can use the reference scene (MipNeRF360 garden) with the bundled `fastergs_garden.yaml`.

---

## 8. Create and edit a config

```bash
python ./scripts/create_config.py -m FasterGS -d Colmap -o fastergs_test
# -> configs/fastergs_test.yaml
```

Apply settings that work headless and keep the output comparable with the toolbox:

```bash
python - <<'EOF'
import yaml, pathlib
p = pathlib.Path('configs/fastergs_test.yaml')
c = yaml.safe_load(p.read_text())
c['DATASET']['PATH'] = '/data/<scene>_undist'   # <-- edit
c['DATASET']['IMAGE_SCALE_FACTOR'] = 1.0        # match the toolbox's training resolution
c['DATASET']['APPLY_PCA'] = False               # keep the COLMAP world frame (toolbox outputs stay aligned)
c['DATASET']['APPLY_PCA_RESCALE'] = False
c['TRAINING']['MODEL_NAME'] = 'fastergs_test'
c['TRAINING']['NUM_ITERATIONS'] = 30000         # match splatfacto MAX_STEPS for a fair comparison
c['TRAINING'].setdefault('WANDB', {})['ACTIVATE'] = False
c['TRAINING'].setdefault('DATA', {})['PRELOADING_LEVEL'] = 2   # images in VRAM (fastest); drop to 1/0 on large datasets
# The no_gui env omits TRAINING.GUI from generated configs, but the FasterGS Trainer reads it
# unconditionally ("Parameter missing in loaded config: GUI"), so add it explicitly.
c['TRAINING']['GUI'] = {**c['TRAINING'].get('GUI', {}), 'ACTIVATE': False, 'RENDER_INTERVAL': 10,
                        'GUI_STATUS_ENABLED': False, 'GUI_STATUS_INTERVAL': 20,
                        'SKIP_GUI_SETUP': True, 'FPS_ROLLING_AVERAGE_SIZE': 100}
p.write_text(yaml.safe_dump(c, sort_keys=False))
EOF
```

Optional feature toggles to test, roughly mapped to toolbox equivalents:

| Faster-GS key | Rough toolbox analogue |
|---|---|
| `TRAINING.USE_MCMC: true` + `MAX_PRIMITIVES` (set `DENSIFICATION_END_ITERATION: 24900`, `MORTON_ORDERING_END_ITERATION: 25000`, opacity/scale reg `0.01`) | splatfacto-mcmc |
| `TRAINING.DENSIFICATION_USE_ABS_GRAD: true` | `ENABLE_ABSGRAD` |
| `TRAINING.FILTER_3D.USE: true` + `RENDERER.PROPER_ANTIALIASING: true` | Mip-Splatting style antialiasing |
| `MODEL.PPISP.USE: true` | Per-image appearance (like splatfacto-w / bilateral grid) |
| `TRAINING.SPEEDYSPLAT_PRUNING.USE: true` | Fewer Gaussians and faster training |
| `TRAINING.USE_RANDOM_BACKGROUND_COLOR: true` | Reduces false transparency |

Start with defaults for the baseline run, then change one toggle per run.

---

## 9. Train, export, evaluate

```bash
# Train (logs timing and VRAM because TIMING.ACTIVATE and WRITE_VRAM_STATS are on)
python ./scripts/train.py -c configs/fastergs_test.yaml

# Output lands in $NERFICG_ROOT/output/<METHOD_TYPE>/<MODEL_NAME>_<timestamp>/
RUN=$(ls -td output/FasterGS/fastergs_test_* | head -1); echo $RUN

# Export the Gaussians to PLY -> $RUN/final.ply
python ./scripts/convert_to_ply.py -d $RUN

# Render the test set and compute metrics (PSNR/SSIM/LPIPS) + FPS benchmark
python ./scripts/inference.py -d $RUN -m -b
```

Check `final.ply` against your toolbox viewers or SPZ conversion:

```bash
python -c "
from plyfile import PlyData; p = PlyData.read('$RUN/final.ply')
print(len(p['vertex']), 'gaussians'); print([x.name for x in p['vertex'].properties])
"
```

The upstream docs only describe the export format as method-specific, so I haven't confirmed it. If the property list matches standard 3DGS (`x y z`, `f_dc_*`, `f_rest_*`, `opacity`, `scale_*`, `rot_*`), the toolbox's existing PLY→SPZ and post-processing steps should accept it as is.

---

## 10. Side-by-side with splatfacto

Run splatfacto on the same undistorted scene in the toolbox container (same instance, same iterations), then record:

| Metric | splatfacto | faster-gs |
|---|---|---|
| Wall-clock train time | | |
| Peak VRAM | | |
| # Gaussians | | |
| Test PSNR / SSIM / LPIPS | | |
| PLY size | | |
| Visual check (floaters, fine detail) | | |

Use `DATASET.TEST_STEP` (default 8, every 8th image held out) and match nerfstudio's eval split so the metrics are comparable.

---

## 11. Notes for toolbox integration (if the test passes)

- Container stack: the toolbox image runs torch 2.12.1 + CUDA 13.2 / Python 3.12, while NeRFICG targets torch 2.8 / cu128 / Py3.11. Building `FasterGSCudaBackend` under CUDA 13.2 hasn't been tested. Try it in the toolbox image early. If it fails, fall back to a separate `--target` env as was done for splatfacto-w.
- Build-time license guard: in the Dockerfile, pin `NERFICG_HASH`/`FASTERGS_HASH`, apply the `knn_utils.py` sed, and fail the build if `pip show simple-knn diff-gaussian-rasterization` succeeds.
- Add `TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9"` to match the existing image.
- Model wiring: add `faster-gs` to `config.json`, `main.py`, the workflow_trigger Lambda, `ASLdefinition.json`, `generate_splat.py`, and the Gradio app. Follow the same end-to-end path as `ENABLE_ABSGRAD`. Feed it the toolbox's existing `images/` + `sparse/0` output, write the YAML config at runtime, then run `train.py` → `convert_to_ply.py` → the existing PLY/SPZ export.
- Attribution: include the Apache-2.0 LICENSE/NOTICE for Faster-GS and PPISP, and the MIT notices for NeRFICG and fused-ssim, in the third-party notices.
