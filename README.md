# Through Van Gogh's Eyes: Global Style Transfer with Diffusion Model

[![arXiv](https://img.shields.io/badge/arXiv-2608.11546-b31b1b.svg)](https://arxiv.org/abs/2608.11546)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
📄 Paper: [arXiv:2608.11546](https://arxiv.org/abs/2608.11546)

---

## :high_brightness: Introduction

<p align="center">
  <img src="figure/global_style_transfer.pdf" alt="Global Style Transfer overview" width="90%">
</p>

**Global Style Transfer (GST)** is an artistic image synthesis paradigm that moves beyond the traditional *One-to-One* style transfer setting — where a single content image is stylized using one or a few reference artworks — toward a **Many-to-One** setting that aggregates an artist's *entire* body of work into a single, coherent style representation.

<cite index="13-1">Conventional style transfer methods stylize a content image using one or a few reference artworks, which works well for artwork-level stylization but struggles to capture the broader stylistic distribution of an artist. Text-to-image diffusion models conditioned on artist names (e.g., "in Van Gogh style") are more flexible but tend to inherit text-induced bias and only reproduce patterns from a handful of iconic pieces.</cite>

<p align="center">
  <img src="figure/main_figure2.pdf" alt="Overall pipeline of GST" width="90%">
</p>

To address this, the paper introduces two core components:

- **Global Style Guidance (GSG)** — <cite index="13-1">learns a residual global style offset in the intermediate feature space (h-space) of a diffusion model under a fixed prompt, capturing artist-level style purely from visual statistics rather than text, which reduces text-dependent bias.</cite> This module is implemented in this repository as the **Style Extraction Function (SEF)**, a lightweight one-hidden-layer MLP `f_t` that transforms a diffusion feature `h_t` into a style-conditioned residual:

  Δh_t = f_t(h_t; θ)

- **Content Alignment Guidance (CAG)** — <cite index="13-1">a training-free perceptual guidance mechanism that preserves the semantic structure of the content image while still allowing artist-specific geometric deformation.</cite>

<p align="center">
  <img src="figure/stylebook.pdf" alt="Overall pipeline of GST" width="90%">
</p>


<cite index="13-1">Experiments on the WikiArt dataset show that GST achieves stronger stylistic fidelity, content preservation, and output diversity compared to existing style transfer and diffusion-based artistic synthesis methods.</cite>



## 🚀 Highlights

- **Many-to-One style aggregation** — captures an artist's global style from their full body of work, not just one or two reference pieces.
- **Text-bias free** — global style is learned from visual statistics in h-space, not from artist-name text conditioning.
- **Training-free content preservation** — Content Alignment Guidance keeps the semantic layout intact while allowing artistic deformation.

## 📋 Requirements

```bash
requirements.txt
```

## 🛠️ Pipeline / Usage

The pipeline consists of four stages: dataset preparation, paired-dataset construction, SEF training, and style-transfer inference.

### 1. Download an Artist's Artwork Dataset

Download the artwork dataset for the target artist whose style you want to transfer (e.g., Van Gogh).

```
ex) vangogh dataset --> vangogh2photo
```

### 2. Build a Label-Paired Dataset

Run [`artist_data_load.py`](artist_data_load.py) to construct the label-paired dataset used for training.

```bash
python3 artist_data_load.py
```

Before running, configure the following paths in `config.py`:

| Argument | Description |
|---|---|
| `args.train_data_dir` | Path to the downloaded artworks dataset (used internally as `source_image_dir`) |
| `args.label_resize_train_data_dir` | Directory where the newly generated paired dataset will be saved (used internally as `root_dir`) |

### 3. Train the Style Extraction Function (SEF)

Once the paired dataset is ready, train the SEF using `train.py`. Upon completion, the trained weights are saved as `unet.pth`.

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py
```

### 4. Generate Style-Transferred Images

With the trained SEF, run global style transfer inference to generate the final stylized images.

```bash
bash test.sh <gpuID>
```

Example:

```bash
bash test.sh 0
```


## 📝 Notes

- Make sure `config.py` is updated with the correct dataset paths (`train_data_dir`, `label_resize_train_data_dir`) before running Step 2.
- Training in Step 3 requires a CUDA-enabled GPU; set `CUDA_VISIBLE_DEVICES` to select the target device.
- The GPU ID passed to `test.sh` in Step 4 determines which device is used for inference.

## 📖 Citation

If you find this repository useful for your research, please cite:

```bibtex
@article{lee2026gst,
  title   = {Through Van Gogh's Eyes: Global Style Transfer with Diffusion Model},
  author  = {Lee, Jeongha and Kim, Yujin and Ali, Ghazanfar and Kim, Suhyun and Hwang, Jae-In},
  journal = {arXiv preprint arXiv:2608.11546},
  year    = {2026}
}
```