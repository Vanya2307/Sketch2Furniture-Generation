# Sketch2Furniture-Generation


Deep learning generation of realistic furniture images from paired edge sketches using Pix2Pix.

**Author:** Vanya Videva  
**Course:** Deep Learning, SoftUni  
**Date:** July-August 2026

---

## Research Question

> Can deep learning transform sketch-like furniture edge maps into plausible RGB furniture images, and does adversarial training improve visual realism compared with reconstruction-only learning?

## Project Overview

This project investigates supervised sketch-to-image generation for furniture design. Each original furniture photograph is paired with a Canny edge representation. The edge map serves as the model input, while the corresponding RGB photograph serves as the target.

Two image-to-image models were compared:

* a reconstruction-only U-Net baseline trained with an L1 objective;
* a Pix2Pix conditional GAN using the same U-Net generator architecture together with a PatchGAN discriminator.

The project focuses on the `beds` and `dressers` categories. These categories are directly relevant to bedroom furniture design and provide a useful contrast between bed structures and predominantly rectilinear dresser forms.

This work represents Phase 2 of a broader three-phase plan. Phase 1, the earlier [Furniture-Sketch-Classifier](https://github.com/Vanya2307/Furniture-Sketch-Classifier) project, used Canny edge detection, HOG features and classical machine-learning models for furniture classification. Phase 2 extends the work from recognizing furniture sketches to generating realistic furniture images from them. Phase 3 is planned as an interactive demonstration in which users can draw or upload furniture sketches and receive category predictions or generated realistic concepts.


## Data Source

Original source: [Bonn Furniture Styles Dataset](https://arxiv.org/abs/1812.03570) - Aggarwal et al. (2018)

The dataset contains furniture photographs organized into six product categories and 17 furniture styles. It was originally provided by the dataset authors upon request for non-commercial educational use. The authors' dataset page is currently unavailable, so new requests may not be possible at this time.

This project focuses on two categories:

- `beds`
- `dressers`

After image-integrity, duplicate and cross-split leakage checks, the preparation stage produced 14318 paired examples. The predefined training, validation and test splits from the original dataset were preserved.

Each pair contains a 256 × 256 Canny edge map as the model input and the corresponding RGB photograph as the target.

The original photographs and generated paired dataset are not included in this repository because of dataset usage restrictions and file size. Instructions for obtaining and preparing the data are provided in the Setup and Reproducibility section.

See References for the full citation.


## Repository Structure

- `notebooks/01_data_exploration.ipynb`
  Dataset loading, metadata validation, integrity checks, duplicate and cross-split leakage analysis, and cleaning

- `notebooks/02_data_preparation.ipynb`
  Canny edge map generation, paired dataset creation, and pair validation

- `notebooks/03_unet_baseline.ipynb`
  Reconstruction-only U-Net trained with an L1 objective  

- `notebooks/04_pix2pix_training.ipynb`
  Pix2Pix training with a PatchGAN discriminator, and checkpoint comparison

- `notebooks/05_model_evaluation.ipynb`
  Test-set evaluation, paired statistical tests, and category-level, structural and qualitative analyses

- `notebooks/06_conclusions_future_work.ipynb`
  Project synthesis, effect sizes, limitations, and roadmap to Phase 3

- `src/`
  Reusable Python modules for data loading and preparation

- `tests/`
  Unit tests for the data pipeline

- `data/raw/`
  Original Bonn dataset images (not tracked)

- `data/processed/`
  Generated Canny edge maps, RGB targets and prepared metadata (not tracked)

- `data/metadata/`
  Generated metadata and split files (not tracked)

- `outputs/figures/`
  Figures from Notebook 01 (committed)

- `outputs/tables/`
  Result tables (committed)

Notebooks 03 to 06 run in Google Colab and write their figures, metric tables and model checkpoints to Google Drive rather than to `outputs/`. All figures produced there are embedded in the committed notebooks.

## Setup and Reproducibility

### Requirements

Local environment (Notebooks 01 and 02): see `requirements.txt`. PyTorch is not required locally, since data preparation uses only OpenCV, Pillow, NumPy and pandas.

Colab environment (Notebooks 03 to 06): PyTorch and torchvision are preinstalled. Notebook 05 additionally installs `torchmetrics` and `lpips` in its first cell.

Tested on Google Colab with PyTorch 2.11.0+cu128 and torchvision 0.26.0+cu128, on a T4 GPU.

### Local Setup

Clone the repository:

```bash
git clone https://github.com/Vanya2307/Sketch2Furniture-Generation.git
cd Sketch2Furniture-Generation
```

On Windows, create and activate the virtual environment with:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

In Command Prompt, use `.venv\Scripts\activate` instead.

On macOS or Linux, use:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Place the Bonn Furniture Styles Dataset in `data/raw/bonn_furniture_styles/`.

### Google Drive Setup

Notebooks 03 to 06 read from and write to a Google Drive folder with the following structure:

```text
MyDrive/Sketch2Furniture/
├── colab_data/                    # paired_dataset.tar
├── checkpoints/
│   ├── unet_baseline/
│   └── pix2pix/
├── figures/
└── results/
```

The path is set once per notebook in the `DRIVE_ROOT` constant and is the only path that normally needs changing.

### Reproducing the Full Project

1. Run Notebooks 01 and 02 locally. These produce `data/processed/paired/` and `prepared_metadata.csv`.

2. Create the paired dataset archive from the project root:

```bash
tar -cf ../paired_dataset.tar data/processed/paired
```

3. Upload `paired_dataset.tar`, which is created in the parent directory, to `MyDrive/Sketch2Furniture/colab_data/`.

4. Run Notebooks 03 to 06 in Google Colab, in order. Notebooks 03 and 04 require a GPU runtime; Notebooks 05 and 06 do not.

Notebooks 03 to 05 extract the archive to Colab's local disk if the paired dataset is not already present in the current runtime. A new Colab runtime requires the archive to be extracted again.

Notebooks 03 and 04 write training checkpoints to Drive after every epoch, allowing an interrupted run to resume from the last completed epoch.

Approximate GPU time: 30 minutes for Notebook 03 and 105 minutes for Notebook 04 on a T4.

### Evaluating Without Training

Notebook 05 can be run using the released model weights instead of training the models. Steps 1 to 3 above are still required because the evaluation loads the test images from the archive.

Download the three files described in Model Weights and Release, place them at the listed Google Drive paths, and then run Notebook 05.

## Model Weights and Release

Trained generator weights are published as a [GitHub release](https://github.com/Vanya2307/Sketch2Furniture-Generation/releases) rather than committed to the repository, because each file is approximately 207 MB.

Each file contains generator weights only, without discriminator or optimizer state, and is therefore sufficient for evaluation but not for resuming training.

| Release asset | Save to | Validation L1 |
|---|---|---|
| `unet_best_generator.pt` | `checkpoints/unet_baseline/best_generator.pt` | 0.2131 |
| `pix2pix_generator_epoch_20.pt` | `checkpoints/pix2pix/checkpoint_epoch_20.pt` | 0.2494 |
| `pix2pix_generator_epoch_48.pt` | `checkpoints/pix2pix/best_generator.pt` | 0.2395 |

Destination paths are relative to the `Sketch2Furniture` folder in Google Drive. The file names differ between the release and the destinations because two of the checkpoints are named `best_generator.pt` within their own directories.

Notebook 05 loads all three models from these paths. The released files cannot be used to resume training because they do not contain discriminator or optimizer state. Remove the released files from the listed destinations before running Notebooks 03 or 04 for a complete retraining run. In particular, Notebook 04 expects `checkpoint_epoch_20.pt` to be a full training checkpoint produced by its own run.

## Results and Main Findings

Both models were evaluated on 2892 previously unseen test pairs.

| Metric | U-Net | Pix2Pix epoch 20 | Pix2Pix epoch 48 |
|---|---|---|---|
| MAE (lower is better) | 0.2164 | 0.2527 | 0.2408 |
| PSNR (higher is better) | 16.99 | 15.33 | 15.73 |
| SSIM (higher is better) | 0.6624 | 0.5534 | 0.5713 |
| LPIPS (lower is better) | 0.3426 | 0.3472 | 0.3421 |
| Edge IoU (higher is better) | 0.6926 | 0.8511 | 0.8465 |

The results establish a trade-off rather than a single better model. The reconstruction-only U-Net reproduces the specific target photograph more accurately, with near-complete rank-based dominance on SSIM against Pix2Pix epoch 48 (rank-biserial r = -0.977). Pix2Pix preserves the contours supplied by the conditioning sketch substantially better (r = 0.837 for edge IoU) and produces sharper, more sketch-aligned structures, but introduces more colour and texture artefacts.

The two models are not statistically distinguishable on LPIPS overall. An exploratory per-category analysis shows why: the U-Net is better for beds and Pix2Pix epoch 48 is better for dressers, with opposing differences of similar magnitude that cancel when pooled.

Neither model recovers colour, material or texture information that is absent from the edge map. Generated images tend toward the wood tones that dominate the training data, so vivid targets are reconstructed in muted colours.

Greater edge density is associated with worse MAE, PSNR, SSIM and edge IoU but better LPIPS, suggesting that edge density acts primarily as a proxy for image complexity rather than as a measure of how informative the sketch is.

Full results, statistical tests and figures are in [Notebook 05](notebooks/05_model_evaluation.ipynb), and the synthesis is in [Notebook 06](notebooks/06_conclusions_future_work.ipynb).

## Limitations and Future Work

- **Scope.** Only beds and dressers from a single dataset of isolated product photographs. The findings may not generalize to other categories, cluttered backgrounds or other image domains.
- **Inputs are extracted edges, not drawings.** Canny edges come from the target photograph and share its exact geometry. Freehand sketches are incomplete, distorted and stylistically variable, so real sketch-to-image generation is harder than the task evaluated here.
- **Colour cannot be recovered.** A binary edge map does not encode hue, material or texture, so the generator can only reproduce the appearance distribution of its training data.
- **Edge IoU can be misleading on sparse sketches.** Agreement on a small number of contours produces a high score even when the generated image is nearly featureless. One test pair reached 0.892 while producing no recognizable furniture.
- **Single run per model.** Each configuration was trained once, without repeated seeds or a hyperparameter search, so run-to-run variation is unmeasured.
- **Dataset quality.** At least one image is filed under the wrong category, and unconfirmed near-duplicate candidates were retained rather than removed.

Future work could test whether the trade-off persists across additional categories, repeated training runs and real freehand sketches, and whether an additional conditioning signal such as a colour hint or the style label already present in the metadata would allow target appearance to be specified rather than inferred. A human preference study and a density-normalized contour metric could evaluate perceptual quality and contour preservation more reliably than the current automated metrics alone.

Phase 3 is planned as an interactive demonstration in which users draw or upload a sketch and receive a generated result.

The full discussion is in [Notebook 06](notebooks/06_conclusions_future_work.ipynb).

## References


* Aggarwal, D., Valiyev, E., Sener, F., & Yao, A. (2018). Learning Style Compatibility for Furniture. *German Conference on Pattern Recognition*, 552–566. Springer. arXiv:1812.03570.

<details>
<summary>BibTeX citation</summary>

```bibtex
@inproceedings{aggarwal2018learning,
  title={Learning Style Compatibility for Furniture},
  author={Aggarwal, Divyansh and Valiyev, Elchin and Sener, Fadime and Yao, Angela},
  booktitle={German Conference on Pattern Recognition},
  pages={552--566},
  year={2018},
  organization={Springer}
}
```
</details>
  

* Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A. (2017). Image-to-image translation with conditional adversarial networks. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 1125–1134.

* Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention (MICCAI), 234–241. Springer.

The complete reference list, including evaluation metrics, statistical methods and software, is in [Notebook 06](notebooks/06_conclusions_future_work.ipynb).


