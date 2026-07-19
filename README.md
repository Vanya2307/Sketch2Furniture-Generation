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

Two generative models will be compared:

- a reconstruction-only U-Net baseline trained with an L1 objective;
- a Pix2Pix conditional GAN using the same U-Net generator architecture together with a PatchGAN discriminator.

The initial scope focuses on the `beds` and `dressers` categories. These categories are directly relevant to bedroom furniture design and provide a useful contrast between bed structures and predominantly rectilinear dresser forms.

The work builds on the earlier [Furniture-Sketch-Classifier](https://github.com/Vanya2307/Furniture-Sketch-Classifier) project, which used Canny edge detection, HOG features, and classical machine learning for furniture classification. The earlier classifier will later be reused as an auxiliary evaluator of category preservation in generated images.

## Data Source

The project uses the Bonn Furniture Styles Dataset introduced by Aggarwal et al. (2018).

The original dataset contains six furniture categories: beds, chairs, dressers, lamps, sofas, and tables. The predefined train / validation / test definitions are preserved as the starting point for the analysis.

Raw images are not included in this repository.

## Setup

Local development and data analysis are performed in VS Code under WSL. GPU training will be performed in Google Colab.

Create and activate the local environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
The notebooks should be executed in numerical order because later stages will use metadata and outputs created by earlier notebooks.

## Current Progress

- [x] Repository structure created
- [x] Local Python environment configured
- [x] Dataset split parser implemented
- [x] Original split records loaded
- [ ] Complete data exploration and cleaning
- [ ] Generate paired edge-photo data
- [ ] Train reconstruction-only U-Net
- [ ] Train Pix2Pix
- [ ] Evaluate and compare both models
- [ ] Complete conclusions and documentation