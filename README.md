<h1 align="center">ContourFeatures-in-CNN: Orientation & Symmetry Analysis</h1>

This repository contains the official codebase for a research project conducted at the BWLab, investigating how convolutional neural networks (CNNs) encode fundamental perceptual principles.

We analyze internal representations of **VGG16** across 73,000 natural images from the Natural Scenes Dataset (NSD), focusing on two core visual properties: **Orientation** and **Symmetry**

---

## Project Overview

<table style="border: 1px solid #ccc; border-collapse: collapse;">
<tr>

<td width="50%" style="border: 1px solid #ccc; vertical-align: top;">

<div align="center">

### Orientation Analysis

We study how CNN feature maps encode orientation information across layers.

</div>

- Analyzed **CONV1-1 layer (64 channels)** and **CONV5-3 layer (512 channels)** feature maps  
- Compared with orientation representations from:
  - contour  
  - line drawing  
  - photo pipelines  
- Used **Pearson correlation** across 180° bins  

<p align="center">
  <img src="docs/orient_intro.png" width="80%">
</p>

</td>

<td width="50%" style="border: 1px solid #ccc; vertical-align: top;">

<div align="center">

### Symmetry Analysis

We investigate whether CNNs capture symmetry-related structure.

</div>

- Analyzed all convolutional layers of VGG16  
- Examined:
  - contour symmetry  
  - medial-axis symmetry  
  - area-based symmetry  
- Tested across:
  - parallel  
  - mirror  
  - taper structures  
- Used **hierarchical (nested) regression**  
- Measured **ΔR² contribution** of symmetry features  

<p align="center">
  <img src="docs/sym_intro.PNG" width="80%">
</p>

</td>

</tr>
</table>

<!-- poster button -->
<p align="center">
  <a href="docs/rop_poster.png" target="_blank" 
     style="background-color:#4CAF50; color:white; padding:10px 20px; text-align:center; text-decoration:none; display:inline-block; border-radius:5px; font-weight:bold;">
    Click to View the Poster
  </a>
</p>
---

## Installation / Setup

Follow these steps to set up and run the project:

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ContourFeatures-in-CNN.git
cd ContourFeatures-in-CNN
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare the dataset**
Place the Natural Scenes Dataset (NSD) under the data/ folder

---
## References
- [Natural Scenes Dataset (NSD)](https://naturalscenesdataset.org/)
---
## Affiliation

Conducted at the **[BWLab, University of Toronto](https://https://www.bwlab.org/)**

---