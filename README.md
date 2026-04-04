<h1 align="center">ContourFeatures-in-CNN: Orientation & Symmetry Analysis</h1>

This repository contains the official codebase for a research project conducted at the BWLab, investigating how convolutional neural networks (CNNs) encode fundamental perceptual principles.

We analyze internal representations of **VGG16** across 73,000 natural images from the Natural Scenes Dataset (NSD), focusing on two core visual properties: **Orientation** and **Symmetry**

---

## Project Overview

<table style="border: 1px solid #ccc; border-collapse: collapse; width:100%;">
<tr>

<td width="50%" style="border: 1px solid #ccc; vertical-align: top; text-align:center;">

<h3>Orientation Analysis</h3>

<img src="docs/orient_intro.PNG" width="80%" style="display:block; margin: 0 auto; vertical-align:middle;">

<p>We study how CNN feature maps encode orientation information across layers.</p>

<ul style="text-align:left;">
  <li>Analyzed <strong>CONV1-1 layer (64 channels)</strong> and <strong>CONV5-3 layer (512 channels)</strong> feature maps</li>
  <li>Compared with orientation representations from:
    <ul>
      <li>contour</li>
      <li>line drawing</li>
      <li>photo pipelines</li>
    </ul>
  </li>
  <li>Used <strong>Pearson correlation</strong> across 180° bins</li>
</ul>

</td>

<td width="50%" style="border: 1px solid #ccc; vertical-align: top; text-align:center;">

<h3>Symmetry Analysis</h3>

<img src="docs/sym_intro.PNG" width="80%" style="display:block; margin: 0 auto; vertical-align:middle;">

<p>We investigate whether CNNs capture symmetry-related structure.</p>

<ul style="text-align:left;">
  <li>Analyzed all convolutional layers of VGG16</li>
  <li>Examined:
    <ul>
      <li>contour symmetry</li>
      <li>medial-axis symmetry</li>
      <li>area-based symmetry</li>
    </ul>
  </li>
  <li>Tested across:
    <ul>
      <li>parallel</li>
      <li>mirror</li>
      <li>taper structures</li>
    </ul>
  </li>
  <li>Used <strong>hierarchical (nested) regression</strong></li>
  <li>Measured <strong>ΔR² contribution</strong> of symmetry features</li>
</ul>

</td>

</tr>
</table>

<!-- poster button -->
<p align="center">
  <a href="docs/rop_poster.pdf" target="_blank" 
     style="background-color:#6031ff; color:black; padding:10px 20px; text-align:center; text-decoration:none; display:inline-block; border-radius:5px; font-weight:bold;">
    Click to View the Poster
  </a>
</p>

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