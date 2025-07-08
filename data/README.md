# Data Overview

This folder contains all data files used in this project. It includes both **labeled** and **unlabeled** statements extracted from open-access scientific articles.

## 📁 Folder Structure

- `labeled/` – Manually coded dataset used to train and evaluate classification models
- `unlabeled/` – Automatically extracted statements for detection and correction tasks

---

## 📌 Labeled Data

The files in `labeled/` include statements that were manually reviewed and labeled based on whether they represent a correct or incorrect interpretation of a nonsignificant p value.

A detailed coding scheme is included in the second sheet (`codebook`) of `labeled_data.xlsx`.

> 💡 The labeled data was created manually by Raphael Merz. For more information about the labeling process, see the final thesis manuscript.

---

## 📂 Unlabeled Data

The files in `unlabeled/` were extracted automatically using the [`papercheck`](https://scienceverse.github.io/papercheck/) R package. These statements have not been manually coded and are used for detection and correction using language models.

---

## ⚠️ Licensing and Fair Use Notice

The statements in both subfolders are excerpted from open-access scientific articles retrieved via the [`papercheck`](https://scienceverse.github.io/papercheck/) package.

- They are provided here **solely for academic research and analysis.**
- **No copyright is claimed** by the author of this repository over the extracted statements.
- Redistribution of the full dataset may be limited depending on the copyright status of the original sources.

> Users are responsible for ensuring that any reuse complies with the original licensing terms or fair use provisions.
