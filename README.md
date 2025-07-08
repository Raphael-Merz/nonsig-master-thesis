# The p stands for Proofread  
_Automated Detection and Correction of Nonsignificance Misinterpretations_

This repository contains the code, data, and documentation for my master’s thesis project. The goal of this project is to develop a pipeline that automatically **detects**, **classifies**, and **corrects** common misinterpretations of nonsignificant p-values (e.g., "p > .05 = no effect").

---

## 🧠 Project Overview

Researchers often misstate the implications of nonsignificant results, interpreting them as proof of an effect’s absence. This project automates the detection and correction of such claims through:

- **Classification**: Using BERT-based classifiers to determine whether a statement correctly interprets a nonsignificant result.
- **Detection and Correction**: Using the R package [`papercheck`](https://scienceverse.github.io/papercheck/) to identify suspect statements and query LLMs for corrected alternatives.

---

## 📁 Repository Structure

```text
data/
├── raw/           # Manually labeled dataset (source: open access articles)
├── processed/     # Cleaned or augmented versions for training/testing

notebooks/
├── python/        # Jupyter notebooks to train and evaluate BERT models
├── r/             # Quarto scripts using the papercheck package and LLM prompts

thesis/              # Proposal, thesis manuscript (in Quarto + apaquarto)
results/           # Model outputs, plots, tables
