# From Detection to Correction 
_A Hybrid NLP Approach to Misinterpretations of Nonsignificant p Values_

This repository contains the code, data, and documentation for my master’s thesis project. The goal of this project is to develop a pipeline that automatically **detects**, **classifies**, and **corrects** common misinterpretations of nonsignificant p-values (e.g., "p > .05 = no effect").

---

## 🧠 Project Overview

Researchers often misinterpret nonsignificant results as evidence for the absence of an effect. This project combines rule-based and machine learning methods to automate the detection and correction of such misstatements through:

- **Detection**: Via Regular Expression (RegEx) searches implemented in the [`papercheck`](https://scienceverse.github.io/papercheck/) R package
- **Classification**: Using BERT-based classifiers to determine whether a statement correctly interprets a nonsignificant result
- **Detection and Correction**: Querying LLMs for corrected alternatives through [`papercheck`](https://scienceverse.github.io/papercheck/).

---

## 📁 Repository Structure

```
data/
├── detection_check/               # Validation of automatic detection against manual annotation
│   ├── article_pdfs/              # 25 article pdfs manually reviewed; including highlighted statements on nonsignificant effects
│   ├── detection_checked.xlsx     # Statements detected automatically + manually, for comparison
│   ├── detection_unchecked.xlsx   # Only automatically detected statements
├── llm_correction_check/          # Spreadsheets for LLM-correction validity checks 
│                                   (original statements, LLM revisions, manual label of correctness)
├── model_performance/             # Classifier training results (confusion matrices, loss curves, etc.)
├── training_data/                 
│   ├── labeled/                   # Classifier training data (statements + manual labels) and annotation notes
│   ├── unlabeled/                 # Statements without labels

notebooks/
├── python/                        # Jupyter notebooks to train/evaluate BERT models
│   ├── archive/                   # Archived versions of classifiers
│   ├── best_model/                # Best model checkpoint for BERT, SciBERT, PubMedBERT
│   ├── results/                   # Epoch-by-epoch models; best moved to 'best_model/' after evaluation
│                                   (intermediate checkpoints deleted to save space)
├── r/                             # Quarto markdown scripts for dataset creation and preprocessing

thesis/                             # Thesis manuscript and related files
├── _extensions/                   # Files for apaquarto extension: https://wjschne.github.io/apaquarto/
├── proposal/                      # Original project proposal (April 2025)
```

---

## 📄 License

- 📄 Code and documentation: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- 📊 Data: Reused under fair use from open-access sources (details in thesis)

## 👤 Contact Info

**Raphael Merz**  
Ruhr University Bochum  
📧 Raphael.Merz@rub.de  
ORCID: 0000-0002-9474-3379  

If you use this project or the data, please consider citing it or getting in touch.