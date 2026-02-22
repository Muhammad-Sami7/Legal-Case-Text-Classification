# Legal-Case-Text-Classification

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)  
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Project Overview
This project applies **supervised machine learning** to classify legal judgments from the **Sindh High Court Case Law Portal** into four jurisdictional categories:

- Civil Appellate Jurisdiction  
- Criminal Appellate Jurisdiction  
- Original Side  
- Writ Jurisdiction  

The pipeline demonstrates **PDF text extraction, TF-IDF feature extraction, Random Forest classification**, and evaluation with a confusion matrix.

---

## Dataset
- Judgments were manually downloaded in PDF format from the [Sindh High Court Case Law Portal](https://caselaw.shc.gov.pk/caselaw/public/rpt-afr).  
- One judge was selected, and 50+ judgments were downloaded.  
- Each judgment was manually labeled into one of the four jurisdictional categories.  
- `labels.csv` maps filenames to their labels.  
- **Note:** Raw PDFs are **not included** due to legal/licensing restrictions.

---
**Data Split:** 80% of the judgments were used for training and 20% for testing.

Classification Report:

Jurisdiction Category           Precision  Recall  F1-Score  Support
---------------------------------------------------------------
Civil Appellate Jurisdiction    1.00      1.00    1.00      3
Criminal Appellate Jurisdiction 1.00      1.00    1.00      2
Original Side                   0.50      0.67    0.57      3
Writ Jurisdiction               0.00      0.00    0.00      2

Overall Metrics:
Accuracy: 70%
Macro Avg F1-Score: 0.64
Weighted Avg F1-Score: 0.67
