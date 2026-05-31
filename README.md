<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=6366f1&height=120&section=header&text=Loss%20Function%20Analysis&fontSize=36&fontColor=ffffff&fontAlignY=38&desc=MSE%20vs%20Cross-Entropy%20on%20Classification%20Tasks&descAlignY=60&descSize=15&descColor=a5b4fc" width="100%"/>

[![License: MIT](https://img.shields.io/badge/License-MIT-6366f1?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-MNIST-14b8a6?style=flat-square)]()

</div>

---

## Overview

A rigorous empirical comparison of Mean Squared Error (MSE) and Cross-Entropy (CE) loss functions on a classification task using a MNIST subset. This experiment addresses a frequently misunderstood question in deep learning: if MSE works for regression, why is it suboptimal for classification?

The answer lies in the gradient dynamics and this repository makes that concrete through controlled experiments.

> *Loss function choice is not a hyperparameter. It is a modeling decision with theoretical consequences.*

---

## The Research Question

MSE and CE are both valid loss functions. Both measure the distance between predictions and targets. Yet CE consistently outperforms MSE on classification tasks. Why?

**The theoretical answer:** CE loss is derived from maximum likelihood estimation under a categorical distribution. Its gradient with respect to the logits is proportional to the prediction error producing well-scaled, non-saturating gradients. MSE's gradient saturates near 0 and 1 under sigmoid/softmax activations, causing the vanishing gradient problem.

**This repository:** makes that theoretical argument empirical and measurable.

---

## Experimental Design

| Setting | Detail |
|---|---|
| **Dataset** | MNIST subset |
| **Architecture** | Identical MLP for both conditions |
| **Loss A** | Mean Squared Error (MSE) |
| **Loss B** | Cross-Entropy (CE) |
| **Controlled variables** | Architecture, optimizer, learning rate, batch size |
| **Measured** | Training loss, validation accuracy, gradient norms, convergence speed |

---

## Key Findings

- CE converges faster and to a lower validation error than MSE on the same architecture
- MSE gradient norms are systematically smaller in early training, confirming the saturation hypothesis
- The gap between CE and MSE widens with deeper networks and longer training

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Deep Learning** | PyTorch |
| **Data** | MNIST (torchvision) |
| **Visualization** | Matplotlib, Jupyter |
| **Language** | Python 3.10+ |

---

## Getting Started

```bash
git clone https://github.com/royxlead/loss-function-analysis-python.git
cd loss-function-analysis-python

pip install -r requirements.txt
jupyter notebook loss_analysis.ipynb
```

---

## Research Context

This experiment is foundational for understanding why standard practice in classification uses CE rather than MSE. The gradient saturation argument connects directly to the broader literature on activation function design, batch normalization, and the vanishing gradient problem. Understanding loss function dynamics at this level is prerequisite knowledge for building reliable production ML systems.

---

## Related Work

- [Self-Diagnosing Neural Models](https://github.com/royxlead/self-diagnosing-neural-models-python) - How loss function choice affects calibration
- [Multi-Objective Feature Selection](https://github.com/royxlead/multi-objective-evolutionary-feature-selection-python) - Optimization in ML systems

---

<div align="center">

**[Portfolio](https://royxlead.netlify.app) · [LinkedIn](https://linkedin.com/in/royxlead) · [ORCID](https://orcid.org/0009-0009-6582-2295)**

<img src="https://capsule-render.vercel.app/api?type=waving&color=6366f1&height=80&section=footer" width="100%"/>

</div>
