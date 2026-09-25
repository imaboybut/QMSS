<div align="center">

# Quantized Model Soup Shake-Up (QMSS)

### Weight Perturbation for Enhanced Ensemble Diversity

**Jinwoo Chung, Sungyeop Jung, Weronika Czorapinska, Jangho Kim**<sup>†</sup>
Kookmin University  ·  <sup>†</sup>Corresponding author

**KDD 2026** · Jeju, Korea

[![Paper](https://img.shields.io/badge/Paper-ACM%20DL-0055A4)](https://dl.acm.org/doi/pdf/10.1145/3770855.3817826)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3770855.3817826-blue)](https://doi.org/10.1145/3770855.3817826)
[![Artifact](https://zenodo.org/badge/DOI/10.5281/zenodo.20496349.svg)](https://doi.org/10.5281/zenodo.20496349)
[![KDD 2026](https://img.shields.io/badge/KDD-2026-orange)](https://doi.org/10.1145/3770855.3817826)

</div>

---

Official implementation of **QMSS**, a **quantized model merging** method that makes **model soup / weight averaging** work under **low-bit quantization-aware training (QAT)**.

> **TL;DR** — Averaging fine-tuned *quantized* models barely helps, because the quantizer maps them all to nearly the same integer weights (*Ensemble Degeneracy*). QMSS "shakes up" a small fraction of weights sitting near quantization-bin boundaries so each candidate lands on different integer values, then averages them. You get ensemble-level gains with **single-model inference cost**.

## Why model soup breaks under quantization

Model soup needs two things: **(1) linear mode connectivity (LMC)** so that averaged weights stay in a low-loss region, and **(2) diversity** among the averaged models. We show that QAT threatens both — and that only one of them is actually the problem:

- **LMC is preserved** if fine-tuning starts from a sufficiently converged checkpoint, which we call the **QAT anchor state**. We give both an analytical argument and empirical evidence.
- **Diversity collapses.** The quantizer $Q(\cdot)$ is many-to-one, so two fine-tuned models with $\|\theta_1 - \theta_2\|_2 > 0$ often satisfy $Q(\theta_1) = Q(\theta_2)$. We call this **Ensemble Degeneracy** and measure it with **Quantized Weight Shift (QWS)**, the fraction of integer assignments that differ from the soup. On CIFAR-100 at 4/4-bit, QMSS raises QWS from 8.83% (standard soup) to 25.69%.

## Method

1. **Anchor.** Train a QAT model to convergence (the QAT anchor state).
2. **Fine-tune candidates.** Fine-tune multiple candidates from the anchor with randomly sampled hyperparameters.
3. **Shake-up.** In each candidate, perturb a small fraction (e.g. **1% or 5%**) of **low-magnitude weights inside a narrow zone around quantization thresholds**, flipping their integer bin. These weights are already the main source of bin oscillation and contribute little to learned features, so perturbing them adds diversity without leaving the loss basin.
4. **Recover.** Briefly re-train each perturbed candidate with QAT.
5. **Average.** Weight-average all candidates into **one** quantized model.

QMSS is quantizer-agnostic (tested with **EWGS** and **LSQ**) and can also be combined with **SWA**.

<!--
## Getting Started

TODO: environment / installation

```bash
git clone https://github.com/imaboybut/QMSS.git
cd QMSS
pip install -r requirements.txt
```

TODO: 1) train QAT anchor  2) fine-tune + shake-up candidates  3) weight-average + evaluate
-->

## Citation

If you find QMSS useful for your research, please cite:

```bibtex
@inproceedings{chung2026quantized,
  title     = {Quantized Model Soup Shake-Up: Weight Perturbation for Enhanced Ensemble Diversity},
  author    = {Chung, Jinwoo and Jung, Sungyeop and Czorapinska, Weronika and Kim, Jangho},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2},
  pages     = {687--698},
  year      = {2026},
  doi       = {10.1145/3770855.3817826}
}
```

## Contact

Questions and issues are welcome via [GitHub Issues](https://github.com/imaboybut/QMSS/issues), or email Jinwoo Chung (imaboybut@kookmin.ac.kr).

**Keywords:** quantized model merging · model soup · weight averaging · quantization-aware training (QAT) · low-bit quantization · ensemble diversity · linear mode connectivity · edge deployment
