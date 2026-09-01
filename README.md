# Quantized Model Soup (QMSS)

Official implementation of **Quantized Model Soup Shake-Up: Weight Perturbation for Enhanced Ensemble Diversity**, accepted at **KDD 2026**.

**Quantized Model Soup (QMSS)** is a method for **quantized model merging** and **model averaging** under low-bit weight quantization. It improves the diversity of quantized models while preserving the efficiency of a single merged model.

## Overview

**Model averaging**, also known as **model soup**, combines multiple models by averaging their weights. However, low-bit quantization can map different full-precision weights to the same discrete values, making the quantized models more similar and reducing the diversity needed for effective model merging.

QMSS addresses this problem by applying targeted weight perturbations to generate diverse quantized models. These models are then combined through weight averaging, producing a single quantized model with improved performance and no additional inference cost.
