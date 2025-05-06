# Plant Phenotype Prediction

This project focuses on plant phenotyping, using machine learning techniques for precise trait analysis to support precision agriculture. The key advancements include object detection, semantic segmentation, and temporal analysis applied to plant growth monitoring, age estimation, and health assessment. The goal is to automate and scale the plant phenotyping pipeline to provide better insights into crop growth, improving agricultural productivity and sustainability.

## Table of Contents
- [Project Overview](#project-overview)
- [Methodology](#methodology)
- [Key Improvements](#key-improvements)
  - [Improvement 1: Robust Preprocessing with HSV Segmentation and Canny Edge Detection](#improvement-1-robust-preprocessing-with-hsv-segmentation-and-canny-edge-detection)
  - [Improvement 2: Using Weighted Contrastive Learning](#improvement-2-using-weighted-contrastive-learning)
- [Mathematical Formulation](#mathematical-formulation)
- [Results](#results)
- [Installation](#installation)
- [References](#references)

## Project Overview

Recent advancements in deep learning have greatly benefited plant phenotyping. By integrating Open-World Semantic Segmentation, Vision Transformers (ViTs), and Graph Neural Networks (GNNs), the model improves the prediction of plant traits, addresses species identification in diverse environments, and models hierarchical plant structures. Furthermore, object detection models like Faster R-CNN, YOLOv8, and DETR, along with semantic segmentation techniques such as DeepLabV3+ and U-Net, enable improved weed-crop differentiation and organ detection. 

Incorporating self-supervised learning techniques and multi-modal learning with RGB and multi-spectral imagery offers enhanced feature extraction, making plant health monitoring more automated and scalable.

## Methodology

This research proposes the application of deep learning for phenotypic analysis, including:
- **Open-World Semantic Segmentation:** Identifying unseen plant species.
- **Graph Neural Networks (GNNs) and Vision Transformers (ViTs):** Analyzing hierarchical plant structures.
- **Object Detection Models:** Detecting plant organs (e.g., wheat spikes).
- **Temporal Analysis:** Using ConvLSTMs and 3D CNNs to model growth over time.
- **Self-Supervised and Multi-Modal Learning:** Improving feature extraction with minimal labeled data and integrating multiple imaging techniques.

## Key Improvements

### Improvement 1: Robust Preprocessing with HSV Segmentation and Canny Edge Detection

The preprocessing pipeline aims to address the challenges posed by fluctuating soil color and ambient lighting. The method includes:

1. **Background Removal via HSV Thresholding:** Converts images to the HSV color space and removes non-leaf pixels.
2. **Canny Edge Detection:** Emphasizes leaf contours and vein patterns to improve model robustness to color or shape variations.

This preprocessing method helps improve noise resilience, shape-centric learning, and computational efficiency.

### Improvement 2: Using Weighted Contrastive Learning

Given that the task of plant age and leaf count prediction can be challenging with limited-view images, we propose an innovative approach using **weighted contrastive learning**. Standard contrastive learning treats all image pairs equally, but we introduce a method to assign higher weights to pairs with more semantic similarity (e.g., images with similar leaf counts). This approach allows the model to focus on distinguishing subtle differences, enhancing the accuracy of predictions.

### Benchmarking

We benchmark three distinct contrastive learning models—**Deep InfoMax (DIM)**, **SimCLR**, and **MoCo**—to evaluate their effectiveness in predicting plant traits. The models' performance is compared against Vision Transformers (ViTs), with a particular focus on weighted contrastive loss functions.

## Mathematical Formulation

The contrastive loss function used in this project is formulated as follows:


Where:
- \( \mathcal{L}_{contrastive} \) is the contrastive loss.
- \( N \) is the total number of image pairs.
- \( z_i \) and \( z_i^+ \) represent positive pairs, and \( z_i^- \) is the negative pair.
- \( \text{d}(z_i, z_j) \) is the distance metric (Euclidean distance).
- \( m \) is the margin that separates positive and negative pairs.

## Results

The performance metrics of the deep learning model after applying preprocessing and contrastive learning techniques are as follows:

| Metric  | Leaf (Train) | Age (Train) | Leaf (Validation) | Age (Validation) | Leaf (Test) | Age (Test) |
|---------|--------------|-------------|-------------------|------------------|-------------|------------|
| MAE     | 8.43         | 10.86       | 6.46              | 9.06             | 9.20        | 4.93       |
| RMSE    | 10.80        | 13.00       | 4.05              | 5.32             | 12.97       | 5.79       |
| R²      | -0.657       | -0.475      | 0.107             | 0.094            | -0.374      | -1.056     |

## Installation

To set up this project on your local machine, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/plant-phenotype.git

# install Dependencies
pip install -r requirements.txt

# run 1st improvement
python first.py

# run 2nd improvement
python second.py
