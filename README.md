# Fatigue-driving-detection

This repository implements a robust video classification pipeline designed to detect fatigue driving behaviors (Normal, Yawning, Microsleep) through spatio-temporal feature extraction.

## Project Overview
Unlike frame-by-frame image classification, video-based behavioral recognition requires understanding continuous temporal dynamics. This project bridges spatial feature extraction with temporal sequence modeling to achieve stable, video-level predictions, explicitly addressing the challenges of subtle behavioral cues and class imbalance.

## Core Architecture
The system employs a cascaded **CNN + RNN** approach:
* **Spatial Feature Extraction:** A pre-trained ResNet18 network acts as the spatial backbone, extracting high-dimensional visual representations from sequentially sampled video frames.
* **Temporal Modeling:** A Bidirectional LSTM (Bi-LSTM) processes the extracted spatial feature sequences, capturing both forward and backward temporal dependencies across frames to differentiate between transient movements and sustained fatigue behaviors.

## Deep Dive: Data Strategy & Engineering Decisions

**1. Mitigating Class Imbalance via Focal Loss**
In real-world fatigue datasets, critical danger events like *Microsleep* are inherently rare compared to *Normal* driving states. Standard Cross-Entropy loss would bias the model towards the majority class. To counter this, a custom **Focal Loss** dynamically scales the loss based on prediction confidence, heavily penalizing misclassifications of the hard-to-detect and highly dangerous *Microsleep* class (enforcing a 2x higher class weight).

**2. Strict Prevention of Data Leakage**
To ensure the model learns generalized behavioral features rather than memorizing subject-specific backgrounds or lighting, the data pipeline strictly enforces subject-independent train/validation/test splits. This guarantees that evaluation metrics reflect true real-world generalization.

**3. Sequence Sampling Strategy**
Instead of processing redundant consecutive frames, the pipeline utilizes a uniform sampling strategy (`SEQ_LENGTH = 16` with `FRAME_SKIP = 2`). This optimizes computational efficiency while preserving the essential kinematic information required for accurate temporal modeling.

## Repository Structure
* `train_lstm.py`: The core script containing the CNN+Bi-LSTM model architecture, Focal Loss implementation, and training loop.
* `data_preprocess.py` / `shuju.py`: Data loaders handling video frame extraction, sequence sampling, and augmentation.
* `eval.py` / `fix_final.py`: Standalone evaluation scripts for generating accuracy metrics and detailed classification reports.
* `疲劳驾驶行为识别.md`: Detailed experimental report documenting methodology, baseline comparisons, and error diagnosis.
* `confusion_matrix.png`: Visual representation of model performance across the three target classes.
