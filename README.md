# LIS_Kaggle
# Multimodal Gesture Classification using RGB and Radar Data

This project addresses the challenge of classifying 126 Italian Sign Language (LIS) gestures by leveraging both visual and radar modalities. It was developed for a Kaggle competition on multimodal learning, and implements a late fusion strategy with temporal modeling using an LSTM network.

## 🔍 Problem Overview

The goal is to recognize LIS gestures performed by different individuals using synchronized RGB video and radar recordings. The dataset is multimodal and requires learning temporal patterns from each modality while effectively combining them for improved performance.

## 🧠 Approach

We use a **late fusion strategy**:
- RGB and radar data are processed independently to extract features.
- Features from both modalities are concatenated.
- A temporal model (LSTM) captures gesture dynamics across time.
- The final output is a gesture classification among 126 classes.

## 📦 Dataset

The dataset consists of:
- RGB video recordings of LIS gestures.
- Radar video signals aligned with RGB.
- Each sample belongs to one of 126 gesture classes.

We preprocess the data as follows:
- Extract **up to 30 frames** from each RGB and radar video. If a video has fewer than 30 frames, we **pad** with zeros.
- Extract features from each frame using:
  - **ResNet50** for RGB
  - A **custom CNN** for radar
- Save features as `.npy` files.
- Use a JSON file to store the original (unpadded) sequence lengths for proper LSTM training with `pack_padded_sequence`.

## 🧪 Fusion Strategy

We explored:
- **Early fusion** (concatenating raw frames): discarded due to alignment issues.
- **Late fusion** (concatenating features): selected as final method.
- **Attention-based fusion**: considered for future work.

## 🏗️ Architecture

- **Feature extractors**:  
  - RGB → ResNet50 → 2048-dim  
  - Radar → Custom CNN → 96-dim  
- **Fusion**: `[RGB | Radar] → 2144-dim`
- **Temporal model**: 2-layer **LSTM**
- **Classifier**: Fully connected + softmax (126 classes)

## 📊 Training

- Optimizer: Adam  
- Loss: CrossEntropyLoss  
- Batch size: 64  
- Sequence padding: Enabled  
- Packed sequences: Used to handle variable-length input  
- Framework: PyTorch
