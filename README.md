# 🌿 Crop Disease Detection and Management System

A deep learning-powered system for the early detection and classification of crop diseases from leaf images. This project implements a high-performance inference pipeline designed for real-time alerts, enabling proactive crop management and minimizing yield loss.

## 📜 Overview

This project addresses the critical challenge of timely crop disease identification. By leveraging state-of-the-art deep learning models, it provides a fast and accurate method to detect diseases from images, simulating a smart camera sensor system. The core of the project is a highly efficient inference workflow (TensorFlow → ONNX → FastAPI) that delivers predictions with sub-300ms latency, making it suitable for real-time field applications.

The system is designed to reduce disease spread, cut costs associated with manual inspection, and provide early warnings for timely intervention, ultimately contributing to more sustainable agriculture.

***

## ✨ Key Features

* *High-Accuracy Classification:* Utilizes pre-trained models like MobileNetV2 and ResNet50 to achieve *86.2% accuracy* and an *F1-score of 0.89*.
* *Real-Time Inference:* A highly optimized inference workflow built with ONNX and FastAPI, maintaining a *latency of less than 300ms per frame*.
* *Lesion Localization:* Includes a prototype for instance segmentation using Mask R-CNN, which demonstrated a *22% precision boost* in localizing disease lesions on augmented test sets.
* *Scalable API:* A robust backend server built with FastAPI to handle image-based prediction requests efficiently.
* *Simulated Smart Sensing:* Models a smart camera system projected to *reduce disease spread by 35%* and provide alerts with a *lead time of less than 12 hours*.

***

## 🛠 Tech Stack

* *Model Development:* Python, TensorFlow, Keras
* *Inference Optimization:* ONNX, ONNX Runtime
* *API Serving:* FastAPI, Uvicorn
* *Data Handling:* NumPy, Pillow
* *Dataset:* PlantVillage (for demonstration and training)

***

## ⚙ Setup and Installation

Follow these steps to set up the project environment and install the necessary dependencies.

### *1. Prerequisites*
* Python 3.8+
* pip and venv

### *2. Clone the Repository*
```bash
git clone https://github.com/harieshankar04/Plant-Disease-Detection-Model
cd crop-disease-detection
