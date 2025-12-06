# Neuromorphic Vision: Spiking Neural Networks on Kubernetes

![Prediction Sample](assets/prediction_sample.gif)
*A sample of the N-MNIST event stream being classified by the SNN.*

## Overview
This project implements a **Spiking Neural Network (SNN)** to classify neuromorphic data (N-MNIST) using a cloud-native pipeline on the **NRP Nautilus** cluster. Unlike standard computer vision which processes static frames, this project processes asynchronous event streams from Dynamic Vision Sensors (DVS), utilizing the efficiency of sparse, time-based data.

## Tech Stack
* **snnTorch:** Gradient-based learning for Spiking Neural Networks.
* **Tonic:** Neuromorphic data loading and transformation.
* **PyTorch:** Core deep learning framework.
* **Kubernetes (K8s):** Container orchestration for GPU training jobs.
* **Rclone:** Data synchronization between S3 object storage and compute nodes.

## Architecture
The project uses a hybrid cloud pipeline:
1.  **S3 Storage:** Hosts the raw dataset (`N-MNIST`) and training scripts.
2.  **Init Container:** A lightweight pod that authenticates with S3 and pulls data to a shared ephemeral volume.
3.  **GPU Container:** Performs runtime installation of libraries, trains the SNN using Leaky Integrate-and-Fire (LIF) neurons, and pushes results (Models & GIFs) back to S3.

## Setup & Usage

### 1. Prerequisites
* Access to a Kubernetes cluster (e.g., NRP Nautilus).
* An S3-compatible Object Storage bucket.
* `kubectl` configured locally.

### 2. Secrets Configuration
Create a Kubernetes secret to store your S3 credentials:

```bash
kubectl create secret generic nautilus-s3-keys \
  --from-literal=access-key=YOUR_ACCESS_KEY \
  --from-literal=secret-key=YOUR_SECRET_KEY
