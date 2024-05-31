# GEMStack_Empty_spot_perception
Part of Automatic Parking Project - Empty Parking Spot Perception in GEMStack. This repository contains the implementation of the Empty Spot Perception task for an automatic parking system using GEMStack. The project utilizes a custom dataset and YOLOv8 for detecting empty parking spots with high accuracy.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Training and Validation](#model-training-and-validation)
- [Results](#results)
- [Usage](#usage)


## Introduction

The Empty Spot Perception project aims to develop an automatic parking system that accurately detects empty parking spots using oriented bounding boxes (OBB). The project leverages a custom-built dataset and the YOLOv8 model to achieve high precision in detecting empty spots in various environments.

## Dataset

### Custom Training Dataset

A custom dataset was created, containing images of empty parking lots with corresponding label files (.txt) indicating the location of annotated bounding boxes in (x, y, w, h, r) format. The labels were generated using tools like Makesense.ai and Roboflow.

- **(x, y)**: Center of the bounding box in the image
- **(w, h)**: Width and height of the bounding box
- **(r)**: Rotating angle of the bounding box

#### Dataset Distribution

| Image Type       | No. of Images | No. of Labels |
|------------------|---------------|---------------|
| Carla Simulation | 175           | 120           |
| Twilight         | 47            | 34            |
| Real-time        | 90            | 80            |
| Test-site (rosbags) | 150         | 110           |

## Model Training and Validation

The training dataset was split into 95% train set, 4% validation set, and 2% test set. The YOLOv8 OBB model was trained to achieve a mean Average Precision (mAP) of 92%.

- **Training Parameters**:
  - Epochs: {50, 70, 80, 100}
  - Batch size: {5, 7, 10}
  - Confidence score: > 0.7
  - Single image and real-time video stream processing (webcam) were tested on rosbags.

The model generated accurate spatial representations in images by detecting bounding boxes with high confidence.

## Results

The YOLOv8 OBB model achieved a high mAP, indicating excellent performance in detecting empty parking spots. The model's performance was validated using both single images and real-time video streams, demonstrating its robustness and reliability.

![Result 1](images/img1.png)

![Result 2](images/img2.png)

![Result 3](images/img3.png)

![Result 4](images/img4.png)

## Usage

To use the model for detecting empty parking spots, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ChandanaGiridhar/GEMstack_Empty_spot_perception.git
   cd GEMstack_Empty_spot_perception
