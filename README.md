# YOLO Object Detection using OpenCV, YOLO, and Streamlit

## Introduction

This repository contains a **YOLO-based object detection** project using **OpenCV and Streamlit**. The application allows users to upload an image and detect multiple objects using **YOLO deep learning models**.

## Features

- **Detects multiple object categories** using a pre-trained YOLO model.
- **Draws bounding boxes** around detected objects with confidence scores.
- **Uses YOLOv3-Tiny for lightweight, fast detection.**
- **Streamlit UI** for seamless interaction.

## Installation

### Prerequisites

Make sure you have Python installed (recommended version: **3.7 or later**).

### Install Required Libraries

Run the following command to install the necessary dependencies:

```bash
pip install opencv-python-headless streamlit numpy pillow
```

For YOLO object detection, also install:

```bash
pip install torch torchvision
```

## How to Run the Application

### Running YOLO Object Detection

Ensure you have the necessary YOLO model files:

- `yolov3-tiny.cfg`
- `yolov3-tiny.weights`
- `coco.names`

Then, run:

```bash
streamlit run yolo_object_detection/yolo_app.py
```

## Usage

1. **Upload an image** for object detection.
2. **Click 'Detect Objects'** to start the YOLO model.
3. The application will display the detected objects with bounding boxes and labels.

## Technologies Used

- **OpenCV**: Used for image processing and feature detection.
- **Streamlit**: Used for building an interactive web-based UI.
- **NumPy**: Used for handling image data arrays.
- **YOLOv3-Tiny**: Used for deep learning-based object detection.
- **PyTorch**: Used for handling the YOLO model in Python.

## Notes

- YOLO object detection works best with clear images and proper lighting.
- Ensure YOLO model files are correctly placed in the project directory before running the app.

## License

This project is open-source and available for modification and distribution under the MIT License.

## Author

Developed by **Karuna Kurewad**.

