# Object Detection with Voice Feedback for Visually Impaired Persons

---

## Overview

This project aims to assist visually impaired individuals by providing real-time object detection and voice feedback. The system uses advanced object detection technology to identify objects in the environment and communicates this information to the user through audio feedback. This greatly enhances their ability to navigate and interact with their surroundings safely and independently.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Detailed Description](#detailed-description)
  - [Model and Prediction](#model-and-prediction)
  - [Bounding Box and Distance Calculation](#bounding-box-and-distance-calculation)
  - [Voice Feedback Implementation](#voice-feedback-implementation)
  - [User Interaction](#user-interaction)
  - [Real-Time Object Detection and Voice Feedback](#real-time-object-detection-and-voice-feedback)
- [Demo](#live-demo)
- [Future Work](#future-work)
- [Contributing](#contributing)

## Features

- Real-time object detection using YOLOv8 model.
- Voice feedback in multiple languages (English, Hindi, Marathi).
- Distance estimation for objects detected in the frame.
- User-friendly interface for selecting preferred language.
- Supports various objects, providing detailed information about the environment.

## Hardware Requirements

- Raspberry Pi 4 Model B
- Raspberry Pi Camera Module
- MicroSD card (minimum 16GB)
- Power supply for Raspberry Pi
- Speaker or headphones for audio output

## Software Requirements

- Python 3.7 or higher
- PyTorch
- OpenCV
- Google Text-to-Speech (gTTS)
- Google Translate API
- YOLOv8 model

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Jay-J-Pithadia/Major_Project_VOI.git
   cd Major_Project_VOI
   ```

2. **Install the required Python packages:**

   ```bash
   pip install -r requirement.txt
   ```

3. **Download the YOLOv8 model:**

   Download the YOLOv8l model weights from ![Ultralytics](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt).

4. **Setup Raspberry Pi:**

   Follow the official Raspberry Pi documentation to set up your Raspberry Pi and install the necessary drivers for the camera module.

## Usage

1. **Connect the camera module to the Raspberry Pi.**
2. **Ensure the speaker or headphones are connected to the Raspberry Pi.**
3. **Run the main script:**

   ```bash
   python main.py
   ```

4. **Follow the on-screen prompts to select your preferred language.**
5. **The system will start detecting objects and providing voice feedback in real-time.**
6. **To Run on Google Colab (for GPU accelerator): Use obj_det_main.ipynb file.**

## Detailed Description

### Model and Prediction

At the core of our system is the YOLOv8 model, which stands for "You Only Look Once." This model is highly efficient and capable of detecting multiple objects within a single frame. The model processes each frame captured by the camera and outputs the detected objects along with their bounding box coordinates, class labels, and confidence scores.

### Bounding Box and Distance Calculation

Once objects are detected, we extract the bounding box coordinates to determine the position and size of each object. For visually impaired users, knowing how close they are to an object is crucial. To provide this information, we calculate the width of the detected object in the image and use it to estimate the distance between the user and the object.

We achieve this by using a reference object, in this case, a mobile phone, whose real-world width is known. By comparing the width of the mobile phone in the image with its actual width, we can calculate the focal length of the camera. This focal length is then used to estimate the distance to any object detected in the frame.

### Voice Feedback Implementation

To make the system user-friendly, we provide voice feedback. This is particularly helpful for visually impaired users who rely on auditory information. The Google Text-to-Speech (gTTS) library is used to convert text information into speech. Users can choose their preferred language for feedback, with current support for English, Hindi, and Marathi.

When an object is detected, the system generates a descriptive sentence, which is then translated into the user's preferred language if needed. This sentence is converted into speech and played back to the user, providing real-time updates on their surroundings.

### User Interaction

The system is designed to be easy to use. When the system starts, the user is prompted to select their preferred language for voice feedback. This ensures that the information provided is accessible and easily understood.

### Real-Time Object Detection and Voice Feedback

The system operates in real-time, continuously capturing video from the camera, processing each frame for object detection, and providing immediate voice feedback. This real-time capability is crucial for helping visually impaired users navigate dynamic environments safely.

## Live Demo

[![Watch the Live Demo](https://github.com/Jay-J-Pithadia/Major_Project_VOI/assets/96567980/1d9b3367-40a3-43fb-a272-b793561ae31d)](https://drive.google.com/file/d/15ccZ_dX0Dgbk9IMN6TuWTrkdUrXc570H/view?usp=sharing)


## Future Work

Looking ahead, future improvements could include:
- Adding support for more languages.
- Enhancing the accuracy of object recognition.
- Integrating additional features like obstacle detection and navigation assistance.
- Optimizing the system for better performance on different hardware setups.

## Contributing

We welcome contributions from the community. If you have any suggestions or improvements, please feel free to submit a pull request or open an issue.
