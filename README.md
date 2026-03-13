# WebAppGreenHouse

## Project Overview
This project focuses on predicting internal greenhouse temperatures using Machine Learning (ML) and Deep Learning (DL) architectures.

## Prerequisites
To run this project, ensure you have the following versions installed:
* **Python**: 3.11
* **TensorFlow**: 2.15.0
* **Keras**: 2.15.0

To install all required libraries, run:
`pip install -r requirements.txt`

---

## Deployment & Usage Instructions

### 1. Model Training
* Navigate to the `src/` directory.
* To **generate** a single, specific model, run:
  `python creazioneModelloManual.py`
* To generate a sequence of multiple models, run:
  `python creazioneSerieModelli.py`

### 2. Model Deployment
* After the training process is complete, copy the output files and move them to the following directory:
  `mysite\static\modelliKeras`

### 3. Server Launch
* Open the terminal and navigate to the project root folder: `webAppGreenHouse\webAppGreenHouse`
* Execute the batch file to start the application:
  `runserver.bat`