# Assessment-of-Parkinsonian-Tremor-Severity-Through-Neurorobotic-Models-and-CNN-based-Classification


## Project Overview

This project extends and builds upon previous work (specifically Timothé Petitjean’s thesis) by developing a comprehensive **neurorobotics computational framework** for **Parkinson's Disease (PD)**. The work combines **computational modeling** of the **Basal Ganglia–Thalamo–Cortical (BG-T-C)** loop, **sensorimotor integration** with the **iCub robot simulator**, **feature extraction** from joint movement data, and the application of **deep learning models (CNNs)** to classify Parkinsonian severity levels.

The project is divided into three main stages:

- **Prototype 1:** Replication of closed-loop DBS control using the BG-T-C computational model.
- **Prototype 2:** Sensorimotor integration of the iCub robot with tremor simulation and closed-loop DBS interventions.
- **Prototype 3:** Feature engineering of robotic kinematic data and CNN-based severity classification.

---
## Prerequisites
Before proceeding, please make sure you have installed and compiled the iCub simulation environment and all necessary dependencies from the following repository:
👉 [Timothé Petitjean's GitHub Repository](https://github.com/Timozegu/ClosedLoopDBS-Parkinson-icub)

You must follow the setup instructions provided there, including installing:

robotology-superbuild

YARP middleware

Gazebo simulation environment

These components are necessary for the successful execution of the neurorobotics simulations developed in this project.

## Docker

To build the Docker container:
```bash
cd docker
bash build-docker.sh
```

If running for the first time, compile the `.mod` files by running:
```bash
nrnivmodl
```

## Computational Model

The computational model needs to be launched outside from the docker.

Download the necessary packages in the requirements.txt beforehand.

To run the computational model:
```bash
python MarmosetBG.py
```

## Launch Gazebo Simulation

Install and compile the [robotology superbuild](https://github.com/robotology/robotology-superbuild) before starting simulation.

To launch the Gazebo iCub simulation, first initiate the yarp server:
```bash
yarpserver
```

Then in another Command window launch the gazebo simulation:
```bash
gazebo tutorial_joint-interface.sdf
```

---

# Additional Contributions

## Sensorimotor Loop Integration
The iCub simulator is connected to the computational model to create a full sensorimotor feedback loop, allowing dynamic modulation of motor behavior based on simulated Parkinsonian neural signals.

## Data Extraction
From the iCub simulations, kinematic data including joint angles, joint velocities, tremor amplitude, and frequency are collected to represent motor symptoms.

## Feature Engineering
Critical features such as joint velocity, angular acceleration, tremor frequency, and amplitude are extracted from the raw time-series data to create clean machine learning-ready datasets.

## MANOVA Statistical Validation
A Multivariate Analysis of Variance (MANOVA) is performed to statistically confirm that the engineered kinematic features significantly differ across different Parkinsonian severity conditions (Healthy, PD without DBS, PD with DBS).

## CNN-based Severity Classification
A Convolutional Neural Network (CNN) is designed, trained, and validated to classify Parkinsonian severity levels based on the engineered features. Model evaluation metrics include accuracy, precision, recall, F1-score, and confusion matrices.

---

# How to Use

1. Clone the repository:
```bash
git clone https://github.com/aatish1741/Assessment-of-Parkinsonian-Tremor-Severity-Through-Neurorobotic-Models-and-CNN-based-Classification.git
cd Assessment-of-Parkinsonian-Tremor-Severity-Through-Neurorobotic-Models-and-CNN-based-Classification
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

3. Run the computational BG-T-C model:
```bash
python MarmosetBG.py
```

4. Run the iCub robot pronation-supination simulation:
```bash
python icub_new.py
```

5. Perform feature extraction:
```bash
python feature_engineering.py
```

6. Visualize features:
```bash
python features_visualization.py
```

7. Run MANOVA validation:
```bash
python manova_results.py
```

8. Train and validate CNN classifiers:
- For dataset with DBS intervention:
```bash
python cnn_with_dbs.py
```
- For dataset without DBS intervention:
```bash
python cnn_no_dbs.py
```

---

# Repository Structure
```
.
├── MarmosetBG.py                  # Computational BG-T-C model simulation
├── icub_new.py                     # iCub robot pronation-supination simulation
├── feature_engineering.py          # Feature extraction scripts
├── features_visualization.py       # Feature visualization scripts
├── manova_results.py               # MANOVA statistical validation
├── cnn_with_dbs.py                 # CNN model training with DBS
├── cnn_no_dbs.py                   # CNN model training without DBS
├── requirements.txt                # Python dependencies
```

---

# Acknowledgements
This work builds upon the foundation of Timothé Petitjean’s "Closed-Loop DBS for Parkinson's disease in the iCub" project. Extended contributions in feature engineering, statistical validation (MANOVA), and CNN-based severity classification have been added.


---

