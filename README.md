# Driver Safety Monitoring System (in-progress)
**A Complete Software Development Life Cycle Project: From Analysis & Design to Implementation with Critical AI Safety Discoveries**

![Status](https://img.shields.io/badge/Status-Research%20Complete-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue) a
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-100%25-brightgreen)
![External Accuracy](https://img.shields.io/badge/External%20Accuracy-60%25-orange)

**Course:** COIT-415 
**Institution:** King Abdulaziz University  
**Semester:** Summer 2024

## Project Overview

### Phase 1: Analysis & Design
- **System Requirements Analysis:** Functional and non-functional requirements specification
- **UML Modeling:** Comprehensive system design using:
  - Class diagrams
  - Use case diagrams  
  - Sequence diagrams
  - Activity diagrams
- **Documentation:** Full analysis and design report (33 pages)

### Phase 2: Implementation & Research Discovery
- **Deep Learning Implementation:** Drowsiness detection using MobileNetV2 transfer learning
- **Critical Discovery:** Despite achieving 100% test accuracy, interpretability analysis revealed the model learned spurious correlations
- **Real-World Validation:** External testing exposed failure in recall and being bias toward alert classification 
- **Alternative Solution:** Designing PERCLOS-based temporal tracking system or fixing the dataset 

**Key Contribution:** This project demonstrates that comprehensive SDLC practices, including interpretability analysis and rigorous validation, are essential for safety-critical AI systems. High benchmark performance alone is insufficient for deployment.

## Problem Statement

### Real-World Safety Crisis

Driving fatigue poses a significant public safety threat:
- **25%** of road accidents caused by driver drowsiness
- **4×** increased accident risk with mobile phone usage  
- Chronic conditions significantly elevate accident likelihood

**Objective:** Develop an AI-powered drowsiness detection system that works reliably in real-world conditions, not just on benchmark datasets.

**Challenge:** Can we trust a model with 100% test accuracy for safety-critical deployment?

## Methodology

### Two-Phase Evaluation Framework

**Phase 1: Benchmark Evaluation**
- Dataset: Driver Drowsiness Dataset (DDD) from Kaggle  
- Original: 41,793 images → Cleaned: 12,823 (69% duplicates removed)
- Split: Train (70%) | Val (15%) | Test (15%) - stratified
- Model: MobileNetV2 with ImageNet transfer learning
- **Result:** 99.96% validation accuracy, 100% test accuracy

**Phase 2: Interpretability & Real-World Validation**
- Grad-CAM analysis to visualize learned features
- External validation on out-of-distribution dashcam images  
- Comparison with rule-based baseline (Eye Aspect Ratio)
- **Result:** Revealed catastrophic failure and spurious correlations

---

## Complete SDLC Documentation

### Analysis & Design Phase

The complete system analysis and design documentation is available in `docs/System_Analysis_Design.pdf` (33 pages), which includes:

**Chapter 1: Introduction**
- Project background and motivation
- Safety statistics (25% of accidents from drowsiness, 4x risk with phone usage)
- Project objectives and scope
- System description

**Chapter 2: System Analysis & Design**

**Requirements Specification:**
- **10 Functional Requirements (FR1-FR10):**
  - FR1-FR3: Account management (register, login, manage account)
  - FR4: Emergency contact management
  - FR5: View reports
  - FR6: Monitoring session management (start/stop, camera connection)
  - FR7-FR8: Alert system (trigger alerts, send SMS)
  - FR9: Critical indicator detection (drowsiness, phone usage, fainting)
  - FR10: Live video capture

- **5 Non-Functional Requirements (NFR1-NFR5):**
  - NFR1: Availability (24/7 system access)
  - NFR2: Usability (clear, easy interface)
  - NFR3: Performance (fast response, handle concurrent users)
  - NFR4: Security (protect sensitive data)
  - NFR5: Speed (immediate emergency response)

**UML Diagrams:**

1. **Class Diagram** (8 classes, 10 relationships)
   - Core classes: Person, Account, EmergencyContact, MonitoringSystem, Camera, Frame, Model, Report
   - Relationships: Generalization, Association, Aggregation, Composition

2. **Use Case Diagram** (11 use cases, 3 actors)
   - Actors: Driver, Camera, Emergency Contact
   - Use cases: Register, Login, Manage Account, Manage Emergency Contact, View Report, Manage Monitoring, Capture Live Video, Detect Critical Indicators, Trigger Alert, Send SMS Alert
   - Relationships: Include, Extend

3. **Sequence Diagrams** (7 scenarios)
   - Register flow
   - Login flow
   - Manage Account flow
   - View Report flow
   - Manage Emergency Contact flow
   - Driver Safety Detection flow (critical path)
   - Manage Monitoring Session flow

4. **Activity Diagrams** (2 workflows)
   - Monitoring Session and Camera Connection workflow
   - Detection workflow (frame processing, alert triggering, SMS notification)

**Design Decisions:**
- Mobile application architecture for accessibility
- Wi-Fi camera integration for real-time monitoring
- Cloud-based ML model for detection 
- SMS emergency notification system
- Report generation for health monitoring

---

## Results

### Dataset Overview

![Class Distribution](results/image_1.png)
*Figure 1: Original dataset contained 22,348 drowsy and 19,445 alert images*

![Sample Images](results/image_2.png)
*Figure 2: Representative samples showing variety in the dataset*

---

### Data Quality Analysis

**Critical Discovery: 69% Duplicates**

The dataset contained a massive number of duplicate images that could cause data leakage. Using perceptual hashing (pHash), it was identified and removed:
- Original dataset: 41,793 images
- After deduplication: 12,823 unique images
- Duplicates removed: 28,970 (69%)

![Image Properties](results/image_3.png)
*Figure 3: All images standardized to 227×227 pixels with aspect ratio 1.0*

![Brightness & Contrast](results/image_4.png)
*Figure 4: Brightness (mean: 104.1) and contrast (mean: 52.7) distribution analysis*

---

### Model Training Performance

![Training Curves](results/image_13.png)
*Figure 5: Training reached 99.93% validation accuracy by epoch 3, with early stopping at epoch 6*

---

### Test Set Evaluation

![Baseline Confusion Matrix](results/image_5.png)
*Figure 6: Baseline EAR model achieved 66.46% accuracy using simple geometric rules*

![Test Confusion Matrix](results/image_6.png)
*Figure 7: MobileNetV2 achieved perfect 100% test accuracy - seemingly flawless performance*


### Interpretability Analysis: The Critical Discovery

![Grad-CAM Analysis](images/image_7.png)
*Figure 8: Grad-CAM reveals model focuses on head angles, forehead, glasses, and mostly not eyes*

### External Real-World Validation

![External Test 1](results/image_8.png)
*Figure 9: Ground truth: Drowsy | MobileNetV2: Alert (WRONG) | Baseline: Drowsy (CORRECT)*

![External Test 2](results/image_9.png)
*Figure 10: Ground truth: Alert | MobileNetV2: Alert | Baseline: Failed*

![External Test 3](results/image_10.png)
*Figure 11: Ground truth: Alert | MobileNetV2: Alert | Baseline: Alert*

![External Test 4](results/image_11.png)
*Figure 12: Ground truth: Alert | MobileNetV2: Alert | Baseline: Alert*

![External Test 5](results/image_12.png)
*Figure 13: Ground truth: Drowsy | MobileNetV2: Alert (WRONG) | Baseline: Failed*

**External Validation Results:**

| Image | Ground Truth | MobileNetV2 | Confidence | Baseline EAR | Winner |
|-------|--------------|-------------|------------|--------------|---------|
| **test_1** (dashcam) | **Drowsy** | Alert (WRONG) | 100.0% | **Drowsy (CORRECT)** (0.062) | **Baseline** |
| test_2 (glasses) | Alert | Alert | 99.6% | Failed (no face) | MobileNetV2 |
| test_3 (turned) | Alert | Alert | 100.0% | **Alert** (0.419) | **Both** |
| test_4 (striped) | Alert | Alert | 100.0% | **Alert** (0.417) | **Both** |
| **test_5** (eyes closed) | **Drowsy** | Alert (WRONG) | 100.0% | Failed (no face) | **Neither** |

## Root Cause Analysis

### Dataset Collection Bias

The Kaggle dataset was likely collected under controlled conditions where multiple features were inadvertently correlated:

**"Drowsy" images captured when:**
- Eyes closed (actual indicator)
- **+ Head tilted down** (spurious)
- **+ Specific head angle/posture** (spurious)
- **+ Mouth slightly open** (spurious)
- **+ Specific lighting conditions** (spurious)
- **+ Similar background setup** (spurious)

**"Alert" images captured when:**
- Eyes open (actual indicator)  
- **+ Head upright/straight** (spurious)
- **+ Different posture** (spurious)
- **+ Different lighting** (spurious)
- **+ Different setup** (spurious)

```

### Why Test Accuracy Was Misleading

```
Train Set    ──┐
Val Set      ──┼──► All share same collection methodology
Test Set     ──┘      ↓
                 Same camera angles
                 Same lighting setup
                 Same capture environment
                 Same systematic biases
                      ↓
                 Model learns biases
                      ↓
                 Biases generalize WITHIN dataset
                      ↓
                 High test accuracy
                      
External Data ──► Different methodology
                      ↓
                 Biases don't transfer
                      ↓
                 Catastrophic failure
```

**note:** Standard train/val/test split prevents **overfitting** but NOT **dataset bias**.

---

## Technical Details

### Dataset

**Source:** Driver Drowsiness Dataset (DDD) -> Kaggle

**Data Quality Pipeline:**
1. **Duplicate Detection:** Perceptual hashing (pHash) identified 69% duplicates
2. **Cleaning:** Reduced 41,793 → 12,823 unique images
3. **Split:** 70% train (13,533) | 15% val (2,900) | 15% test (2,900) - stratified
4. **Augmentation:** Rotation (±15°), horizontal flip, color jitter, affine transforms
5. **Normalization:** ImageNet mean/std [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

**Class Distribution:**
- Drowsy: 7,226 images (56.3%)
- Alert: 5,597 images (43.7%)

---

### Model Architecture

**MobileNetV2 with Transfer Learning**

```
Input (224×224×3)
    ↓
MobileNetV2 Backbone (ImageNet pretrained)
    ├─ Depthwise Separable Convolutions
    ├─ Inverted Residual Blocks
    └─ Global Average Pooling
    ↓
Fully Connected Layer (2 classes)
    ↓
Softmax → [P(Drowsy), P(Alert)]
```

**Training Configuration:**
- **Parameters:** 2.26M trainable
- **Optimizer:** Adam (lr=0.001, β₁=0.9, β₂=0.999)
- **Loss:** CrossEntropyLoss with class weights [1.0, 1.29]
- **Batch Size:** 32
- **Early Stopping:** Patience=3 epochs
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=2)
- **Hardware:** NVIDIA L4 GPU (23.8GB VRAM)
- **Training Time:** 53.59 minutes (6 epochs)

### Baseline: Eye Aspect Ratio (EAR)

**Rule-Based Geometric Approach:**

```python
def calculate_ear(eye_landmarks):
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
    
    Where:
    - p1, p4: Horizontal eye corners
    - p2, p3, p5, p6: Vertical eye points
    """
    # Vertical eye distances
    A = euclidean_distance(p2, p6)
    B = euclidean_distance(p3, p5)
    
    # Horizontal eye distance
    C = euclidean_distance(p1, p4)
    
    EAR = (A + B) / (2.0 * C)
    return EAR

# Classification
if EAR < 0.25:
    prediction = "Drowsy"
else:
    prediction = "Alert"
```

**Advantages:**
- Uses correct semantic feature (eye geometry)
- Camera-agnostic (works across different setups)
- Lighting-robust (relative measurements)
- Explainable (pure mathematics, no black box)
- Fast (60+ FPS, no GPU required)
- Edge-deployable (Raspberry Pi compatible)

**Limitations:**
- Requires accurate face detection
- Fails on extreme head angles
- Sensitive to occlusions (hands, shadows)

**Validation Performance:**
- Test Accuracy: 66.46%
- External Accuracy: 100% when face detected (3/3 correct)
- External Failure: 40% face detection failure rate

---

### Alternative Solution: PERCLOS System

After discovering the deep learning failure, designed a temporal tracking system based on transportation safety research:

**System Components:**
1. **Temporal drowsiness detection** (PERCLOS-based)
2. **Fainting detection** (greater than 10s continuous eye closure)
3. **Phone usage detection** (planned: YOLO integration)
4. **Emergency notification** (SMS to registered contacts)
5. **Real-time dashboard** (EAR, PERCLOS, alert status)

**Comparison:**

| Feature | MobileNetV2 | Baseline EAR | PERCLOS |
|---------|-------------|--------------|---------|
| **Performance** |
| Test Accuracy | 100.0% | 66.5% | N/A (rule-based) |
| External Accuracy | 60.0% | 60.0%* | Expected: High |
| Drowsy Recall (ext) | 0.0% | 100%* | Expected: 100% |
| **Interpretability** |
| Explainable | Black box | Full | Full |
| Feature Used | Head angle/posture | Eye geometry | Temporal eye closure |
| **Deployment** |
| GPU Required | Yes | No | No |
| FPS | 20-30 | 30-60 | 60+ |
| Edge Device | Difficult | Easy | Easy |
| Camera-Agnostic | No | Yes | Yes |
| Robust to Lighting | No | Moderate | Yes |

*When face detection succeeds

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/driver-safety-system.git
cd driver-safety-system

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
opencv-python>=4.8.0
mediapipe>=0.10.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
numpy>=1.24.0
pillow>=10.0.0
imagehash>=4.3.0
scipy>=1.11.0
```

---

### 2. Dataset Setup

**Option A: Use Kaggle Dataset**

```python
import kagglehub

# Download dataset
path = kagglehub.dataset_download("ismailnasri20/driver-drowsiness-dataset-ddd")
print(f"Dataset downloaded to: {path}")
```

### 3. Run the Pipeline

```bash
jupyter notebook notebooks/Driver_Safety_Monitoring_System.ipynb
```

**Execute sections in order:**

| Section | Description | Output |
|---------|-------------|--------|
| 1-2 | Setup & Data Loading | Dataset statistics |
| 3 | Data Cleaning | Remove 69% duplicates |
| 4 | EDA | Understand data properties |
| 5 | Train/Val/Test Split | 70/15/15 stratified split |
| 6 | Data Preprocessing | Augmentation & normalization |
| 7 | Baseline EAR Model | 66.46% validation accuracy |
| 8 | MobileNetV2 Training | 99.93% validation accuracy |
| 9 | Test Evaluation | 100% test accuracy |
| **10** | **Grad-CAM Analysis** | **Discover spurious correlations** |
| **11** | **External Validation** | **Reveal catastrophic failure** |
| 12 | PERCLOS Alternative | Robust temporal tracking |

---

### 4. Reproduce Key Findings

**Test on External Images:**

```python
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = timm.create_model('mobilenetv2_100', pretrained=False, num_classes=2)
model.load_state_dict(torch.load('models/best_model.pth'))
model = model.to(device)
model.eval()

# Test image
img = Image.open('data/external_test/test_1.png').convert('RGB')
img_tensor = val_test_transform(img).unsqueeze(0).to(device)

# Prediction
with torch.no_grad():
    output = model(img_tensor)
    probs = F.softmax(output, dim=1)
    pred_class = torch.argmax(probs).item()
    confidence = probs[0, pred_class].item()

print(f"Prediction: {'Drowsy' if pred_class==0 else 'Alert'}")
print(f"Confidence: {confidence*100:.1f}%")

# Run Grad-CAM
gradcam = GradCAM(model, last_conv_layer)
heatmap = gradcam.generate_cam(img_tensor, pred_class)

# Visualize attention
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.imshow(img); plt.title('Original')
plt.subplot(1, 3, 2); plt.imshow(heatmap, cmap='jet'); plt.title('Attention')
plt.subplot(1, 3, 3); plt.imshow(overlay); plt.title('Overlay')
plt.show()

# Observe: Model focuses on head angle, NOT eyes
```

**Run PERCLOS System:**

```python
from src.perclos_system import DriverSafetySystem
import cv2

# Initialize system
system = DriverSafetySystem()

# Process video stream
cap = cv2.VideoCapture(0)  # Webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame
    processed_frame, status, metrics = system.process_frame(frame)
    
    # Display
    cv2.imshow('PERCLOS Driver Safety Monitor', processed_frame)
    
    # Print metrics
    print(f"Status: {status} | EAR: {metrics['EAR']:.3f} | PERCLOS: {metrics['PERCLOS']:.1f}%")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Future Work

### Immediate Next Steps

- Collect diverse external validation dataset (1,000+ images from multiple sources)
- Validate PERCLOS system on real-time video streams
- Implement phone detection using YOLO object detection
- A/B test: MobileNetV2 vs PERCLOS on identical external test set
- Conduct user studies on alert system effectiveness and false alarm tolerance

### Research Directions

- **Hybrid Architecture:** Combine DL feature extraction with rule-based validation
  - Use CNN for robust face/eye detection
  - Use EAR/PERCLOS for actual drowsiness classification
  
- **Attention Mechanisms:** Force model to attend to relevant regions
  - Spatial attention gates on eye regions
  - Penalize attention on irrelevant features
  
- **Causal Learning:** Explicitly model causal relationships
  - Structural causal models for drowsiness
  - Counterfactual reasoning to identify spurious correlations
  
- **Domain Adaptation:** Train to be robust across distributions
  - Adversarial training on camera types, angles, lighting
  - Meta-learning for quick adaptation to new setups
  
- **Synthetic Data Generation:** Create diverse training data
  - 3D face models with controllable eye closure
  - Vary camera angles, lighting, backgrounds systematically

### Deployment Considerations

- Real-world pilot study in vehicles (100+ hours of driving)
- Edge deployment on Raspberry Pi / NVIDIA Jetson
- Integration with vehicle CAN bus for automatic interventions
- Regulatory compliance testing (ISO 26262 for automotive)
- Long-term monitoring for model drift and performance degradation
- Privacy-preserving architecture (on-device processing only)

## License

This project is licensed under the MIT License - see LICENSE file for details.


### The Three-Part Validation Framework

```
┌─────────────────────────────────────────────────┐
│          BENCHMARK PERFORMANCE                   │
│                                                  │
│  High test accuracy (100%)                      │
│  But what did the model actually learn?         │
└──────────────────┬──────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────┐
│      INTERPRETABILITY ANALYSIS                   │
│                                                  │
│  Grad-CAM reveals spurious correlations         │
│  Model ignores eyes, focuses on artifacts       │
└──────────────────┬──────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────┐
│       EXTERNAL VALIDATION                        │
│                                                  │
│  Catastrophic failure (0% drowsy recall)        │
│  40% accuracy drop on OOD data                  │
└─────────────────────────────────────────────────┘

```

**High accuracy does not equal correct learning does not equal real-world reliability**

---

*Last Updated: November 2025*
