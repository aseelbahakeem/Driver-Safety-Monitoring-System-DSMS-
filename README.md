# Driver Safety Monitoring System
**A Complete SDLC Project: When 100% Test Accuracy Misleads**

![Status](https://img.shields.io/badge/Status-Complete-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue) 
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-100%25-brightgreen)
![External Accuracy](https://img.shields.io/badge/External%20Accuracy-60%25-orange)

- **Course:** COIT-415 
- **Institution:** King Abdulaziz University  
- **Semester:** Summer 2024
- **Project Status:** In-progress
---

## Project Overview

This project follows a complete Software Development Life Cycle for a Driver Safety Monitoring System:

### Phase 1: Analysis & Design
- System requirements specification (10 functional, 5 non-functional)
- UML modeling: Class, Use Case, Sequence, and Activity diagrams
- Full documentation (33 pages) available in `plan, design and analysis/`

### Phase 2: Implementation & Critical Discovery
- Drowsiness detection using MobileNetV2 transfer learning
- Achieved +99% accuracy
-  However, Grad-CAM analysis revealed model learned spurious correlations instead of eye closure --> External validation exposed catastrophic failure: 0% drowsy recall

 
### Phase 3: In progress


---

## Problem Statement
Driving fatigue poses a significant public safety threat:
- **25%** of road accidents caused by driver drowsiness
- **4×** increased accident risk with mobile phone usage  
- Chronic conditions significantly elevate accident likelihood

**Objective:** Develop an AI-powered drowsiness detection system that works reliably in real-world conditions, not just on benchmark datasets.

**Challenge:** Can we trust a model with +99% test accuracy for safety-critical deployment?

---

## Methodology

**Phase 1: Benchmark Evaluation**
- Dataset: Driver Drowsiness Dataset (Kaggle) -> 41,793 images
- Data cleaning: Removed 69% duplicates --> 12,823 unique images
- Split: 70% train | 15% val | 15% test (stratified)
- Model: MobileNetV2 with ImageNet transfer learning
- Result: 99.93% validation accuracy, 100% test accuracy

**Phase 2: Interpretability & External Validation**
- Grad-CAM visualization of learned features
- External testing on out-of-distribution dashcam images
- Baseline comparison with rule-based Eye Aspect Ratio (EAR)
 
---

## Key Results

### Training Performance

![Training Curves](results/image_13.png)
*Training reached 99.93% validation accuracy by epoch 3, early stopping at epoch 6*

---

### Test Set Performance

![Test Confusion Matrix](results/image_6.png)
*MobileNetV2 achieved 100% test accuracy*

---

### Interpretability

![Grad-CAM Analysis](results/image_7.png)
*Grad-CAM reveals model focuses on head angle, forehead, and mostly not the eyes*
---

### External Validation 

![External Results](results/image_8.png)
*Test 1: Ground truth = Drowsy | MobileNetV2 = Alert (wrong) | Baseline EAR = Drowsy (correct)*

**Performance Summary:**

| Metric | Test Set | External Images | Change |
|--------|----------|-----------------|---------|
| Overall Accuracy | 100.0% | 60.0% | **-40%** |
| Drowsy Recall | 100.0% | **0.0%** | **-100%** |
| Alert Precision | 100.0% | 100.0% | 0% |
| Prediction Pattern | Balanced | **100% Alert** | Systematic bias |

**Critical Finding:** Model predicted "Alert" for ALL 5 external images with >99% confidence, including clearly drowsy drivers with closed eyes.

---

## Root Cause: Dataset Collection Bias

The dataset was collected under controlled conditions where spurious features were correlated:

**Drowsy images:** Eyes closed + Head tilted + Specific lighting + Similar setup  
**Alert images:** Eyes open + Head upright + Different lighting + Different setup

Standard train/val/test split prevents overfitting but not dataset-wide systematic biases.
---

## Alternative Solutions

After discovering the spurious correlation problem, several approaches could address this:

### 1. Dataset Remediation (Root Cause)
- Collect diverse data with varied camera angles, lighting, head positions for both classes

### 2. YOLOv8 Multi-Task Detection 
- Use YOLOv8 for simultaneous detection: eyes, phone, head pose

### 3. PERCLOS Temporal Tracking (Implemented)
- Rule-based EAR tracking over time (90 frames = 3 seconds). It is Proof-of-concept demonstrating interpretable alternative
---

## Key Lessons

### 1. Benchmark Performance ≠ Real-World Reliability
Test accuracy dropped from 100% → 60% on external data. High performance on held-out test sets doesn't guarantee correct feature learning or real-world generalization.

### 2. Interpretability Is Not Optional
Without Grad-CAM, we would have deployed a model with 100% test accuracy that fails catastrophically in production. Interpretability analysis must be performed regardless of benchmark performance.

### 3. Dataset Bias ≠ Overfitting
```
High train + High test + Catastrophic external = Dataset Bias
```
Standard splits prevent overfitting but NOT systematic biases shared across all data.

### 4. External Validation Is Essential
Standard ML: Dataset → Split → Train/Val/Test → Deploy  
**Safety-Critical ML:** Dataset → Split → Train/Val/Test → Interpretability → External OOD Validation → Deploy

### 5. Simple Methods Can Outperform Complex Models
When deep learning learns spurious correlations, rule-based methods using correct features may be more reliable.

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/yourusername/driver-safety-system.git
cd driver-safety-system
pip install -r requirements.txt

# Download dataset
import kagglehub
path = kagglehub.dataset_download("ismailnasri20/driver-drowsiness-dataset-ddd")

# Run notebook
jupyter notebook notebooks/Driver_Safety_Monitoring_System.ipynb
```
---

## Future Work

**Immediate Priorities:**
1. YOLOv8 integration for multi-task detection (eyes, phone, head pose)
2. Diverse dataset collection (multiple cameras, lighting, angles)
3. Hybrid system combining YOLOv8 + PERCLOS validation
4. Real-world pilot study (100+ hours driving)

**Research Directions:**
- Attention mechanisms forcing focus on eye regions
- Causal learning to identify spurious correlations
- Domain adaptation for camera/lighting robustness
- Synthetic data generation with 3D face models

---
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
--- 
*Last Updated: November 2024*
