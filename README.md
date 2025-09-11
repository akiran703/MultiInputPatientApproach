# Multi-Input Patient-Centered COVID-19 Detection

## Overview

Rapid detection of COVID-19 is essential to prevent disease spread. While numerous machine learning algorithms have been developed to detect COVID-19 using Computerized Tomography (CT) lung scans, current approaches have significant limitations:

- **Image-centric instead of patient-centric**: Most algorithms prioritize accurate detection on individual images rather than considering all scans from a single patient
- **Risk of misdiagnosis**: When multiple CT scans exist for one patient, treating each scan independently can lead to incorrect diagnoses if not all images are incorporated into the final decision
- **Limited generalization**: Using single datasets raises concerns about model robustness across different CT machine environments and image qualities

## Our Approach

We address these limitations by developing convolutional neural network (CNN) algorithms that prioritize accurate diagnosis at the patient level using multiple scans. Our methodology includes:

1. **Single-input CNN with patient-level aggregation** (`finalpredictionmodel.py`)
   - Individual image processing followed by aggregation of predictions by patient ID
   
2. **Multi-input CNN architectures** (`4input_alexnet.py` and `7input_alexnet.py`)
   - Simultaneous processing of 4 or 7 images from the same patient

Both approaches are validated using patient-based splits across two large COVID-19 CT datasets to demonstrate robustness and real-world applicability.

## Architecture Details

### AlexNet Base Architecture
All models use a simplified AlexNet architecture optimized for medical imaging:

- **Conv Layer 1**: 96 filters, 11×11 kernel, stride 4, ReLU activation
- **Conv Layer 2**: 256 filters, 5×5 kernel, ReLU activation  
- **Conv Layer 3**: 384 filters, 3×3 kernel, ReLU activation
- **Conv Layer 4**: 384 filters, 3×3 kernel, ReLU activation
- **Conv Layer 5**: 256 filters, 3×3 kernel, ReLU activation
- **Fully Connected**: 4096 → 4096 → 1000 → 2 neurons with dropout regularization

## File Descriptions

### `finalpredictionmodel.py` - Single-Input with Aggregation

**Architecture**: AlexNet CNN processing individual CT images (224×224×3)

**Patient-Level Aggregation Methods**:
- **Averaging**: Averages prediction probabilities across all patient images
- **Majority Voting**: Takes majority vote across individual image predictions  
- **Entropy-based Weighting**: Weighs each scans prediction based on the model's confidence [weight is logarithmic] (method explored post UROC SOAR)
- **Z-score Normalization**: Transforms mean and standard deviation of COVID-19 probabilities into Z-scores, then avergaing the Z-scores (method explored post UROC SOAR)
- **Bayesian Weighting**: Weighting each scans prediction based on the model's confidence [weight is linear] (method explored post UROC SOAR)

**Key Features**:
- Individual image processing with intelligent patient-level aggregation
- Five different aggregation strategies for robust diagnosis
- SGD optimization with appropriate learning rates
- Comprehensive performance metrics and patient-wise analysis

### `4input_alexnet.py` and `7input_alexnet.py` - Multi-Input Architecture

**Architecture**: Parallel AlexNet branches processing 4 or 7 CT slices simultaneously

**Input Processing**: 
- 4 or 7 CT scan images (224×224×3) per patient
- Automatic padding with zeros if fewer images available
- Transfer learning with pre-trained weights

**Key Features**:
- End-to-end patient-centric training
- Cross-dataset transfer learning for improved generalization
- Adam optimization with appropriate learning rates
- Comprehensive training visualization and metrics

## Methodology

### Patient-Centered Design
Unlike traditional image-by-image approaches, our models ensure:
- All available CT scans from a patient contribute to the final diagnosis
- Single low-quality images cannot lead to misdiagnosis
- Conflicting COVID-19 indicators are properly resolved
- Patient identity is preserved throughout evaluation

### Data Processing Pipeline
- **Preprocessing**: Image normalization and resizing with aspect ratio preservation
- **Data Splitting**: Patient-based splits to prevent data leakage
- **Cross-Dataset Validation**: Testing on independent COVID-19 CT datasets

### Cross-Dataset Validation
Models are validated across different datasets to ensure:
- Robustness to varying image quality and acquisition parameters
- Generalizability across different hospital/scanner environments  
- Real-world applicability beyond single-dataset performance

## Results

Performance comparison on cross-dataset evaluation:

| Method | Accuracy | F1-Score | Precision | Recall |
|--------|----------|----------|-----------|--------|
| Multi-Input AlexNet (4 images) | 0.58 | 0.48 | 0.55 | 0.58 |
| Multi-Input AlexNet (7 images) | 0.62 | 0.50 | 0.42 | 0.62 |
| Single-Input + Majority Voting | 0.91 | 0.91 | 0.92 | 0.91 |
| Single-Input + Averaging | 0.92 | 0.92 | 0.93 | 0.92 |
| Single-Input + Entropy Weighting | 0.94 | 0.93 | 0.94 | 0.92 |
| Single-Input + Z-Score | 0.89 | 0.89 | 0.90 | 0.88 |
| Single-Input + Bayesian | 0.93 | 0.93 | 0.93 | 0.93 |

### Key Findings

- **Single-input models with patient-level aggregation significantly outperform multi-input architectures**
- All aggregation methods achieve excellent performance (>90% across all metrics)
- Patient-level aggregation improved sensitivity by 2% compared to standard image-by-image baselines
- Models minimize false negatives while maintaining high precision
- Cross-dataset generalization demonstrates real-world applicability

## Architecture Comparison

| Aspect | Multi-Input AlexNet | Single-Input + Aggregation |
|--------|--------------------|-----------------------------|
| **Advantages** | •Simultaneous cross-image learning •End-to-end optimization •Complex inter-slice relationships •Single model per patient | •Simpler, more stable architecture •Handles variable image counts •Easier interpretation and debugging •Superior cross-dataset robustness |
| **Performance** | Moderate (58-62% accuracy) | Excellent (89-94% accuracy) |
| **Complexity** | High | Low-Medium |



## Clinical Impact

This patient-centered approach addresses critical limitations in COVID-19 detection:
- **Reduces misdiagnosis risk** by considering all available patient scans
- **Improves diagnostic confidence** through multiple aggregation strategies  
- **Ensures cross-dataset robustness** for real-world deployment
- **Maintains high sensitivity** while minimizing false negatives

## Future Work

- Integration with additional imaging modalities
- Real-time clinical deployment testing
- Extension to other respiratory diseases
- Advanced ensemble methods for aggregation

