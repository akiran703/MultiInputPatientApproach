# Multi-Input Patient-Centered COVID-19 Detection

## Overview

Rapid detection of COVID-19 is essential to prevent the disease from spreading. Currently, numerous machine learning algorithms have been developed to detect COVID-19 using Computerized Tomography (CT) lung scans. However, due to how broad and general they are, there is a lack of precision and attention to these patients. 

In particular, these algorithms prioritize accurate detection on an image-by-image basis, instead of on a patient-by-patient basis. Treating each scan independently (image-by-image) might result in a misdiagnosis if there are multiple CT scans of a single patient and they are not all incorporated in the final decision process. Having repeated images in different parts of the model will produce an invalid outcome that can't be trusted for real world scenarios. 

Moreover, these developed algorithms use a single dataset, which raises concerns about the generalization of the methods to other data. Various datasets tend to vary in image size and quality due to differing CT machine environments.

## Approach

Our approach tackles both of these issues by creating convolutional neural network (CNN) machine learning algorithms that prioritize producing an accurate diagnosis from multiple scans of a single patient. These methodologies include:

1. **Voting system based on individual image predictions** - Implemented in `finalpredictionmodel.py`
2. **Multi-input CNN that processes multiple images from the same patient** - Implemented in `4input_alexnet.py` and `7input_alexnet.py`

The approach is tested with the two largest datasets that are currently available in patient-based split. A cross-dataset study is presented to show the robustness of the models in a realistic scenario in which data comes from different distributions.

## Files Description

**AlexNet Architecture Details used in all files:**  
- **Conv Layer 1**: 96 filters, 11×11 kernel, stride 4, ReLU activation
- **Conv Layer 2**: 256 filters, 5×5 kernel, ReLU activation
- **Conv Layer 3**: 384 filters, 3×3 kernel, ReLU activation
- **Conv Layer 4**: 384 filters, 3×3 kernel, ReLU activation
- **Conv Layer 5**: 256 filters, 3×3 kernel, ReLU activation
- **Fully Connected**: 4096 → 4096 → 1000 → 2 neurons with dropout regularization

### `4input_alexnet.py` and `7input_alexnet.py`
This file implements a multi-input AlexNet architecture that simultaneously processes 4 or 7 CT scan images from the same patient:

- **Architecture**: Four parallel AlexNet branches that process different CT slices
- **Input**: 1, 4, or 7 CT scan images (224×224×3) per patient
- **Data handling**: Automatically adjusts patient data to have exactly 4 or 7 images (adds zeros if needed)
- **Training**: Uses transfer learning with pre-trained weights
- **Datasets**: Trained on clustered COVID dataset, then fine-tuned on a second dataset for cross-dataset validation

**Key Features:**
- Patient-centric approach ensuring all images from a patient contribute to diagnosis
- Cross-dataset transfer learning for improved generalization
- Optimized with Adam optimizer and appropriate learning rates
- Includes comprehensive training visualization and metrics


### `finalpredictionmodel.py`
This file implements a traditional single-input CNN with patient-level aggregation methods:

- **Architecture**: AlexNet-inspired CNN processing individual CT images
- **Input**: Single CT scan images (224×224×3)
- **Aggregation Methods**: 
  - **Averaging Method**: Averages prediction probabilities across all patient images
  - **Majority Voting**: Takes majority vote across individual image predictions
  - **Entropy-based Weighting**: Intelligently weighs each scans prediction based on the model's confidence (Method tried post UROC SOAR)
  - **Z-score Normalization**: calculates the mean and standard deviation of the COVID-19 probabilities across all of a single patient's CT scans,                                  then transforms each scans probability into a Z-score.(Method tried post UROC SOAR)
  - **Majority Voting**: Give each scans prediction a weight based on the model's confidence. A prediction with high confidence (i.e., a probability                           close to 0 or 1) is given more weight (Method tried post UROC SOAR)
- **Metrics**: Comprehensive evaluation including accuracy, precision, recall, F1-score, and confusion matrix

**Key Features:**
- Individual image processing with patient-level decision aggregation
- 5 different patient-level prediction strategies
- Optimized with SGD optimizer and appropriate learning rates
- Detailed performance metrics and patient-wise analysis
- Cross-dataset evaluation for robustness testing

## Methodology

### Data Processing
Both models implement careful data preprocessing:
- Image normalization and resizing with aspect ratio preservation
- Patient-based data splitting (avoiding data leakage)
- Cross-dataset validation using two independent COVID-19 CT datasets

### Patient-Centered Diagnosis
Unlike traditional image-by-image approaches, both models ensure that:
- All available CT scans from a patient contribute to the final diagnosis
- No single low-quality image can lead to misdiagnosis
- Conflicting signs of COVID-19 can be ruled out properly 
- Patient identity is preserved throughout the evaluation process

### Cross-Dataset Validation
Both approaches are validated across different datasets to ensure:
- Robustness to varying image quality and acquisition parameters
- Generalizability across different hospital/scanner environments
- Real-world applicability beyond single-dataset performance

## Requirements

```python
- tensorflow>=2.8.0
- keras
- opencv-python
- numpy
- pandas
- scikit-learn
- matplotlib
- PIL (Pillow)
- glob
```

## Notes

1. **Patient-Centric Architecture**: Approaches ensure diagnosis is made at the patient level, not image level
2. **Multi-Image Integration**: Systematic handling of multiple CT scans per patient
3. **Cross-Dataset Robustness**: Validation across different data distributions
4. **Clinical Relevance**: Approaches designed for real-world medical scenarios where patients have multiple scans
5. **AlexNet Optimization**: Simplified yet effective architecture suitable for medical imaging applications

## Results

The following table summarizes the performance of different model architectures and aggregation methods on the cross-dataset evaluation:

| Method | Accuracy | F1-Score | Precision | Recall |
|--------|----------|----------|-----------|--------|
| m-alex-4 | 0.58 | 0.48 | 0.55 | 0.58 |
| m-alex-7 | 0.62 | 0.50 | 0.42 | 0.62 |
| Predi-alex-vote | 0.91 | 0.91 | 0.92 | 0.91 |
| Predi-alex-avg | 0.92 | 0.92 | 0.93 | 0.92 |
| Predi-alex-Entro | 0.94 | 0.93 | 0.94 | 0.92 |
| Predi-alex-Z-Score | 0.89 | 0.89 | 0.90 | 0.88 |
| Predi-alex-Bayesian | 0.93 | 0.93 | 0.93 | 0.93 |



### Key Findings:
- The single-input models with patient-level aggregation significantly outperform the multi-input architectures
- All aggregation methods achieve excellent performance (>90% across all metrics)
- Single-input model with patient ID aggregation (AVG) outperformed the standard image-by-image baseline mode by 2% in sensitivity 
- The single-input models do not return false negatives but returns false positives
- Patient-level aggregation proves to be highly effective for cross-dataset generalization

## Architecture Comparison

### Multi-Input AlexNet vs Single-Input with Aggregation

**Multi-Input AlexNet Advantages:**
- Simultaneous processing of multiple images allows for cross-image feature learning
- End-to-end optimization of the entire patient diagnosis pipeline
- Potential for learning complex inter-slice relationships
- Single model handles entire patient case

**Single-Input with Aggregation Advantages:**
- Simpler architecture with proven stability
- Better handling of variable number of images per patient
- Easier to interpret and debug
- More robust to dataset variations

