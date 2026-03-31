# Project Documentation Template (Completed Draft)

This document follows the mandatory structure from the project brief.
Complete all TODO fields with your final values, plots, and screenshots before submission.

## 1. Project Idea and Methodology

### 1.1 Problem Definition and Objectives

Problem:
- Predict apartment monthly rent (CHF) more realistically by combining structured apartment metadata with visual apartment cues from images.

Objectives:
- Build an integrated AI application with ML Numeric Data + Computer Vision.
- Use image-derived features (brightness, style score) as direct inputs to the numeric prediction process.
- Provide transparent output that explains image influence on the final prediction.

### 1.2 Use Case Motivation

- Real-estate listings often miss latent quality cues in tabular fields.
- Apartment photos can contain additional information (light, perceived modernity).
- Combining both modalities improves practical utility and explainability.

### 1.3 Block Combination and Integration

Selected blocks:
- ML Numeric Data
- Computer Vision

Technical integration:
- Uploaded image is converted to numeric image features.
- These features are injected into engineered numeric features.
- The regression model predicts final rent from the combined representation.

### 1.4 Scope and Assumptions

Scope:
- Inference-focused web app using pretrained/fitted artifacts.
- Location represented by predefined Zurich location profiles.

Assumptions:
- Brightness and style are useful rent proxies.
- CLIP style score (modern vs old) provides a meaningful signal.
- The fitted scaler/model expect feature format stored in features.pkl.

## 2. Data and Preprocessing

### 2.1 Data Sources

Source A (Numeric):
- Name: Zurich apartment analytics dataset
- File: original_apartment_data_analytics_hs24.csv
- Type: structured/tabular
- Size: TODO (rows, columns)

Source B (Visual):
- Name: user-uploaded apartment images (inference-time)
- Type: RGB images
- Size: dynamic (provided by user)

Source C (Pretrained model knowledge):
- Model: openai/clip-vit-base-patch32
- Usage: zero-shot style scoring

### 2.2 Data Cleaning and Preparation

Numeric preprocessing (training phase summary):
- Handle missing values: TODO
- Outlier treatment: TODO
- Transformations: log target transform via expm1 inverse at inference
- Scaling: scaler.pkl applied before regression prediction

Image preprocessing:
- Convert to RGB
- Compute normalized brightness and contrast from pixel statistics
- Compute modern_score via CLIP (or heuristic fallback)

### 2.3 Feature Engineering / Augmentation

Numeric engineered features:
- area_per_room
- pop_to_emp_ratio
- wealth_indicator
- density_category
- room_category

Image-derived engineered features:
- brightness_score
- contrast_score
- modern_score
- old_style_score

Integration engineering:
- image_quality_factor derived from modern_score and brightness_score
- Injected into regression input by adjusting area_per_room and wealth_indicator

### 2.4 EDA

Provide EDA plots and findings:
- Distribution of target rent
- Correlation heatmap of numeric predictors
- Rooms vs area vs rent trend
- Location profile comparisons
- Any anomalies and handling

TODO: Insert figures and interpretation.

## 3. Modeling and Implementation

### 3.1 Model/Prompt Selection Justification

Numeric model:
- Loaded from best_model.pkl (regression artifact from prior training)
- Chosen due to prior evaluation performance on rental data

Vision model:
- CLIP zero-shot classifier (modern vs old interior labels)
- Chosen for lightweight integration without additional CV fine-tuning

### 3.2 Training / Development Strategy

- Training performed offline for numeric model (stored artifacts)
- Inference app integrates CV features at runtime
- Fallback heuristic keeps app robust if CLIP cannot load

### 3.3 Comparison of Models / Approaches

Minimum required comparison (to include in final report):
- Numeric-only baseline vs numeric+image integrated approach
- Optional: compare CLIP-based style score vs heuristic style score

TODO: Add metrics table and interpretation.

### 3.4 Iterations and Improvements

Iteration examples:
1. Numeric-only prediction
2. Add brightness/contrast features
3. Add CLIP style score and transparency output
4. Add stronger error handling and input validation

### 3.5 Technical Implementation Details

Core files:
- app.py: integrated Gradio app
- best_model.pkl, scaler.pkl, features.pkl: inference artifacts

Main functions:
- extract_image_features(image)
- predict_price(numeric_inputs, image_features)
- run_prediction(...) wrapper for UI and error handling

Libraries:
- gradio, numpy, pandas, joblib, pillow, transformers, torch

## 4. Evaluation and Analysis

### 4.1 Evaluation Strategy

Quantitative:
- Compare baseline prediction pipeline and integrated pipeline
- Metrics: MAE, RMSE, R2 (if ground-truth test labels available)

Qualitative:
- Visual inspection with multiple apartment photos
- Check whether style/brightness adjustments are plausible

Data splits:
- TODO: Document train/validation/test split used during numeric model training

### 4.2 Model Performance Analysis

TODO:
- Insert final baseline metrics
- Insert integrated metrics
- Discuss whether integration improved performance and where

### 4.3 Error Analysis

Required examples to include:
- Cases where bright but old apartment is overestimated
- Cases where dark modern apartment is underestimated
- Location profile mismatch errors

TODO: Add at least 3 concrete failure cases with screenshots/inputs.

### 4.4 Interpretation of Results

- Explain sensitivity to image quality factor
- Explain effect size distribution (CHF delta) across test inputs
- Clarify limitations and assumptions

### 4.5 Block-Specific Evaluation

ML Numeric Data:
- Quantitative metrics on holdout data
- Residual analysis

Computer Vision:
- Style score sanity checks on selected images
- Visual examples where score aligns/misaligns with human judgment

## 5. Deployment

### 5.1 Working Deployment

- Platform: Hugging Face Spaces (Gradio)
- URL: TODO

### 5.2 Separation of Training and Inference

- Training artifacts are precomputed and loaded from files
- Deployed app performs inference only

### 5.3 Screenshots

Include:
1. Main UI with inputs
2. Prediction output with explanation
3. JSON transparency output (image features)
4. One error-handling example

TODO: Insert screenshot references.

## 6. Execution Instructions

### 6.1 Local Run

```bash
pip install -r requirements.txt
python app.py
```

### 6.2 Reproducibility Notes

- Keep artifact files in project root:
  - best_model.pkl
  - scaler.pkl
  - features.pkl
- Keep dataset file in project root:
  - original_apartment_data_analytics_hs24.csv

## 7. Requirement Mapping Checklist

A. General Requirements
- [x] At least two blocks combined
- [x] Meaningful technical integration
- [x] Multiple data sources
- [x] Realistic use case
- [x] Independent implementation and documentation

B. Documentation (this file)
- [x] Sections 1-6 covered
- [ ] Final TODO values completed

E.1 ML Numeric Data
- [x] Structured dataset used
- [ ] EDA figures inserted
- [x] Feature engineering included
- [ ] Comparison of at least two numeric models documented with metrics
- [ ] Quantitative evaluation + error analysis completed
- [x] Integration with CV explained

E.3 Computer Vision
- [x] Image data used
- [x] Image preprocessing implemented
- [x] Vision model applied (CLIP)
- [ ] CV evaluation examples documented
- [x] Visual features integrated into numeric model

## 8. Submission Notes

Deadline:
- 07 June 2026, 18:00

GitHub link submission:
- TODO: Insert repository URL

Required GitHub users to add:
- Jasmin Heierli (jasminh)
- Benjamin Kuhnis (bkuehnis)

Final pre-submission check:
- [ ] Deployment URL added
- [ ] Screenshots added
- [ ] Metrics tables finalized
- [ ] Error analysis completed
- [ ] Collaborators added on GitHub
