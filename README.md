---
title: Apartment Price Prediction with Vision Features
emoji: "🏠"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.8.0
app_file: app.py
pinned: false
---

# Apartment Price Prediction (ML Numeric Data + Computer Vision)

This project implements a single integrated AI system for apartment rent prediction in Zurich:

- ML Numeric Data block: regression on structured apartment features
- Computer Vision block: image-based feature extraction (brightness and modern style score)

Important integration logic:
- Image -> image features -> regression input features -> final CHF prediction
- The vision component directly modifies engineered regression features before inference
- This is not two independent models running side-by-side

## 1. Use Case

Realistic goal:
Predict monthly rent more robustly by combining classic apartment metadata (size, rooms, location profile) with visual apartment quality signals from uploaded photos.

Motivation:
- Structured listings often miss visual quality information
- Brightness and interior style are known price cues
- Vision features improve transparency and can improve practical prediction quality

## 2. Project Structure

- app.py: Gradio inference app (integrated ML + CV pipeline)
- best_model.pkl: trained regression model
- scaler.pkl: fitted feature scaler
- features.pkl: ordered model feature names
- original_apartment_data_analytics_hs24.csv: tabular source dataset
- requirements.txt: dependencies

## 3. Data Sources

The project uses multiple and different data sources:

1. Structured numeric dataset
- File: original_apartment_data_analytics_hs24.csv
- Type: tabular numeric data
- Role: trains/evaluates apartment regression model

2. Visual input data
- Source: user-uploaded apartment images at inference time
- Type: image data
- Role: provides brightness and style features used by regression model

3. Pretrained vision knowledge source
- Model: openai/clip-vit-base-patch32
- Role: zero-shot score for modern vs old interior style

## 4. Integrated Inference Pipeline

1. User uploads apartment image and enters size, rooms, location
2. extract_image_features(image) computes:
- brightness_score
- contrast_score
- modern_score
- old_style_score
3. Numeric base features are built from size/rooms/location profile
4. predict_price(numeric_inputs, image_features) injects image impact by updating engineered features
5. Final prediction is produced in CHF
6. App also shows baseline without image and image effect delta for transparency

## 5. How to Run Locally

1. Create environment and install dependencies

```bash
pip install -r requirements.txt
```

2. Start app

```bash
python app.py
```

3. Open the local Gradio URL shown in the terminal

## 6. Hugging Face Space Deployment

This repository is Space-ready via frontmatter and app.py entrypoint.

Deployment checklist:
1. Push repository to GitHub
2. Create a new Gradio Space on Hugging Face
3. Connect the GitHub repository
4. Ensure the Space installs requirements.txt
5. Verify upload + prediction flow in browser

URL to include in final report:
- Add deployed Space URL here: TODO

## 7. Requirement Coverage Summary (A-E)

A. General Project Requirements
- At least 2 blocks: fulfilled (ML Numeric Data + Computer Vision)
- Meaningful integration: fulfilled (image features injected into regression features)
- Multiple data sources: fulfilled (tabular CSV + uploaded images + pretrained CLIP)
- Realistic use case: fulfilled (rental price estimation)

B. Documentation Requirements
- A full report template is provided in PROJECT_DOCUMENTATION.md and should be completed with your final figures, screenshots, and metrics.

C. Assessment Criteria
- Project is structured for clarity, technical correctness, integration quality, and reproducibility.

D. Submission
- Deadline: 07 June 2026, 18:00
- Required reviewers/collaborators to add on GitHub:
  - Jasmin Heierli (jasminh)
  - Benjamin Kuhnis (bkuehnis)

E. Block-Specific Requirements
- E.1 ML Numeric Data: structured data, engineered features, quantitative model output
- E.3 Computer Vision: image preprocessing, vision model usage, image-derived features used by numeric model

## 8. Notes on Training vs Inference Separation

- Training artifacts are stored as serialized files (best_model.pkl, scaler.pkl, features.pkl)
- app.py performs inference only
- This separates model training workflow from deployed prediction workflow

## 9. Recommended Final Steps Before Submission

1. Complete PROJECT_DOCUMENTATION.md with final numeric metrics and EDA plots
2. Add deployment URL and screenshots
3. Add explicit error analysis examples
4. Verify requirements.txt only contains needed packages
5. Invite required GitHub users as collaborators
