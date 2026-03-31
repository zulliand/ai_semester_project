"""Integrated apartment price prediction app with image-derived features."""

from typing import Dict, List, Tuple

import gradio as gr
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from transformers import pipeline


MODEL_PATH = "best_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "features.pkl"


LOCATION_PROFILES = {
    "Zurich Center": {"pop_dens": 4700.0, "frg_pct": 34.0, "tax_income": 98000.0},
    "Zurich Suburb": {"pop_dens": 2200.0, "frg_pct": 26.0, "tax_income": 84000.0},
    "Winterthur": {"pop_dens": 1600.0, "frg_pct": 29.0, "tax_income": 78000.0},
    "Uster": {"pop_dens": 1200.0, "frg_pct": 24.0, "tax_income": 76000.0},
    "Rural Zurich": {"pop_dens": 550.0, "frg_pct": 18.0, "tax_income": 72000.0},
}


def _load_regression_assets() -> Tuple[object, object, List[str]]:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, scaler, features


REG_MODEL, REG_SCALER, REG_FEATURES = _load_regression_assets()

try:
    # CLIP is optional for style scoring; app still runs with heuristic fallback.
    CLIP_STYLE = pipeline(
        task="zero-shot-image-classification",
        model="openai/clip-vit-base-patch32",
    )
    CLIP_AVAILABLE = True
except Exception:
    CLIP_STYLE = None
    CLIP_AVAILABLE = False


def _clip_style_score(image: Image.Image) -> float:
    if not CLIP_AVAILABLE:
        return 0.5

    labels = ["modern apartment interior", "old apartment interior"]
    results = CLIP_STYLE(image, candidate_labels=labels)
    modern_score = 0.5

    for item in results:
        label = item.get("label", "").lower()
        score = float(item.get("score", 0.0))
        if "modern" in label:
            modern_score = score
            break

    return float(np.clip(modern_score, 0.0, 1.0))


def extract_image_features(image: Image.Image) -> Dict[str, float]:
    """Convert apartment image into numeric features for the regression model."""
    if image is None:
        raise ValueError("Please upload an apartment image.")

    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32)

    brightness_score = float(arr.mean() / 255.0)
    contrast_score = float(arr.std() / 255.0)

    modern_score = _clip_style_score(rgb)
    if not CLIP_AVAILABLE:
        modern_score = float(np.clip((0.65 * brightness_score) + (0.35 * contrast_score), 0.0, 1.0))

    old_style_score = float(1.0 - modern_score)
    style_label = "modern" if modern_score >= 0.5 else "old"

    return {
        "brightness_score": round(brightness_score, 4),
        "contrast_score": round(contrast_score, 4),
        "modern_score": round(modern_score, 4),
        "old_style_score": round(old_style_score, 4),
        "style_label": style_label,
        "clip_used": float(CLIP_AVAILABLE),
    }


def _build_base_features(size_m2: float, rooms: float, location: str) -> Dict[str, float]:
    profile = LOCATION_PROFILES[location]
    pop_dens = float(profile["pop_dens"])
    frg_pct = float(profile["frg_pct"])
    tax_income = float(profile["tax_income"])

    area_per_room = size_m2 / rooms if rooms > 0 else 0.0
    pop = pop_dens * 10.0
    emp = pop * 0.3
    pop_to_emp_ratio = pop / (emp + 1.0)
    wealth_indicator = (tax_income / 100000.0) * (emp / 100000.0)

    density_category = int(min(pop_dens // 1000, 4))
    if rooms < 2:
        room_category = 0
    elif rooms < 3.5:
        room_category = 1
    elif rooms < 5:
        room_category = 2
    else:
        room_category = 3

    return {
        "rooms": float(rooms),
        "area": float(size_m2),
        "pop_dens": pop_dens,
        "frg_pct": frg_pct,
        "tax_income": tax_income,
        "area_per_room": area_per_room,
        "pop_to_emp_ratio": pop_to_emp_ratio,
        "wealth_indicator": wealth_indicator,
        "density_category": float(density_category),
        "room_category": float(room_category),
    }


def _predict_from_features(feature_dict: Dict[str, float]) -> float:
    if isinstance(REG_FEATURES, list) and len(REG_FEATURES) > 0:
        ordered = {key: feature_dict[key] for key in REG_FEATURES if key in feature_dict}
        if len(ordered) != len(REG_FEATURES):
            missing = [name for name in REG_FEATURES if name not in ordered]
            raise ValueError(f"Missing model features: {missing}")
        model_input = pd.DataFrame([ordered])
    else:
        model_input = pd.DataFrame([feature_dict])

    scaled = REG_SCALER.transform(model_input)
    pred_log = float(REG_MODEL.predict(scaled)[0])
    return float(np.expm1(pred_log))


def predict_price(numeric_inputs: Dict[str, float], image_features: Dict[str, float]) -> Dict[str, float]:
    """Predict apartment price with numeric inputs and image-derived signals."""
    size_m2 = float(numeric_inputs["size_m2"])
    rooms = float(numeric_inputs["rooms"])
    location = str(numeric_inputs["location"])

    base_features = _build_base_features(size_m2=size_m2, rooms=rooms, location=location)
    baseline_price = _predict_from_features(base_features)

    brightness = float(image_features["brightness_score"])
    modern = float(image_features["modern_score"])

    image_quality_factor = 1.0 + (0.10 * (modern - 0.5)) + (0.06 * (brightness - 0.5))
    image_quality_factor = float(np.clip(image_quality_factor, 0.9, 1.1))

    combined_features = dict(base_features)
    combined_features["area_per_room"] = base_features["area_per_room"] * image_quality_factor
    combined_features["wealth_indicator"] = base_features["wealth_indicator"] * (1.0 + 0.12 * (modern - 0.5))

    final_price = _predict_from_features(combined_features)

    effect_chf = final_price - baseline_price
    effect_pct = (effect_chf / baseline_price) * 100.0 if baseline_price > 0 else 0.0

    return {
        "predicted_price_chf": final_price,
        "baseline_price_chf": baseline_price,
        "image_effect_chf": effect_chf,
        "image_effect_pct": effect_pct,
        "image_quality_factor": image_quality_factor,
    }


def _validate_inputs(image: Image.Image, size_m2: float, rooms: float, location: str) -> None:
    if image is None:
        raise ValueError("Image is required.")
    if size_m2 <= 10 or size_m2 > 400:
        raise ValueError("Size must be between 10 and 400 m2.")
    if rooms <= 0 or rooms > 12:
        raise ValueError("Rooms must be between 0.5 and 12.")
    if location not in LOCATION_PROFILES:
        raise ValueError("Please select a valid location.")


def run_prediction(image: Image.Image, size_m2: float, rooms: float, location: str):
    try:
        _validate_inputs(image, size_m2, rooms, location)
        image_features = extract_image_features(image)

        numeric_inputs = {
            "size_m2": float(size_m2),
            "rooms": float(rooms),
            "location": location,
        }
        prediction = predict_price(numeric_inputs=numeric_inputs, image_features=image_features)

        price = prediction["predicted_price_chf"]
        delta = prediction["image_effect_chf"]
        delta_sign = "+" if delta >= 0 else ""

        explanation = (
            f"### Predicted Monthly Rent: CHF {price:,.0f}\n\n"
            f"Image -> features -> prediction\n"
            f"- Brightness score: {image_features['brightness_score']:.3f}\n"
            f"- Modern score: {image_features['modern_score']:.3f} ({image_features['style_label']})\n"
            f"- Image quality factor applied inside regression features: {prediction['image_quality_factor']:.3f}\n"
            f"- Baseline (without image impact): CHF {prediction['baseline_price_chf']:,.0f}\n"
            f"- Image impact on final prediction: {delta_sign}CHF {delta:,.0f} ({prediction['image_effect_pct']:+.2f}%)"
        )

        feature_output = {
            "numeric_inputs": numeric_inputs,
            "location_profile": LOCATION_PROFILES[location],
            "image_features": image_features,
            "prediction_details": {
                "baseline_price_chf": round(prediction["baseline_price_chf"], 2),
                "predicted_price_chf": round(prediction["predicted_price_chf"], 2),
                "image_effect_chf": round(prediction["image_effect_chf"], 2),
                "image_effect_pct": round(prediction["image_effect_pct"], 3),
            },
        }

        return explanation, feature_output
    except Exception as exc:
        return f"Error: {exc}", {"error": str(exc)}


with gr.Blocks(title="Apartment Price Predictor with Image Features") as demo:
    gr.Markdown(
        """
        # Apartment Price Prediction (Numeric + Vision)

        Upload an apartment image and provide structured apartment data.
        The app extracts image features and injects them into the regression pipeline.
        """
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Apartment Image")
            size_input = gr.Number(value=75, label="Size (m2)")
            rooms_input = gr.Number(value=3.0, label="Number of Rooms")
            location_input = gr.Dropdown(
                choices=list(LOCATION_PROFILES.keys()),
                value="Zurich Suburb",
                label="Location",
            )
            predict_button = gr.Button("Predict Price", variant="primary")

        with gr.Column():
            prediction_output = gr.Markdown(label="Prediction")
            features_output = gr.JSON(label="Extracted Image Features and Model Inputs")

    predict_button.click(
        fn=run_prediction,
        inputs=[image_input, size_input, rooms_input, location_input],
        outputs=[prediction_output, features_output],
    )


if __name__ == "__main__":
    demo.launch()