# utils.py
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import feature_audio

# =====================================================
# LOAD MODEL & LABEL MAP
# =====================================================

MODEL_PATH = "./text_model"
LABEL_MAP_PATH = "./text_model_label_mapping.json"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load model from safetensors format
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)
model.eval()

# Load label mapping
with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)

# Reverse mapping
index_to_label = {v: k for k, v in label_map.items()}


# =====================================================
# TEXT PREDICTION
# =====================================================

def predict_text_class(text):
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits.squeeze()
    probs = torch.softmax(logits, dim=0).numpy()

    pred_index = int(np.argmax(probs))
    pred_label = index_to_label[pred_index]

    label_probabilities = {
        index_to_label[i]: float(probs[i])
        for i in range(len(probs))
    }

    return {
        "probs": probs.tolist(),
        "label_probabilities": label_probabilities,
        "pred_index": pred_index,
        "pred_label": pred_label
    }



# =====================================================
# AUDIO RISK FROM PROSODY
# =====================================================

def audio_risk_from_prosody(feats):
    pause = feats.get("pause_ratio", 0)
    energy = feats.get("energy", 0)
    onset = feats.get("onset_rate", 0)

    pause_score = min(1.0, pause * 2.0)
    energy_score = 1.0 - min(1.0, energy * 50)
    onset_score = 1.0 - min(1.0, onset / 5.0)

    final = 0.45 * pause_score + 0.35 * energy_score + 0.20 * onset_score
    return float(np.clip(final, 0, 1))


# =====================================================
# FUSION SCORE
# =====================================================

def fuse_scores(text_out, audio_score=None):
    text_probs = text_out["probs"]

    severe_risk = text_probs[label_map["Depression"]]
    suicidal_risk = text_probs[label_map["Suicidal"]]

    text_risk = float(max(severe_risk, suicidal_risk))

    if audio_score is not None:
        score = 0.6 * text_risk + 0.4 * audio_score
    else:
        score = text_risk

    return float(score * 100)


def risk_category(score):
    if score >= 80:
        return "Critical Risk"
    if score >= 60:
        return "High Risk"
    if score >= 40:
        return "Moderate Risk"
    return "Low Risk"
