# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import utils
import feature_audio
import os
import uuid

app = Flask(__name__, static_url_path="")
CORS(app)

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# =====================================================
# 1) INDEX (OPTIONAL FRONTEND)
# =====================================================

@app.route("/")
def index():
    return send_from_directory(".", "frontend.html")


# =====================================================
# 2) TEXT-ONLY PREDICTION
# =====================================================

@app.route("/predict_text", methods=["POST"])
def predict_text():
    data = request.json
    text = data.get("text", "")

    if text.strip() == "":
        return jsonify({"error": "Input text is empty"}), 400

    text_out = utils.predict_text_class(text)
    risk_score = utils.fuse_scores(text_out, audio_score=None)
    category = utils.risk_category(risk_score)

    return jsonify({
        "text_prediction": text_out["pred_label"],
        "text_probabilities": text_out["probs"],
        "label_probabilities": text_out["label_probabilities"],
        "final_risk_score": risk_score,
        "risk_category": category
    })


# =====================================================
# 3) AUDIO-ONLY PREDICTION
# =====================================================

@app.route("/predict_audio", methods=["POST"])
def predict_audio():
    if "file" not in request.files:
        return jsonify({"error": "No audio file found"}), 400

    audio_file = request.files["file"]

    fname = str(uuid.uuid4()) + ".wav"
    path = os.path.join(UPLOAD_DIR, fname)
    audio_file.save(path)

    # Extract audio features
    pros = feature_audio.extract_prosodic(path)
    audio_score = utils.audio_risk_from_prosody(pros)
    risk_score = audio_score * 100
    category = utils.risk_category(risk_score)

    return jsonify({
        "prosodic_features": pros,
        "audio_risk_score": risk_score,
        "risk_category": category
    })


# =====================================================
# 4) COMBINED (TEXT + AUDIO) FUSION PREDICTION
# =====================================================

@app.route("/predict_both", methods=["POST"])
def predict_both():
    text = request.form.get("text", "")

    if text.strip() == "":
        return jsonify({"error": "Text input is required"}), 400

    audio_file = request.files.get("file", None)

    # 1) TEXT PREDICTION
    text_out = utils.predict_text_class(text)

    audio_score = None
    pros = None

    # 2) AUDIO PREDICTION (IF AUDIO GIVEN)
    if audio_file:
        fname = str(uuid.uuid4()) + ".wav"
        path = os.path.join(UPLOAD_DIR, fname)
        audio_file.save(path)

        pros = feature_audio.extract_prosodic(path)
        audio_score = utils.audio_risk_from_prosody(pros)

    # 3) FUSION
    final_score = utils.fuse_scores(text_out, audio_score)
    category = utils.risk_category(final_score)

    return jsonify({
        "text_prediction": text_out["pred_label"],
        "text_probabilities": text_out["probs"],
        "label_probabilities": text_out.get("label_probabilities", {}),
        "audio_features": pros,
        "final_risk_score": final_score,
        "risk_category": category
    })


# =====================================================
# 5) SERVER START
# =====================================================

if __name__ == "__main__":
    print("🚀 Starting Mental Health Detection API...")
    print("➡ Visit: http://127.0.0.1:8000/")
    app.run(host="0.0.0.0", port=8000, debug=True)
