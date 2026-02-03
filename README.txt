===========================================================
                  README — MENTAL HEALTH AI PROJECT
===========================================================

This project is an AI-driven Mental Health Detection System that uses
both TEXT (NLP) and AUDIO (speech-based features) to classify mental
health states and generate a final risk score.

===========================================================
1. PROJECT STRUCTURE
===========================================================

mental_health_mvp/
│
├── app.py
│     - Flask backend API
│     - Provides endpoints for text, audio, and combined prediction
│
├── utils.py
│     - Loads trained DistilBERT model (model.safetensors)
│     - Performs text prediction
│     - Combines text + audio to generate final risk score
│
├── feature_audio.py
│     - Extracts audio features such as:
│         * Pause Ratio
│         * Speech Energy (RMS)
│         * Onset Rate (speech rate)
│
├── frontend.html
│     - Frontend UI (HTML, CSS, JavaScript)
│     - Allows:
│         * Text input
│         * Voice recording
│         * Combined analysis
│
├── text_model/
│     - Folder containing trained NLP model files:
│         * model.safetensors       (model weights)
│         * config.json
│         * tokenizer.json
│         * vocab.txt
│         * tokenizer_config.json
│
├── text_model_label_mapping.json
│     - Mapping of mental health labels to numeric IDs
│
├── uploads/
│     - Stores user-uploaded audio files temporarily
│
└── README.txt
      - Project documentation (this file)

===========================================================
2. TECHNOLOGIES USED
===========================================================

1) Programming Language:
    - Python 3.10 (REQUIRED)
      (PyTorch does not support Python 3.12 or 3.13)

2) Backend Framework:
    - Flask (REST API)
    - Flask-CORS (for frontend-backend communication)

3) Machine Learning / NLP:
    - HuggingFace Transformers
    - DistilBERT model (fine-tuned)
    - safetensors format for storing model weights

4) Audio Processing:
    - Librosa
    - SoundFile
    - NumPy
    - Custom audio feature extraction (prosody-based)

5) Data Handling:
    - Pandas
    - NumPy
    - Datasets (HuggingFace)

6) Frontend:
    - HTML
    - CSS
    - JavaScript (Fetch API, MediaRecorder API)

===========================================================
3. MODEL INFORMATION
===========================================================

- NLP model: DistilBERT-base-uncased
- Trained on custom mental-health dataset (9,000+ entries)
- Output labels:
      * Anxiety
      * Bipolar
      * Depression
      * Normal
      * Personality Disorder
      * Stress
      * Suicidal

Model outputs probabilities; highest probability = prediction.

===========================================================
4. RISK SCORE CALCULATION
===========================================================

Text Risk Score:
    - Based on probabilities of Depression and Suicidal classes

Audio Risk Score:
    - Based on:
        * pause ratio
        * energy
        * onset rate

Final Score:
    final = (0.6 * text_score) + (0.4 * audio_score)

Risk Categories:
    0–39      : Low Risk
    40–59     : Moderate Risk
    60–79     : High Risk
    80–100    : Critical Risk

===========================================================
5. RUNNING THE PROJECT
===========================================================

1. Install Python 3.10
2. Create and activate virtual environment:
       python -m venv venv
       venv\Scripts\activate
3. Install dependencies:
       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
       pip install transformers safetensors pandas numpy librosa soundfile flask flask-cors
4. Run backend:
       python app.py
5. Open frontend (frontend.html) in a browser.

===========================================================
6. NOTES
===========================================================

- The model uses "model.safetensors"; the traditional
  "pytorch_model.bin" is not required and should not be created.
- The project works fully offline once the model folder is available.
- Python 3.10 is mandatory due to PyTorch compatibility.

===========================================================
END OF README
===========================================================
