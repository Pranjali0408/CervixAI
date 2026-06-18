# CervixAI — Multimodal Cervical Cancer Screening Decision-Support System

CervixAI is a research/demo application for cervical cancer screening support. It combines a Pap smear microscopy image with selected clinical risk factors and returns a three-class screening output:

- **Normal**
- **LSIL** — Low-grade Squamous Intraepithelial Lesion
- **HSIL** — High-grade Squamous Intraepithelial Lesion

The current version includes:

- PyTorch image model inference using a fine-tuned CNN backbone.
- Optional clinical risk-factor encoder and attention-based fusion model.
- Out-of-distribution / invalid image checks before classification.
- Grad-CAM visual explanation for image regions.
- SHAP visual explanation for clinical risk-factor contribution.
- White professional medical UI.
- Login/dashboard/history/logout flow for demo use.
- SQLite-backed case history.
- Stored original image, Grad-CAM image, and SHAP image paths for history reuse.
- PDF report generation from the frontend.

> **Important clinical disclaimer:** This project is a research and decision-support prototype. It is not validated for clinical deployment and must not be used as a standalone diagnostic system.

---

## 1. Current project structure

The expected working structure is:

```text
upload/
├── backend/
│   ├── app.py                    # Current Flask API with SQLite history
│   ├── app2.py                   # Older/alternate backend file, not the main runtime
│   ├── build_ood_reference.py     # Builds calibrated OOD reference bank
│   ├── ood_detector.py            # OOD / invalid image detector
│   ├── test_ood_detector.py       # OOD detector tests / checks
│   ├── cervixai.db                # Auto-created SQLite DB after first run
│   └── storage/                   # Auto-created media storage
│       ├── uploads/               # Original uploaded Pap smear images
│       ├── gradcam/               # Grad-CAM output images
│       ├── shap/                  # SHAP output images
│       └── reports/               # Reserved for future backend-generated reports
│
├── frontend/
│   └── index.html                 # Current white UI dashboard
│
├── src/
│   ├── augment_dataset.py         # Offline image augmentation
│   ├── cnn_models.py              # CNN model helpers / variants
│   ├── config.py                  # Paths, constants, class mappings
│   ├── evaluate.py                # Metrics, Grad-CAM, SHAP utilities
│   ├── models.py                  # CNNClassifier, ClinicalMLP, AttentionFusion
│   ├── preprocess.py              # Image and clinical preprocessing
│   └── train.py                   # Full training pipeline
│
├── models/                        # Expected trained model artifacts
│   ├── best_arch.pkl
│   ├── scaler.pkl
│   ├── imputer.pkl
│   ├── feature_names.pkl
│   ├── resnet50_finetuned_best.pt or resnet50_finetuned_final.pt
│   ├── clinical_mlp_best.pt
│   ├── fusion_model_final.pt
│   ├── shap_background.npz
│   └── ood_reference.pkl
│
├── reports/                       # Training/evaluation reports and example plots
├── requirements.txt
└── README.md
```

`backend/app.py` and `frontend/index.html` are the two primary files changed in the current SQLite-enabled version.

---

## 2. What the system does end-to-end

At runtime, the user flow is:

1. User opens the web UI.
2. User logs in using demo login.
3. User uploads a Pap smear image.
4. User fills clinical risk factors.
5. Frontend sends image + clinical fields to `POST /predict`.
6. Backend validates the image.
7. Backend runs OOD checks.
8. Backend runs CNN image inference.
9. If available, backend also runs clinical MLP + attention fusion model.
10. Backend returns prediction result and saves a case record in SQLite.
11. User can generate Grad-CAM using `POST /explain`.
12. User can generate SHAP using `POST /shap`.
13. Grad-CAM / SHAP outputs are saved under backend `storage/` and linked to the SQLite case.
14. UI displays result, class likelihood bars, Grad-CAM, SHAP, and history.
15. User can download a PDF report from the frontend.

---

## 3. Dataset and class mapping

### 3.1 Image dataset

The image pipeline was designed around SIPaKMeD-style cropped cervical cell images.

Original SIPaKMeD classes are mapped into clinically simpler screening classes:

| SIPaKMeD folder/class | Mapped output class | Meaning |
|---|---|---|
| Superficial-Intermediate | Normal | Normal epithelial cell morphology |
| Parabasal | Normal | Normal/benign class grouping |
| Koilocytotic | LSIL | Low-grade lesion-associated cellular changes |
| Metaplastic | LSIL | Grouped under low-grade / abnormal but not high-grade |
| Dyskeratotic | HSIL | High-grade abnormality grouping |

This mapping reduces five source image categories into three clinically easier categories for the demo UI.

### 3.2 Clinical risk-factor dataset

The clinical branch expects features based on the UCI cervical cancer risk-factor dataset style. The frontend currently exposes a simplified subset:

- Age
- Number of sexual partners
- Number of pregnancies
- First sexual intercourse age
- Smokes
- HPV positive
- IUD
- STDs

The backend preprocessing uses the saved training artifacts:

- `feature_names.pkl`
- `imputer.pkl`
- `scaler.pkl`

These make runtime clinical inputs compatible with the clinical MLP trained earlier.

---

## 4. Model architecture

### 4.1 CNN image branch

The image branch uses `CNNClassifier` from `src/models.py`.

The training pipeline originally compared multiple CNN backbones such as:

- VGG16
- ResNet50
- EfficientNet
- Xception / timm-based variants depending on installed support

The current deployed backend loads the selected best architecture from:

```text
models/best_arch.pkl
```

If no value is available, the backend defaults to:

```text
resnet50
```

The backend then tries to load one of these files:

```text
models/{best_arch}_finetuned_best.pt
models/{best_arch}_finetuned_final.pt
models/{best_arch}_phase1_best.pt
models/{best_arch}_phase1.pt
```

The CNN produces:

- image feature embedding
- class logits
- softmax probabilities for Normal / LSIL / HSIL

### 4.2 Clinical MLP branch

The clinical branch uses `ClinicalMLP`.

It receives normalized clinical inputs and produces a compact clinical feature embedding.

This branch is optional at runtime. If the MLP artifact is unavailable, the backend can still run image-only inference.

Expected artifact:

```text
models/clinical_mlp_best.pt
```

### 4.3 Attention fusion model

The attention fusion model uses:

```text
AttentionFusion(IMG_FEATURE_DIM, CLINICAL_FEAT_DIM, number_of_classes)
```

It combines:

- CNN image features
- MLP clinical features

The intention is to allow the model to consider both visual morphology and patient-level risk-factor context.

Expected artifact:

```text
models/fusion_model_final.pt
```

If both `fusion_model_final.pt` and `clinical_mlp_best.pt` are present, `/predict` uses multimodal fusion. Otherwise, it falls back to CNN-only prediction.

---

## 5. Image validation and OOD protection

The backend now has two levels of protection before producing a prediction.

### 5.1 Basic image-quality validation

Implemented inside `validate_image()` in `backend/app.py`.

It rejects:

- corrupted/unreadable files
- nearly black images
- nearly white/blank images
- extremely unusual aspect ratios
- near-grayscale / very low-color images

Why this exists:

The model was trained on stained microscopy cell images. If a user uploads a random photo, icon, black image, document scan, or non-cell image, the neural network may still produce a softmax class. That would make the UI look confident even when the input is invalid. The validation layer reduces that risk.

### 5.2 OOD detector

The OOD detector is implemented in:

```text
backend/ood_detector.py
```

The current backend attempts to load:

```text
models/ood_reference.pkl
```

When available, it uses a calibrated reference bank of SIPaKMeD-like embeddings. The detector combines signals such as:

1. kNN/cosine similarity to known cervical-cell embeddings.
2. Free-energy score from CNN logits.
3. Softmax confidence and entropy gates.

If `ood_reference.pkl` is missing, the backend still creates an `OODDetector()` instance but has weaker fallback protection.

Recommended command to build the OOD reference bank:

```bash
cd upload/backend
python build_ood_reference.py
```

---

## 6. Confidence handling and UI wording

Earlier UI versions showed results as “accuracy,” sometimes with unrealistic `100%` display. This was changed because a single prediction confidence is not the same as model accuracy.

The current UI avoids saying:

```text
100% accuracy
```

Instead, it uses safer wording:

```text
Decision-support confidence
Class likelihood
Confidence band
Review required
```

Backend fields added/used:

```json
{
  "confidence": 0.82,
  "confidence_band": "Moderate",
  "review_required": true
}
```

### Why this was done

- Softmax confidence can be overconfident.
- A predicted class probability is not the same as validated clinical accuracy.
- During evaluation, repeatedly showing `100% accuracy` can raise concerns.
- “Decision-support confidence” is more appropriate for a prototype.

---

## 7. Risk-level logic

The base class-to-risk mapping is:

| Stage | Base risk |
|---|---|
| Normal | Low |
| LSIL | Medium |
| HSIL | High |

The backend also computes a simple clinical risk score using selected frontend fields:

```python
if age > 40: add 0.15
if sexual_partners > 3: add 0.15
if hpv_positive == 1: add 0.30
if smokes == 1: add 0.15
if stds == 1: add 0.15
```

Then `_elevate_risk()` may increase the displayed risk if the clinical risk score is high.

This is an intentionally simple rules-based risk adjustment, not a validated medical scoring system.

---

## 8. Grad-CAM implementation

Grad-CAM is triggered through:

```text
POST /explain
```

It can work in two modes:

1. With an uploaded image file.
2. With an existing `case_id`, where the backend loads the stored original image from SQLite/media storage.

The backend:

1. Validates the image.
2. Runs OOD check.
3. Runs `gradcam_single()` from `src/evaluate.py`.
4. Converts the Grad-CAM map to a colored heatmap.
5. Blends it with the original image.
6. Returns base64 PNG to the frontend.
7. Saves it to:

```text
backend/storage/gradcam/{case_id}_gradcam.png
```

The frontend supports:

- static Grad-CAM display
- popup/full-screen preview
- hover overlay on the original uploaded image

The hover overlay uses an HTML canvas positioned over the original image.

---

## 9. SHAP implementation

SHAP is triggered through:

```text
POST /shap
```

There are two SHAP modes in the project:

### 9.1 Training-time/global SHAP

Generated during training/evaluation and saved in `reports/`, for example:

```text
reports/shap_summary_bar.png
reports/shap_beeswarm.png
reports/shap_waterfall_0.png
reports/shap_top10_features.json
```

Purpose:

- explain which clinical features matter globally across the dataset
- produce academic/reporting artifacts

### 9.2 Runtime/per-patient SHAP

Generated when the user clicks “Explain Risk Factors.”

Requires:

```text
models/shap_background.npz
models/clinical_mlp_best.pt
models/feature_names.pkl
```

The backend calls:

```python
shap_single_prediction(...)
```

It returns a base64 waterfall PNG and saves it to:

```text
backend/storage/shap/{case_id}_shap.png
```

Purpose:

- explain this specific patient’s clinical risk-factor contribution
- include the output in the UI and PDF report

---

## 10. SQLite history implementation

The current version replaces browser-only localStorage history with SQLite.

Why SQLite was added:

- Browser localStorage has a small limit, usually around a few MB.
- Base64 images quickly filled it and caused quota errors.
- SQLite allows stable case history with image paths.
- It looks more professional for evaluation.
- It allows old cases to be re-opened with stored Grad-CAM/SHAP media.

SQLite file:

```text
backend/cervixai.db
```

Auto-created at backend startup.

Main table:

```sql
CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL,
    age INTEGER,
    sexual_partners INTEGER,
    pregnancies INTEGER,
    first_intercourse INTEGER,
    smokes INTEGER,
    hpv_positive INTEGER,
    iud INTEGER,
    stds INTEGER,
    stage TEXT,
    risk_level TEXT,
    confidence REAL,
    confidence_band TEXT,
    probabilities_json TEXT,
    stage_classes_json TEXT,
    model TEXT,
    fusion_used INTEGER,
    review_required INTEGER,
    clinical_risk_score REAL,
    original_image_path TEXT,
    gradcam_path TEXT,
    shap_path TEXT,
    report_path TEXT
);
```

Media files are not stored as blobs in SQLite. SQLite stores only file paths.

This is better because:

- database stays small
- image files remain easy to inspect/debug
- browser quota issue disappears
- reports/images can be served through `/media/...`

---

## 11. Backend endpoints

### `GET /` and `GET /app`

Serves the frontend HTML.

### `GET /health`

Basic liveness check.

Example response:

```json
{
  "status": "ok",
  "models_loaded": 4
}
```

### `GET /status`

Returns model and OOD loading status.

Useful to verify whether CNN, MLP, fusion, SHAP background, and OOD reference are available.

### `POST /predict`

Input:

- image file field: `image`
- clinical form fields:
  - `age`
  - `num_sexual_partners`
  - `num_pregnancies`
  - `first_sexual_intercourse`
  - `smokes`
  - `stds_hpv`
  - `iud`
  - `stds`

Output:

```json
{
  "case_id": "CXAI-20260428-ABC12345",
  "stage": "LSIL",
  "risk_level": "Medium",
  "confidence": 0.76,
  "confidence_band": "Moderate",
  "review_required": true,
  "probabilities": [0.14, 0.76, 0.10],
  "stage_classes": ["Normal", "LSIL", "HSIL"],
  "fusion_used": true,
  "model": "resnet50",
  "clinical_risk_score": 0.3,
  "original_image_url": "/media/uploads/CXAI-..._original.png"
}
```

Also saves the case to SQLite.

### `POST /explain`

Generates Grad-CAM.

Input:

- `case_id`, recommended
- optionally `image`

Output:

```json
{
  "case_id": "CXAI-...",
  "gradcam_png": "base64...",
  "gradcam_url": "/media/gradcam/CXAI-..._gradcam.png",
  "predicted_stage": "LSIL"
}
```

### `POST /shap`

Generates per-patient SHAP explanation.

Input:

- `case_id`
- same clinical form fields used in `/predict`

Output:

```json
{
  "case_id": "CXAI-...",
  "shap_png": "base64...",
  "shap_url": "/media/shap/CXAI-..._shap.png"
}
```

### `GET /history?limit=15`

Returns latest case history entries from SQLite.

### `GET /history/<case_id>`

Returns one full case record.

### `DELETE /history/<case_id>`

Deletes one case and associated media files.

### `DELETE /history`

Clears all history and associated media files.

### `GET /media/<folder>/<filename>`

Serves stored media from:

- uploads
- gradcam
- shap
- reports

---

## 12. Frontend UI features

The current `frontend/index.html` contains a single-file white medical dashboard UI.

Major UI features:

- Demo login screen
- Dashboard layout
- New analysis screen
- Pap smear image upload preview
- Clinical risk-factor form
- Result card
- Class likelihood bars
- Risk-level display
- Confidence band / review flag
- Grad-CAM generation
- Grad-CAM hover overlay on original image
- Grad-CAM popup preview
- SHAP generation
- SHAP popup preview
- PDF report download
- SQLite-backed history list
- Case restore/view from history
- Logout flow

### Why single-file HTML was used

The current app is a lightweight Flask demo. A single HTML file makes it easy to deploy quickly with Flask static serving and ngrok without adding a React/Vite build step.

For production, this can later be moved to React or another structured frontend framework.

---

## 13. PDF report generation

The report is currently generated in the browser using:

```html
jspdf 2.5.1
```

The frontend builds the PDF using current in-memory data:

- prediction result
- class likelihoods
- clinical fields
- uploaded image
- Grad-CAM image if generated
- SHAP image if generated
- disclaimer

Why frontend PDF was kept:

- faster to implement for demo
- avoids backend PDF dependency issues
- works immediately from the visible UI state

Limitation:

For old history cases, if the frontend does not reload image media into memory, PDF image embedding may require either regenerating visuals or fetching the stored media. Backend-side PDF generation would be more robust for a production version.

---

## 14. Installation

### 14.1 Create environment

```bash
cd upload
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate      # Windows
```

### 14.2 Install dependencies

CPU example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

GPU example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

If using Xception/timm backbone:

```bash
pip install timm
```

---

## 15. Training pipeline

Run training from:

```bash
cd upload/src
python train.py
```

The training pipeline performs:

1. CNN comparison.
2. Fine-tuning best CNN.
3. Clinical MLP training.
4. Feature extraction.
5. Attention fusion training.
6. Evaluation artifact generation.

Useful commands:

```bash
python train.py --status
python train.py --from-step 3
python train.py --only-step 5
python train.py --force
```

Before training, if needed, run augmentation:

```bash
cd upload/src
python augment_dataset.py
```

---

## 16. Running the current app

From project root:

```bash
cd upload/backend
python app.py
```

Then open:

```text
http://127.0.0.1:5000/app
```

For ngrok, expose the Flask port and update the frontend `API` constant if your file is opened separately.

If Flask serves the frontend directly from `/app`, use a relative API or the same backend origin.

---

## 17. SQLite history workflow

After first run, backend automatically creates:

```text
backend/cervixai.db
backend/storage/uploads/
backend/storage/gradcam/
backend/storage/shap/
backend/storage/reports/
```

A typical case lifecycle:

1. `/predict` creates case row and saves original image.
2. `/explain` updates same row with Grad-CAM path.
3. `/shap` updates same row with SHAP path.
4. `/history` lists recent cases.
5. `/history/<case_id>` loads full details.
6. `/media/...` serves stored images.

---

## 18. What changed from the previous version

The previous README primarily described the training/model pipeline. The current project now includes a more complete application layer.

Major updates:

| Area | Previous version | Current version |
|---|---|---|
| UI | Basic inference page | White professional dashboard |
| History | Browser localStorage | SQLite backend history |
| Image storage | Base64 in browser memory/history | Files saved in backend storage folders |
| Grad-CAM | Static heatmap | Static + hover overlay + popup |
| SHAP | Training-time plots mainly | Runtime per-patient SHAP endpoint + popup |
| Report | Basic PDF behavior | UI PDF with current image/Grad-CAM/SHAP |
| Confidence wording | Could look like “accuracy” | Decision-support confidence / class likelihood |
| Invalid input | Basic validation | Basic validation + OOD detector |
| Case tracking | Not persistent | `case_id` based case lifecycle |

---

## 19. Assumptions, approximations, and workarounds

This section is important for explaining the system honestly.

### 19.1 Three-class mapping approximation

The original SIPaKMeD dataset has more granular cell categories. These were mapped to three UI classes: Normal, LSIL, HSIL.

This is useful for a simplified screening demo, but it is still an approximation and should be explained as a research grouping.

### 19.2 Clinical risk score is heuristic

The `_clinical_risk_score()` function is rules-based. It is not a clinically validated scoring system.

It was added to make the UI risk level more context-aware and to avoid relying only on image class.

### 19.3 Softmax probability is not true clinical certainty

The displayed confidence is based on model probability. Neural networks can be overconfident.

That is why the UI uses “decision-support confidence” and “class likelihood,” not “accuracy.”

### 19.4 Probability redistribution / UI smoothing

During UI development, the goal was to avoid presenting every prediction as `100% accuracy`. If any frontend probability smoothing is enabled, it should be treated as a presentation safeguard, not a scientific recalibration.

The recommended final approach is to preserve backend probabilities but label them carefully. If smoothing remains in the UI, mention it as a visualization workaround only.

### 19.5 OOD fallback is weaker without reference bank

If `models/ood_reference.pkl` is missing, the detector falls back to softmax/entropy checks. This is weaker than calibrated kNN/energy-based detection.

For serious demos, build the OOD reference bank.

### 19.6 SHAP depends on saved background data

Runtime SHAP requires `shap_background.npz`. If the file is missing, `/shap` returns an error.

This is expected because SHAP needs representative background samples.

### 19.7 SQLite is not a hospital-grade data layer

SQLite is appropriate for local demos, evaluation, and small deployments.

For production multi-user use, replace it with PostgreSQL/MySQL and proper authentication.

### 19.8 Login is demo-level

The frontend login is a lightweight demo mechanism. It should not be considered secure authentication.

Production should use backend sessions, JWT, OAuth, or hospital SSO.

### 19.9 Frontend PDF is convenient but not ideal for production

Browser PDF generation works well for demos. A production system should generate reports on the backend so reports can be reproduced exactly from stored case data and media.

### 19.10 Clinical validation is not complete

The model may show strong technical metrics, but clinical deployment requires:

- external validation
- prospective testing
- pathologist review
- bias analysis
- scanner/stain/site variation testing
- regulatory review where applicable

---

## 20. Recommended demo/testing checklist

Before showing the application:

1. Start Flask backend.
2. Open `/status` and confirm models loaded.
3. Upload one valid Pap smear image.
4. Confirm `/predict` returns `case_id`.
5. Generate Grad-CAM.
6. Confirm hover overlay works on original image.
7. Open Grad-CAM popup.
8. Generate SHAP.
9. Open SHAP popup.
10. Download PDF and confirm images are included.
11. Go to History and confirm the case is listed.
12. Re-open the case and confirm stored media paths load.
13. Test an invalid/random image and confirm it is rejected.

---

## 21. Common issues and fixes

### 21.1 `Model not loaded. Run train.py first.`

Required `.pt` files are missing from `models/`.

Check:

```text
models/best_arch.pkl
models/resnet50_finetuned_best.pt or resnet50_finetuned_final.pt
```

### 21.2 SHAP error: background missing

Run training step that generates:

```text
models/shap_background.npz
```

### 21.3 Browser localStorage quota error

Old UI versions saved base64 images in browser localStorage. Current SQLite version should not do this.

Clear old browser storage once:

```js
localStorage.removeItem("cervixai_history_v3_white");
location.reload();
```

### 21.4 Grad-CAM not visible on uploaded image

Make sure:

1. An image is uploaded.
2. Analysis is run.
3. Grad-CAM is generated.
4. The original image preview is visible.
5. Hover over the original preview image.

### 21.5 CORS / ngrok issue

If frontend is opened from a different origin than the Flask API, update the `API` constant in `frontend/index.html` to match the ngrok backend URL.

---

## 22. Security and production notes

For production, add:

- real backend authentication
- role-based access
- HTTPS only
- audit logs
- PHI-safe storage
- encryption at rest
- signed media URLs
- database migrations
- external database instead of SQLite
- model versioning
- proper clinical disclaimers
- report versioning
- user/session tracking

---

## 23. Files changed in the current SQLite update

```text
backend/app.py
frontend/index.html
README.md
```

`backend/app.py` now contains:

- SQLite initialization
- case insertion/update helpers
- media storage helpers
- history APIs
- media serving API
- updated `/predict`, `/explain`, `/shap` case handling

`frontend/index.html` now contains:

- SQLite-backed history calls
- no base64 image storage in browser history
- Grad-CAM popup
- SHAP popup
- hover Grad-CAM overlay
- white medical dashboard UI

---

## 24. High-level explanation for evaluation defense

CervixAI is best described as:

> A multimodal cervical screening decision-support prototype that combines cytology image morphology with structured clinical risk factors. The system uses a fine-tuned CNN for Pap smear image classification, an MLP encoder for clinical features, and an attention-based fusion module when both modalities are available. It includes explainability through Grad-CAM for image regions and SHAP for patient-specific clinical risk factors. The application layer includes SQLite-backed case history, stored visual outputs, and PDF report generation for demo/evaluation workflows.

Key defense points:

- The system is not claiming standalone diagnosis.
- It is a decision-support prototype.
- Confidence is presented as class likelihood, not clinical certainty.
- Invalid image/OOD checks reduce misuse.
- Explainability helps users understand model behavior.
- SQLite was selected for reliable local case history without browser quota issues.
- Current limitations and assumptions are explicitly documented.

---

## 25. Quick start summary

```bash
cd upload
pip install -r requirements.txt
cd backend
python app.py
```

Open:

```text
http://127.0.0.1:5000/app
```

Check backend status:

```text
http://127.0.0.1:5000/status
```

