<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=220&section=header&text=DeepScan&fontSize=80&fontColor=ffffff&fontAlignY=38&desc=A%20Synthetic%20Data-Augmented%20Deepfake%20Detection&descAlignY=58&descSize=20&animation=fadeIn" width="100%"/>

<br/>

[![Node](https://img.shields.io/badge/Node.js-18%2B-339933?style=flat-square&logo=node.js&logoColor=white)](.)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](.)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](.)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react&logoColor=black)](.)
[![License](https://img.shields.io/badge/license-academic-8B5CF6?style=flat-square)](.)

<br/>

> **DeepScan: A Synthetic Data-Augmented Deepfake Detection.**
> Upload an image or video and get REAL, AI-GENERATED, DEEPFAKE or UNCERTAIN, with the evidence behind the verdict.

</div>

---

## ✦ What DeepScan Does

| | |
|---|---|
|  **Synthetic data augmentation** | Five synthetic face manipulations (blend/warp, frequency perturbation, compression artifacts, colour perturbation, autoencoder swap); the face detector is trained with and without them and compared on unseen manipulations |
|  **Face-manipulation detector** | Effort CLIP ViT-L/14 backbone (FaceForensics++) with a classification head trained on FF++ and real portraits; dlib 5-point face alignment |
|  **AI-image detector** | Community Forensics Vision Transformer (ViT-S/16) for fully AI-generated images |
|  **Decision engine** | Thresholds measured on validation data; REAL only with positive evidence; UNCERTAIN when evidence is weak |
|  **Provenance evidence** | EXIF, XMP, IPTC and ICC metadata; C2PA manifest detection (not cryptographically verified); SynthID adapter (no verifier connected) |
|  **Images and videos** | JPG, PNG, WEBP, MP4, MOV, WEBM; videos are checked on 8 evenly sampled frames |
|  **History** | Results stored in MongoDB |

---

## ✦ Quick Start (macOS / Linux)

```bash
cd Ai_deepfake-main
bash scripts/start-all.sh
```

| Service | URL |
|:---|---:|
| Web app | `http://localhost:3000` |
| API | `http://localhost:5050` |
| ML inference | `http://127.0.0.1:7070/health` |

Full instructions, training and benchmarking: see [HOW_TO_RUN.md](HOW_TO_RUN.md).

---

## ✦ Architecture

```
┌─ Browser ──────────────────────────────┐
│  React 19 · React Router · Axios        │
└────────────────┬────────────────────────┘
                 │  POST /api/analyze · /api/analyze-video
                 ▼
┌─ Node.js (Express) ─────────────────────┐
│  Multer · Rate limiter · Helmet · CORS   │──► MongoDB (history)
└────────────────┬────────────────────────┘
                 │  multipart
                 ▼
┌─ FastAPI inference service ─────────────────────────────────────┐
│  Community Forensics ViT ─► p_ai                                  │
│  dlib face alignment ─► Effort ViT-L/14 + augmented head ─► p_df  │
│  EXIF / XMP / IPTC / C2PA / SynthID ─► provenance                 │
│                     ▼                                             │
│  Decision engine ─► REAL · AI-GENERATED · DEEPFAKE · UNCERTAIN    │
└──────────────────────────────────────────────────────────────────┘
```

---

## ✦ Synthetic Data Augmentation

`deepscan-backend/ml_server/synthetic/` contains the five techniques. Each takes an aligned
face crop and returns a manipulated copy, deterministically for a given seed.

| Technique | Imitates |
|---|---|
| `blend_warp` | Warping and blending at the face boundary |
| `freq_perturb` | Frequency-domain traces of GAN upsampling |
| `compression_artifact` | Re-encoding traces |
| `color_perturb` | Colour and lighting mismatch in the face region |
| `autoencoder_swap` | Reconstruction artifacts of a small autoencoder |

`train_face_detector.py` generates synthetic fakes only from real **training** faces
(ratio 0.3), trains a baseline head and an augmented head on identical data otherwise,
and compares them on FaceForensics++ test images, on FaceShifter (never used in training)
and in a leave-one-method-out setting. Results: `test_results/augmentation_study.json`.

---

## ✦ API

### `POST /api/analyze` (field `image`)  ·  `POST /api/analyze-video` (field `video`)

```json
{
  "result": "DEEPFAKE",
  "confidence": 0.91,
  "ai_probability": 0.012,
  "deepfake_probability": 0.91,
  "face_detected": true,
  "decision_reason": "Face detected and face-manipulation probability 0.910 >= calibrated threshold ...",
  "evidence_summary": { "ai_forensic_evidence": "LOW", "deepfake_evidence": "HIGH", "c2pa": "NOT_FOUND", "synthid": "UNAVAILABLE" },
  "model_scores": { "community_forensics_vit": 0.012, "deepscan_face_effort_augmented": 0.91 },
  "verdict": "SYNTHETIC",
  "final_score": 91.0
}
```

### `GET /api/results` · `GET /api/results/:id`

---

## ✦ Project Structure

```
deepscan-backend/
├── server.js                     # Express entry point
├── routes/analyze.js             # Image/video analysis and history endpoints
├── services/mlservice.js         # Client for the inference service
└── ml_server/
    ├── image_server.py           # FastAPI inference (images and videos)
    ├── decision_engine.py        # Verdict rules and calibration
    ├── effort_model.py           # Effort CLIP ViT-L/14 loader and face alignment
    ├── synthetic/                # Five synthetic manipulation techniques
    ├── train_face_detector.py    # Augmented training + baseline comparison
    ├── detector_benchmark.py     # Standalone benchmark (--real / --fake)
    ├── provenance.py             # EXIF/XMP/IPTC/ICC and C2PA detection
    ├── synthid.py                # SynthID adapter
    └── test_sets/acceptance/     # Held-out test images

deepscan-frontend/src/
├── components/                   # UploadZone, ResultCard, MetadataPanel, ...
├── pages/                        # Checker, History, Features, ...
└── services/api.js
```

---

 **Naman Singh** | Backend & Testing |

**Supervisor:** Mr. Abhishek Singh · **Submitted To:** Mr. Sanjay Madaan

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=120&section=footer" width="100%"/>

**DeepScan: A Synthetic Data-Augmented Deepfake Detection**

</div>
