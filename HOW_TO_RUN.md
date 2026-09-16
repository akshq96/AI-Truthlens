# How to Run DeepScan

**DeepScan: A Synthetic Data-Augmented Deepfake Detection**

## Requirements

- macOS or Linux, 8 GB RAM minimum (16 GB recommended)
- Python 3.9 virtual environment at `deepscan-backend/ml_server/.venv` (already set up on this machine)
- Node.js 18 or later
- MongoDB running on `localhost:27017` (needed only for saving analysis history)
- Model files in `deepscan-backend/ml_server/trained_models/`:
  - `effort/effort_clip_L14_trainOn_FaceForensic.pth` and `effort/shape_predictor_81_face_landmarks.dat`
  - `face_head/face_head.pt` (created by `train_face_detector.py`)
  - The Community Forensics ViT is loaded from the local Hugging Face cache

## Start everything

```bash
cd ~/Downloads/Ai_deepfake-main
bash scripts/start-all.sh
```

Then open **http://localhost:3000**, sign in, and open the checker page.

| Service | Address | Log |
|---|---|---|
| Web app (React) | http://localhost:3000 | `logs/frontend.log` |
| API (Node/Express) | http://localhost:5050 | `logs/api.log` |
| ML inference (FastAPI) | http://127.0.0.1:7070/health | `logs/ml_server.log` |

The ML server needs about a minute to load its models. `/health` lists which detectors are loaded.

## Stop everything

```bash
bash scripts/stop-all.sh
```

## Start the services by hand

```bash
cd ~/Downloads/Ai_deepfake-main/deepscan-backend/ml_server && .venv/bin/python -m uvicorn image_server:app --host 127.0.0.1 --port 7070
```

```bash
cd ~/Downloads/Ai_deepfake-main/deepscan-backend && node server.js
```

```bash
cd ~/Downloads/Ai_deepfake-main/deepscan-frontend && npm start
```

## Train the face detector with synthetic augmentation

```bash
cd ~/Downloads/Ai_deepfake-main/deepscan-backend/ml_server && .venv/bin/python train_face_detector.py
```

This trains a baseline head and an augmented head, writes the comparison to
`test_results/augmentation_study.json`, and saves the chosen head to
`trained_models/face_head/face_head.pt` with its threshold in `calibration_face.json`.
Restart the ML server afterwards.

## Benchmark a detector on your own folders

```bash
cd ~/Downloads/Ai_deepfake-main/deepscan-backend/ml_server && .venv/bin/python detector_benchmark.py --real <folder of real images> --fake <folder of fake images> --detector cf_vit --out test_results/my_run
```

Detectors: `cf_vit`, `effort_face`, `effort_aigi`, `gend`, `old_deepscan` (needs the ML server running).

## Troubleshooting

- **Port already in use:** run `bash scripts/stop-all.sh`. On macOS, ports 5000 and 7000 are used by AirPlay, which is why DeepScan uses 5050 and 7070.
- **"Too many requests":** the API allows 300 requests per 15 minutes; change `API_RATE_LIMIT_MAX` in `deepscan-backend/.env`.
- **Slow or crashing ML server:** close other large apps; the face model needs about 1.2 GB of memory.

The Windows scripts `scripts/start-all.ps1` and `scripts/restart-stack.ps1` are from an older version and use ports 5000/7000.
