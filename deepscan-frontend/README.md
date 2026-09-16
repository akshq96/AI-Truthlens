<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=220&section=header&text=Deepfake&fontSize=90&fontColor=ffffff&fontAlignY=40&desc=AI-Generated%20Media%20Verifier&descAlignY=62&descSize=22&animation=fadeIn" width="100%"/>

<br/>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&pause=1000&color=A78BFA&center=true&vCenter=true&width=600&lines=Can+you+tell+what%27s+real%3F+%F0%9F%A4%94;We+can.+%F0%9F%94%8D;Deepfake+detection+powered+by+AI+%F0%9F%A4%96;Upload.+Analyze.+Verify.+%E2%9C%85)](https://git.io/typing-svg)

<br/>

[![Status](https://img.shields.io/badge/Status-In%20Development-yellow?style=for-the-badge&logo=statuspage)](.)
[![Version](https://img.shields.io/badge/Version-1.0%20Light-blue?style=for-the-badge)](.)
[![Node.js](https://img.shields.io/badge/Node.js-18+-339933?style=for-the-badge&logo=node.js&logoColor=white)](.)
[![MongoDB](https://img.shields.io/badge/MongoDB-Connected-47A248?style=for-the-badge&logo=mongodb&logoColor=white)](.)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](.)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](.)
[![License](https://img.shields.io/badge/License-Academic-8B5CF6?style=for-the-badge)](.)

<br/>

> 💀 **Deepfakes are getting scary good.**
> Deepfake fights back — upload any image, get a verdict in seconds.
> *Because not everything you see is real.*

<br/>

**Department of Computer Science & Engineering (CSED) · Section 2FH**

<br/>

</div>

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Workflow](#-workflow)
- [Tech Stack](#-tech-stack)
- [EXIF Metadata Scoring](#-exif-metadata-scoring)
- [API Endpoints](#-api-endpoints)
- [Getting Started](#-getting-started)
- [Docker Setup](#-docker-setup)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Project Scope](#-project-scope)
- [Implementation Plan](#-implementation-plan)
- [Team](#-team)
- [References](#-references)

---

## 🧠 About the Project

**Deepfake** is a lightweight, web-based deepfake and AI-generated media verifier built as an academic project. As synthetic media becomes increasingly indistinguishable from reality, tools that help everyday users verify digital content are more critical than ever.

Deepfake analyzes uploaded images through a **3-layer detection pipeline** — no GPU, no heavy setup, just results:

| Layer | What it does |
|---|---|
| 🎨 **Visual Artifact Analysis** | Color variance, edge sharpness, brightness anomalies |
| 🔎 **Metadata Forensics** | EXIF parsing — camera info, AI software detection, timestamps |
| ⚖️ **Score Aggregation** | Weighted fusion → final probability score + verdict |

The system outputs a **probability score** (e.g., `72% — Likely Synthetic`) with confidence rating and detailed breakdown — no technical expertise required.

> ⚠️ **Light Version** — built for academic and demonstration purposes. Not legal or forensic-grade.

---

## ✨ Features

| Feature | Description | Status |
|---|---|---|
| 🖼️ **Image Upload** | Drag-and-drop or click — JPEG, PNG, WEBP supported | ✅ Done |
| 🤖 **Artifact Detection** | Heuristic scan — color variance, edges, brightness | ✅ Done |
| 📊 **AI Probability Score** | 0–100% confidence score with Real / Synthetic verdict | ✅ Done |
| 🔎 **Metadata Forensics** | EXIF parsing — flags missing camera, AI software, timestamps | ✅ Done |
| ⚖️ **Weighted Score Fusion** | 3 signals combined into one final score intelligently | ✅ Done |
| 🗄️ **Result Logging** | Every scan saved to MongoDB with timestamp & breakdown | ✅ Done |
| 🎨 **React UI** | Reusable components — responsive, fast, state-driven | ✅ Done |
| ⚡ **Fast Pipeline** | End-to-end analysis completes in under 3 seconds | ✅ Done |
| 🐳 **Docker Support** | Full stack containerized — one command to run everything | ✅ Done |
| 🔁 **CI/CD Pipeline** | GitHub Actions — auto test, build, and publish on every push | ✅ Done |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 1 — FRONTEND                       │
│      React.js  ·  Component UI  ·  Upload Widget  ·  Axios  │
└─────────────────────┬───────────────────────────────────────┘
                      │  HTTP Request (multipart/form-data)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 LAYER 2 — API GATEWAY                       │
│      Node.js / Express.js  ·  Multer Validator  ·  Router   │
└──────────┬──────────────────────────────┬───────────────────┘
           │                              │
           ▼                              ▼
┌──────────────────────┐      ┌───────────────────────────────┐
│  METADATA FORENSICS  │      │     DETECTION ENGINE          │
│  EXIF Parser (exifr) │      │  Sharp Image Processor        │
│  Timestamp Checker   │      │  Color Variance Analysis      │
│  AI Software Flags   │      │  Edge Sharpness Heuristics    │
│  Camera Info Check   │      │  Brightness Anomaly Check     │
└──────────┬───────────┘      └──────────────┬────────────────┘
           │                                  │
           └──────────────┬───────────────────┘
                          │  Weighted Score Fusion
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  SCORE AGGREGATOR                           │
│     Model (50%)  +  Artifact (30%)  +  Metadata (20%)       │
│                  → Final Probability %                      │
│         → Verdict: REAL / UNCERTAIN / SYNTHETIC             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
              Result returned to UI
         [ 72% — ⚠ LIKELY SYNTHETIC · Medium Confidence ]

          ┌────────────────────────┐
          │        MongoDB         │
          │  Store results & logs  │
          └────────────────────────┘
```

### Docker Architecture

```
  Browser (localhost:3001)
          │
          ▼
  [ Frontend Container ]     React / Nginx — Port 3001→3000
          │
          │ HTTP API calls
          ▼
  [ Backend Container ]      Node.js + Express — Port 5000
    /api/analyze
    /api/results
          │
          │ Mongoose ODM
          ▼
  [ MongoDB Container ]      mongo:7 — Port 27017 — Volume: mongo_data
```

---

## 🔄 Workflow

```
User Uploads Image
       │
       ▼
┌─────────────┐
│  Validate   │  ← File type · Size limit · Multer
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Preprocess  │  ← Resize · Normalize · Extract Metadata
└──────┬──────┘
       │
       ├──────────────────────────┐
       ▼                          ▼
┌─────────────┐          ┌────────────────┐
│  Artifact   │          │    Metadata    │
│  Analysis   │          │   Forensics    │
│  (Sharp)    │          │   (exifr)      │
└──────┬──────┘          └───────┬────────┘
       │                          │
       └──────────┬───────────────┘
                  ▼
         ┌────────────────┐
         │Score Aggregator│  ← Weighted fusion
         └───────┬────────┘
                 │
                 ▼
         ┌──────────────┐
         │  Save to DB  │  ← MongoDB logging
         └───────┬──────┘
                 │
                 ▼
        Display Result to User
      ┌─────────────────────────┐
      │  Final Score: 72%       │
      │  Verdict: ⚠ SYNTHETIC   │
      │  Confidence: Medium     │
      │  Breakdown: shown       │
      └─────────────────────────┘
```

**Step-by-step breakdown:**

1. 📤 **Upload** — JPEG, PNG, WEBP · max 5MB accepted
2. ✅ **Validate** — Multer checks file type and size before processing
3. 🎨 **Artifact Analysis** — Sharp scans color variance, edge sharpness, and brightness irregularities
4. 🔎 **Metadata Forensics** — EXIF flags missing camera info, AI software signatures, and timestamp anomalies
5. ⚖️ **Score Fusion** — Model 50% + Artifact 30% + Metadata 20% combined into a single probability score
6. 🗄️ **DB Logging** — Full result saved to MongoDB for scan history
7. 📊 **Response** — Score, verdict, confidence level, and detailed breakdown returned to the frontend

---

## 🛠️ Tech Stack

<div align="center">

![Node.js](https://img.shields.io/badge/Node.js-339933?style=for-the-badge&logo=node.js&logoColor=white)
![Express.js](https://img.shields.io/badge/Express.js-000000?style=for-the-badge&logo=express&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
![Nginx](https://img.shields.io/badge/Nginx-009639?style=for-the-badge&logo=nginx&logoColor=white)

</div>

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | React.js · Axios · CSS3 · Nginx | Interactive UI, API communication, production serving |
| **Backend** | Node.js · Express.js · Multer | REST API, file handling, routing |
| **Image Analysis** | Sharp · exifr · Heuristic Detection | Pixel-level processing, metadata extraction |
| **Database** | MongoDB · Mongoose | Scan result persistence and history |
| **DevOps** | Docker · Docker Compose · GitHub Actions | Containerization, orchestration, CI/CD |
| **Dev Tools** | Postman · VS Code · Git · GitHub | API testing, development, version control |

---

## 🔬 EXIF Metadata Scoring

The metadata forensics layer analyzes hidden EXIF data embedded in every real camera photo. AI-generated images typically have none of this data — making its absence a strong synthetic signal.

```
Base Score: 50  (neutral starting point)

┌─────────────────────────────────────────────────────────────────┐
│                     EXIF SIGNAL TABLE                          │
├────────────────────────────┬────────────────────────────────────┤
│  Signal                    │  Score Impact                      │
├────────────────────────────┼────────────────────────────────────┤
│  No EXIF data at all       │  → score = 85  (instant flag)      │
│  No camera make/model      │  → score += 20                     │
│  Camera make/model found   │  → score -= 20                     │
│  AI software detected *    │  → score += 40                     │
│  No timestamp              │  → score += 10                     │
│  Timestamp found           │  → score -= 10                     │
│  No GPS data               │  → score += 5                      │
│  GPS data found            │  → score -= 10                     │
├────────────────────────────┼────────────────────────────────────┤
│  Final score               │  clamped between 0 and 100         │
└────────────────────────────┴────────────────────────────────────┘

* AI software detection covers:
  Stable Diffusion · Midjourney · DALL-E · Adobe Firefly
```

> 💡 **Note:** Images shared via WhatsApp or Telegram have their EXIF stripped, which inflates the metadata score even for real photos. Always upload the **original file directly via USB** for accurate results.

---

## 📡 API Endpoints

### `POST /api/analyze`
Upload an image for deepfake analysis.

**Request**
```
Body    : form-data
Key     : image (type: File)
Allowed : .jpg · .jpeg · .png · .webp
Max Size: 5MB
```

**Response**
```json
{
  "id": "69b03d5a06d049d960e7937c",
  "final_score": 72,
  "verdict": "LIKELY SYNTHETIC",
  "confidence": "Medium",
  "breakdown": {
    "model_score": 68,
    "artifact_score": 75,
    "metadata_score": 85
  },
  "flags": ["No EXIF data found — strong synthetic signal"],
  "analyzed_at": "2026-03-10T15:48:42.235Z"
}
```

### `GET /api/results`
Fetch last 20 scan results from the database.

**Response**
```json
{
  "results": [ ...array of past scans... ]
}
```

---

## 🚀 Getting Started

### Option 1: Docker (Recommended)
See [Docker Setup](#-docker-setup) below — one command runs everything.

### Option 2: Manual Setup

**Prerequisites:** Node.js v18+ · MongoDB running locally

```bash
git clone https://github.com/namann5/Ai_deepfake.git
cd Ai_deepfake
```

**Backend:**
```bash
cd deepfake-backend
npm install
node server.js
```

Create `.env` inside `deepfake-backend/`:
```env
PORT=5000
MONGODB_URI=mongodb://localhost:27017/deepfake
```

**Frontend:**
```bash
cd deepfake-frontend
npm install
npm start
```

**Access:** Frontend → `http://localhost:3000` · Backend → `http://localhost:5000`

---

## 🐳 Docker Setup

> **Prerequisites:** Install [Docker Desktop](https://www.docker.com/products/docker-desktop)

### Start everything (dev mode with hot-reload)
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

### Stop everything (keep data)
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml down
```

### Stop and delete all data (fresh start)
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml down -v
```

### Rebuild after Dockerfile changes
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
```

### View live logs
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f
```

### Access URLs (Docker)

| Service | URL |
|---|---|
| Frontend App | http://localhost:3001 |
| Backend API | http://localhost:5000 |
| Backend Health | http://localhost:5000/ |
| All Results | http://localhost:5000/api/results |
| MongoDB | mongodb://localhost:27017/deepfake |

### Why Docker?

| Before Docker | After Docker |
|---|---|
| Install Node + MongoDB manually | `docker compose up` — done |
| 3 separate terminals to start | One command starts everything |
| "Works on my machine" bugs | Identical environment everywhere |
| Hours to onboard a new dev | ~5 minutes from clone to running |
| Manual MongoDB setup | MongoDB runs in a container automatically |

---

## 🔁 CI/CD Pipeline

Every push or pull request to `main` automatically triggers the GitHub Actions pipeline:

```
Push to main
     │
     ├── Job 1: Backend Test    → npm ci → run Jest tests
     ├── Job 2: Frontend Build  → npm ci → npm run build
     └── Job 3: Docker Push     → build images → push to GHCR
                                  (only on direct push, not PRs)
```

Pipeline file: `.github/workflows/ci.yml`

---

## 📐 Project Scope

### ✅ In Scope
- Detection of AI-generated and deepfake **images**
- **Visual artifact** and **metadata** analysis
- **Probability scores** with verdict and confidence rating
- Clean **web-based UI** accessible to non-technical users
- **Scan history** stored in MongoDB
- **Docker** containerization for consistent cross-platform environments
- **CI/CD** automated testing and deployment pipeline

### ❌ Out of Scope
- Real-time **video** deepfake detection
- **Legal or forensic-grade** accuracy guarantees
- Detection of AI models released after training data cutoff
- Audio deepfake analysis

---

## 📅 Implementation Plan

```
Phase 1  ██████████  Requirement Analysis & Problem Understanding   ✅ Done
Phase 2  ██████████  Dataset & Pre-trained Model Selection          ✅ Done
Phase 3  ██████████  System Architecture Design                     ✅ Done
Phase 4  ██████████  Backend Development                            ✅ Done
Phase 5  ██████████  Frontend Development                           ✅ Done
Phase 6  ██████████  Frontend–Backend Integration                   ✅ Done
Phase 7  ████░░░░░░  Testing & Performance Evaluation               🔄 In Progress
Phase 8  ██░░░░░░░░  Deployment & Documentation                     🔄 In Progress
```

---

## 👥 Team

| Name | Role | ID |
|---|---|---|
| **Anurag Singh** | Backend Development & System Analysis | 12515990006 |
| **Arpita Raj** | Frontend Development & UI Design | 12515990007 |
| **Harshita Nagpal** | Frontend Development & Documentation | 12515990016 |
| **Naman Singh** | Backend Development & Testing | 12515990024 |

**Supervisor:** Mr. Abhishek Singh *(Technical Trainer)*
**Submitted To:** Mr. Sanjay Madaan

---

## 📚 References

1. Research papers on Deepfake Detection
2. Open-source AI-generated image detection models
3. IEEE and ACM digital libraries
4. Sharp.js documentation — image processing
5. exifr documentation — EXIF metadata parsing
6. FaceForensics++ dataset — Kaggle
7. Docker documentation — containerization
8. GitHub Actions documentation — CI/CD
9. Online resources — computer vision & deep learning

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=120&section=footer" width="100%"/>

*Department of Computer Science & Engineering (CSED) · Section 2FH · Academic Project*

**Deepfake** — Fighting synthetic misinformation, one pixel at a time. 🔍

</div>