# DeepScan API Documentation

Base URL: http://localhost:5000

## POST /api/analyze
Upload an image for deepfake analysis.

**Request**
- Method: POST
- Body: form-data
- Key: image (type: File)
- Allowed: .jpg, .jpeg, .png, .webp (max 5MB)

**Response**
{
  "id": "mongodb_id",
  "final_score": 44,
  "verdict": "UNCERTAIN",
  "confidence": "Low",
  "breakdown": {
    "model_score": 35,
    "artifact_score": 37,
    "metadata_score": 85
  },
  "flags": ["No EXIF data found"],
  "analyzed_at": "2026-03-10T15:48:42.235Z"
}

## GET /api/results
Fetch last 20 scan results.

**Request**
- Method: GET
- No body required

**Response**
{
  "results": [ ...array of past scans... ]
}