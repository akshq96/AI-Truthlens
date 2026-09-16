# Start DeepScan stack (Windows): image ML :7000, video ML :5500, backend :5000, frontend :3000.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$ml = Join-Path $root "deepscan-backend\ml_server"
$be = Join-Path $root "deepscan-backend"
$fe = Join-Path $root "deepscan-frontend"

# Keep image threshold unchanged; set dedicated video-as-image calibration defaults.
if (-not $env:DEEPSCAN_VIDEO_SAMPLE_SECOND) { $env:DEEPSCAN_VIDEO_SAMPLE_SECOND = "3.0" }
if (-not $env:DEEPSCAN_VIDEO_AS_IMAGE_POSITIVE_CLASS) { $env:DEEPSCAN_VIDEO_AS_IMAGE_POSITIVE_CLASS = "real" }
if (-not $env:DEEPSCAN_VIDEO_AS_IMAGE_DEEPFAKE_THRESHOLD) { $env:DEEPSCAN_VIDEO_AS_IMAGE_DEEPFAKE_THRESHOLD = "0.016" }
if (-not $env:DEEPSCAN_VIDEO_AS_IMAGE_NUM_SAMPLES) { $env:DEEPSCAN_VIDEO_AS_IMAGE_NUM_SAMPLES = "8" }
if (-not $env:DEEPSCAN_VIDEO_AS_IMAGE_DECISION_THRESHOLD) { $env:DEEPSCAN_VIDEO_AS_IMAGE_DECISION_THRESHOLD = "0.157" }

Write-Host "Starting ML servers from $ml ..."
Start-Process python -ArgumentList "image_server.py" -WorkingDirectory $ml -WindowStyle Minimized
Start-Process python -ArgumentList "video_server.py" -WorkingDirectory $ml -WindowStyle Minimized

Write-Host "Starting Node backend from $be ..."
# Prefer `node server.js` so .env is picked up reliably (npm.cmd can be flaky from Start-Process).
Start-Process node -ArgumentList "server.js" -WorkingDirectory $be -WindowStyle Minimized

Write-Host "Starting React frontend from $fe ..."
Start-Process npm -ArgumentList "start" -WorkingDirectory $fe -WindowStyle Minimized

Write-Host @"

Started four minimized windows. Wait for models to load (can take 1–2 minutes), then open:
  Frontend   http://localhost:3000
  Backend    http://localhost:5000
  Image ML   http://localhost:7000/health
  Video ML   http://localhost:5500/health

Sanity tests:  cd `"$be`"  &&  npm run sanity
"@
