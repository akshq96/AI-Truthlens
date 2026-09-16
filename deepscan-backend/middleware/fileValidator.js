const multer = require('multer');
const path = require('path');
const crypto = require('crypto');

// ─── Storage Configuration ─────────────────────────────────────────────────
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    // Use an absolute path so the server works regardless of cwd
    cb(null, path.join(__dirname, '..', 'uploads'));
  },
  filename: (req, file, cb) => {
    // crypto.randomBytes gives collision-safe unique names
    const randomHex = crypto.randomBytes(16).toString('hex');
    const ext = path.extname(file.originalname).toLowerCase();
    cb(null, `${randomHex}${ext}`);
  }
});

// ─── File Filter ───────────────────────────────────────────────────────────
const ALLOWED_MIME_TYPES = [
  // Images
  'image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'image/gif', 'image/tiff',
  // Videos
  'video/mp4', 'video/x-msvideo', 'video/quicktime', 'video/x-matroska',
  'video/webm', 'video/x-flv', 'video/x-ms-wmv',
];

const fileFilter = (req, file, cb) => {
  if (ALLOWED_MIME_TYPES.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(new Error('Only JPEG, PNG, WEBP images and MP4, AVI, MOV, MKV, WEBM videos are allowed'), false);
  }
};

// ─── Video File Filter ─────────────────────────────────────────────────────
const ALLOWED_VIDEO_TYPES = ['video/mp4', 'video/webm', 'video/quicktime', 'video/x-msvideo'];

const videoFileFilter = (req, file, cb) => {
  if (ALLOWED_VIDEO_TYPES.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(new Error('Only MP4, WEBM, AVI, and MOV videos are allowed'), false);
  }
};

// ─── Multer Instances ──────────────────────────────────────────────────────
const upload = multer({
  storage,
  fileFilter,
  limits: { fileSize: parseInt(process.env.UPLOAD_LIMIT, 10) || 50 * 1024 * 1024 }
});

const uploadVideo = multer({
  storage,
  fileFilter: videoFileFilter,
  limits: { fileSize: parseInt(process.env.UPLOAD_LIMIT, 10) || 50 * 1024 * 1024 }
});

// ─── Shared Multer Error Handler ───────────────────────────────────────────
// Attach to any router that uses multer so errors return clean JSON.
const handleMulterError = (err, req, res, next) => {
  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      const maxMB = Math.round(((parseInt(process.env.UPLOAD_LIMIT, 10) || 50 * 1024 * 1024) / 1024 / 1024) * 10) / 10;
      return res.status(400).json({ error: `File too large. Maximum size is ${maxMB} MB.` });
    }
    return res.status(400).json({ error: `Upload error: ${err.message}` });
  }
  if (err) {
    return res.status(400).json({ error: err.message });
  }
  next();
};

module.exports = { upload, uploadVideo, handleMulterError };