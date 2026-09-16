const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const mongoose = require('mongoose');
const rateLimit = require('express-rate-limit');
const dotenv = require('dotenv');

dotenv.config();

const app = express();

// Ensure uploads directory exists
const fs_sync = require('fs');
const path = require('path');
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs_sync.existsSync(uploadsDir)) {
  fs_sync.mkdirSync(uploadsDir);
  console.log('📁 Created uploads directory');
}

// ─── Security & Logging ────────────────────────────────────────────────────
app.use(helmet());
app.use(morgan('dev'));

// ─── CORS ──────────────────────────────────────────────────────────────────
const corsOrigins = (process.env.CORS_ORIGIN || 'http://localhost:3000,http://localhost:3001')
  .split(',')
  .map(o => o.trim());
// Dev servers (CRA, etc.) often get reassigned to a random port when their
// preferred one is taken, so also allow any localhost/127.0.0.1 origin
// unless explicitly disabled — avoids CORS breaking every time that happens.
const allowAnyLocalhost = (process.env.CORS_ALLOW_ANY_LOCALHOST || 'true').trim().toLowerCase() !== 'false';
const localhostOriginPattern = /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/;
app.use(cors({
  origin: (origin, callback) => {
    // Allow requests with no origin (e.g. curl, Postman)
    if (!origin || corsOrigins.includes(origin)) return callback(null, true);
    if (allowAnyLocalhost && localhostOriginPattern.test(origin)) return callback(null, true);
    callback(new Error(`CORS blocked: ${origin}`));
  }
}));

// ─── Body Parsers ──────────────────────────────────────────────────────────
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// ─── Rate Limiting ─────────────────────────────────────────────────────────
const apiLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  // 30 requests per 15 min (shared with history/page loads) returned 429 after a
  // handful of test uploads, which looked like a broken analysis. Configurable.
  max: Number(process.env.API_RATE_LIMIT_MAX) || 300,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many requests. Please try again later.' }
});
app.use('/api', apiLimiter);

// ─── MongoDB Connection ────────────────────────────────────────────────────
let mongoConnected = false;
mongoose
  .connect(process.env.MONGODB_URI)
  .then(() => {
    mongoConnected = true;
    console.log('✅ MongoDB Connected');
  })
  .catch((err) => {
    console.error('❌ MongoDB Connection Error:', err.message);
    console.warn('⚠️  Server will run without database — analysis results will not be persisted.');
  });

// Middleware to reject DB-dependent routes when MongoDB is down
app.use('/api/results', (req, res, next) => {
  if (!mongoConnected) {
    return res.status(503).json({ error: 'Database unavailable. Results cannot be retrieved.' });
  }
  next();
});

// ─── Routes ────────────────────────────────────────────────────────────────
const analyzeRoute = require('./routes/analyze');
app.use('/api', analyzeRoute);

// ─── Health Check ──────────────────────────────────────────────────────────
app.get('/', (req, res) => {
  res.json({ status: 'Deepfake Backend Running 🚀', timestamp: new Date().toISOString() });
});

// ─── 404 Handler ───────────────────────────────────────────────────────────
app.use((req, res) => {
  res.status(404).json({ error: 'Route not found' });
});

// ─── Global Error Handler ──────────────────────────────────────────────────
// eslint-disable-next-line no-unused-vars
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err.message);
  const status = err.status || 500;
  res.status(status).json({ error: err.message || 'Internal Server Error' });
});

// ─── Start Server ──────────────────────────────────────────────────────────
const PORT = process.env.PORT || 5000;
const server = app.listen(PORT, () => {
  console.log(`🚀 Deepfake Server running on http://localhost:${PORT}`);
});

// ─── Graceful Shutdown ─────────────────────────────────────────────────────
const shutdown = async (signal) => {
  console.log(`\n🛑 ${signal} received. Shutting down Deepfake gracefully...`);
  server.close(async () => {
    await mongoose.connection.close();
    console.log('✅ MongoDB connection closed. Exiting.');
    process.exit(0);
  });
};

process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT', () => shutdown('SIGINT'));