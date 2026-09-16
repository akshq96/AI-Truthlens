# ═══════════════════════════════════════════════════════════════
# DeepScan Backend — Production Dockerfile
# Multi-stage build: install deps → slim production image
# ═══════════════════════════════════════════════════════════════

# ── Stage 1: Install production dependencies ─────────────────
FROM node:20-alpine AS deps

WORKDIR /app

# Copy only package files first (Docker layer caching)
COPY package.json package-lock.json ./

# Install production-only dependencies (no devDependencies)
# Use --ignore-scripts to avoid platform-specific postinstall issues
RUN npm ci --omit=dev --ignore-scripts

# Reinstall sharp with correct linux-musl binary for Alpine
RUN npm install --cpu=x64 --os=linux --libc=musl sharp

# ── Stage 2: Production image ────────────────────────────────
FROM node:20-alpine AS runner

WORKDIR /app

# Security: create non-root user
RUN addgroup -S deepscan && adduser -S deepscan -G deepscan

# Copy production node_modules from deps stage
COPY --from=deps /app/node_modules ./node_modules

# Copy application source code
COPY . .

# Create uploads directory with correct ownership
RUN mkdir -p uploads && chown -R deepscan:deepscan /app

# Switch to non-root user
USER deepscan

# Expose the API port
EXPOSE 5000

# Health check — verifies the server is responding
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:5000/ || exit 1

# Start the server
CMD ["node", "server.js"]
