# ═══════════════════════════════════════════════════════════════
# DeepScan Frontend — Production Dockerfile
# Stage 1: Build React app → Stage 2: Serve with Nginx
# Final image: ~25MB (nginx-alpine)
# ═══════════════════════════════════════════════════════════════

# ── Stage 1: Build the React application ─────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files first (Docker layer caching)
COPY package.json package-lock.json ./

# Install all dependencies (need devDeps for build)
RUN npm ci --legacy-peer-deps

# Copy source code
COPY . .

# CRA bakes environment variables at build time
# This ARG lets us inject the API URL during docker build
ARG REACT_APP_API_BASE_URL=http://localhost:5000
ENV REACT_APP_API_BASE_URL=$REACT_APP_API_BASE_URL

# Build optimized production bundle
RUN npm run build

# ── Stage 2: Serve with Nginx ────────────────────────────────
FROM nginx:1.27-alpine AS runner

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy our custom nginx config (SPA routing + gzip + caching)
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built static files from builder stage
COPY --from=builder /app/build /usr/share/nginx/html

EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
