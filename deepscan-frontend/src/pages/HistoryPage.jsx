import React, { useEffect, useState } from 'react';
import { fetchResults } from '../services/api';

function getVerdictColor(verdict) {
  switch (verdict) {
    case 'REAL': return '#059669';
    case 'UNCERTAIN': return '#d97706';
    case 'LIKELY SYNTHETIC': return '#ea580c';
    case 'SYNTHETIC': return '#dc2626';
    default: return '#94a3b8';
  }
}

function getVerdictEmoji(verdict) {
  switch (verdict) {
    case 'REAL': return '✅';
    case 'UNCERTAIN': return '🤔';
    case 'LIKELY SYNTHETIC': return '⚠️';
    case 'SYNTHETIC': return '🚨';
    default: return '—';
  }
}

export default function HistoryPage() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);
  const limit = 10;

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await fetchResults(page, limit);
        if (!cancelled) {
          setResults(data.results || []);
          setTotalPages(data.totalPages || 1);
          setTotal(data.total || 0);
        }
      } catch (err) {
        if (!cancelled) {
          setError('Failed to load results. Make sure the backend is running.');
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => { cancelled = true; };
  }, [page]);

  return (
    <>
      <div className="app__hero">
        <h1 className="page__title--glow">Analysis History</h1>
        <p>
          Browse past deepfake detection results. Showing {total} total analyses.
        </p>
      </div>

      <div className="history">
        {loading && (
          <div className="history__loading">
            <div className="history__spinner" />
            Loading results…
          </div>
        )}

        {error && <div className="history__error">{error}</div>}

        {!loading && !error && results.length === 0 && (
          <div className="history__empty">
            <div className="history__empty-icon">📭</div>
            <p>No analysis results yet. Upload an image or video to get started!</p>
          </div>
        )}

        {!loading && results.length > 0 && (
          <>
            <div className="history__grid">
              {results.map((r) => (
                <div key={r._id} className="history__card">
                  <div className="history__card-header">
                    <span className={`history__type-badge history__type-badge--${r.media_type}`}>
                      {r.media_type === 'video' ? '🎬' : '🖼️'} {r.media_type}
                    </span>
                    <span className="history__date">
                      {new Date(r.analyzed_at).toLocaleDateString(undefined, {
                        month: 'short', day: 'numeric', year: 'numeric',
                        hour: '2-digit', minute: '2-digit',
                      })}
                    </span>
                  </div>
                  <div className="history__card-name">{r.originalName || r.filename}</div>
                  <div className="history__card-verdict">
                    <span style={{ color: getVerdictColor(r.verdict) }}>
                      {getVerdictEmoji(r.verdict)} {r.verdict}
                    </span>
                    <span className="history__card-score">
                      {r.final_score}%
                    </span>
                  </div>
                  <div className="history__card-bar">
                    <div
                      className="history__card-bar-fill"
                      style={{
                        width: `${r.final_score}%`,
                        background: getVerdictColor(r.verdict),
                      }}
                    />
                  </div>
                  <div className="history__card-meta">
                    <span>Model: {r.breakdown?.model_score ?? '—'}</span>
                    <span>Metadata: {r.breakdown?.metadata_score ?? '—'}</span>
                    {r.frames_analyzed != null && <span>Frames: {r.frames_analyzed}</span>}
                  </div>
                </div>
              ))}
            </div>

            {totalPages > 1 && (
              <div className="history__pagination">
                <button
                  className="history__page-btn"
                  disabled={page <= 1}
                  onClick={() => setPage(p => p - 1)}
                >
                  ← Previous
                </button>
                <span className="history__page-info">
                  Page {page} of {totalPages}
                </span>
                <button
                  className="history__page-btn"
                  disabled={page >= totalPages}
                  onClick={() => setPage(p => p + 1)}
                >
                  Next →
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </>
  );
}
