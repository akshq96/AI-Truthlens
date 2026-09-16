import React, { useCallback, useEffect, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { analyzeMedia } from '../services/api';
import ResultCard from './ResultCard';
import MetadataPanel from './MetadataPanel';

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function UploadZone({ id }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loadingDot, setLoadingDot] = useState(0);

  const [description, setDescription] = useState('');

  const onDrop = useCallback((acceptedFiles) => {
    const [selectedFile] = acceptedFiles;
    setFile(selectedFile || null);
    setResult(null);
    setError(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: false,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp'],
      'video/*': ['.mp4', '.mov', '.webm']
    },
  });

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return undefined;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  // Loading dots animation
  useEffect(() => {
    if (!loading) return;
    const interval = setInterval(() => {
      setLoadingDot(d => (d + 1) % 4);
    }, 400);
    return () => clearInterval(interval);
  }, [loading]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file || loading) return;
    setLoading(true);
    setError(null);
    try {
      const data = await analyzeMedia(file, description);
      setResult(data);
    } catch (err) {
      const serverMessage = err?.response?.data?.error || err?.response?.data?.message;
      setError(serverMessage || 'Failed to analyze the file. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleRemoveFile = (e) => {
    e.stopPropagation();
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setDescription('');
  };

  const isVideo = file && file.type.startsWith('video/');
  const loadingText = 'Analyzing' + '.'.repeat(loadingDot);

  return (
    <section className="upload-zone" id={id}>
      <div className="upload-zone__card">

        {!file && (
          <div
            className={`upload-zone__drop ${isDragActive ? 'is-active' : ''}`}
            {...getRootProps()}
          >
            <input {...getInputProps()} />
            <div className="upload-zone__drop-icon-wrap">
              <div className="upload-zone__drop-ring" style={{ borderColor: 'var(--primary-glow)' }} />
              <div className="upload-zone__drop-ring upload-zone__drop-ring--2" style={{ borderColor: 'var(--cyan-glow)' }} />
              <svg className="upload-zone__drop-icon" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M32 8L40 20H24L32 8Z" fill="var(--primary)" opacity="0.9"/>
                <rect x="14" y="20" width="36" height="28" rx="4" stroke="var(--primary)" strokeWidth="2" fill="var(--primary-dim)"/>
                <circle cx="32" cy="34" r="8" stroke="var(--cyan)" strokeWidth="1.5" fill="none"/>
                <circle cx="32" cy="34" r="3" fill="var(--cyan)"/>
                <line x1="14" y1="34" x2="20" y2="34" stroke="var(--primary)" strokeWidth="1.5" strokeDasharray="3 2" opacity="0.5"/>
                <line x1="44" y1="34" x2="50" y2="34" stroke="var(--primary)" strokeWidth="1.5" strokeDasharray="3 2" opacity="0.5"/>
                <line x1="32" y1="20" x2="32" y2="26" stroke="var(--primary)" strokeWidth="1.5" strokeDasharray="3 2" opacity="0.5"/>
                <line x1="32" y1="42" x2="32" y2="48" stroke="var(--primary)" strokeWidth="1.5" strokeDasharray="3 2" opacity="0.5"/>
              </svg>
            </div>
            <div className="upload-zone__prompt">
              {isDragActive ? '⚡ Release to Upload' : 'Drag & drop media here'}
            </div>
            <div className="upload-zone__hint" style={{ fontFamily: "'JetBrains Mono', monospace", opacity: 0.6 }}>
              JPG · PNG · WEBP · MP4 · MOV (MAX_50MB)
            </div>
            <div className="upload-zone__or">
              <span>or</span>
            </div>
            <div className="upload-zone__browse-btn">Select File</div>
          </div>
        )}

        {previewUrl && (
          <div className="upload-zone__preview-wrap">
            <div className="upload-zone__preview-inner">
              {isVideo ? (
                <video src={previewUrl} controls className="upload-zone__preview-media" />
              ) : (
                <img src={previewUrl} alt="Uploaded preview" className="upload-zone__preview-media" />
              )}
              <button
                type="button"
                className="upload-zone__remove-btn"
                onClick={handleRemoveFile}
                title="Remove file"
              >
                ✕
              </button>
            </div>
            <div className="upload-zone__file-info" style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.75rem' }}>
              <span className="upload-zone__file-name">{file.name}</span>
              <span className="upload-zone__file-size" style={{ borderColor: 'var(--border2)' }}>{formatFileSize(file.size)}</span>
            </div>
          </div>
        )}

        <div className="upload-zone__input-group" style={{ marginTop: '20px', padding: '0 20px' }}>
          <label style={{ display: 'block', fontSize: '0.65rem', fontFamily: "'JetBrains Mono', monospace", color: 'var(--muted)', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
            Analysis_Context
          </label>
          <textarea
            className="upload-zone__desc"
            placeholder="Enter scene details or metadata overrides..."
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            style={{
              width: '100%',
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid var(--border2)',
              borderRadius: '12px',
              padding: '12px 16px',
              color: 'var(--text)',
              fontFamily: "'Inter', sans-serif",
              fontSize: '0.9rem',
              minHeight: '80px',
              resize: 'none',
              outline: 'none',
              transition: 'all 0.2s',
              borderBottom: '2px solid transparent'
            }}
            onFocus={(e) => e.target.style.borderColor = 'var(--primary)'}
            onBlur={(e) => e.target.style.borderColor = 'var(--border2)'}
          />
        </div>

        <button
          className={`upload-zone__button ${loading ? 'is-loading' : ''} ${file ? 'is-ready' : ''}`}
          onClick={handleSubmit}
          disabled={!file || loading}
          type="button"
          style={{boxShadow: file ? 'var(--shadow-fire)' : 'none'}}
        >
          {loading ? (
            <>
              <span className="upload-zone__spinner" style={{ borderTopColor: 'var(--primary)' }} />
              {loadingText}
            </>
          ) : (
            <>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" style={{ opacity: 0.8 }}>
                <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
              </svg>
              Run Inference
            </>
          )}
        </button>

        {error && (
          <div className="upload-zone__error">
            <span>⚠️</span>
            {error}
          </div>
        )}
      </div>

      {result && (
        <div className="upload-zone__results">
          <ResultCard
            score={result.final_score}
            verdict={result.verdict}
            result={result.result}
            confidence={result.confidence}
            evidence_summary={result.evidence_summary}
            ai_probability={result.ai_probability}
            deepfake_probability={result.deepfake_probability}
            metadata_provenance={result.metadata_provenance}
            trace={result.trace}
            real_score={result.real_score}
            fake_score={result.fake_score}
            signals={result.signals}
            unavailable_signals={result.unavailable_signals}
            decision_reason={result.decision_reason}
            model_score={result.breakdown?.model_score}
            metadata_score={result.breakdown?.metadata_score}
          />
          <MetadataPanel metadata={result.raw_metadata} />
        </div>
      )}
    </section>
  );
}

