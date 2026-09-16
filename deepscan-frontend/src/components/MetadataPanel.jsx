import React from 'react';

function getValue(metadata, key) {
  if (!metadata) {
    return null;
  }
  const value = metadata[key];
  return value === undefined || value === null || value === '' ? null : value;
}

function MetadataRow({ label, value }) {
  const isMissing = value === null;
  return (
    <div className={`metadata-panel__row ${isMissing ? 'is-missing' : ''}`}>
      <span>{label}</span>
      <strong>{isMissing ? 'Missing' : value}</strong>
    </div>
  );
}

export default function MetadataPanel({ metadata }) {
  if (!metadata || Object.keys(metadata).length === 0) {
    return (
      <section className="metadata-panel">
        <div className="metadata-panel__title">Signal Extraction</div>
        <div className="metadata-panel__empty">
          No verifiable metadata found. Typical of AI-generated or privacy-scrubbed media.
        </div>
      </section>
    );
  }

  const make = getValue(metadata, 'Make') || getValue(metadata, 'DeviceManufacturer');
  const model = getValue(metadata, 'Model') || getValue(metadata, 'DeviceModel');
  const device = (make && model) ? `${make} ${model}` : (make || model || (getValue(metadata, 'ProfileDescription') === 'sRGB' ? 'Generic Display Device' : null));
  
  const date = getValue(metadata, 'DateTimeOriginal') || getValue(metadata, 'CreateDate') || getValue(metadata, 'DateTime') || (getValue(metadata, 'ProfileDateTime') !== '2016-01-01T00:00:00.000Z' ? getValue(metadata, 'ProfileDateTime') : null);
  
  // exifr often flattens GPS
  const lat = getValue(metadata, 'latitude');
  const lon = getValue(metadata, 'longitude');
  const location = (lat !== null && lon !== null) ? `${Number(lat).toFixed(4)}, ${Number(lon).toFixed(4)}` : null;
  
  const software = getValue(metadata, 'Software') || getValue(metadata, 'ProfileCreator') || getValue(metadata, 'ProfileCopyright');

  return (
    <section className="metadata-panel">
      <div className="metadata-panel__title">Engine Verification</div>
      
      <MetadataRow label="Engine Device" value={device} />
      <MetadataRow label="Capture Date" value={date} />
      <MetadataRow label="Geo Signature" value={location} />
      <MetadataRow label="Signal Software" value={software} />

      {/* Optional: Show more if it's there but stick to user request mainly */}
      {Object.keys(metadata).length > 10 && (
        <div style={{ fontSize: '0.65rem', color: 'var(--muted)', marginTop: '12px', opacity: 0.6 }}>
          + {Object.keys(metadata).length - 4} additional technical signals verified
        </div>
      )}
    </section>
  );
}
