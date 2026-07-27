import React from 'react';
import type { LocationInfo, ObservationInfo } from '../types/aqi';
import { HelpCircle, Layers } from 'lucide-react';

interface AqiCardProps {
  location: LocationInfo;
  observation: ObservationInfo;
  onShowWhy: () => void;
  loadingWhy?: boolean;
}

function formatAsOf(asOfStr?: string): string {
  if (!asOfStr) return '';
  try {
    const d = new Date(asOfStr);
    if (isNaN(d.getTime())) return asOfStr;
    return d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  } catch (_) {
    return asOfStr;
  }
}

export const AqiCard: React.FC<AqiCardProps> = ({
  location,
  observation,
  onShowWhy,
  loadingWhy = false
}) => {
  const { aqi, category, category_color, category_text_color, primary_pollutant } = observation;
  const formattedTime = formatAsOf(observation.as_of);

  return (
    <div className="aqi-card-glass">
      <div className="aqi-header">
        <div>
          <h2 className="location-name">{location.name}</h2>
        </div>
        <span
          className="category-pill"
          style={{ backgroundColor: category_color, color: category_text_color }}
        >
          {category}
        </span>
      </div>

      <div className="aqi-body">
        <div className="aqi-number-container">
          <span className="aqi-value" style={{ color: category_color }}>{aqi}</span>
          <span className="aqi-label">US AQI</span>
        </div>

        <div className="aqi-meta">
          <div className="meta-item">
            <Layers size={16} className="text-zinc-400" />
            <span>Primary: <strong>{primary_pollutant}</strong></span>
          </div>
          <div className="meta-item">
            <span>{observation.category_description}</span>
          </div>
          <div className="meta-item" style={{ fontSize: '0.75rem', opacity: 0.7, marginTop: '2px' }}>
            <span>Source: {observation.source || 'AirNow'}{formattedTime ? ` • ${formattedTime}` : ''}</span>
          </div>
        </div>
      </div>

      <div className="aqi-footer">
        <button className="show-why-btn" onClick={onShowWhy} disabled={loadingWhy}>
          <HelpCircle size={18} />
          <span>{loadingWhy ? 'Loading...' : 'Show Why'}</span>
        </button>
      </div>
    </div>
  );
};
