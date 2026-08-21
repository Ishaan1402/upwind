import type { AqiResponse, LocationInfo } from '../types/aqi';

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api';

/**
 * Fire-and-forget user-behavior ping for the observability dashboard.
 * Never blocks or throws; keepalive so it survives page unload.
 */
export function trackEvent(event: string, detail?: string): void {
  try {
    fetch(`${API_BASE}/events`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ event, detail: detail || null }),
      keepalive: true,
    }).catch(() => {});
  } catch (_) { /* observability must never break the app */ }
}

/** Human-readable location detail for event pings. */
export function locationDetail(location: Pick<LocationInfo, 'lat' | 'lon' | 'name' | 'zip_code'>): string {
  return (
    location?.zip_code ||
    location?.name ||
    (location ? `${Number(location.lat).toFixed(4)},${Number(location.lon).toFixed(4)}` : '')
  );
}

async function fetchWithTimeout(url: string, options: RequestInit = {}, timeoutMs = 8000): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    clearTimeout(timeoutId);
    return res;
  } catch (err: any) {
    clearTimeout(timeoutId);
    if (err.name === 'AbortError') {
      throw new Error('API request timed out. Please check backend server.');
    }
    throw new Error('Unable to connect to backend API server. Please check the backend connection.');
  }
}

export async function fetchAqiData(query: string): Promise<AqiResponse> {
  const isZip = /^\d{5}$/.test(query.trim());
  const paramKey = isZip ? 'zip' : 'query';
  const res = await fetchWithTimeout(`${API_BASE}/aqi?${paramKey}=${encodeURIComponent(query)}`);
  
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to fetch AQI' }));
    throw new Error(err.detail || 'Failed to fetch AQI data');
  }
  
  return res.json();
}

export async function fetchAqiByCoords(lat: number, lon: number): Promise<AqiResponse> {
  const res = await fetchWithTimeout(`${API_BASE}/aqi?lat=${lat}&lon=${lon}`);
  
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to fetch AQI' }));
    throw new Error(err.detail || 'Failed to fetch AQI data');
  }

  return res.json();
}

export interface StreamCallbacks {
  onToolStart?: (data: { step: string; label: string }) => void;
  onToolDone?: (data: { step: string; label: string; duration_ms: number; status: string; result?: string }) => void;
  onSignalsReady?: (data: any) => void;
  onToken?: (token: string) => void;
  onComplete?: (data: { narrative: string; execution_trace: any[] }) => void;
  onError?: (err: any) => void;
}

export function streamWhyExplanation(location: any, observation: any, callbacks: StreamCallbacks, observationToken?: string | null): () => void {
  const params = new URLSearchParams({
    lat: String(location.lat),
    lon: String(location.lon),
    zip_code: location.zip_code || '',
    city: location.city || '',
    state: location.state || '',
    name: location.name || '',
    country_code: location.country_code || '',
    observation_token: observationToken || '',
    aqi: String(observation.aqi ?? 50),
    primary_pollutant: observation.primary_pollutant || 'PM2.5',
    category: observation.category || 'Moderate'
  });

  const url = `${API_BASE}/why/stream?${params.toString()}`;
  const eventSource = new EventSource(url);
  let completed = false;

  eventSource.addEventListener('tool_start', (e) => {
    try { callbacks.onToolStart?.(JSON.parse(e.data)); } catch (_) {}
  });

  eventSource.addEventListener('tool_done', (e) => {
    try { callbacks.onToolDone?.(JSON.parse(e.data)); } catch (_) {}
  });

  eventSource.addEventListener('signals_ready', (e) => {
    try { callbacks.onSignalsReady?.(JSON.parse(e.data)); } catch (_) {}
  });

  eventSource.addEventListener('llm_token', (e) => {
    try {
      const payload = JSON.parse(e.data);
      if (payload.token) callbacks.onToken?.(payload.token);
    } catch (_) {}
  });

  eventSource.addEventListener('complete', (e) => {
    completed = true;
    try {
      callbacks.onComplete?.(JSON.parse(e.data));
    } catch (_) {}
    eventSource.close();
  });

  eventSource.onerror = (err) => {
    if (completed || eventSource.readyState === EventSource.CLOSED) return;
    callbacks.onError?.(err);
    eventSource.close();
  };

  return () => {
    eventSource.close();
  };
}
