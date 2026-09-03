export interface LocationInfo {
  lat: number;
  lon: number;
  name: string;
  zip_code?: string | null;
  state?: string | null;
  city?: string | null;
  country_code?: string | null;
  country?: string | null;
}

export interface CoverageInfo {
  country_code?: string | null;
  mode: 'us' | 'international' | 'unknown';
  aqi_index: string;
  sources: string[];
  disclaimer: string;
}

export interface ObservationInfo {
  source: string;
  aqi: number;
  primary_pollutant: string;
  category: string;
  category_color: string;
  category_text_color: string;
  category_description: string;
  reporting_area: string;
  pollutants: Record<string, number | null>;
  as_of: string;
}

export interface AqiResponse {
  location: LocationInfo;
  observation: ObservationInfo;
  coverage?: CoverageInfo;
  observation_token?: string | null;
}

export interface SignalItem {
  id: string;
  label: string;
  status: 'present' | 'absent' | 'unavailable';
  details?: string;
  extra?: Record<string, any>;
}

export interface HypothesisItem {
  id: 'wildfire_smoke' | 'ozone_episode' | 'winter_stagnation' | 'windblown_dust' | 'urban_industrial_pm' | 'unclear';
  title: string;
  confidence: 'low' | 'medium' | 'high';
  score: number;
  support: string[];
  against: string[];
  place?: {
    bearing?: string;
    approx_km?: number;
    description?: string;
  };
}

export interface ExecutionTraceStep {
  step: string;
  label: string;
  duration_ms: number;
  status: string;
  result?: string;
}

export interface WhyResponse {
  location: LocationInfo;
  observation: ObservationInfo;
  signals: SignalItem[];
  hypotheses: HypothesisItem[];
  open_questions: string[];
  narrative: string;
  execution_trace?: ExecutionTraceStep[];
  coverage?: CoverageInfo;
  total_ms?: number;
  map_layers?: {
    firms_hotspots?: Array<{ lat: number; lon: number; frp?: number; distance_km?: number }>;
    hms_polygons?: any; // GeoJSON
  };
}
