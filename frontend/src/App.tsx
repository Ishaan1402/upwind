import React, { useState, useEffect } from 'react';
import type { CoverageInfo, LocationInfo, ObservationInfo, WhyResponse } from './types/aqi';
import { Wind } from 'lucide-react';
import { SearchBar } from './components/SearchBar';
import { MapView } from './components/MapView';
import { AqiCard } from './components/AqiCard';
import { WhyDrawer } from './components/WhyDrawer';
import { fetchAqiData, fetchAqiByCoords, trackEvent, locationDetail } from './services/api';

export const App: React.FC = () => {
  const [location, setLocation] = useState<LocationInfo | null>(null);
  const [observation, setObservation] = useState<ObservationInfo | null>(null);
  const [coverage, setCoverage] = useState<CoverageInfo | null>(null);
  const [observationToken, setObservationToken] = useState<string | null>(null);
  const [whyData, setWhyData] = useState<WhyResponse | null>(null);
  const [loadingAqi, setLoadingAqi] = useState<boolean>(false);
  const [errorAqi, setErrorAqi] = useState<string | null>(null);

  const [drawerOpen, setDrawerOpen] = useState<boolean>(false);
  const reqIdRef = React.useRef(0);

  // Load default location on initial mount (e.g. 90210 Beverly Hills / Los Angeles)
  useEffect(() => {
    handleSearch('90210');
  }, []);

  // Enforce tab title to Upwind
  useEffect(() => {
    document.title = 'Upwind';
  }, []);

  // Auto-clear error toast after 4 seconds
  useEffect(() => {
    if (errorAqi) {
      const timer = setTimeout(() => setErrorAqi(null), 4000);
      return () => clearTimeout(timer);
    }
  }, [errorAqi]);

  const handleSearch = async (query: string) => {
    const reqId = ++reqIdRef.current;
    setLoadingAqi(true);
    setErrorAqi(null);
    try {
      const res = await fetchAqiData(query);
      if (reqId !== reqIdRef.current) return;
      setLocation(res.location);
      setObservation(res.observation);
      setCoverage(res.coverage ?? null);
      setObservationToken(res.observation_token ?? null);
      setWhyData(null);
      trackEvent('aqi_view', query.trim());
    } catch (err: any) {
      if (reqId !== reqIdRef.current) return;
      setErrorAqi(err.message || 'Upwind currently covers US states & territories only.');
    } finally {
      if (reqId === reqIdRef.current) setLoadingAqi(false);
    }
  };

  const handleMapClick = async (lat: number, lon: number) => {
    const reqId = ++reqIdRef.current;
    setLoadingAqi(true);
    setErrorAqi(null);
    try {
      const res = await fetchAqiByCoords(lat, lon);
      if (reqId !== reqIdRef.current) return;
      setLocation(res.location);
      setObservation(res.observation);
      setCoverage(res.coverage ?? null);
      setObservationToken(res.observation_token ?? null);
      setWhyData(null);
      trackEvent('aqi_view', `${Number(lat).toFixed(4)},${Number(lon).toFixed(4)}`);
    } catch (err: any) {
      if (reqId !== reqIdRef.current) return;
      setErrorAqi(err.message || 'Upwind currently covers US states & territories only.');
    } finally {
      if (reqId === reqIdRef.current) setLoadingAqi(false);
    }
  };

  const handleShowWhy = () => {
    if (location) trackEvent('why_open', locationDetail(location));
    setDrawerOpen(true);
  };

  return (
    <div className="app-layout">
      {/* Header Bar */}
      <header className="app-header">
        <div className="header-brand">
          <div className="brand-logo">
            <Wind size={22} className="text-accent" />
          </div>
          <div>
            <h1 className="brand-title">Upwind</h1>
          </div>
        </div>

        <SearchBar
          onSearch={handleSearch}
          onLocate={handleMapClick}
          onError={(msg) => setErrorAqi(msg)}
          loading={loadingAqi}
          displayQuery={location?.zip_code || location?.name || ''}
        />
      </header>

      {/* Main Map View */}
      <main className="app-main">
        <MapView
          location={location}
          observation={observation}
          whyData={whyData}
          onMapClick={handleMapClick}
        />

        {errorAqi && (
          <div className="error-toast">
            <span>{errorAqi}</span>
          </div>
        )}

        {/* AQI Badge Card floating on top of map */}
        {location && observation && (
          <div className={`aqi-card-overlay ${drawerOpen ? 'is-drawer-open' : ''}`}>
            <AqiCard
              location={location}
              observation={observation}
              coverage={coverage}
              onShowWhy={handleShowWhy}
              loadingWhy={false}
            />
          </div>
        )}
      </main>

      {/* Real-time SSE Explanation Drawer */}
      <WhyDrawer
        isOpen={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        location={location}
        observation={observation}
        coverage={coverage}
        observationToken={observationToken}
        onSignalsReady={(data) => {
          if (location && observation) {
            setWhyData({
              location,
              observation,
              signals: data.signals || [],
              hypotheses: data.hypotheses || [],
              open_questions: data.open_questions || [],
              narrative: '',
              map_layers: data.map_layers,
              execution_trace: data.execution_trace,
              coverage: data.coverage
            });
          }
        }}
      />
    </div>
  );
};

export default App;
