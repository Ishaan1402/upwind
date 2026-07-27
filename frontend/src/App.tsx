import React, { useState, useEffect } from 'react';
import type { LocationInfo, ObservationInfo, WhyResponse } from './types/aqi';
import { Wind } from 'lucide-react';
import { SearchBar } from './components/SearchBar';
import { MapView } from './components/MapView';
import { AqiCard } from './components/AqiCard';
import { WhyDrawer } from './components/WhyDrawer';
import { fetchAqiData, fetchAqiByCoords } from './services/api';
import { getRecents, pushRecent, type RecentPlace } from './services/recentPlaces';

export const App: React.FC = () => {
  const [location, setLocation] = useState<LocationInfo | null>(null);
  const [observation, setObservation] = useState<ObservationInfo | null>(null);
  const [whyData, setWhyData] = useState<WhyResponse | null>(null);
  const [loadingAqi, setLoadingAqi] = useState<boolean>(false);
  const [errorAqi, setErrorAqi] = useState<string | null>(null);
  const [recents, setRecents] = useState<RecentPlace[]>(getRecents());

  const [drawerOpen, setDrawerOpen] = useState<boolean>(false);

  // Load default location on initial mount (e.g. 90210 Beverly Hills / Los Angeles)
  useEffect(() => {
    handleSearch('90210');
  }, []);

  // Auto-clear error toast after 4 seconds
  useEffect(() => {
    if (errorAqi) {
      const timer = setTimeout(() => setErrorAqi(null), 4000);
      return () => clearTimeout(timer);
    }
  }, [errorAqi]);

  const handleSearch = async (query: string) => {
    setLoadingAqi(true);
    setErrorAqi(null);
    try {
      const res = await fetchAqiData(query);
      setLocation(res.location);
      setObservation(res.observation);
      setWhyData(null);
      const updated = pushRecent({
        name: res.location.name,
        lat: res.location.lat,
        lon: res.location.lon,
        zip_code: res.location.zip_code,
        query
      });
      setRecents(updated);
    } catch (err: any) {
      setErrorAqi(err.message || 'Upwind currently covers US states & territories only.');
    } finally {
      setLoadingAqi(false);
    }
  };

  const handleMapClick = async (lat: number, lon: number) => {
    setLoadingAqi(true);
    setErrorAqi(null);
    try {
      const res = await fetchAqiByCoords(lat, lon);
      setLocation(res.location);
      setObservation(res.observation);
      setWhyData(null);
      const updated = pushRecent({
        name: res.location.name,
        lat: res.location.lat,
        lon: res.location.lon,
        zip_code: res.location.zip_code
      });
      setRecents(updated);
    } catch (err: any) {
      setErrorAqi(err.message || 'Upwind currently covers US states & territories only.');
    } finally {
      setLoadingAqi(false);
    }
  };

  const handleShowWhy = () => {
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

        <div className="search-section flex flex-col gap-1">
          <SearchBar
            onSearch={handleSearch}
            onLocate={handleMapClick}
            onError={(msg) => setErrorAqi(msg)}
            loading={loadingAqi}
          />
          {recents.length > 0 && (
            <div className="recent-chips-container flex items-center gap-1 overflow-x-auto py-0.5">
              <span className="recent-label text-zinc-500 text-xs shrink-0 mr-1" style={{ fontSize: '0.7rem' }}>Recents:</span>
              {recents.map((r, i) => (
                <button
                  key={i}
                  onClick={() => r.zip_code ? handleSearch(r.zip_code) : handleMapClick(r.lat, r.lon)}
                  className="recent-chip px-2 py-0.5 rounded text-xs text-zinc-300 bg-zinc-800/80 hover:bg-zinc-700/80 border border-zinc-700/50 shrink-0 transition-colors cursor-pointer"
                  style={{ fontSize: '0.72rem' }}
                >
                  {r.name.split(',')[0]}
                </button>
              ))}
            </div>
          )}
        </div>
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
          <div className="aqi-card-overlay">
            <AqiCard
              location={location}
              observation={observation}
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
              execution_trace: data.execution_trace
            });
          }
        }}
      />
    </div>
  );
};

export default App;
