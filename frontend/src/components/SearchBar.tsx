import React, { useState, useEffect } from 'react';
import { Search, Loader2, Locate } from 'lucide-react';

interface SearchBarProps {
  onSearch: (query: string) => void;
  onLocate?: (lat: number, lon: number) => void;
  onError?: (msg: string) => void;
  loading: boolean;
  displayQuery?: string;
}

export const SearchBar: React.FC<SearchBarProps> = ({ onSearch, onLocate, onError, loading, displayQuery }) => {
  const [term, setTerm] = useState('');
  const [locating, setLocating] = useState(false);

  useEffect(() => {
    if (displayQuery !== undefined) {
      setTerm(displayQuery);
    }
  }, [displayQuery]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (term.trim() && !loading && !locating) {
      onSearch(term.trim());
    }
  };

  const handleLocateClick = () => {
    if (!navigator.geolocation) {
      onError?.('Geolocation is not supported by your browser.');
      return;
    }
    setLocating(true);
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setLocating(false);
        onLocate?.(pos.coords.latitude, pos.coords.longitude);
      },
      (err) => {
        setLocating(false);
        onError?.(err.message || 'Unable to access your location. Please check browser permissions.');
      },
      { timeout: 10000, enableHighAccuracy: false }
    );
  };

  return (
    <form onSubmit={handleSubmit} className="search-bar-container">
      <div className="search-input-wrapper">
        <Search className="search-icon" size={18} />
        <input
          type="text"
          value={term}
          onChange={(e) => setTerm(e.target.value)}
          placeholder="Enter US ZIP code or City, State"
          className="search-input"
        />
        {onLocate && (
          <button
            type="button"
            className="locate-btn"
            onClick={handleLocateClick}
            disabled={loading || locating}
            title="Use my location"
          >
            {locating ? <Loader2 className="animate-spin text-accent" size={16} /> : <Locate size={18} />}
          </button>
        )}
        <button type="submit" className="search-btn" disabled={loading || locating || !term.trim()}>
          {loading ? <Loader2 className="animate-spin" size={16} /> : 'Search'}
        </button>
      </div>
    </form>
  );
};
