import React, { useEffect, useRef, useState } from 'react';
import * as maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import type { LocationInfo, ObservationInfo, WhyResponse } from '../types/aqi';
import { MonitorX, RefreshCw } from 'lucide-react';

interface MapViewProps {
  location: LocationInfo | null;
  observation: ObservationInfo | null;
  whyData?: WhyResponse | null;
  onMapClick: (lat: number, lon: number) => void;
}

const checkWebGlSupport = (): boolean => {
  try {
    const canvas = document.createElement('canvas');
    return !!(window.WebGL2RenderingContext && canvas.getContext('webgl2'));
  } catch (e) {
    return false;
  }
};

export const MapView: React.FC<MapViewProps> = ({ location, observation, whyData, onMapClick }) => {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapInstance = useRef<maplibregl.Map | null>(null);
  const markerRef = useRef<maplibregl.Marker | null>(null);
  const fireMarkersRef = useRef<maplibregl.Marker[]>([]);

  const [webGlSupported, setWebGlSupported] = useState<boolean>(true);

  useEffect(() => {
    if (!mapContainer.current || mapInstance.current) return;

    if (!checkWebGlSupport()) {
      setWebGlSupported(false);
      return;
    }

    let resizeObserver: ResizeObserver | null = null;

    try {
      // Initialize MapLibre map centered on North America / US
      const map = new maplibregl.Map({
        container: mapContainer.current,
        style: {
          version: 8,
          sources: {
            'carto-dark': {
              type: 'raster',
              tiles: [
                'https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
                'https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
                'https://c.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png'
              ],
              tileSize: 256,
              attribution: '&copy; <a href="https://carto.com/">CARTO</a> &copy; <a href="https://openstreetmap.org">OpenStreetMap</a>'
            }
          },
          layers: [
            {
              id: 'carto-dark-layer',
              type: 'raster',
              source: 'carto-dark',
              minzoom: 0,
              maxzoom: 19
            }
          ]
        },
        center: [-98.5795, 39.8283], // Geographical center of US
        zoom: 4,
        dragPan: true,
        scrollZoom: true,
        renderWorldCopies: true
      });

      map.addControl(new maplibregl.NavigationControl(), 'top-right');

      map.on('click', (e: maplibregl.MapMouseEvent) => {
        const { lat, lng } = e.lngLat;
        onMapClick(lat, lng);
      });

      mapInstance.current = map;

      // Force canvas resize once map is loaded
      map.once('load', () => {
        try { map.resize(); } catch (_) {}
      });

      // ResizeObserver to handle container height changes dynamically (e.g. mobile Safari URL bar collapse/expand)
      if (typeof ResizeObserver !== 'undefined' && mapContainer.current) {
        resizeObserver = new ResizeObserver(() => {
          if (mapInstance.current) {
            try { mapInstance.current.resize(); } catch (_) {}
          }
        });
        resizeObserver.observe(mapContainer.current);
      }
    } catch (err) {
      console.warn("[MapLibre WebGL Init Warning]:", err);
      setWebGlSupported(false);
    }

    return () => {
      if (resizeObserver) {
        try { resizeObserver.disconnect(); } catch (_) {}
      }
      if (markerRef.current) {
        try { markerRef.current.remove(); } catch (e) {}
        markerRef.current = null;
      }
      if (mapInstance.current) {
        try {
          mapInstance.current.remove();
        } catch (e) {
          // Ignore WebGL context destruction error on failed init
        }
        mapInstance.current = null;
      }
    };
  }, []);

  // Update main location AQI pin
  useEffect(() => {
    if (!mapInstance.current || !location || !observation) return;

    const map = mapInstance.current;

    // Remove existing main marker to prevent leaks
    if (markerRef.current) {
      try { markerRef.current.remove(); } catch (e) {}
      markerRef.current = null;
    }

    // Create custom pin element
    const el = document.createElement('div');
    el.className = 'custom-aqi-pin';
    el.style.backgroundColor = observation.category_color;
    el.style.color = observation.category_text_color;
    el.innerHTML = `
      <div class="pin-inner">
        <span class="pin-aqi">${observation.aqi}</span>
      </div>
    `;

    // Stop propagation so clicking the pin doesn't trigger map onMapClick
    const stopProp = (e: Event) => e.stopPropagation();
    el.addEventListener('click', stopProp);
    el.addEventListener('touchstart', stopProp);

    markerRef.current = new maplibregl.Marker({ element: el })
      .setLngLat([location.lon, location.lat])
      .addTo(map);

    map.flyTo({
      center: [location.lon, location.lat],
      zoom: 8,
      duration: 1200
    });
  }, [location, observation]);

  // Update FIRMS hotspots overlay markers when whyData loads
  useEffect(() => {
    if (!mapInstance.current) return;
    const map = mapInstance.current;

    // Clear existing fire markers
    fireMarkersRef.current.forEach(m => {
      try { m.remove(); } catch (e) {}
    });
    fireMarkersRef.current = [];

    const hotspots = whyData?.map_layers?.firms_hotspots;
    if (hotspots && hotspots.length > 0) {
      hotspots.forEach(spot => {
        const fireEl = document.createElement('div');
        fireEl.className = 'fire-hotspot-marker';
        fireEl.title = `NASA FIRMS Hotspot (${spot.distance_km ? spot.distance_km + ' km away' : 'upwind'})`;
        fireEl.innerHTML = '🔥';
        fireEl.style.fontSize = '18px';
        fireEl.style.cursor = 'pointer';

        // Stop propagation so clicking a fire hotspot doesn't trigger map onMapClick
        const stopProp = (e: Event) => e.stopPropagation();
        fireEl.addEventListener('click', stopProp);
        fireEl.addEventListener('touchstart', stopProp);

        const marker = new maplibregl.Marker({ element: fireEl })
          .setLngLat([spot.lon, spot.lat])
          .addTo(map);
        
        fireMarkersRef.current.push(marker);
      });
    }
  }, [whyData]);

  if (!webGlSupported) {
    return (
      <div className="webgl-fallback-container">
        <div className="webgl-fallback-card">
          <MonitorX size={38} className="text-zinc-400" />
          <h3 className="fallback-title">WebGL is Not Supported</h3>
          <p className="fallback-desc">
            Interactive map rendering requires WebGL. Please enable hardware acceleration in your browser settings or try a different browser.
          </p>
          <button onClick={() => window.location.reload()} className="reload-btn">
            <RefreshCw size={15} />
            <span>Retry</span>
          </button>
        </div>
      </div>
    );
  }

  return <div ref={mapContainer} className="map-viewport" />;
};
