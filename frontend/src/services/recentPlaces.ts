export interface RecentPlace {
  name: string;
  lat: number;
  lon: number;
  zip_code?: string | null;
  query?: string;
}

const STORAGE_KEY = 'upwind_recent';
const MAX_RECENTS = 5;

export function getRecents(): RecentPlace[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      return parsed;
    }
  } catch (_) {}
  return [];
}

export function pushRecent(place: RecentPlace): RecentPlace[] {
  if (!place || (!place.name && (!place.lat || !place.lon))) return getRecents();
  try {
    const current = getRecents();
    // Filter out existing matching place by zip or lat,lon proximity
    const filtered = current.filter(p => {
      if (place.zip_code && p.zip_code === place.zip_code) return false;
      if (Math.abs(p.lat - place.lat) < 0.05 && Math.abs(p.lon - place.lon) < 0.05) return false;
      return true;
    });

    const updated = [place, ...filtered].slice(0, MAX_RECENTS);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    return updated;
  } catch (_) {
    return getRecents();
  }
}
