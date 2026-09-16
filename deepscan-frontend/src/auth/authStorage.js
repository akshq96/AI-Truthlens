const STORAGE_KEY = 'aideepfake_auth_v1';

export function getAuth() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed.email === 'string' && parsed.email.length > 3) {
      return parsed;
    }
    return null;
  } catch {
    return null;
  }
}

export function setAuth(payload) {
  const safe = {
    email: payload?.email || '',
    createdAt: payload?.createdAt || new Date().toISOString(),
  };
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(safe));
  return safe;
}

export function clearAuth() {
  window.localStorage.removeItem(STORAGE_KEY);
}

