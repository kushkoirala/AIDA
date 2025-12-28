export async function fetchRunwayConfig() {
  try {
    const res = await fetch('./runway_config.json', { cache: 'no-cache' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error('[viewer] failed to load runway_config.json:', err);
    // Minimal fallback to keep viewer running
    return {
      runway: { start: [0, 0, 0], length: 150, width: 12 },
      property: { size: 400 },
    };
  }
}
