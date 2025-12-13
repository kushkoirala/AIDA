// Utility to fetch runway_config.json and return config object
export async function fetchRunwayConfig() {
  const resp = await fetch('/runway_config.json');
  if (!resp.ok) throw new Error('Failed to load runway_config.json');
  return await resp.json();
}
