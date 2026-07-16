/* Shared artifact capability registry.
 *
 * A registered type is a promise that Mio can actually render it. Aliases
 * resolve to one canonical MIME key, so every consumer (label, preview,
 * download, help, and future share/export paths) sees the same capability.
 */
(function () {
  'use strict';

  const Mio = (window.Mio = window.Mio || {});
  const MIME_RE = /^[a-z0-9][a-z0-9.+-]*\/[a-z0-9][a-z0-9.+-]*$/;
  const definitions = new Map();
  const aliases = new Map();

  function mime(value) {
    const normalized = String(value || '').trim().toLowerCase();
    if (!MIME_RE.test(normalized) || normalized.length > 128) {
      throw new TypeError(`Invalid artifact MIME type: ${normalized || '(empty)'}`);
    }
    return normalized;
  }

  function normalize(type) {
    let current;
    try { current = mime(type); }
    catch (_) { return String(type || '').trim().toLowerCase(); }
    const seen = new Set();
    while (aliases.has(current) && !seen.has(current)) {
      seen.add(current);
      const next = aliases.get(current);
      if (next === current) break;
      current = next;
    }
    return current;
  }

  function register(raw) {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
      throw new TypeError('Artifact definition must be an object.');
    }
    const type = mime(raw.type);
    if (definitions.has(type)) throw new TypeError(`Artifact type already registered: ${type}`);
    if (typeof raw.render !== 'function') {
      throw new TypeError(`Artifact type has no renderer: ${type}`);
    }
    const label = String(raw.label || '').trim();
    if (!label || label.length > 120) throw new TypeError(`Artifact type needs a bounded label: ${type}`);

    const definition = Object.freeze({
      type,
      label,
      description: String(raw.description || '').trim().slice(0, 300),
      category: String(raw.category || 'Other').trim().slice(0, 80),
      render: raw.render,
      download: typeof raw.download === 'function' ? raw.download : null,
      standalone: typeof raw.standalone === 'function' ? raw.standalone : null,
    });

    const claimed = [type, ...(Array.isArray(raw.aliases) ? raw.aliases : [])].map(mime);
    for (const alias of claimed) {
      const owner = aliases.get(alias);
      if (owner && owner !== type) throw new TypeError(`Artifact alias already registered: ${alias}`);
    }
    definitions.set(type, definition);
    for (const alias of claimed) aliases.set(alias, type);
    return definition;
  }

  function definition(type) {
    return definitions.get(normalize(type)) || null;
  }

  function render(body, artifact) {
    const resolved = definition(artifact?.type);
    if (!resolved) return false;
    const canonical = { ...artifact, type: resolved.type };
    return resolved.render(body, canonical) !== false;
  }

  function download(artifact) {
    const resolved = definition(artifact?.type);
    if (!resolved?.download) return null;
    return resolved.download({ ...artifact, type: resolved.type }) || null;
  }

  function catalog() {
    return Array.from(definitions.values(), (entry) => ({
      type: entry.type,
      label: entry.label,
      description: entry.description,
      category: entry.category,
      downloadable: Boolean(entry.download),
      standalone: Boolean(entry.standalone),
    })).sort((left, right) => left.type.localeCompare(right.type));
  }

  Mio.artifactTypes = Object.freeze({
    catalog,
    definition,
    download,
    label: (type) => definition(type)?.label || '',
    normalize,
    register,
    render,
    supports: (type) => Boolean(definition(type)),
  });
})();
