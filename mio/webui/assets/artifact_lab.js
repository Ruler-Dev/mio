/* Native, MLX-specific Mio artifacts.
 *
 * These renderers deliberately avoid remote libraries and executable HTML.
 * Payload values enter the parent document only through textContent.
 */
(function () {
  'use strict';

  const Mio = (window.Mio = window.Mio || {});
  const MAX_BYTES = 512 * 1024;
  const MAX_RUNS = 48;
  const MAX_SPANS = 256;

  const CATALOG = Object.freeze([
    Object.freeze({
      type: 'application/vnd.pimio.benchmark+json',
      label: 'MLX benchmark comparison',
      description: 'Prefill, decode, TTFT, memory, and acceptance across matched runs.',
      category: 'MLX research',
    }),
    Object.freeze({
      type: 'application/vnd.pimio.model-card+json',
      label: 'Model compatibility card',
      description: 'Checkpoint identity, quantization, memory, context, and drafter support.',
      category: 'MLX research',
    }),
    Object.freeze({
      type: 'application/vnd.pimio.inference-trace+json',
      label: 'Inference trace',
      description: 'A stage-by-stage timing trace for load, prefill, draft, verify, and decode.',
      category: 'MLX research',
    }),
  ]);

  const BY_TYPE = new Map(CATALOG.map((entry) => [entry.type, entry]));

  function element(tag, className, text) {
    const value = document.createElement(tag);
    if (className) value.className = className;
    if (text !== undefined) value.textContent = String(text);
    return value;
  }

  function parsePayload(content) {
    const source = String(content || '');
    if (new TextEncoder().encode(source).length > MAX_BYTES) {
      throw new Error('Payload exceeds the 512 KiB artifact limit.');
    }
    let value;
    try { value = JSON.parse(source); }
    catch (_) { throw new Error('Payload must be valid JSON.'); }
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      throw new Error('Payload must be a JSON object.');
    }
    return value;
  }

  function boundedText(value, fallback = '—', limit = 160) {
    if (value === undefined || value === null || value === '') return fallback;
    return String(value).slice(0, limit);
  }

  function finite(value, fallback = null) {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  }

  function metric(value, unit, digits = 1) {
    const number = finite(value);
    return number === null ? '—' : `${number.toFixed(digits)}${unit ? ` ${unit}` : ''}`;
  }

  function header(root, eyebrow, title, subtitle) {
    const head = element('header', 'mio-lab-head');
    head.append(element('div', 'mio-lab-eyebrow', eyebrow));
    head.append(element('h2', 'mio-lab-title', title));
    if (subtitle) head.append(element('p', 'mio-lab-subtitle', subtitle));
    root.append(head);
  }

  function summaryCard(label, value, detail) {
    const card = element('div', 'mio-lab-summary-card');
    card.append(element('span', 'mio-lab-summary-label', label));
    card.append(element('strong', 'mio-lab-summary-value', value));
    if (detail) card.append(element('span', 'mio-lab-summary-detail', detail));
    return card;
  }

  function renderBenchmark(root, payload) {
    if (!Array.isArray(payload.runs) || !payload.runs.length) {
      throw new Error('MLX benchmark requires a non-empty runs array.');
    }
    if (payload.runs.length > MAX_RUNS) throw new Error(`At most ${MAX_RUNS} runs are supported.`);
    const runs = payload.runs.map((raw, index) => {
      if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
        throw new Error(`Run ${index + 1} must be an object.`);
      }
      return {
        label: boundedText(raw.label || raw.name, `Run ${index + 1}`, 80),
        prefill: finite(raw.prefill_tps ?? raw.prompt_tps),
        decode: finite(raw.decode_tps ?? raw.generation_tps),
        ttft: finite(raw.ttft_ms),
        memory: finite(raw.memory_gb ?? raw.peak_memory_gb),
        acceptance: finite(raw.acceptance ?? raw.acceptance_ratio),
      };
    });
    if (!runs.some((run) => run.prefill !== null || run.decode !== null || run.ttft !== null)) {
      throw new Error('Each benchmark needs at least one timing metric.');
    }

    header(
      root,
      'MLX benchmark',
      boundedText(payload.title, 'Matched run comparison', 120),
      boundedText(payload.subtitle || payload.workload, '', 240),
    );
    const bestDecode = runs.filter((run) => run.decode !== null).sort((a, b) => b.decode - a.decode)[0];
    const bestPrefill = runs.filter((run) => run.prefill !== null).sort((a, b) => b.prefill - a.prefill)[0];
    const bestTtft = runs.filter((run) => run.ttft !== null).sort((a, b) => a.ttft - b.ttft)[0];
    const summary = element('section', 'mio-lab-summary');
    summary.setAttribute('aria-label', 'Benchmark highlights');
    summary.append(
      summaryCard('Fastest decode', bestDecode ? metric(bestDecode.decode, 'tok/s') : '—', bestDecode?.label),
      summaryCard('Fastest prefill', bestPrefill ? metric(bestPrefill.prefill, 'tok/s') : '—', bestPrefill?.label),
      summaryCard('Lowest TTFT', bestTtft ? metric(bestTtft.ttft, 'ms') : '—', bestTtft?.label),
      summaryCard('Compared runs', String(runs.length), boundedText(payload.device, 'local device', 80)),
    );
    root.append(summary);

    const maxDecode = Math.max(1, ...runs.map((run) => run.decode || 0));
    const maxPrefill = Math.max(1, ...runs.map((run) => run.prefill || 0));
    const table = element('div', 'mio-lab-run-table');
    table.setAttribute('role', 'table');
    table.setAttribute('aria-label', 'MLX benchmark runs');
    for (const run of runs) {
      const row = element('div', 'mio-lab-run');
      row.setAttribute('role', 'row');
      const name = element('div', 'mio-lab-run-name', run.label);
      name.setAttribute('role', 'rowheader');
      const bars = element('div', 'mio-lab-bars');
      for (const [label, value, maximum, tone] of [
        ['Decode', run.decode, maxDecode, 'decode'],
        ['Prefill', run.prefill, maxPrefill, 'prefill'],
      ]) {
        const line = element('div', 'mio-lab-bar-line');
        line.append(element('span', 'mio-lab-bar-label', label));
        const track = element('span', 'mio-lab-bar-track');
        const fill = element('span', `mio-lab-bar-fill ${tone}`);
        fill.style.width = `${Math.max(0, Math.min(100, ((value || 0) / maximum) * 100))}%`;
        track.append(fill);
        line.append(track, element('span', 'mio-lab-bar-value', metric(value, 'tok/s')));
        bars.append(line);
      }
      const facts = element('div', 'mio-lab-run-facts');
      facts.append(
        element('span', '', `TTFT ${metric(run.ttft, 'ms')}`),
        element('span', '', `Memory ${metric(run.memory, 'GB')}`),
        element('span', '', `Accept ${run.acceptance === null ? '—' : metric(run.acceptance <= 1 ? run.acceptance * 100 : run.acceptance, '%')}`),
      );
      row.append(name, bars, facts);
      table.append(row);
    }
    root.append(table);
  }

  function renderModelCard(root, payload) {
    const name = boundedText(payload.name || payload.model, '', 160);
    if (!name) throw new Error('Model card requires name or model.');
    header(root, 'Model card', name, boundedText(payload.description, '', 320));
    const grid = element('dl', 'mio-lab-fact-grid');
    const facts = [
      ['Family', payload.family], ['Parameters', payload.parameters],
      ['Quantization', payload.quantization], ['Format', payload.format || 'MLX'],
      ['Context', payload.context_window ? `${payload.context_window} tokens` : null],
      ['Checkpoint size', payload.size_gb ? `${payload.size_gb} GB` : null],
      ['Peak memory', payload.memory_gb ? `${payload.memory_gb} GB` : null],
      ['Revision', payload.revision],
    ];
    for (const [label, value] of facts) {
      const item = element('div', 'mio-lab-fact');
      item.append(element('dt', '', label), element('dd', '', boundedText(value)));
      grid.append(item);
    }
    root.append(grid);
    const tags = [];
    for (const value of [...(Array.isArray(payload.features) ? payload.features : []), ...(Array.isArray(payload.drafters) ? payload.drafters : [])]) {
      if (tags.length >= 24) break;
      tags.push(boundedText(value, '', 64));
    }
    if (tags.length) {
      const list = element('div', 'mio-lab-tags');
      list.setAttribute('aria-label', 'Features and compatible drafters');
      tags.forEach((tag) => list.append(element('span', 'mio-lab-tag', tag)));
      root.append(list);
    }
    if (payload.sha256 || payload.source) {
      const provenance = element('section', 'mio-lab-provenance');
      provenance.append(element('h3', '', 'Provenance'));
      if (payload.source) provenance.append(element('p', '', boundedText(payload.source, '', 300)));
      if (payload.sha256) provenance.append(element('code', '', boundedText(payload.sha256, '', 128)));
      root.append(provenance);
    }
  }

  function renderTrace(root, payload) {
    if (!Array.isArray(payload.spans) || !payload.spans.length) {
      throw new Error('Inference trace requires a non-empty spans array.');
    }
    if (payload.spans.length > MAX_SPANS) throw new Error(`At most ${MAX_SPANS} spans are supported.`);
    const spans = payload.spans.map((raw, index) => {
      const start = finite(raw?.start_ms, 0);
      const duration = finite(raw?.duration_ms);
      if (!raw || duration === null || start < 0 || duration < 0) {
        throw new Error(`Span ${index + 1} needs non-negative start_ms and duration_ms.`);
      }
      return {
        name: boundedText(raw.name, `Span ${index + 1}`, 80), start, duration,
        category: boundedText(raw.category, 'other', 32).toLowerCase().replace(/[^a-z0-9-]/g, ''),
        detail: boundedText(raw.detail, '', 180),
      };
    }).sort((a, b) => a.start - b.start);
    const total = Math.max(1, finite(payload.total_ms, 0), ...spans.map((span) => span.start + span.duration));
    header(root, 'Inference trace', boundedText(payload.title, 'Request timeline', 120), `${metric(total, 'ms')} total · ${spans.length} stages`);
    const timeline = element('div', 'mio-lab-trace');
    timeline.setAttribute('role', 'list');
    for (const span of spans) {
      const row = element('div', 'mio-lab-trace-row');
      row.setAttribute('role', 'listitem');
      const copy = element('div', 'mio-lab-trace-copy');
      copy.append(element('strong', '', span.name), element('span', '', span.detail || span.category));
      const track = element('div', 'mio-lab-trace-track');
      const bar = element('div', `mio-lab-trace-bar ${span.category}`);
      bar.style.left = `${Math.min(100, (span.start / total) * 100)}%`;
      bar.style.width = `${Math.max(0.5, Math.min(100 - (span.start / total) * 100, (span.duration / total) * 100))}%`;
      bar.title = `${span.name}: ${metric(span.duration, 'ms')}`;
      track.append(bar);
      row.append(copy, track, element('code', 'mio-lab-trace-time', metric(span.duration, 'ms')));
      timeline.append(row);
    }
    root.append(timeline);
  }

  const RENDERERS = Object.freeze({
    'application/vnd.pimio.benchmark+json': renderBenchmark,
    'application/vnd.pimio.model-card+json': renderModelCard,
    'application/vnd.pimio.inference-trace+json': renderTrace,
  });

  function render(body, artifact) {
    const renderer = RENDERERS[artifact?.type];
    if (!renderer) return false;
    const root = element('article', 'mio-lab-artifact');
    root.dataset.artifactType = artifact.type;
    try { renderer(root, parsePayload(artifact.content)); }
    catch (error) {
      root.classList.add('has-error');
      header(root, 'Artifact error', boundedText(artifact.title, 'Cannot render artifact'), 'Open Source to inspect and edit the JSON payload.');
      root.append(element('p', 'mio-lab-error', boundedText(error.message, 'Invalid artifact payload.', 300)));
    }
    body.append(root);
    return true;
  }

  function download(artifact) {
    if (!BY_TYPE.has(artifact?.type)) return null;
    let content = String(artifact.content || '');
    try { content = JSON.stringify(parsePayload(content), null, 2) + '\n'; } catch (_) {}
    return { content, extension: '.json', mime: 'application/json' };
  }

  Mio.artifactLab = Object.freeze({
    catalog: () => CATALOG.slice(),
    download,
    label: (type) => BY_TYPE.get(type)?.label || '',
    render,
    supports: (type) => BY_TYPE.has(type),
  });
})();
