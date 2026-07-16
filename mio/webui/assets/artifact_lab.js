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
  const MAX_ATLAS_POSITIONS = 64;
  const MAX_ATLAS_PHASES = 24;
  const SPECULATIVE_ATLAS_TYPE = 'application/vnd.pimio.speculative-acceptance-atlas+json';
  const SPECULATIVE_ATLAS_ALIASES = Object.freeze([
    'application/vnd.pimio.speculative-atlas+json',
    'application/vnd.pimio.acceptance-atlas+json',
  ]);
  const SPECULATIVE_ATLAS_SCHEMA = Object.freeze({
    id: 'pimio.speculative-acceptance-atlas',
    version: 1,
    mime: SPECULATIVE_ATLAS_TYPE,
    aliases: SPECULATIVE_ATLAS_ALIASES,
  });

  // A copyable, explicitly synthetic example doubles as the executable schema
  // fixture. It demonstrates the contract and must not be cited as a benchmark.
  // Consumers receive a serialized clone through artifactLab.sample().
  const SPECULATIVE_ATLAS_SAMPLE = Object.freeze({
    schema: SPECULATIVE_ATLAS_SCHEMA.id,
    version: SPECULATIVE_ATLAS_SCHEMA.version,
    title: 'Illustrative mixture of drafters · schema example',
    subtitle: 'Synthetic values for renderer validation — replace every metric with a matched Mio benchmark before drawing conclusions.',
    baseline: Object.freeze({
      label: 'Base decode', prefill_tps: 612.4, decode_tps: 31.8, peak_memory_gb: 18.6,
    }),
    candidate: Object.freeze({
      label: 'Adaptive dFlash + dSpark', prefill_tps: 649.7, decode_tps: 57.9, peak_memory_gb: 20.1,
    }),
    positions: Object.freeze([
      Object.freeze({ position: 1, acceptance: 0.91, samples: 4096 }),
      Object.freeze({ position: 2, acceptance: 0.82, samples: 3728 }),
      Object.freeze({ position: 3, acceptance: 0.71, samples: 3054 }),
      Object.freeze({ position: 4, acceptance: 0.57, samples: 2188 }),
      Object.freeze({ position: 5, acceptance: 0.42, samples: 1320 }),
      Object.freeze({ position: 6, acceptance: 0.29, samples: 714 }),
    ]),
    phases: Object.freeze([
      Object.freeze({ name: 'Short context', from_token: 0, to_token: 2048, acceptance: 0.78, speedup: 1.91, memory_gb: 19.2, samples: 36 }),
      Object.freeze({ name: 'Long context', from_token: 2049, to_token: 8192, acceptance: 0.68, speedup: 1.63, memory_gb: 20.1, samples: 36 }),
      Object.freeze({ name: 'Tool-heavy turns', from_token: 0, to_token: 8192, acceptance: 0.61, speedup: 1.48, memory_gb: 19.8, samples: 24 }),
    ]),
    reliability: Object.freeze({
      runs: 96, confidence: 0.95, speedup_ci: Object.freeze([1.72, 1.91]), regression_rate: 0.031,
    }),
    decision: Object.freeze({
      status: 'promote',
      rationale: 'Illustrative rule: promote only when every measured phase stays positive and the lower confidence bound clears the configured threshold.',
    }),
  });

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
    Object.freeze({
      type: SPECULATIVE_ATLAS_TYPE,
      aliases: SPECULATIVE_ATLAS_ALIASES,
      label: 'Speculative acceptance atlas',
      description: 'Decision-grade speedup, acceptance depth, phase behavior, memory cost, and confidence.',
      category: 'MLX research',
    }),
  ]);

  const BY_TYPE = new Map(CATALOG.map((entry) => [entry.type, entry]));
  const TYPE_ALIASES = new Map();
  for (const entry of CATALOG) {
    TYPE_ALIASES.set(entry.type, entry.type);
    for (const alias of entry.aliases || []) TYPE_ALIASES.set(alias, entry.type);
  }

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

  function strictFinite(value) {
    if (typeof value === 'number') return Number.isFinite(value) ? value : null;
    if (typeof value !== 'string' || !value.trim()) return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function positive(value, label, { allowZero = false } = {}) {
    const number = strictFinite(value);
    if (number === null || (allowZero ? number < 0 : number <= 0)) {
      throw new Error(`${label} must be ${allowZero ? 'non-negative' : 'greater than zero'}.`);
    }
    return number;
  }

  function fraction(value, label) {
    const number = strictFinite(value);
    if (number === null || number < 0 || number > 1) {
      throw new Error(`${label} must be between 0 and 1.`);
    }
    return number;
  }

  function positiveInteger(value, label) {
    const number = strictFinite(value);
    if (!Number.isInteger(number) || number <= 0) throw new Error(`${label} must be a positive integer.`);
    return number;
  }

  function nonNegativeInteger(value, label) {
    const number = strictFinite(value);
    if (!Number.isInteger(number) || number < 0) throw new Error(`${label} must be a non-negative integer.`);
    return number;
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

  function renderAcceptanceAtlas(root, payload) {
    if (payload.schema !== SPECULATIVE_ATLAS_SCHEMA.id || payload.version !== SPECULATIVE_ATLAS_SCHEMA.version) {
      throw new Error(`Acceptance atlas requires schema ${SPECULATIVE_ATLAS_SCHEMA.id} version ${SPECULATIVE_ATLAS_SCHEMA.version}.`);
    }
    if (!payload.baseline || typeof payload.baseline !== 'object' || Array.isArray(payload.baseline)) {
      throw new Error('Acceptance atlas requires a baseline object.');
    }
    if (!payload.candidate || typeof payload.candidate !== 'object' || Array.isArray(payload.candidate)) {
      throw new Error('Acceptance atlas requires a candidate object.');
    }

    const baseline = {
      label: boundedText(payload.baseline.label, 'Baseline', 80),
      prefill: positive(payload.baseline.prefill_tps, 'baseline.prefill_tps'),
      decode: positive(payload.baseline.decode_tps, 'baseline.decode_tps'),
      memory: positive(payload.baseline.peak_memory_gb, 'baseline.peak_memory_gb', { allowZero: true }),
    };
    const candidate = {
      label: boundedText(payload.candidate.label, 'Candidate', 80),
      prefill: positive(payload.candidate.prefill_tps, 'candidate.prefill_tps'),
      decode: positive(payload.candidate.decode_tps, 'candidate.decode_tps'),
      memory: positive(payload.candidate.peak_memory_gb, 'candidate.peak_memory_gb', { allowZero: true }),
    };

    if (!Array.isArray(payload.positions) || !payload.positions.length) {
      throw new Error('Acceptance atlas requires a non-empty positions array.');
    }
    if (payload.positions.length > MAX_ATLAS_POSITIONS) {
      throw new Error(`At most ${MAX_ATLAS_POSITIONS} draft positions are supported.`);
    }
    const seenPositions = new Set();
    const positions = payload.positions.map((raw, index) => {
      if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
        throw new Error(`Position ${index + 1} must be an object.`);
      }
      const position = positiveInteger(raw.position, `positions[${index}].position`);
      if (seenPositions.has(position)) throw new Error(`Draft position ${position} is duplicated.`);
      seenPositions.add(position);
      return {
        position,
        acceptance: fraction(raw.acceptance, `positions[${index}].acceptance`),
        samples: positiveInteger(raw.samples, `positions[${index}].samples`),
      };
    }).sort((left, right) => left.position - right.position);

    if (!Array.isArray(payload.phases) || !payload.phases.length) {
      throw new Error('Acceptance atlas requires a non-empty phases array.');
    }
    if (payload.phases.length > MAX_ATLAS_PHASES) {
      throw new Error(`At most ${MAX_ATLAS_PHASES} phases are supported.`);
    }
    const phases = payload.phases.map((raw, index) => {
      if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
        throw new Error(`Phase ${index + 1} must be an object.`);
      }
      const from = nonNegativeInteger(raw.from_token, `phases[${index}].from_token`);
      const to = nonNegativeInteger(raw.to_token, `phases[${index}].to_token`);
      if (to < from) throw new Error(`phases[${index}].to_token must not precede from_token.`);
      return {
        name: boundedText(raw.name, `Phase ${index + 1}`, 80),
        from,
        to,
        acceptance: fraction(raw.acceptance, `phases[${index}].acceptance`),
        speedup: positive(raw.speedup, `phases[${index}].speedup`),
        memory: positive(raw.memory_gb, `phases[${index}].memory_gb`, { allowZero: true }),
        samples: positiveInteger(raw.samples, `phases[${index}].samples`),
      };
    });

    const evidence = payload.reliability;
    if (!evidence || typeof evidence !== 'object' || Array.isArray(evidence)) {
      throw new Error('Acceptance atlas requires a reliability object.');
    }
    const runs = positiveInteger(evidence.runs, 'reliability.runs');
    const confidence = fraction(evidence.confidence, 'reliability.confidence');
    const regressionRate = fraction(evidence.regression_rate, 'reliability.regression_rate');
    if (!Array.isArray(evidence.speedup_ci) || evidence.speedup_ci.length !== 2) {
      throw new Error('reliability.speedup_ci must contain lower and upper bounds.');
    }
    const confidenceLow = positive(evidence.speedup_ci[0], 'reliability.speedup_ci[0]');
    const confidenceHigh = positive(evidence.speedup_ci[1], 'reliability.speedup_ci[1]');
    if (confidenceHigh < confidenceLow) throw new Error('The speedup confidence interval is reversed.');

    const rawDecision = payload.decision;
    if (!rawDecision || typeof rawDecision !== 'object' || Array.isArray(rawDecision)) {
      throw new Error('Acceptance atlas requires a decision object.');
    }
    const allowedDecisions = new Set(['promote', 'hold', 'reject', 'collect-more']);
    const decision = String(rawDecision.status || '').trim().toLowerCase();
    if (!allowedDecisions.has(decision)) {
      throw new Error('decision.status must be promote, hold, reject, or collect-more.');
    }

    const decodeSpeedup = candidate.decode / baseline.decode;
    const prefillSpeedup = candidate.prefill / baseline.prefill;
    const memoryDelta = candidate.memory - baseline.memory;
    const weightedAcceptance = positions.reduce((total, point) => total + point.acceptance * point.samples, 0)
      / positions.reduce((total, point) => total + point.samples, 0);

    header(
      root,
      `Speculative acceptance atlas · schema v${SPECULATIVE_ATLAS_SCHEMA.version}`,
      boundedText(payload.title, `${candidate.label} against ${baseline.label}`, 140),
      boundedText(payload.subtitle, 'Measured draft acceptance and serving trade-offs.', 300),
    );

    const summary = element('section', 'mio-lab-summary');
    summary.setAttribute('aria-label', 'Speculative decoding highlights');
    summary.append(
      summaryCard('Decode speedup', `${decodeSpeedup.toFixed(2)}×`, `${baseline.label} → ${candidate.label}`),
      summaryCard('Prefill speedup', `${prefillSpeedup.toFixed(2)}×`, `${metric(candidate.prefill, 'tok/s')} candidate`),
      summaryCard('Weighted acceptance', metric(weightedAcceptance * 100, '%'), `${positions.length} draft positions`),
      summaryCard('Peak memory delta', `${memoryDelta >= 0 ? '+' : ''}${memoryDelta.toFixed(1)} GB`, `${metric(candidate.memory, 'GB')} peak`),
    );
    root.append(summary);

    const decisionPanel = element('section', `mio-lab-atlas-decision is-${decision}`);
    decisionPanel.setAttribute('aria-label', 'Experiment decision');
    const decisionCopy = element('div', 'mio-lab-atlas-decision-copy');
    decisionCopy.append(
      element('span', 'mio-lab-atlas-decision-label', 'Evidence decision'),
      element('strong', 'mio-lab-atlas-decision-value', decision.replace('-', ' ')),
    );
    const rationale = element('p', '', boundedText(rawDecision.rationale, 'No rationale supplied.', 420));
    decisionPanel.append(decisionCopy, rationale);
    root.append(decisionPanel);

    const positionSection = element('section', 'mio-lab-atlas-section');
    const positionHead = element('div', 'mio-lab-atlas-section-head');
    positionHead.append(
      element('h3', '', 'Acceptance by draft position'),
      element('p', '', 'Depth decay reveals where verification work stops paying back.'),
    );
    const positionChart = element('div', 'mio-lab-atlas-positions');
    positionChart.setAttribute('role', 'list');
    positionChart.setAttribute('aria-label', 'Acceptance ratio by draft position');
    for (const point of positions) {
      const percent = point.acceptance * 100;
      const item = element('div', 'mio-lab-atlas-position');
      item.setAttribute('role', 'listitem');
      item.setAttribute('aria-label', `Position ${point.position}: ${percent.toFixed(1)} percent acceptance across ${point.samples} samples`);
      const value = element('strong', '', `${percent.toFixed(1)}%`);
      const track = element('span', 'mio-lab-atlas-position-track');
      const bar = element('span', `mio-lab-atlas-position-bar ${percent >= 70 ? 'is-high' : percent >= 45 ? 'is-mid' : 'is-low'}`);
      bar.style.height = `${percent}%`;
      track.append(bar);
      item.append(value, track, element('span', '', `P${point.position}`), element('small', '', `${point.samples} n`));
      positionChart.append(item);
    }
    positionSection.append(positionHead, positionChart);
    root.append(positionSection);

    const phaseSection = element('section', 'mio-lab-atlas-section');
    const phaseHead = element('div', 'mio-lab-atlas-section-head');
    phaseHead.append(
      element('h3', '', 'Phase robustness'),
      element('p', '', 'Matched slices prevent a fast aggregate from hiding a weak workload regime.'),
    );
    const phaseTable = element('div', 'mio-lab-atlas-phases');
    phaseTable.setAttribute('role', 'table');
    phaseTable.setAttribute('aria-label', 'Acceptance atlas phases');
    const phaseHeader = element('div', 'mio-lab-atlas-phase is-header');
    phaseHeader.setAttribute('role', 'row');
    for (const label of ['Phase', 'Token range', 'Acceptance', 'Speedup', 'Memory', 'Samples']) {
      const cell = element('span', '', label);
      cell.setAttribute('role', 'columnheader');
      phaseHeader.append(cell);
    }
    phaseTable.append(phaseHeader);
    for (const phase of phases) {
      const row = element('div', 'mio-lab-atlas-phase');
      row.setAttribute('role', 'row');
      for (const value of [
        phase.name,
        `${phase.from}–${phase.to}`,
        metric(phase.acceptance * 100, '%'),
        `${phase.speedup.toFixed(2)}×`,
        metric(phase.memory, 'GB'),
        String(phase.samples),
      ]) {
        const cell = element('span', '', value);
        cell.setAttribute('role', 'cell');
        row.append(cell);
      }
      phaseTable.append(row);
    }
    phaseSection.append(phaseHead, phaseTable);
    root.append(phaseSection);

    const reliability = element('section', 'mio-lab-atlas-reliability');
    reliability.setAttribute('aria-label', 'Reliability and uncertainty');
    const reliabilityCopy = element('div', 'mio-lab-atlas-section-head');
    reliabilityCopy.append(
      element('h3', '', 'Reliability envelope'),
      element('p', '', 'Speedup uncertainty and regression frequency travel with the headline result.'),
    );
    const reliabilityGrid = element('dl', 'mio-lab-atlas-reliability-grid');
    for (const [label, value] of [
      ['Confidence', metric(confidence * 100, '%')],
      ['Speedup interval', `${confidenceLow.toFixed(2)}× – ${confidenceHigh.toFixed(2)}×`],
      ['Matched runs', String(runs)],
      ['Regression rate', metric(regressionRate * 100, '%')],
    ]) {
      const fact = element('div', 'mio-lab-atlas-reliability-fact');
      fact.append(element('dt', '', label), element('dd', '', value));
      reliabilityGrid.append(fact);
    }
    reliability.append(reliabilityCopy, reliabilityGrid);
    root.append(reliability);
  }

  const RENDERERS = Object.freeze({
    'application/vnd.pimio.benchmark+json': renderBenchmark,
    'application/vnd.pimio.model-card+json': renderModelCard,
    'application/vnd.pimio.inference-trace+json': renderTrace,
    [SPECULATIVE_ATLAS_TYPE]: renderAcceptanceAtlas,
  });

  function normalizeType(type) {
    return TYPE_ALIASES.get(String(type || '').trim().toLowerCase()) || String(type || '').trim().toLowerCase();
  }

  function render(body, artifact) {
    const type = normalizeType(artifact?.type);
    const renderer = RENDERERS[type];
    if (!renderer) return false;
    const root = element('article', 'mio-lab-artifact');
    root.dataset.artifactType = type;
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
    if (!BY_TYPE.has(normalizeType(artifact?.type))) return null;
    let content = String(artifact.content || '');
    try { content = JSON.stringify(parsePayload(content), null, 2) + '\n'; } catch (_) {}
    return { content, extension: '.json', mime: 'application/json' };
  }

  if (Mio.artifactTypes?.register) {
    for (const entry of CATALOG) {
      Mio.artifactTypes.register({
        ...entry,
        render,
        download,
      });
    }
  }

  // Compatibility facade for older Mio modules. New consumers use the
  // shared artifactTypes registry so label/render/download cannot diverge.
  Mio.artifactLab = Object.freeze({
    catalog: () => CATALOG.slice(),
    download,
    label: (type) => BY_TYPE.get(normalizeType(type))?.label || '',
    render,
    sample: (type) => normalizeType(type) === SPECULATIVE_ATLAS_TYPE
      ? JSON.stringify(SPECULATIVE_ATLAS_SAMPLE, null, 2) + '\n'
      : '',
    schema: (type) => normalizeType(type) === SPECULATIVE_ATLAS_TYPE ? SPECULATIVE_ATLAS_SCHEMA : null,
    supports: (type) => BY_TYPE.has(normalizeType(type)),
  });
})();
