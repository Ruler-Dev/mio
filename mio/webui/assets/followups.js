// Smart follow-up suggestions — after each assistant reply, show 3
// contextual follow-up prompts as chips beneath the message. Heuristic
// picker: pattern-match on reply text and choose from a pool. No model
// call needed, so it's instantaneous.
(function () {
  const NS = (window.Mio = window.Mio || {});

  // Each rule: { match: RegExp, suggestions: [strings] }
  // First 3 matching rules' suggestions are pooled.
  const RULES = [
    { match: /\b(code|function|python|javascript|typescript|rust|go)\b/i,
      suggestions: [
        "Can you write tests for that?",
        "What are potential edge cases?",
        "Refactor for readability",
        "Add type annotations",
        "Explain the time complexity",
      ]},
    { match: /\b(pdf|report|document|letter|brochure|flyer|invoice)\b/i,
      suggestions: [
        "Make it more minimal",
        "Try a different color palette",
        "Use a warmer tone",
        "Export this as a Word doc too",
        "Add a cover page",
      ]},
    { match: /\b(chart|graph|plot|data|table)\b/i,
      suggestions: [
        "Break it down by region",
        "Convert this to a different chart type",
        "Export the data as CSV",
        "Highlight the outliers",
      ]},
    { match: /\b(weather|forecast|rain|temperature)\b/i,
      suggestions: [
        "Compare with another city",
        "What should I pack?",
        "When's the best time to go out?",
      ]},
    { match: /\b(recipe|ingredient|cup|tsp|tbsp|bake|cook)\b/i,
      suggestions: [
        "Scale this for 6 people",
        "Make a vegan version",
        "What wine pairs with this?",
        "Turn this into a weekly meal plan",
      ]},
    { match: /\b(anime|manga|film|movie|show|game)\b/i,
      suggestions: [
        "Recommend something similar",
        "What's the plot in 2 sentences?",
        "Who's the main character?",
        "Is there a sequel?",
      ]},
    { match: /\b(explain|how does|what is|why)\b/i,
      suggestions: [
        "Explain it like I'm 8",
        "Give me a real-world example",
        "What are the downsides?",
        "How does this compare to X?",
      ]},
    { match: /\b(search|news|latest|today)\b/i,
      suggestions: [
        "Summarize the key points",
        "What's the counter-argument?",
        "Dig deeper on the first result",
      ]},
    { match: /\b(3d|scene|render|visualize)\b/i,
      suggestions: [
        "Add lighting effects",
        "Make it interactive",
        "Try a different camera angle",
      ]},
    { match: /\b(error|bug|issue|failing)\b/i,
      suggestions: [
        "What's the minimal repro?",
        "How would I test the fix?",
        "Is this a regression?",
      ]},
  ];

  // Universal fallbacks if no rule matches
  const FALLBACK = [
    "Elaborate on the first point",
    "What should I do next?",
    "Can you give me an example?",
    "Compare this to the alternative",
    "Summarize in 3 bullets",
    "Where's the risk in this?",
  ];

  function pickSuggestions(replyText) {
    const pool = [];
    for (const r of RULES) {
      if (r.match.test(replyText)) {
        for (const s of r.suggestions) {
          if (!pool.includes(s)) pool.push(s);
        }
      }
    }
    if (pool.length < 3) {
      for (const s of FALLBACK) if (!pool.includes(s)) pool.push(s);
    }
    // Shuffle a copy, take 3
    const copy = pool.slice();
    for (let i = copy.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [copy[i], copy[j]] = [copy[j], copy[i]];
    }
    return copy.slice(0, 3);
  }

  function renderFollowups(messageEl, replyText) {
    if (!messageEl || !replyText) return;
    const body = messageEl.querySelector('.msg-body');
    if (!body || body.querySelector('.followups')) return;
    const suggestions = pickSuggestions(replyText);
    if (!suggestions.length) return;
    const chips = document.createElement('div');
    chips.className = 'followups';
    chips.innerHTML = suggestions.map(s =>
      `<button class="followup-chip" onclick="Mio.followups.send(this)">${escapeHTML(s)}</button>`
    ).join('');
    body.appendChild(chips);
  }

  function send(btn) {
    const text = btn.textContent;
    const input = document.getElementById('inputArea');
    if (!input) return;
    input.value = text;
    if (window.sendMessage) window.sendMessage();
  }

  function escapeHTML(s) {
    return String(s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  function injectCSS() {
    if (document.getElementById('followups-css')) return;
    const css = document.createElement('style');
    css.id = 'followups-css';
    css.textContent = `
      .followups { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
      .followup-chip { background: transparent; border: 1px solid var(--border); color: var(--text-secondary); padding: 6px 12px; border-radius: 99px; font-size: 12px; cursor: pointer; transition: all 120ms; }
      .followup-chip:hover { border-color: var(--accent); color: var(--accent); background: var(--accent-subtle, rgba(59,130,246,0.08)); }
    `;
    document.head.appendChild(css);
  }

  injectCSS();
  NS.followups = { render: renderFollowups, send, pick: pickSuggestions };
})();
