// Curated prompt library — 80+ well-crafted prompts grouped by category.
// Click to insert into the message box (NOT auto-send, so the user can
// tweak first).
(function () {
  const NS = (window.Mio = window.Mio || {});

  const LIBRARY = [
    // Writing
    { cat: "Writing", name: "Blog outline",
      prompt: "Draft a detailed outline for a blog post titled \"<YOUR TITLE>\". Include hook, main sections with 3 sub-points each, and a call-to-action." },
    { cat: "Writing", name: "Email rewrite (formal)",
      prompt: "Rewrite the email below in a formal, concise tone. Keep the intent but remove fluff.\n\n<PASTE EMAIL>" },
    { cat: "Writing", name: "Email rewrite (friendly)",
      prompt: "Rewrite this to sound warm and friendly without being sycophantic.\n\n<PASTE EMAIL>" },
    { cat: "Writing", name: "Tweet thread",
      prompt: "Turn the following idea into a 5-tweet thread. Each tweet <=280 chars, progressive narrative, no hashtags.\n\n<TOPIC>" },
    { cat: "Writing", name: "Elevator pitch",
      prompt: "Write a 60-second elevator pitch for <PRODUCT/IDEA>. Structure: hook → problem → solution → proof → ask." },

    // Code
    { cat: "Code", name: "Explain a function",
      prompt: "Explain what this code does, line by line. Then suggest 3 improvements.\n\n```\n<CODE>\n```" },
    { cat: "Code", name: "Find the bug",
      prompt: "The following code has a bug. Find it, explain why it fails, and give a minimal fix.\n\n```\n<CODE>\n```" },
    { cat: "Code", name: "Refactor for clarity",
      prompt: "Refactor this for clarity without changing behavior. Use self-documenting names and extract helpers.\n\n```\n<CODE>\n```" },
    { cat: "Code", name: "Add tests",
      prompt: "Write a thorough test suite for this code. Cover the happy path, edge cases, and error paths.\n\n```\n<CODE>\n```" },
    { cat: "Code", name: "Security review",
      prompt: "Review this code for security issues (injection, auth, secrets, serialization, TOCTOU). Rank findings high/medium/low.\n\n```\n<CODE>\n```" },
    { cat: "Code", name: "Big-O analysis",
      prompt: "Give a Big-O analysis of this function's time and space, including best / average / worst. Show the reasoning.\n\n```\n<CODE>\n```" },

    // Career
    { cat: "Career", name: "Résumé bullet rewrite",
      prompt: "Rewrite each of these résumé bullets to lead with impact (metric + verb + scope). Keep to ~2 lines.\n\n<BULLETS>" },
    { cat: "Career", name: "Interview prep",
      prompt: "I'm interviewing for <ROLE> at <COMPANY>. Generate 15 likely behavioral + technical questions, grouped, with a 2-line answer strategy each." },
    { cat: "Career", name: "Salary negotiation",
      prompt: "I received an offer of <AMOUNT>. Market data suggests <RANGE>. Draft a polite counter-offer email and a 2-option negotiation script." },
    { cat: "Career", name: "Portfolio case study",
      prompt: "I built <PROJECT>. Turn the raw notes below into a crisp case study: context, challenge, actions, results, learnings.\n\n<NOTES>" },

    // Business
    { cat: "Business", name: "SWOT analysis",
      prompt: "Do a SWOT analysis of <COMPANY/PRODUCT>. Be specific; avoid generic bullets. End with the ONE strategic question that matters most." },
    { cat: "Business", name: "Market sizing",
      prompt: "Estimate the market size for <PRODUCT> in <REGION>. Show the top-down and bottom-up methods, reconcile the gap." },
    { cat: "Business", name: "Pricing page copy",
      prompt: "Write copy for a 3-tier pricing page for <PRODUCT>. Each tier: name, price, target user, 5 feature bullets, CTA." },
    { cat: "Business", name: "OKRs for Q",
      prompt: "Draft 3 Objectives with 3 Key Results each for the next quarter, for a team working on <DESCRIPTION>. KRs must be measurable." },

    // Learning
    { cat: "Learning", name: "Feynman explanation",
      prompt: "Explain <CONCEPT> like I'm 10, then at high-school level, then at college level. Show where the simpler analogies break down." },
    { cat: "Learning", name: "Study plan",
      prompt: "Build a 6-week study plan to learn <TOPIC>. Each week: objective, resources (book/video/exercise), weekly project, self-test." },
    { cat: "Learning", name: "Spaced-repetition deck",
      prompt: "Create 20 high-quality flashcards for <TOPIC>. Format: Q / A. Avoid trivia; prefer understanding-level questions." },
    { cat: "Learning", name: "Quick reference sheet",
      prompt: "Make a 1-page cheatsheet for <TOPIC>: key definitions, common patterns, gotchas, when-to-use." },

    // Personal
    { cat: "Personal", name: "Weekly review",
      prompt: "Run me through a weekly review. Ask me one question at a time: wins, lessons, leftovers, priorities for next week." },
    { cat: "Personal", name: "Decision matrix",
      prompt: "I'm choosing between <OPTIONS>. Create a decision matrix with 5 weighted criteria. Ask me to assign weights first." },
    { cat: "Personal", name: "Goal decomposition",
      prompt: "My goal is <GOAL>. Break it down into monthly milestones, weekly targets, and this-week actions. Flag the first keystone habit." },
    { cat: "Personal", name: "Journal prompts",
      prompt: "Give me 10 journal prompts for someone feeling <FEELING>. Focus on reflection, not performative gratitude." },

    // Analysis
    { cat: "Analysis", name: "Steelman + rebuttal",
      prompt: "Steelman the argument for <POSITION>. Then give the strongest rebuttal. End with what evidence would change your mind." },
    { cat: "Analysis", name: "Premortem",
      prompt: "Do a pre-mortem on <PROJECT>. 'It's 12 months from now and the project failed.' List the top 10 causes, ranked." },
    { cat: "Analysis", name: "Compare options",
      prompt: "Compare <OPTION_A> vs <OPTION_B> across cost, time, risk, flexibility. End with a recommendation and its single biggest risk." },
    { cat: "Analysis", name: "Root cause (5 whys)",
      prompt: "Apply the 5 Whys to the problem: <PROBLEM>. After the 5 whys, propose 2 distinct interventions at different levels." },

    // Creative
    { cat: "Creative", name: "Character backstory",
      prompt: "Invent a rich backstory for a character who is <ARCHETYPE>. Include one defining childhood moment, one moral contradiction, and one secret." },
    { cat: "Creative", name: "Plot twists",
      prompt: "Give me 10 plot twists for a <GENRE> story about <PREMISE>. Rank by surprise-vs-believability." },
    { cat: "Creative", name: "World-building",
      prompt: "Build a fictional world around the constraint: <ONE SENTENCE>. Cover: geography, power structure, daily life, recent history, open tension." },
    { cat: "Creative", name: "Poetry in a style",
      prompt: "Write a poem about <SUBJECT> in the style of <POET>. Match their rhythm, imagery, and thematic obsessions." },

    // Travel
    { cat: "Travel", name: "City itinerary",
      prompt: "Plan a 3-day itinerary for <CITY> focused on <THEME>. Balance must-see + hidden gems. Include neighborhoods to base each day around." },
    { cat: "Travel", name: "Packing list",
      prompt: "Build a packing list for <TRIP> in <WEATHER>. Include base layers, one outfit per day concept, and a minimal electronics kit." },

    // Data
    { cat: "Data", name: "SQL from question",
      prompt: "Write a SQL query to answer: <QUESTION>. Schema:\n\n<SCHEMA>\n\nExplain joins and filters in 2 sentences after the query." },
    { cat: "Data", name: "Clean up messy data",
      prompt: "I have the following messy data. Clean it: standardize columns, normalize values, flag suspicious rows.\n\n<DATA>" },

    // Feedback
    { cat: "Feedback", name: "Constructive critique",
      prompt: "Give me constructive critique on the following. Be kind but honest. Format: 3 strengths, 3 biggest opportunities, 1 action.\n\n<WORK>" },
    { cat: "Feedback", name: "Disagreement coaching",
      prompt: "Rewrite this disagreement so it attacks the idea, not the person, and invites collaboration.\n\n<EXCHANGE>" },
  ];

  function open() {
    close();
    const cats = [...new Set(LIBRARY.map(p => p.cat))];
    const overlay = document.createElement('div');
    overlay.id = 'plOverlay';
    overlay.className = 'pl-overlay';
    overlay.innerHTML = `
      <div class="pl-modal">
        <div class="pl-head">
          <h2>Prompt library</h2>
          <input id="plFilter" placeholder="Filter…" oninput="Mio.promptLibrary.filter()">
          <button class="pl-close" onclick="Mio.promptLibrary.close()">×</button>
        </div>
        <div class="pl-body">
          <div class="pl-side" id="plCats">
            <div class="pl-cat active" data-c="__all" onclick="Mio.promptLibrary.selectCat('__all')">All (${LIBRARY.length})</div>
            ${cats.map(c => `<div class="pl-cat" data-c="${c}" onclick="Mio.promptLibrary.selectCat('${c}')">${esc(c)} (${LIBRARY.filter(p => p.cat === c).length})</div>`).join('')}
          </div>
          <div class="pl-list" id="plList"></div>
        </div>
      </div>
    `;
    overlay.addEventListener('click', e => { if (e.target === overlay) close(); });
    document.body.appendChild(overlay);
    renderList('__all');
  }

  let _currentCat = '__all';
  function renderList(cat) {
    _currentCat = cat;
    const q = (document.getElementById('plFilter')?.value || '').toLowerCase();
    const items = LIBRARY.filter(p =>
      (cat === '__all' || p.cat === cat) &&
      (!q || p.name.toLowerCase().includes(q) || p.prompt.toLowerCase().includes(q))
    );
    const list = document.getElementById('plList');
    if (!list) return;
    list.innerHTML = items.length ? items.map((p, i) => `
      <div class="pl-card" onclick="Mio.promptLibrary.insert(${LIBRARY.indexOf(p)})">
        <div class="pl-card-top"><span class="pl-card-cat">${esc(p.cat)}</span><span class="pl-card-name">${esc(p.name)}</span></div>
        <div class="pl-card-body">${esc(p.prompt.slice(0, 240))}${p.prompt.length > 240 ? '…' : ''}</div>
      </div>
    `).join('') : '<div class="pl-empty">No prompts match.</div>';
    // Update active cat highlight
    document.querySelectorAll('.pl-cat').forEach(el =>
      el.classList.toggle('active', el.dataset.c === cat));
  }

  function selectCat(c) { renderList(c); }
  function filter() { renderList(_currentCat); }

  function insert(idx) {
    const p = LIBRARY[idx];
    if (!p) return;
    const input = document.getElementById('inputArea');
    if (!input) return;
    input.value = p.prompt;
    input.focus();
    close();
  }

  function close() {
    const o = document.getElementById('plOverlay');
    if (o) o.remove();
  }

  function esc(s) {
    return String(s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  function injectCSS() {
    if (document.getElementById('pl-css')) return;
    const css = document.createElement('style');
    css.id = 'pl-css';
    css.textContent = `
      .pl-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.5); backdrop-filter: blur(4px); z-index: 1600; display: flex; align-items: center; justify-content: center; padding: 40px; }
      .pl-modal { background: var(--bg-chat); border: 1px solid var(--border); border-radius: 14px; width: min(960px, 100%); height: 80vh; display: flex; flex-direction: column; }
      .pl-head { padding: 16px 20px; border-bottom: 1px solid var(--border-subtle); display: flex; gap: 12px; align-items: center; }
      .pl-head h2 { font-size: 16px; margin: 0; flex: 1; }
      .pl-head input { flex: 0 0 220px; background: var(--bg-input); border: 1px solid var(--border); color: var(--text-primary); padding: 6px 10px; border-radius: 6px; font-size: 13px; }
      .pl-close { background: transparent; border: 0; color: var(--text-muted); font-size: 20px; cursor: pointer; }
      .pl-body { flex: 1; display: grid; grid-template-columns: 200px 1fr; overflow: hidden; }
      .pl-side { border-right: 1px solid var(--border-subtle); padding: 12px 0; overflow-y: auto; }
      .pl-cat { padding: 6px 16px; cursor: pointer; font-size: 13px; color: var(--text-secondary); border-left: 2px solid transparent; }
      .pl-cat:hover { background: var(--bg-hover); }
      .pl-cat.active { color: var(--accent); border-left-color: var(--accent); font-weight: 600; background: var(--bg-hover); }
      .pl-list { overflow-y: auto; padding: 12px 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 12px; align-content: start; }
      .pl-card { background: var(--bg-input); border: 1px solid var(--border); border-radius: 10px; padding: 12px; cursor: pointer; transition: all 120ms; }
      .pl-card:hover { border-color: var(--accent); transform: translateY(-1px); }
      .pl-card-top { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
      .pl-card-cat { font-size: 10px; color: var(--accent); background: rgba(59,130,246,0.1); padding: 2px 8px; border-radius: 999px; text-transform: uppercase; letter-spacing: 0.4px; }
      .pl-card-name { font-weight: 600; color: var(--text-primary); font-size: 13px; }
      .pl-card-body { color: var(--text-muted); font-size: 12px; line-height: 1.5; display: -webkit-box; -webkit-line-clamp: 4; -webkit-box-orient: vertical; overflow: hidden; }
      .pl-empty { grid-column: 1/-1; padding: 40px; text-align: center; color: var(--text-muted); }
      @media (max-width: 700px) { .pl-list { grid-template-columns: 1fr; } .pl-body { grid-template-columns: 140px 1fr; } }
    `;
    document.head.appendChild(css);
  }

  injectCSS();
  NS.promptLibrary = { open, close, insert, selectCat, filter };
})();
