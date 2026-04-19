// Text snippets — type ::shortcut to expand into a larger block.
// Saved to localStorage. Manage via /snippet add|rm|list.
(function () {
  const NS = (window.Mio = window.Mio || {});
  const KEY = 'mio-snippets';

  function load() {
    try { return JSON.parse(localStorage.getItem(KEY) || '{}'); }
    catch (e) { return {}; }
  }
  function save(d) { localStorage.setItem(KEY, JSON.stringify(d)); }

  function run(text) {
    const m = text.match(/^\/snippet\s+(add|rm|list)(?:\s+(\S+))?\s*([\s\S]*)$/);
    if (!m) return;
    const cmd = m[1];
    const name = m[2];
    const body = (m[3] || '').trim();
    const snips = load();
    if (cmd === 'add') {
      if (!name || !body) { if (window.toast) window.toast('Usage: /snippet add <name> <body>'); return; }
      snips[name] = body;
      save(snips);
      if (window.toast) window.toast('Snippet ::' + name + ' saved');
      return;
    }
    if (cmd === 'rm') {
      delete snips[name];
      save(snips);
      if (window.toast) window.toast('Snippet ::' + name + ' removed');
      return;
    }
    if (cmd === 'list') {
      const keys = Object.keys(snips);
      const body = keys.length
        ? keys.map(k => '• ::' + k + ' → ' + snips[k].slice(0, 80) + (snips[k].length > 80 ? '…' : '')).join('\n')
        : 'No snippets yet. Try `/snippet add sig "Best, Alex"`.';
      if (window.appendSystemMessage) window.appendSystemMessage('**Snippets:**\n\n' + body);
      return;
    }
  }

  function watchInput() {
    const input = document.getElementById('inputArea');
    if (!input) return setTimeout(watchInput, 300);
    input.addEventListener('input', () => {
      const snips = load();
      if (!Object.keys(snips).length) return;
      let v = input.value;
      let changed = false;
      for (const [k, body] of Object.entries(snips)) {
        const tag = '::' + k;
        // Only expand when terminated by a non-word char or end-of-string
        const re = new RegExp(tag.replace(/([^\w])/g, '\\$1') + '(?=\\s|$|[^\\w])', 'g');
        if (re.test(v)) {
          v = v.replace(re, body);
          changed = true;
        }
      }
      if (changed) {
        const pos = input.selectionStart;
        input.value = v;
        input.selectionStart = input.selectionEnd = v.length;
      }
    });
  }

  watchInput();

  NS.snippets = { run, load, save };
})();
