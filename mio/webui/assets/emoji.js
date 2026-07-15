// Small emoji picker — button in the input row + :shortcode autocomplete.
(function () {
  const NS = (window.Mio = window.Mio || {});

  const EMOJIS = {
    smileys: ["😀","😃","😄","😁","😆","😂","🤣","😊","😇","🙂","🙃","😉","😌","😍","🥰","😘","😗","😙","😚","😋","😛","😝","😜","🤪","🤨","🧐","🤓","😎"],
    gestures: ["👍","👎","👌","✌️","🤞","🤟","🤘","🤙","👈","👉","👆","👇","☝️","✋","🤚","🖐","🖖","👋","🤝","🙏","✍️","💅"],
    hearts: ["❤️","🧡","💛","💚","💙","💜","🖤","🤍","🤎","💔","❣️","💕","💞","💓","💗","💖","💘","💝","💟"],
    objects: ["🎯","🔥","🎉","💡","🔑","🗝","📌","🔖","⭐","🌟","✨","⚡","💎","🚀","🎈","🎁","📦","📫","📖","📚","✏️","🖊","📝","📋","📊","📈","📉","💻","📱","⌚","📷","🎥","🎮","🎵","🎨"],
    food: ["🍕","🍔","🍟","🌮","🌯","🍣","🍜","🍝","🍱","🍙","🍘","🍢","🍡","🍨","🍦","🍰","🎂","🍪","🍫","🍩","🍵","☕","🍷","🍺","🥃","🍾"],
    animals: ["🐶","🐱","🐭","🐹","🐰","🦊","🐻","🐼","🐨","🐯","🦁","🐮","🐷","🐸","🐵","🐔","🐧","🐦","🦆","🦉","🐺","🐴","🦄","🐝","🪲","🐢","🐍","🐙","🦀","🐟","🐠","🐳","🦈"],
    nature: ["🌱","🌲","🌳","🌴","🌵","🌷","🌸","🌹","🌻","🌼","🌾","🍀","🍁","🍂","🍃","🌍","🌎","🌏","🌙","☀️","⭐","☁️","🌈","❄️","🔥","💧","🌊"],
    symbols: ["✅","❌","❓","❗","⚠️","🚫","🔔","🔕","💯","✔️","➕","➖","🔄","♻️","⚙️","🔧","🔨","🛠"],
  };

  const SHORTCODES = {
    ":+1:": "👍", ":-1:": "👎", ":fire:": "🔥", ":star:": "⭐", ":heart:": "❤️",
    ":rocket:": "🚀", ":smile:": "😊", ":laugh:": "😂", ":cry:": "😢", ":party:": "🎉",
    ":tada:": "🎉", ":idea:": "💡", ":100:": "💯", ":eye:": "👀", ":thinking:": "🤔",
    ":check:": "✅", ":x:": "❌", ":warning:": "⚠️", ":clap:": "👏", ":pray:": "🙏",
    ":wave:": "👋", ":point_right:": "👉", ":ok:": "👌", ":sparkles:": "✨", ":zap:": "⚡",
    ":cookie:": "🍪", ":coffee:": "☕", ":beer:": "🍺", ":pizza:": "🍕", ":taco:": "🌮",
    ":cat:": "🐱", ":dog:": "🐶", ":sun:": "☀️", ":moon:": "🌙", ":rainbow:": "🌈",
  };

  function openPicker() {
    closePicker();
    const input = document.getElementById('inputArea');
    if (!input) return;
    const cats = Object.keys(EMOJIS);
    const pop = document.createElement('div');
    pop.id = 'emojiPicker';
    pop.className = 'emoji-picker';
    pop.innerHTML = `
      <div class="emoji-tabs">
        ${cats.map((c, i) => `<span class="emoji-tab ${i===0?'active':''}" data-cat="${c}" onclick="Mio.emoji.selectCat('${c}')">${EMOJIS[c][0]}</span>`).join('')}
      </div>
      <div class="emoji-grid" id="emojiGrid"></div>
    `;
    document.body.appendChild(pop);
    const r = input.getBoundingClientRect();
    pop.style.left = (r.left + 40) + 'px';
    pop.style.top = (r.top - pop.offsetHeight - 8) + 'px';
    selectCat(cats[0]);
    setTimeout(() => document.addEventListener('click', _dismiss, true), 0);
  }

  function _dismiss(e) {
    const p = document.getElementById('emojiPicker');
    if (!p) return;
    if (!e.target.closest('#emojiPicker') && !e.target.closest('.emoji-btn')) closePicker();
  }

  function closePicker() {
    const p = document.getElementById('emojiPicker');
    if (p) p.remove();
    document.removeEventListener('click', _dismiss, true);
  }

  function selectCat(cat) {
    const grid = document.getElementById('emojiGrid');
    if (!grid) return;
    grid.innerHTML = EMOJIS[cat].map(e =>
      `<span class="emoji-cell" onclick="Mio.emoji.insert('${e}')">${e}</span>`
    ).join('');
    document.querySelectorAll('.emoji-tab').forEach(el =>
      el.classList.toggle('active', el.dataset.cat === cat));
  }

  function insert(e) {
    const input = document.getElementById('inputArea');
    if (!input) return;
    const s = input.selectionStart || input.value.length;
    input.value = input.value.slice(0, s) + e + input.value.slice(input.selectionEnd || s);
    input.focus();
    input.selectionStart = input.selectionEnd = s + e.length;
    closePicker();
  }

  function expandShortcodes() {
    const input = document.getElementById('inputArea');
    if (!input) return setTimeout(expandShortcodes, 300);
    input.addEventListener('input', () => {
      let v = input.value;
      let changed = false;
      for (const [code, emo] of Object.entries(SHORTCODES)) {
        if (v.includes(code)) {
          v = v.split(code).join(emo);
          changed = true;
        }
      }
      if (changed) {
        const pos = input.selectionStart;
        input.value = v;
        input.selectionStart = input.selectionEnd = pos;
      }
    });
  }

  function addButton() {
    const input = document.getElementById('inputArea');
    if (!input) return setTimeout(addButton, 300);
    const hints = document.querySelector('.input-hints');
    if (!hints || document.querySelector('.emoji-btn')) return;
    const actions = document.querySelector('.input-actions');
    if (!actions) return;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'emoji-btn';
    btn.title = 'Emoji picker';
    btn.setAttribute('aria-label', 'Emoji picker');
    btn.textContent = '😀';
    btn.onclick = (e) => { e.stopPropagation(); openPicker(); };
    // A descendant button is not a valid `insertBefore` reference for the
    // actions container.  `prepend` is stable even when another module wraps
    // the existing voice/send controls before this deferred module loads.
    actions.prepend(btn);
  }

  function injectCSS() {
    if (document.getElementById('emoji-css')) return;
    const css = document.createElement('style');
    css.id = 'emoji-css';
    css.textContent = `
      .emoji-btn { background: transparent; border: 0; font-size: 18px; cursor: pointer; padding: 4px 8px; border-radius: 6px; }
      .emoji-btn:hover { background: var(--bg-hover); }
      .emoji-picker { position: fixed; background: var(--bg-chat); border: 1px solid var(--border); border-radius: 10px; padding: 8px; width: 320px; z-index: 100; box-shadow: 0 12px 40px rgba(0,0,0,0.4); }
      .emoji-tabs { display: flex; gap: 2px; margin-bottom: 8px; border-bottom: 1px solid var(--border-subtle); padding-bottom: 6px; }
      .emoji-tab { flex: 1; text-align: center; padding: 4px; border-radius: 6px; cursor: pointer; font-size: 16px; }
      .emoji-tab:hover { background: var(--bg-hover); }
      .emoji-tab.active { background: var(--accent-subtle); }
      .emoji-grid { display: grid; grid-template-columns: repeat(8, 1fr); gap: 2px; max-height: 220px; overflow-y: auto; }
      .emoji-cell { text-align: center; padding: 6px 2px; cursor: pointer; font-size: 18px; border-radius: 4px; }
      .emoji-cell:hover { background: var(--bg-hover); }
    `;
    document.head.appendChild(css);
  }

  injectCSS();
  addButton();
  expandShortcodes();

  NS.emoji = { openPicker, closePicker, selectCat, insert };
})();
