// Chat import — drop-in importer for ChatGPT / Claude / Mio JSON exports.
// Parses client-side and POSTs to /ui/api/chats/import.
(function () {
  const NS = (window.Mio = window.Mio || {});

  function pickFile() {
    const inp = document.createElement('input');
    inp.type = 'file';
    inp.accept = '.json,application/json';
    inp.onchange = async (e) => {
      const f = e.target.files[0];
      if (!f) return;
      let data;
      try {
        data = JSON.parse(await f.text());
      } catch (err) {
        if (window.toast) window.toast('Invalid JSON: ' + err.message);
        return;
      }
      // ChatGPT export is an array; Claude may be {conversations: [...]}.
      let source = 'auto';
      let payload = data;
      if (Array.isArray(data) && data.length && data[0].mapping) source = 'chatgpt';
      if (data && data.conversations) { payload = data.conversations; source = 'claude'; }
      if (data && data.chat_messages) source = 'claude';
      const r = await fetch('/ui/api/chats/import', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ source, data: payload }),
      }).then(r => r.json());
      if (r.error) {
        if (window.toast) window.toast('Import failed: ' + r.error);
      } else {
        if (window.toast) window.toast('Imported ' + r.created + ' chats');
        if (window.loadSessionList) window.loadSessionList();
      }
    };
    inp.click();
  }

  NS.chatImport = { pickFile };
})();
