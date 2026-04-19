// Paste images directly into the chat input — Cmd+V on an image copies
// it as a binary blob; we convert to a File and send through the normal
// upload path.
(function () {
  const NS = (window.Mio = window.Mio || {});

  function attach() {
    const input = document.getElementById('inputArea');
    if (!input) return;
    input.addEventListener('paste', async (e) => {
      if (!e.clipboardData) return;
      const items = Array.from(e.clipboardData.items);
      const imgs = items.filter(it => it.type && it.type.startsWith('image/'));
      if (!imgs.length) return;
      e.preventDefault();
      for (const it of imgs) {
        const blob = it.getAsFile();
        if (!blob) continue;
        const ext = (it.type.split('/')[1] || 'png').replace('jpeg', 'jpg');
        const file = new File([blob], 'pasted-' + Date.now() + '.' + ext, { type: it.type });
        if (window.uploadFile) await window.uploadFile(file);
      }
      if (window.renderAttachmentChips) window.renderAttachmentChips();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', attach);
  } else {
    attach();
  }

  NS.imagePaste = { attach };
})();
