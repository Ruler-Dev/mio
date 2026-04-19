// Artifact screenshot — uses html2canvas loaded on demand; snapshots the
// artifact iframe to PNG and offers a download. Lazy-loads the lib so
// startup cost stays zero.
(function () {
  const NS = (window.Mio = window.Mio || {});

  let _html2canvas = null;

  async function ensureLibrary() {
    if (_html2canvas) return _html2canvas;
    if (window.html2canvas) { _html2canvas = window.html2canvas; return _html2canvas; }
    return new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = 'https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js';
      s.onload = () => { _html2canvas = window.html2canvas; resolve(_html2canvas); };
      s.onerror = () => reject(new Error('Failed to load html2canvas'));
      document.head.appendChild(s);
    });
  }

  async function screenshot() {
    const iframe = document.querySelector('.artifact-iframe');
    if (!iframe) {
      if (window.toast) window.toast('No artifact open');
      return;
    }
    try {
      const h2c = await ensureLibrary();
      const doc = iframe.contentDocument || iframe.contentWindow.document;
      if (!doc || !doc.body) throw new Error('iframe body unreachable');
      const canvas = await h2c(doc.body, {
        backgroundColor: getComputedStyle(doc.body).backgroundColor || '#ffffff',
        scale: 2,
        useCORS: true,
        allowTaint: true,
        logging: false,
      });
      canvas.toBlob(blob => {
        if (!blob) throw new Error('Snapshot blob empty');
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const art = window.allArtifacts && window.activeArtifactId
          ? window.allArtifacts[window.activeArtifactId] : null;
        const stem = (art && art.title || 'artifact').replace(/[^a-z0-9]+/gi, '-').slice(0, 40);
        a.download = stem + '-' + Date.now() + '.png';
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 2000);
      }, 'image/png');
      if (window.toast) window.toast('Snapshot saved to Downloads');
    } catch (e) {
      if (window.toast) window.toast('Snapshot failed: ' + e.message);
    }
  }

  NS.artifactExport = { screenshot };
})();
