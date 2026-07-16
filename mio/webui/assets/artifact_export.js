// Local PNG export for artifacts rendered in Mio's parent DOM.
//
// Sandboxed artifact frames intentionally have an opaque origin. This module
// never reads their document; it reports that boundary and points the user to
// source download or the operating-system screenshot tool instead.
(function () {
  'use strict';

  const NS = (window.Mio = window.Mio || {});
  if (NS.artifactExport) return;

  const LIMITS = Object.freeze({
    maxWidth: 2048,
    maxHeight: 4096,
    maxPixels: 8 * 1024 * 1024,
    maxNodes: 1800,
    maxEmbeddedImageBytes: 4 * 1024 * 1024,
    maxSvgBytes: 10 * 1024 * 1024,
    maxPngBytes: 16 * 1024 * 1024,
  });

  const STYLE_PROPERTIES = Object.freeze([
    'display', 'visibility', 'box-sizing', 'position', 'inset', 'top', 'right', 'bottom', 'left',
    'width', 'min-width', 'max-width', 'height', 'min-height', 'max-height',
    'margin', 'margin-top', 'margin-right', 'margin-bottom', 'margin-left',
    'padding', 'padding-top', 'padding-right', 'padding-bottom', 'padding-left',
    'border', 'border-width', 'border-style', 'border-color', 'border-radius',
    'border-top', 'border-right', 'border-bottom', 'border-left',
    'background', 'background-color', 'background-image', 'background-position', 'background-size',
    'background-repeat', 'box-shadow', 'color', 'opacity',
    'font', 'font-family', 'font-size', 'font-style', 'font-weight', 'font-variant',
    'line-height', 'letter-spacing', 'text-align', 'text-decoration', 'text-transform',
    'text-indent', 'text-overflow', 'text-shadow', 'white-space', 'word-break', 'overflow-wrap',
    'overflow', 'overflow-x', 'overflow-y', 'object-fit', 'object-position',
    'list-style', 'list-style-position', 'list-style-type',
    'table-layout', 'border-collapse', 'border-spacing', 'caption-side',
    'flex', 'flex-basis', 'flex-direction', 'flex-flow', 'flex-grow', 'flex-shrink', 'flex-wrap',
    'align-content', 'align-items', 'align-self', 'justify-content', 'justify-items', 'justify-self',
    'order', 'gap', 'row-gap', 'column-gap',
    'grid', 'grid-template', 'grid-template-columns', 'grid-template-rows', 'grid-auto-flow',
    'grid-auto-columns', 'grid-auto-rows', 'grid-column', 'grid-row',
    'transform', 'transform-origin', 'vertical-align', 'z-index',
  ]);

  const UNSAFE_CLONE_NODES = 'script,noscript,style,iframe,object,embed,link,meta';
  const ALTERNATIVE = 'Use Download as file, or use your system screenshot tool.';

  class SnapshotError extends Error {
    constructor(code, message, alternative = ALTERNATIVE) {
      super(message);
      this.name = 'SnapshotError';
      this.code = code;
      this.alternative = alternative;
    }
  }

  async function screenshot() {
    const selected = activeArtifact();
    if (!selected) {
      return failure('no-artifact', 'No artifact is open.', 'Open an artifact, then run the command again.');
    }

    const body = document.getElementById('artifactBody');
    if (!body) {
      return failure('missing-surface', 'The artifact preview surface is unavailable.');
    }

    // A sandbox without allow-same-origin must stay opaque. Never inspect a
    // frame DOM, even when the frame was populated through srcdoc.
    if (body.querySelector('iframe')) {
      return failure(
        'sandboxed-frame',
        'PNG snapshot is unavailable for this sandboxed artifact.',
        ALTERNATIVE,
      );
    }

    const target = body.firstElementChild || body;
    if (!target || typeof target.cloneNode !== 'function') {
      return failure('empty-surface', 'The artifact has no rendered parent-DOM content.');
    }

    try {
      const png = await renderNodeToPng(target);
      const filename = artifactFilename(selected.artifact);
      triggerDownload(png.blob, filename);
      notify(`Saved ${filename}`);
      return Object.freeze({
        ok: true,
        code: 'saved',
        filename,
        width: png.width,
        height: png.height,
        bytes: png.blob.size,
      });
    } catch (error) {
      const detail = error instanceof SnapshotError
        ? error
        : new SnapshotError('render-failed', 'This artifact could not be converted to a local PNG.');
      return failure(detail.code, detail.message, detail.alternative);
    }
  }

  function activeArtifact() {
    const store = NS.store && typeof NS.store === 'object' ? NS.store : {};
    const id = store.activeArtifactId ?? window.activeArtifactId;
    const artifacts = store.allArtifacts ?? window.allArtifacts;
    if (!id || !artifacts || typeof artifacts !== 'object' || Array.isArray(artifacts)) return null;
    if (!Object.prototype.hasOwnProperty.call(artifacts, id)) return null;
    const artifact = artifacts[id];
    if (!artifact || typeof artifact !== 'object') return null;
    return { id: String(id), artifact };
  }

  async function renderNodeToPng(target) {
    const size = measure(target);
    const clone = await cloneForSnapshot(target, size);
    const background = opaqueBackground(target);
    const svg = serializeForeignObject(clone, size, background);
    const svgBlob = new Blob([svg], { type: 'image/svg+xml;charset=utf-8' });
    if (svgBlob.size > LIMITS.maxSvgBytes) {
      throw new SnapshotError('snapshot-too-complex', 'The rendered artifact is too complex for a safe local snapshot.');
    }

    let sourceUrl = '';
    let canvas = null;
    try {
      sourceUrl = URL.createObjectURL(svgBlob);
      const image = await loadImage(sourceUrl);
      const scale = outputScale(size.width, size.height);
      canvas = document.createElement('canvas');
      canvas.width = Math.max(1, Math.round(size.width * scale));
      canvas.height = Math.max(1, Math.round(size.height * scale));
      const context = canvas.getContext('2d', { alpha: false });
      if (!context) throw new SnapshotError('canvas-unavailable', 'Canvas PNG export is unavailable in this browser.');
      context.setTransform(scale, 0, 0, scale, 0, 0);
      context.fillStyle = background;
      context.fillRect(0, 0, size.width, size.height);
      context.drawImage(image, 0, 0, size.width, size.height);
      const png = await canvasToBlob(canvas);
      if (png.size > LIMITS.maxPngBytes) {
        throw new SnapshotError('png-too-large', 'The PNG exceeds Mio’s safe 16 MiB export limit.');
      }
      return { blob: png, width: canvas.width, height: canvas.height };
    } catch (error) {
      if (error instanceof SnapshotError) throw error;
      throw new SnapshotError('foreign-object-unsupported', 'This browser could not render the parent-DOM artifact as PNG.');
    } finally {
      if (sourceUrl) URL.revokeObjectURL(sourceUrl);
      if (canvas) {
        canvas.width = 1;
        canvas.height = 1;
      }
    }
  }

  function measure(target) {
    const rect = typeof target.getBoundingClientRect === 'function'
      ? target.getBoundingClientRect()
      : { width: 0, height: 0 };
    const width = Math.ceil(Math.max(Number(rect.width) || 0, target.clientWidth || 0, target.scrollWidth || 0));
    const height = Math.ceil(Math.max(Number(rect.height) || 0, target.clientHeight || 0, target.scrollHeight || 0));
    if (width < 1 || height < 1) {
      throw new SnapshotError('empty-surface', 'The artifact has no visible parent-DOM content.');
    }
    if (width > LIMITS.maxWidth || height > LIMITS.maxHeight) {
      throw new SnapshotError(
        'dimensions-exceeded',
        `The artifact is ${width}×${height}; local snapshots are limited to ${LIMITS.maxWidth}×${LIMITS.maxHeight}.`,
      );
    }
    return { width, height };
  }

  async function cloneForSnapshot(target, size) {
    const clone = target.cloneNode(true);
    const sourceNodes = [target, ...target.querySelectorAll('*')];
    const cloneNodes = [clone, ...clone.querySelectorAll('*')];
    if (sourceNodes.length > LIMITS.maxNodes) {
      throw new SnapshotError('node-limit', `The artifact exceeds Mio’s ${LIMITS.maxNodes}-node snapshot limit.`);
    }
    if (sourceNodes.length !== cloneNodes.length) {
      throw new SnapshotError('clone-failed', 'The rendered artifact could not be cloned safely.');
    }

    for (let index = 0; index < sourceNodes.length; index += 1) {
      copyComputedStyles(sourceNodes[index], cloneNodes[index]);
      copyControlValue(sourceNodes[index], cloneNodes[index]);
    }
    await inlineMedia(sourceNodes, cloneNodes);

    clone.querySelectorAll(UNSAFE_CLONE_NODES).forEach(node => node.remove());
    clone.style.setProperty('box-sizing', 'border-box', 'important');
    clone.style.setProperty('margin', '0', 'important');
    clone.style.setProperty('width', `${size.width}px`, 'important');
    clone.style.setProperty('min-width', `${size.width}px`, 'important');
    clone.style.setProperty('max-width', `${size.width}px`, 'important');
    clone.style.setProperty('min-height', `${size.height}px`, 'important');
    clone.style.setProperty('overflow', 'hidden', 'important');
    return clone;
  }

  function copyComputedStyles(source, clone) {
    const computed = getComputedStyle(source);
    const backgroundImage = computed.getPropertyValue('background-image');
    if (backgroundImage && backgroundImage !== 'none' && /url\s*\(/i.test(backgroundImage)) {
      throw new SnapshotError(
        'css-image',
        'The artifact uses a CSS background image that cannot be embedded safely.',
      );
    }
    for (const property of STYLE_PROPERTIES) {
      const value = computed.getPropertyValue(property);
      if (value) clone.style.setProperty(property, value, computed.getPropertyPriority(property));
    }
    clone.style.setProperty('animation', 'none', 'important');
    clone.style.setProperty('transition', 'none', 'important');
    clone.style.setProperty('caret-color', 'transparent', 'important');
  }

  function copyControlValue(source, clone) {
    const tag = String(source.tagName || '').toLowerCase();
    if (tag === 'textarea') {
      clone.textContent = source.value || '';
    } else if (tag === 'input') {
      clone.setAttribute('value', source.value || '');
      if (source.checked) clone.setAttribute('checked', 'checked');
      else clone.removeAttribute('checked');
    } else if (tag === 'select') {
      const sourceOptions = Array.from(source.options || []);
      const cloneOptions = Array.from(clone.options || []);
      sourceOptions.forEach((option, index) => {
        if (!cloneOptions[index]) return;
        if (option.selected) cloneOptions[index].setAttribute('selected', 'selected');
        else cloneOptions[index].removeAttribute('selected');
      });
    }
  }

  async function inlineMedia(sourceNodes, cloneNodes) {
    let embeddedBytes = 0;
    for (let index = 0; index < sourceNodes.length; index += 1) {
      const source = sourceNodes[index];
      const clone = cloneNodes[index];
      const tag = String(source.tagName || '').toLowerCase();

      if (tag === 'canvas') {
        let dataUrl;
        try { dataUrl = source.toDataURL('image/png'); }
        catch (_) {
          throw new SnapshotError('tainted-canvas', 'The artifact contains a canvas with non-local pixels.');
        }
        embeddedBytes += estimatedDataUrlBytes(dataUrl);
        enforceEmbeddedImageLimit(embeddedBytes);
        const replacement = document.createElement('img');
        replacement.src = dataUrl;
        replacement.alt = '';
        replacement.style.cssText = clone.style.cssText;
        clone.replaceWith(replacement);
      } else if (tag === 'img') {
        const sourceUrl = source.currentSrc || source.getAttribute('src') || '';
        const dataUrl = await imageDataUrl(sourceUrl);
        embeddedBytes += estimatedDataUrlBytes(dataUrl);
        enforceEmbeddedImageLimit(embeddedBytes);
        clone.removeAttribute('srcset');
        clone.setAttribute('src', dataUrl);
      } else if (tag === 'video' || tag === 'audio') {
        throw new SnapshotError('live-media', 'Live video or audio cannot be represented in a still artifact PNG.');
      }
    }
  }

  async function imageDataUrl(raw) {
    if (!raw) throw new SnapshotError('image-missing', 'An artifact image has no readable source.');
    if (/^data:image\//i.test(raw)) {
      enforceEmbeddedImageLimit(estimatedDataUrlBytes(raw));
      return raw;
    }

    let url;
    try { url = new URL(raw, location.href); }
    catch (_) { throw new SnapshotError('image-url', 'An artifact image URL is invalid.'); }
    if (url.protocol !== 'blob:' && url.origin !== location.origin) {
      throw new SnapshotError(
        'cross-origin-image',
        'The artifact contains an external image that Mio will not fetch for a snapshot.',
      );
    }

    let response;
    try { response = await fetch(url.href, { credentials: 'same-origin' }); }
    catch (_) { throw new SnapshotError('image-fetch', 'A local artifact image could not be read.'); }
    if (!response.ok) throw new SnapshotError('image-fetch', 'A local artifact image could not be read.');
    const blob = await response.blob();
    if (!String(blob.type || '').toLowerCase().startsWith('image/')) {
      throw new SnapshotError('image-type', 'A snapshot resource is not an image.');
    }
    if (blob.size > LIMITS.maxEmbeddedImageBytes) {
      throw new SnapshotError('image-too-large', 'An artifact image exceeds Mio’s safe 4 MiB embedding limit.');
    }
    return blobDataUrl(blob);
  }

  function enforceEmbeddedImageLimit(bytes) {
    if (bytes > LIMITS.maxEmbeddedImageBytes) {
      throw new SnapshotError('images-too-large', 'Embedded artifact images exceed Mio’s safe 4 MiB snapshot limit.');
    }
  }

  function estimatedDataUrlBytes(value) {
    const comma = value.indexOf(',');
    if (comma < 0) return value.length;
    return Math.ceil((value.length - comma - 1) * 0.75);
  }

  function blobDataUrl(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ''));
      reader.onerror = () => reject(new SnapshotError('image-read', 'A local artifact image could not be encoded.'));
      reader.readAsDataURL(blob);
    });
  }

  function serializeForeignObject(clone, size, background) {
    const markup = new XMLSerializer().serializeToString(clone);
    return [
      `<svg xmlns="http://www.w3.org/2000/svg" width="${size.width}" height="${size.height}" viewBox="0 0 ${size.width} ${size.height}">`,
      `<foreignObject x="0" y="0" width="100%" height="100%">`,
      `<div xmlns="http://www.w3.org/1999/xhtml" style="width:${size.width}px;height:${size.height}px;overflow:hidden;background:${escapeAttribute(background)}">`,
      markup,
      '</div></foreignObject></svg>',
    ].join('');
  }

  function escapeAttribute(value) {
    return String(value).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
  }

  function opaqueBackground(target) {
    let current = target;
    while (current && current.nodeType === 1) {
      const color = getComputedStyle(current).backgroundColor;
      if (color && !/^(?:transparent|rgba?\(\s*0\s*,\s*0\s*,\s*0\s*,\s*0\s*\))$/i.test(color)) {
        return color;
      }
      current = current.parentElement;
    }
    return '#ffffff';
  }

  function outputScale(width, height) {
    const requested = Math.min(2, Math.max(1, Number(window.devicePixelRatio) || 1));
    return Math.min(requested, Math.sqrt(LIMITS.maxPixels / (width * height)));
  }

  function loadImage(url) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      const timeout = setTimeout(() => {
        image.onload = null;
        image.onerror = null;
        reject(new SnapshotError('svg-timeout', 'The browser timed out while drawing the artifact.'));
      }, 10000);
      image.onload = () => { clearTimeout(timeout); resolve(image); };
      image.onerror = () => {
        clearTimeout(timeout);
        reject(new SnapshotError('foreign-object-unsupported', 'This browser cannot render the artifact DOM as an image.'));
      };
      image.decoding = 'async';
      image.src = url;
    });
  }

  function canvasToBlob(canvas) {
    return new Promise((resolve, reject) => {
      try {
        canvas.toBlob(blob => {
          if (blob) resolve(blob);
          else reject(new SnapshotError('png-empty', 'The browser returned an empty PNG snapshot.'));
        }, 'image/png');
      } catch (_) {
        reject(new SnapshotError('canvas-security', 'The browser blocked PNG encoding for this artifact.'));
      }
    });
  }

  function artifactFilename(artifact) {
    const raw = String(artifact.title || artifact.id || 'artifact');
    const stem = raw.normalize('NFKD').replace(/[^a-z0-9]+/gi, '-').replace(/^-+|-+$/g, '').slice(0, 48) || 'artifact';
    return `${stem}-${Date.now()}.png`;
  }

  function triggerDownload(blob, filename) {
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    anchor.hidden = true;
    try {
      document.body.appendChild(anchor);
      anchor.click();
    } catch (error) {
      URL.revokeObjectURL(url);
      throw new SnapshotError(
        'download-blocked',
        'The browser blocked the local PNG download.',
        'Allow downloads for Mio, then try again.',
      );
    } finally {
      anchor.remove();
    }
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  function failure(code, message, alternative = ALTERNATIVE) {
    const text = alternative ? `${message} ${alternative}` : message;
    notify(text, true);
    return Object.freeze({ ok: false, code, message, alternative });
  }

  function notify(message, isError = false) {
    if (typeof window.toast === 'function') window.toast(message);
    else if (isError) console.warn('[Mio artifact export]', message);
    else console.info('[Mio artifact export]', message);
  }

  NS.artifactExport = Object.freeze({ screenshot, limits: LIMITS });
})();
