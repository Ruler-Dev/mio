// /compress — summarize the older half of the chat into one concise
// system note, freeing context window for the rest of the conversation.
// Runs the summarization through a one-shot WS call and replaces the
// summarized messages with a single system-role entry.
(function () {
  const NS = (window.Mio = window.Mio || {});

  async function run() {
    const msgs = window.chatMessages || [];
    if (msgs.length < 10) {
      if (window.toast) window.toast('Too short to compress — need 10+ messages');
      return;
    }
    if (window.isStreaming) {
      if (window.toast) window.toast('Wait for current generation to finish');
      return;
    }
    // Keep the most recent N messages verbatim; summarize everything older
    const KEEP = 4;
    const older = msgs.slice(0, msgs.length - KEEP);
    const recent = msgs.slice(-KEEP);
    if (!older.length) {
      if (window.toast) window.toast('Nothing to compress');
      return;
    }
    const transcript = older.map(m =>
      '[' + (m.role === 'user' ? 'USER' : 'ASSISTANT') + '] ' +
      (m.content || '').replace(/<antArtifact[\s\S]*?<\/antArtifact>/g, '[artifact]')
    ).join('\n\n');

    const prompt =
      "Condense the following conversation into a concise, third-person " +
      "summary that preserves every meaningful fact, preference, name, " +
      "number, and unresolved question. Aim for ~20% of the original " +
      "length. Use bullet points where helpful. Do not invent or editorialize.\n\n" +
      "---\n" + transcript + "\n---";

    if (window.toast) window.toast('Compressing older messages…');

    // Fire a one-shot WS call that bypasses the main chat flow
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(proto + '//' + location.host + '/ui/ws/chat');
    let buf = '';
    ws.onopen = () => {
      ws.send(JSON.stringify({
        action: 'chat',
        messages: [{ role: 'user', content: prompt }],
        max_tokens: 2000,
        skills: false,
      }));
    };
    ws.onmessage = (e) => {
      const m = JSON.parse(e.data);
      if (m.type === 'token' && m.text) buf += m.text;
      if (m.type === 'done' || m.type === 'error') {
        const summary = (m.full_text || buf || '').trim();
        if (!summary) {
          if (window.toast) window.toast('Compression produced empty result');
          ws.close(); return;
        }
        // Replace older messages with ONE compressed marker
        const compressedMsg = {
          role: 'user',
          content: '[COMPRESSED HISTORY — ' + older.length + ' messages summarized:]\n\n' + summary,
        };
        window.chatMessages = [compressedMsg, ...recent];
        if (window.renderAllMessages) window.renderAllMessages();
        if (window.autoSave) window.autoSave();
        if (window.toast) window.toast('Compressed ' + older.length + ' messages into a summary');
        ws.close();
      }
    };
    ws.onerror = () => { if (window.toast) window.toast('Compression failed'); };
  }

  NS.compress = { run };
})();
