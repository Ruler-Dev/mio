// Auto-extracted from mio_ui.html — 100+ slash-command templates
window.SLASH_TEMPLATES = {
  // --- Web / info ---
  'search':      'Search the web for: {{ARG}}',
  'fetch':       'Fetch and summarize this URL: {{ARG}}',
  'weather':     'Get the current weather in {{ARG}} as an animated widget.',
  'news':        'Search the latest news about: {{ARG}}',
  'images':      'Find images of: {{ARG}}',
  'youtube':     'Find YouTube videos about: {{ARG}}',
  'translate':   'Translate this text: {{ARG}}',

  // --- Media recommendations ---
  'anime':       'Recommend anime: {{ARG}}',
  'manga':       'Recommend manga: {{ARG}}',
  'movie':       'Recommend a movie or TV show: {{ARG}}',
  'tv':          'Recommend a TV show: {{ARG}}',
  'game':        'Recommend a video game: {{ARG}}',

  // --- Documents: PDFs ---
  'pdf':         'Generate a professional PDF report about: {{ARG}}',
  'report':      'Generate a PDF report about: {{ARG}}',
  'letter':      'Draft a formal letter about: {{ARG}}',
  'certificate': 'Create a certificate for: {{ARG}}',
  'flyer':       'Design a flyer for: {{ARG}}',
  'menu':        'Design a restaurant menu for: {{ARG}}',
  'brochure':    'Design a tri-fold brochure for: {{ARG}}',
  'newsletter':  'Create a newsletter about: {{ARG}}',
  'card':        'Make a business card for: {{ARG}}',
  'resume':      'Generate a professionally-formatted resume PDF for: {{ARG}}',
  'cv':          'Generate a CV for: {{ARG}}',
  'invoice':     'Generate an invoice PDF for: {{ARG}}',
  'contract':    'Draft a contract for: {{ARG}}',
  'proposal':    'Write a project proposal for: {{ARG}}',

  // --- Office files ---
  'docx':        'Generate a Word document about: {{ARG}}',
  'xlsx':        'Generate an Excel spreadsheet with: {{ARG}}',
  'pptx':        'Generate a slide deck about: {{ARG}}',
  'slides':      'Generate a slide deck about: {{ARG}}',
  'csv':         'Export this data as CSV: {{ARG}}',
  'sqlite':      'Generate a SQLite database of: {{ARG}}',
  'ical':        'Generate an .ics calendar event for: {{ARG}}',
  'markdown':    'Save this as a Markdown note: {{ARG}}',
  'obsidian':    'Save this as an Obsidian note: {{ARG}}',
  'note':        'Write a note about: {{ARG}}',

  // --- Utilities ---
  'qr':          'Generate a QR code for: {{ARG}}',
  'chart':       'Generate a chart for: {{ARG}}',
  'python':      'Run this Python code: {{ARG}}',
  'bash':        'Write a bash script for: {{ARG}}',

  // --- Visual / diagrams ---
  'mindmap':     'Create a mindmap artifact for: {{ARG}}',
  'mermaid':     'Create a mermaid diagram for: {{ARG}}',
  'diagram':     'Create a graphviz diagram for: {{ARG}}',
  'flowchart':   'Create a flowchart diagram for: {{ARG}}',
  'sequence':    'Create a sequence diagram for: {{ARG}}',
  'gantt':       'Create a gantt-chart diagram for: {{ARG}}',
  'tree':        'Create a tree / hierarchy diagram for: {{ARG}}',
  'timeline':    'Create a timeline artifact for: {{ARG}}',
  'map':         'Show {{ARG}} on a Leaflet map artifact.',
  'math':        'Render these equations as a math artifact: {{ARG}}',
  'excalidraw':  'Create an Excalidraw whiteboard of: {{ARG}}',
  'svg':         'Create an SVG illustration of: {{ARG}}',

  // --- 3D / WebGL ---
  '3d':          'Create a three.js 3D scene: {{ARG}}',
  'threejs':     'Create a three.js 3D scene: {{ARG}}',
  'shader':      'Create a WebGL shader artifact: {{ARG}}',
  'jscad':       'Create a JSCAD parametric 3D model: {{ARG}}',
  'stl':         'Create an STL / 3D-printable model: {{ARG}}',

  // --- Interactive / React ---
  'react':       'Create a React component artifact for: {{ARG}}',
  'html':        'Create an interactive HTML artifact for: {{ARG}}',
  'p5':          'Create a p5.js sketch for: {{ARG}}',
  'chartjs':     'Create a Chart.js interactive chart for: {{ARG}}',
  'game-artifact': 'Create a playable mini-game: {{ARG}}',
  'simulator':   'Create an interactive simulator for: {{ARG}}',
  'playground':  'Create an interactive playground for: {{ARG}}',
  'dashboard':   'Create an interactive dashboard artifact for: {{ARG}}',
  'explain-visual': 'Explain {{ARG}} with an interactive visual artifact — NOT a PDF.',

  // --- Coding-oriented artifacts ---
  'code':        'Write code for: {{ARG}}',
  'regex':       'Create a regex tester artifact for: {{ARG}}',
  'diff':        'Show a diff viewer for: {{ARG}}',
  'json':        'Render this JSON as an interactive tree: {{ARG}}',
  'json2ts':     'Convert JSON to TypeScript interfaces: {{ARG}}',
  'table':       'Create a sortable / filterable table artifact for: {{ARG}}',
  'pyrepl':      'Create a Python REPL artifact for: {{ARG}}',
  'terminal':    'Create a terminal-simulator artifact: {{ARG}}',

  // --- Audio / music ---
  'piano':       'Create a piano keyboard artifact to play: {{ARG}}',
  'synth':       'Create a Tone.js synthesizer artifact for: {{ARG}}',
  'drumkit':     'Create a drum machine artifact.',
  'abc':         'Render this ABC music notation: {{ARG}}',
  'audio':       'Embed an audio player for: {{ARG}}',

  // --- Learning ---
  'flashcards':  'Create flashcards for: {{ARG}}',
  'quiz':        'Create a quiz about: {{ARG}}',
  'kanban':      'Create a kanban board for: {{ARG}}',
  'recipe':      'Write a recipe for: {{ARG}}',
  'checklist':   'Create a checklist for: {{ARG}}',
  'notes':       'Take structured notes on: {{ARG}}',

  // --- Presentation ---
  'reveal':      'Create a reveal.js slide deck about: {{ARG}}',
  'presentation':'Create a slide deck about: {{ARG}}',

  // --- Life & work ---
  'todo':        'Add a todo: {{ARG}}',
  'todos':       'List open todos.',
  'habit':       'Add or check in a habit: {{ARG}}',
  'habits':      'List tracked habits + streaks.',
  'journal':     'Append to today\'s journal: {{ARG}}',
  'journal-read':'Read today\'s journal.',
  'analyze-json':'Analyze a JSON blob: {{ARG}}',
  'analyze-csv': 'Analyze a CSV: {{ARG}}',
  'regex-explain': 'Explain this regex token by token: {{ARG}}',
  'hn':            'Show Hacker News top stories.',
  'reddit':        'Show top Reddit posts from: {{ARG}}',
  'quote':         'Random famous quote. Optional topic: {{ARG}}',
  'convert':       'Convert currency: {{ARG}}',
  'preview':       'URL preview card for: {{ARG}}',
  'scale-recipe':  'Scale a recipe: {{ARG}}',
  'bookmark':     'Save a URL to reading list: {{ARG}}',
  'bookmarks':    'List saved bookmarks.',
  'palette':      'Generate a color palette from seed: {{ARG}}',
  'describe':     'Describe an attached image file: {{ARG}}',
  'review':       'Review this code: {{ARG}}',
  'notes':        'Extract meeting notes from transcript: {{ARG}}',

  // --- Local folder RAG ---
  'index':       'Index a local folder of text files for full-text search: {{ARG}}',
  'search-local':'Search your indexed local folders: {{ARG}}',
  'list-indexes':'List all indexed folders.',

  // --- Developer / dashboards ---  'keys':                'Show every keyboard shortcut and slash command (⌘/).',
  'shortcuts':           'Show every keyboard shortcut and slash command (⌘/).',
  'tour':                'Replay the onboarding tour.',
  'templates':           'Browse / load / manage saved chat templates.',
  'save-template':       'Save the current chat as a reusable template.',
  'export-html':         'Export the current chat as a standalone HTML bundle.',
  'export-pdf':          'Export the current chat as a PDF.',
  'import-chats':        'Import chats from ChatGPT / Claude JSON export.',
  'library':             'Browse the curated prompt library.',
  'prompts':             'Browse the curated prompt library.',
  'surprise':            'Surprise me — random prompt + random persona, auto-sent.',
  'chat-prompt':         'Set a per-chat system prompt that overrides the global one.',
  'compress':            'Summarize older messages to save context window.',
  'accent':              'Pick a custom accent color.',
  'color':               'Pick a custom accent color.',
  'alias':               'Manage slash-command aliases. /alias add <name> <template>',
  'pomodoro':            'Open a pomodoro timer in a popup.',
  'emoji':               'Open the emoji picker.',
  'worldclock':          'Open the multi-timezone clock popup.',
  'zen':                 'Launch a meditation / breathing timer.',
  'roll':                'Roll dice in NdM notation: {{ARG}}',
  'flip':                'Flip N coins (default 1): {{ARG}}',
  'pick':                'Pick a random item from: {{ARG}}',
  'names':               'Generate names: {{ARG}}',
  'wordle':              'Wordle helper: {{ARG}}',
  'wiki':                'Wikipedia summary for: {{ARG}}',
  'http':                'Issue an HTTP request (/http GET https://…).',
  'briefing':            'Summarize your reading list.',
  'snippet':             'Manage text snippets: /snippet add|rm|list',
  'stats':               'Open the stats dashboard (messages, skills, artifacts).',
  'attachments':         'Browse every generated / uploaded file in Downloads.',
  'playground':          'Open the skill playground (try every skill with a live form).',
  'dashboard':           'Open the Mio dashboard (schedules / webhooks / indexed folders).',
  'compare':             'Side-by-side model compare (two tiers, same prompt).',
  'schedules':           'Manage scheduled prompts.',
  'webhooks':            'Manage webhook triggers.',

  // --- Artifacts ---
  'screenshot-artifact': 'Save a PNG screenshot of the currently open artifact.',
  'present':             'Launch presentation mode across every artifact in this chat.',

  // --- Clipboard ---
  'paste-context': 'Paste clipboard as hidden context for the next message (⌘⇧V).',
  'clipboard':     'Paste clipboard as hidden context for the next message.',

  // --- Density ---
  'density':     'Cycle chat density (comfortable → compact → cozy).',
  'compact':     'Set compact density.',
  'cozy':        'Set cozy density.',
  'comfortable': 'Set comfortable density (default).',

  // --- Personas (see /as-list for the full roster) ---
  'as':          'Switch persona: /as teacher | /as skeptic | /as chef | /as pirate | /as haiku | etc.',
  'as-list':     'List all available personas.',
  'personas':    'List all available personas.',

  // --- Creative ---
  'story':       'Write a short story about: {{ARG}}',
  'haiku':       'Write a haiku about: {{ARG}}',
  'poem':        'Write a poem about: {{ARG}}',
  'brainstorm':  'Brainstorm ideas for: {{ARG}}',
  'explain':     'Explain {{ARG}} clearly, step by step.',
  'summarize':   'Summarize: {{ARG}}',
  'critique':    'Critique: {{ARG}}',
  'rewrite':     'Rewrite this more clearly: {{ARG}}',
  'plan':        'Plan this: {{ARG}}',
  'compare':     'Compare: {{ARG}}',
};
