"""Mio UI — Web interface for Mio.

Provides a FastAPI sub-application with:
  - Chat UI served at /ui
  - WebSocket streaming at /ui/ws/chat
  - Chat session persistence at /ui/api/sessions
  - Model/tier control at /ui/api/config
  - Skills (web search, PDF generation) at /ui/api/skills
"""
