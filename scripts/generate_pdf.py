#!/usr/bin/env python3
"""Generate Mio slide deck presentation with bar chart benchmarks."""

from __future__ import annotations

from fpdf import FPDF

# --- Dark theme palette ---
BG = (12, 12, 20)
CARD = (22, 25, 40)
CYAN = (0, 200, 255)
PURPLE = (130, 90, 255)
GREEN = (0, 220, 130)
ORANGE = (255, 170, 40)
RED = (255, 75, 75)
WHITE = (240, 240, 248)
DIM = (130, 135, 155)
MUTED = (70, 75, 95)
YELLOW = (255, 210, 50)

# Hardware colors
M4_CLR = CYAN
C5080 = GREEN
C5090 = ORANGE
CLOUD = PURPLE
GRAY_BAR = (55, 60, 80)


class Deck(FPDF):
    def __init__(self):
        super().__init__(orientation="L", unit="mm", format="A4")
        self.set_auto_page_break(auto=False)
        self.slide_num = 0

    def slide(self, title=None, subtitle=None):
        self.add_page()
        self.slide_num += 1
        self.set_fill_color(*BG)
        self.rect(0, 0, 297, 210, "F")
        # Bottom bar
        self.set_fill_color(*CARD)
        self.rect(0, 200, 297, 10, "F")
        self.set_font("Helvetica", "", 6)
        self.set_text_color(*DIM)
        self.text(8, 206, "Mio  |  Federico Lupo  |  April 2026")
        self.text(272, 206, f"{self.slide_num}")
        # Accent line
        self._gradient_line(12)
        if title:
            self.set_font("Helvetica", "B", 24)
            self.set_text_color(*WHITE)
            self.set_xy(15, 16)
            self.cell(0, 10, title)
        if subtitle:
            self.set_font("Helvetica", "", 11)
            self.set_text_color(*DIM)
            self.set_xy(15, 28)
            self.cell(0, 6, subtitle)

    def _gradient_line(self, y, w=297):
        for i in range(w):
            f = i / w
            r = int(CYAN[0] * (1 - f) + PURPLE[0] * f)
            g = int(CYAN[1] * (1 - f) + PURPLE[1] * f)
            b = int(CYAN[2] * (1 - f) + PURPLE[2] * f)
            self.set_fill_color(r, g, b)
            self.rect(i, y, 1.2, 1.5, "F")

    def stat_box(self, x, y, value, label, color=CYAN, w=55, h=32):
        self.set_fill_color(*CARD)
        self.rect(x, y, w, h, "F")
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(*color)
        self.set_xy(x, y + 4)
        self.cell(w, 10, value, align="C")
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*DIM)
        self.set_xy(x, y + 18)
        self.cell(w, 6, label, align="C")

    def text_block(self, x, y, lines, size=10, color=DIM, line_h=6):
        self.set_font("Helvetica", "", size)
        self.set_text_color(*color)
        for i, line in enumerate(lines):
            self.set_xy(x, y + i * line_h)
            self.cell(0, line_h, line)

    def bold_text(self, x, y, txt, size=12, color=WHITE):
        self.set_font("Helvetica", "B", size)
        self.set_text_color(*color)
        self.set_xy(x, y)
        self.cell(0, 8, txt)

    def bar_chart(self, x, y, w, h, bars, unit="", max_val=None, title=None):
        """Draw horizontal bar chart.
        bars: [(label, value, (r,g,b)), ...]
        """
        if not bars:
            return
        if max_val is None:
            max_val = max(b[1] for b in bars) * 1.12

        label_w = 72
        chart_w = w - label_w - 15
        bar_h = min(10, (h - 15) / len(bars) - 2)
        spacing = max(2, (h - 15 - bar_h * len(bars)) / max(len(bars) - 1, 1))

        if title:
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(*WHITE)
            self.set_xy(x, y)
            self.cell(w, 5, title)
            y += 8

        # Grid lines
        for frac in [0.25, 0.5, 0.75, 1.0]:
            gx = x + label_w + chart_w * frac
            self.set_draw_color(*MUTED)
            self.set_line_width(0.15)
            self.line(gx, y, gx, y + len(bars) * (bar_h + spacing))
            val = max_val * frac
            self.set_font("Helvetica", "", 5.5)
            self.set_text_color(*MUTED)
            self.text(gx - 4, y + len(bars) * (bar_h + spacing) + 4,
                      f"{val:.0f}" if val >= 10 else f"{val:.1f}")

        by = y
        for label, value, color in bars:
            # Label
            self.set_font("Helvetica", "", 7)
            self.set_text_color(*DIM)
            self.set_xy(x, by + 1)
            self.cell(label_w - 3, bar_h, label, align="R")

            # Bar
            bw = max((value / max_val) * chart_w, 1) if max_val > 0 else 1
            self.set_fill_color(*color)
            self.rect(x + label_w, by + 0.5, bw, bar_h - 1, "F")

            # Value label
            self.set_font("Helvetica", "B", 6.5)
            self.set_text_color(*color)
            self.text(x + label_w + bw + 2, by + bar_h - 2.5,
                      f"{value:g} {unit}")

            by += bar_h + spacing

    def grouped_bars(self, x, y, w, h, groups, legend, unit="", max_val=None, title=None):
        """Draw grouped horizontal bars.
        groups: [(label, [(value, (r,g,b)), ...]), ...]
        legend: [(name, (r,g,b)), ...]
        """
        n_groups = len(groups)
        n_per = max(len(g[1]) for g in groups)
        label_w = 60
        chart_w = w - label_w - 15

        if max_val is None:
            max_val = max(v for _, vals in groups for v, _ in vals) * 1.12

        bar_h = 6
        group_gap = 6
        group_h = n_per * (bar_h + 1.5) + group_gap

        if title:
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(*WHITE)
            self.set_xy(x, y)
            self.cell(w, 5, title)
            y += 7

        # Legend
        lx = x + label_w
        for name, clr in legend:
            self.set_fill_color(*clr)
            self.rect(lx, y, 5, 4, "F")
            self.set_font("Helvetica", "", 6)
            self.set_text_color(*DIM)
            self.text(lx + 7, y + 3, name)
            lx += len(name) * 2.8 + 18
        y += 8

        # Grid
        for frac in [0.25, 0.5, 0.75, 1.0]:
            gx = x + label_w + chart_w * frac
            self.set_draw_color(*MUTED)
            self.set_line_width(0.1)
            self.line(gx, y, gx, y + n_groups * group_h)
            val = max_val * frac
            self.set_font("Helvetica", "", 5)
            self.set_text_color(*MUTED)
            self.text(gx - 3, y + n_groups * group_h + 3,
                      f"{val:.0f}" if val >= 10 else f"{val:.1f}")

        gy = y
        for label, vals in groups:
            self.set_font("Helvetica", "", 7)
            self.set_text_color(*DIM)
            mid_y = gy + (n_per * (bar_h + 1.5)) / 2 - 1
            self.set_xy(x, mid_y)
            self.cell(label_w - 3, 5, label, align="R")

            for j, (value, clr) in enumerate(vals):
                bw = max((value / max_val) * chart_w, 0.5)
                by = gy + j * (bar_h + 1.5)
                self.set_fill_color(*clr)
                self.rect(x + label_w, by, bw, bar_h, "F")
                self.set_font("Helvetica", "B", 5.5)
                self.set_text_color(*clr)
                self.text(x + label_w + bw + 2, by + bar_h - 1.5, f"{value:g}")

            gy += group_h


def build():
    pdf = Deck()

    # ==========================================================
    # SLIDE 1: TITLE
    # ==========================================================
    pdf.add_page()
    pdf.set_fill_color(*BG)
    pdf.rect(0, 0, 297, 210, "F")
    pdf._gradient_line(82)
    pdf._gradient_line(84)

    pdf.set_font("Helvetica", "B", 52)
    pdf.set_text_color(*WHITE)
    pdf.set_xy(30, 28)
    pdf.cell(0, 20, "Mio")

    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(*DIM)
    pdf.set_xy(30, 52)
    pdf.cell(0, 8, "Fast Local LLM Inference for Apple Silicon")

    pdf.set_font("Helvetica", "I", 12)
    pdf.set_text_color(*CYAN)
    pdf.set_xy(30, 64)
    pdf.cell(0, 8, '"ci sono tanti agenti ma questo e mio!"')
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*DIM)
    pdf.set_xy(30, 73)
    pdf.cell(0, 6, "there are many agents but this one is mine")

    pdf.stat_box(30, 100, "4.1x", "DFlash Speedup", CYAN)
    pdf.stat_box(95, 100, "3.6x", "Cache Compress", GREEN)
    pdf.stat_box(160, 100, "1M", "Max Context", ORANGE)
    pdf.stat_box(225, 100, "10.7x", "Faster vs Ollama", PURPLE)

    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*WHITE)
    pdf.set_xy(30, 145)
    pdf.cell(0, 8, "Federico Lupo")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*DIM)
    pdf.set_xy(30, 155)
    pdf.cell(0, 6, "v0.1.0  |  April 2026  |  macOS Apple Silicon  |  MIT License")

    # Bottom bar
    pdf.set_fill_color(*CARD)
    pdf.rect(0, 200, 297, 10, "F")

    # ==========================================================
    # SLIDE 2: THE PROBLEM
    # ==========================================================
    pdf.slide("The Problem", "Local models are slow. Cloud models leak your code.")

    pdf.bar_chart(15, 42, 265, 80, [
        ("Cloud: GPT-4o", 80, PURPLE),
        ("Cloud: Claude Opus", 60, PURPLE),
        ("Mio 27B DFlash+TQ", 56, CYAN),
        ("mlx-lm 27B 4bit", 28, GRAY_BAR),
        ("LM Studio 27B", 26, GRAY_BAR),
        ("Ollama 27B", 24, GRAY_BAR),
        ("llama.cpp 27B", 20, GRAY_BAR),
    ], unit="tok/s", max_val=95,
       title="27B Model Generation Speed (tok/s)")

    pdf.bold_text(15, 130, "Cloud is fast but costs money and leaks code.", 11, ORANGE)
    pdf.bold_text(15, 140, "Standard local is private but painfully slow.", 11, RED)
    pdf.bold_text(15, 152, "Mio: fast AND local AND private.", 13, GREEN)

    pdf.text_block(15, 166, [
        "DFlash MLX speculative decoding:  1.7-4.1x speedup, 89% acceptance",
        "TurboQuant V2 KV compression:  3.6x less cache memory",
        "Combined:  fast generation + massive context windows",
    ], size=9, color=DIM)

    # ==========================================================
    # SLIDE 3: HOW IT WORKS
    # ==========================================================
    pdf.slide("How It Works", "Two orthogonal accelerations that stack")

    # DFlash card
    pdf.set_fill_color(*CARD)
    pdf.rect(15, 40, 130, 65, "F")
    pdf.bold_text(22, 43, "DFlash Speculative Decoding", 12, CYAN)
    pdf.text_block(22, 56, [
        "Small draft model proposes 16 tokens at once",
        "Target verifies entire block in 1 forward pass",
        "Accept longest correct prefix (89% acceptance)",
        "Result: 1.7-4.1x speedup depending on model",
    ], size=8.5)
    pdf.bold_text(22, 92, "Up to 4.1x generation speedup", 11, GREEN)

    # TQ card
    pdf.set_fill_color(*CARD)
    pdf.rect(152, 40, 130, 65, "F")
    pdf.bold_text(159, 43, "TurboQuant V2 KV Cache", 12, PURPLE)
    pdf.text_block(159, 56, [
        "Quantize KV cache: fp16 -> 4-bit on the fly",
        "Hardware-accelerated via MLX Metal kernel",
        "Random rotation for uniform quantization",
        "< 1% quality loss at 4-bit",
    ], size=8.5)
    pdf.bold_text(159, 92, "3.6x cache compression", 11, GREEN)

    # Combined
    pdf.set_fill_color(20, 35, 55)
    pdf.rect(15, 115, 267, 25, "F")
    pdf.bold_text(22, 118, "Combined Pipeline", 12, WHITE)
    pdf.text_block(22, 129, [
        "Prompt -> DFlash draft (16 tok) -> Target verify (1 pass) -> TQ compress KV -> Accept + repeat"
    ], size=9, color=CYAN)

    pdf.text_block(15, 150, [
        "Draft cost: ~0.5-1.5 GB     Target: 3-27 GB     KV cache: 75% smaller than fp16",
        "Ollama/LM Studio replacement:  same API, 2-8x faster, 4x longer context",
    ], size=8.5)

    # ==========================================================
    # SLIDE 4: SPEED  - Mio vs Standard Local
    # ==========================================================
    pdf.slide("Speed: Mio vs. Standard Local", "M4 Max 48GB  |  DFlash+TQ vs mlx-lm / Ollama / LM Studio")

    pdf.bar_chart(15, 42, 265, 150, [
        ("Mio 35B-A3B MoE (DFlash+TQ)", 204, ORANGE),
        ("mlx-lm 35B-A3B baseline", 120, GRAY_BAR),
        ("Mio 9B (DFlash+TQ)", 108, CYAN),
        ("mlx-lm 9B baseline", 26, GRAY_BAR),
        ("Ollama 9B", 24, GRAY_BAR),
        ("Mio 27B (DFlash+TQ)", 56, PURPLE),
        ("mlx-lm 27B baseline", 28, GRAY_BAR),
        ("LM Studio 27B", 26, GRAY_BAR),
        ("Ollama 27B", 24, GRAY_BAR),
    ], unit="tok/s", max_val=240,
       title="Generation Speed: Mio vs Standard Inference (tok/s)")

    pdf.bold_text(15, 192, "35B-A3B MoE: 35B brain, 3B active/token = 204 tok/s. Fast AND smart.", 10, GREEN)

    # ==========================================================
    # SLIDE 4b: SPEED  - Local vs Cloud
    # ==========================================================
    pdf.slide("Speed: Local vs. Cloud", "Mio 35B-A3B and 9B both exceed cloud API speeds")

    pdf.bar_chart(15, 42, 265, 150, [
        ("Mio 35B-A3B MoE (DFlash+TQ)", 204, ORANGE),
        ("Mio 9B (DFlash+TQ)", 108, CYAN),
        ("GPT-4o (cloud)", 80, PURPLE),
        ("Claude Opus (cloud)", 60, PURPLE),
        ("Mio 27B (DFlash+TQ)", 56, (100, 130, 220)),
        ("Ollama 27B", 24, GRAY_BAR),
    ], unit="tok/s", max_val=240,
       title="Generation Speed: Local Mio vs Cloud APIs (tok/s)")

    pdf.bold_text(15, 188, "35B-A3B: 204 tok/s > GPT-4o: 80 tok/s. Smarter than 9B, faster than cloud.", 10, ORANGE)

    # ==========================================================
    # SLIDE 5: KV CACHE  - Compression
    # ==========================================================
    pdf.slide("KV Cache: TurboQuant Compression", "How much cache memory at each context length?")

    pdf.grouped_bars(15, 42, 265, 150,
        [
            ("32K ctx",  [(2.0, RED), (0.5, CYAN), (0.2, GREEN)]),
            ("128K ctx", [(8.0, RED), (2.0, CYAN), (1.0, GREEN)]),
            ("265K ctx", [(16.6, RED), (4.1, CYAN), (2.1, GREEN)]),
            ("512K ctx", [(32.0, RED), (8.0, CYAN), (4.0, GREEN)]),
            ("1M ctx",   [(62.5, RED), (15.6, CYAN), (7.8, GREEN)]),
        ],
        [("fp16 (no TQ)", RED), ("TQ 4-bit", CYAN), ("TQ 2-bit", GREEN)],
        unit="GB", max_val=70,
        title="KV Cache VRAM: Qwen3.5-27B (GB, lower is better)")

    pdf.bold_text(15, 185, "Without TQ: 1M ctx = 63 GB (impossible)  |  With TQ 2-bit: 1M ctx = 8 GB (easy!)", 9, GREEN)

    # ==========================================================
    # SLIDE 5b: KV CACHE  - Total VRAM
    # ==========================================================
    pdf.slide("KV Cache: Total VRAM Impact", "Model weights + DFlash draft + KV cache + overhead")

    pdf.bar_chart(15, 42, 265, 150, [
        ("27B + TQ4 @ 128K", 18.1, CYAN),
        ("27B + TQ4 @ 265K", 20.2, CYAN),
        ("27B + TQ2 @ 512K", 20.1, GREEN),
        ("27B + TQ2 @ 1M", 23.9, GREEN),
        ("27B + fp16 @ 128K", 24.1, RED),
        ("27B + fp16 @ 265K", 32.7, RED),
        ("27B + fp16 @ 512K", 48.1, RED),
        ("48 GB M4 Max limit", 48, GRAY_BAR),
    ], unit="GB", max_val=55,
       title="Total System VRAM: Qwen3.5-27B 4-bit (GB)")

    pdf.bold_text(15, 192, "TQ4: 265K in 20 GB  |  TQ2: 1M in 24 GB  |  fp16: OOM at 512K", 9, ORANGE)

    # ==========================================================
    # SLIDE 6: CONTEXT  - How Far Can You Go
    # ==========================================================
    pdf.slide("Context: Up to 1M Tokens", "5x more context than any cloud agent. 100% local.")

    pdf.bar_chart(15, 42, 265, 150, [
        ("Mio 27B (TQ 2-bit)", 1000, CYAN),
        ("Mio 27B (TQ 4-bit)", 512, (80, 170, 220)),
        ("Mio 9B (TQ 4-bit)", 512, GREEN),
        ("Claude Opus (cloud)", 200, PURPLE),
        ("Ollama 27B", 200, GRAY_BAR),
        ("LM Studio 27B", 200, GRAY_BAR),
        ("GPT-4o (cloud)", 128, PURPLE),
        ("Cursor", 128, GRAY_BAR),
        ("Aider", 128, GRAY_BAR),
    ], unit="K tokens", max_val=1100,
       title="Maximum Context Window (K tokens, higher is better)")

    pdf.bold_text(15, 192, "1M tokens = 50K lines of code = entire codebase in one prompt", 9, ORANGE)

    # ==========================================================
    # SLIDE 7: vs OLLAMA / LM STUDIO  - Speed
    # ==========================================================
    pdf.slide("Ollama / LM Studio Replacement: Speed", "Same OpenAI API. 2-8x faster. Drop-in swap.")

    pdf.grouped_bars(15, 42, 265, 120,
        [
            ("Qwen3.5-27B gen", [(56, CYAN), (24, GRAY_BAR), (26, (70, 80, 100))]),
            ("Qwen3.5-9B gen",  [(108, CYAN), (24, GRAY_BAR), (26, (70, 80, 100))]),
            ("Qwen3.5-35B-A3B gen",  [(204, CYAN), (24, GRAY_BAR), (26, (70, 80, 100))]),
        ],
        [("Mio (DFlash+TQ)", CYAN), ("Ollama", GRAY_BAR), ("LM Studio", (70, 80, 100))],
        unit="tok/s", max_val=240,
        title="Generation Speed Comparison (tok/s, higher is better)")

    pdf.bold_text(15, 172, "Migration: change one URL", 12, GREEN)
    pdf.set_fill_color(*CARD)
    pdf.rect(15, 183, 267, 14, "F")
    pdf.set_font("Courier", "B", 8)
    pdf.set_text_color(*DIM)
    pdf.text(22, 190, 'Before: base_url="http://localhost:11434/v1"')
    pdf.set_text_color(*GREEN)
    pdf.text(165, 190, 'After: base_url="http://localhost:9090/v1"')

    # ==========================================================
    # SLIDE 7b: vs OLLAMA / LM STUDIO  - Context
    # ==========================================================
    pdf.slide("Ollama / LM Studio Replacement: Context", "5x longer context windows with TurboQuant KV compression")

    pdf.grouped_bars(15, 42, 265, 140,
        [
            ("27B TQ 4-bit", [(512, CYAN), (200, GRAY_BAR), (200, (70, 80, 100))]),
            ("27B TQ 2-bit", [(1000, CYAN), (200, GRAY_BAR), (200, (70, 80, 100))]),
            ("9B TQ 4-bit",  [(512, CYAN), (200, GRAY_BAR), (200, (70, 80, 100))]),
            ("9B TQ 2-bit",  [(1000, CYAN), (200, GRAY_BAR), (200, (70, 80, 100))]),
        ],
        [("Mio", CYAN), ("Ollama", GRAY_BAR), ("LM Studio", (70, 80, 100))],
        unit="K tokens", max_val=1100,
        title="Maximum Context Window (K tokens, higher is better)")

    pdf.bold_text(15, 192, "Ollama/LM Studio: 200K max  |  Mio: up to 1M tokens. Same API.", 9, ORANGE)

    # ==========================================================
    # SLIDE 8: CUDA  - Generation Speed
    # ==========================================================
    pdf.slide("CUDA: Generation Speed Estimates", "Mio for NVIDIA  (in development)")

    pdf.grouped_bars(15, 42, 265, 130,
        [
            ("Qwen3.5-27B 4bit", [(56, M4_CLR), (95, C5080), (140, C5090)]),
            ("Qwen3.5-9B bf16",  [(108, M4_CLR), (180, C5080), (270, C5090)]),
            ("Qwen3.5-35B-A3B MoE",  [(204, M4_CLR), (340, C5080), (510, C5090)]),
        ],
        [("M4 Max 48GB", M4_CLR), ("RTX 5080 16GB", C5080), ("RTX 5090 32GB", C5090)],
        unit="tok/s", max_val=580,
        title="Estimated tok/s with Speculative Decoding + KV Quantization")

    pdf.bold_text(15, 178, "RTX 5090: 140 tok/s on 27B  - first consumer GPU to beat cloud APIs.", 11, ORANGE)
    pdf.text_block(15, 190, [
        "Cloud reference: GPT-4o ~80 tok/s  |  Claude Opus ~60 tok/s  |  Both exceeded by RTX 5090 locally",
    ], size=8.5)

    # ==========================================================
    # SLIDE 9: CUDA  - RTX 5080 Detail
    # ==========================================================
    pdf.slide("CUDA: RTX 5080 (16 GB GDDR7, ~960 GB/s)", "Best price-to-performance  |  $1,000  |  2.4x M4 Max bandwidth")

    pdf.bar_chart(15, 42, 265, 140, [
        ("35B-A3B MoE + spec + KVq", 340, C5080),
        ("9B 4bit + spec + KVq", 180, C5080),
        ("27B 4bit + spec + KVq", 95, C5080),
        ("27B 4bit + spec (no KVq)", 85, (120, 200, 140)),
        ("27B 4bit baseline (no spec)", 36, GRAY_BAR),
        ("Cloud: GPT-4o", 80, PURPLE),
        ("Cloud: Claude Opus", 60, PURPLE),
    ], unit="tok/s", max_val=400,
       title="RTX 5080 Estimated Generation Speed (tok/s)")

    pdf.text_block(15, 188, [
        "16 GB VRAM: 27B fits in 4-bit only, max ~128K context (TQ4)  |  9B is the sweet spot on this GPU",
    ], size=8.5)

    # ==========================================================
    # SLIDE 10: CUDA  - RTX 5090 Detail
    # ==========================================================
    pdf.slide("CUDA: RTX 5090 (32 GB GDDR7, ~1,792 GB/s)", "The speed king  |  $2,000  |  4.5x M4 Max bandwidth")

    pdf.bar_chart(15, 42, 265, 140, [
        ("35B-A3B MoE + spec + KVq", 510, C5090),
        ("9B 4bit + spec + KVq", 270, C5090),
        ("27B 4bit + spec + KVq", 140, C5090),
        ("27B fp16 + spec + KVq", 75, (200, 170, 80)),
        ("27B 4bit baseline (no spec)", 65, GRAY_BAR),
        ("Cloud: GPT-4o", 80, PURPLE),
        ("Cloud: Claude Opus", 60, PURPLE),
    ], unit="tok/s", max_val=580,
       title="RTX 5090 Estimated Generation Speed (tok/s)")

    pdf.bold_text(15, 188, "27B @ 140 tok/s local  >  GPT-4o @ 80 tok/s cloud.  Zero API cost.", 10, GREEN)

    # ==========================================================
    # SLIDE 11: CUDA  - Context Windows
    # ==========================================================
    pdf.slide("CUDA: Context Windows by Hardware", "M4 Max wins on memory. RTX 5090 wins on speed.")

    pdf.grouped_bars(15, 42, 265, 130,
        [
            ("27B TQ 4-bit", [(512, M4_CLR), (128, C5080), (512, C5090)]),
            ("27B TQ 2-bit", [(1000, M4_CLR), (256, C5080), (1000, C5090)]),
            ("35B-A3B TQ 4-bit",  [(384, M4_CLR), (128, C5080), (384, C5090)]),
            ("9B TQ 4-bit",  [(512, M4_CLR), (512, C5080), (512, C5090)]),
        ],
        [("M4 Max 48GB", M4_CLR), ("RTX 5080 16GB", C5080), ("RTX 5090 32GB", C5090)],
        unit="K tokens", max_val=1100,
        title="Maximum Context Window (K tokens, higher is better)")

    pdf.text_block(15, 180, [
        "M4 Max: 48GB unified = 1M ctx portable  |  5080: 16GB = limited 27B ctx, great for 9B",
        "RTX 5090: 32GB = 1M ctx + 140 tok/s = the ultimate local setup",
    ], size=8.5)

    # ==========================================================
    # SLIDE 12: Model Tier Speed
    # ==========================================================
    pdf.slide("Model Lineup: Speed by Tier", "All Qwen 3.5 + DFlash drafts  |  TurboQuant V2 default")

    pdf.bar_chart(15, 42, 265, 150, [
        ("MoE:    Qwen3.5-35B-A3B (3B active)", 204, ORANGE),
        ("Medium: Qwen3.5-9B bf16", 108, CYAN),
        ("Large:  Qwen3.5-27B Opus 4-bit", 56, PURPLE),
        ("Large alt: Qwen3.5-27B std 8-bit", 28, GRAY_BAR),
    ], unit="tok/s", max_val=240,
       title="Generation Speed by Tier (tok/s, M4 Max, DFlash+TQ)")

    pdf.bold_text(15, 188, "35B-A3B MoE: 35B total, 3B active/token = fast like small, smart like large", 9, ORANGE)

    # ==========================================================
    # SLIDE 13: Model Tier VRAM
    # ==========================================================
    pdf.slide("Model Lineup: VRAM Budget", "All three tiers fit simultaneously for tandem routing")

    pdf.bar_chart(15, 42, 265, 150, [
        ("35B-A3B MoE 4-bit weights", 17, ORANGE),
        ("35B-A3B DFlash draft", 0.5, (200, 170, 80)),
        ("9B 4-bit weights", 5.5, CYAN),
        ("9B DFlash draft", 1.0, (100, 170, 220)),
        ("35B-A3B MoE 4-bit weights", 17, ORANGE),
        ("35B-A3B DFlash draft", 0.5, (200, 170, 80)),
        ("27B Opus 4-bit weights", 14.1, PURPLE),
        ("27B DFlash draft", 1.5, (150, 120, 220)),
        ("48 GB M4 Max limit", 48, (50, 50, 65)),
    ], unit="GB", max_val=52,
       title="VRAM Budget (GB)")

    # ==========================================================
    # SLIDE 14: SYSTEM PROMPT - Caveman Ultra
    # ==========================================================
    pdf.slide("Caveman Ultra System Prompt", "Reduce token usage ~75% with zero quality loss on code output")

    pdf.set_fill_color(*CARD)
    pdf.rect(15, 40, 267, 65, "F")
    pdf.bold_text(22, 43, "What is Caveman Mode?", 11, CYAN)
    pdf.text_block(22, 55, [
        "A system prompt that forces terse, technical-only output from local models.",
        "Strips: articles (a/an/the), filler (just/really/basically), pleasantries (sure/certainly),",
        "hedging, and fluff. Code blocks and commits are always written normally.",
        "",
        'Not: "Sure! I\'d be happy to help you with that. The issue you\'re experiencing is likely..."',
        'Yes: "Bug in auth middleware. Token expiry check use < not <=. Fix:"',
    ], size=8, color=DIM, line_h=7)

    pdf.bold_text(22, 108, "4 intensity levels:", 10, WHITE)
    pdf.bar_chart(15, 118, 265, 72, [
        ("ultra (default)", 75, CYAN),
        ("full (classic caveman)", 60, GREEN),
        ("lite (professional tight)", 35, ORANGE),
        ("off (normal prose)", 0, GRAY_BAR),
    ], unit="% saved", max_val=85,
       title="Estimated Token Reduction by Caveman Level (%)")

    pdf.bold_text(15, 192, "Toggle: /caveman off|lite|full|ultra  |  Auto-clarity for security warnings", 8, DIM)

    # ==========================================================
    # SLIDE 15: TOKEN SAVINGS BREAKDOWN
    # ==========================================================
    pdf.slide("Token Savings: Why It Matters Locally", "Every saved token = faster response + more context budget")

    pdf.bar_chart(15, 42, 265, 70, [
        ("Standard verbose LLM output", 850, RED),
        ("Caveman lite", 550, ORANGE),
        ("Caveman full", 340, GREEN),
        ("Caveman ultra (default)", 215, CYAN),
    ], unit="tokens", max_val=950,
       title="Typical Response Length: 'Explain why this React component re-renders' (tokens)")

    pdf.bold_text(15, 118, "Impact on local inference:", 11, WHITE)
    pdf.bar_chart(15, 130, 265, 60, [
        ("Time @ 56 tok/s (verbose)", 15.2, RED),
        ("Time @ 56 tok/s (ultra)", 3.8, CYAN),
        ("Time @ 108 tok/s (verbose)", 7.9, (180, 100, 100)),
        ("Time @ 108 tok/s (ultra)", 2.0, GREEN),
    ], unit="sec", max_val=18,
       title="Response Time: 27B and 9B Models (seconds, lower is better)")

    pdf.bold_text(15, 192, "4x fewer tokens = 4x faster responses = 4x more turns in context window", 9, ORANGE)

    # ==========================================================
    # SLIDE 16: BEST vs WORST CASE - 10K Coding Task
    # ==========================================================
    pdf.slide("10K Coding Task: Best vs Worst Case", "Vanilla no-system-prompt vs Mio maxed out")

    # Scenario: user sends 10K token coding prompt, model generates response
    # Vanilla: verbose model, no spec decode, fp16 KV, standard 27B
    # Mio: caveman ultra, DFlash+TQ, 27B Opus

    pdf.bold_text(15, 40, "Scenario: 10,000-token coding prompt, 27B model, M4 Max 48GB", 10, WHITE)

    # Response tokens
    pdf.bar_chart(15, 52, 125, 55, [
        ("Vanilla (verbose)", 2400, RED),
        ("Mio (ultra)", 600, CYAN),
    ], unit="tok", max_val=2700,
       title="Response Tokens Generated")

    # Time to respond
    pdf.bar_chart(150, 52, 135, 55, [
        ("Vanilla (no spec)", 120, RED),
        ("Mio (DFlash+TQ)", 11.2, CYAN),
    ], unit="sec", max_val=140,
       title="Time to Full Response (sec)")

    # KV cache used
    pdf.bar_chart(15, 112, 125, 55, [
        ("Vanilla (fp16 KV)", 0.77, RED),
        ("Mio (TQ 4-bit)", 0.19, CYAN),
    ], unit="GB", max_val=0.9,
       title="KV Cache for 10K Prompt (GB)")

    # Effective context remaining
    pdf.bar_chart(150, 112, 135, 55, [
        ("Vanilla (fp16)", 190, RED),
        ("Mio (TQ4)", 750, CYAN),
    ], unit="K left", max_val=850,
       title="Context Budget Remaining (K tokens)")

    pdf.set_fill_color(*CARD)
    pdf.rect(15, 172, 267, 24, "F")
    pdf.bold_text(22, 174, "Summary for a single 10K coding task:", 10, WHITE)
    pdf.text_block(22, 184, [
        "Vanilla: 2,400 tok response, 120 sec, 0.77 GB cache, 190K ctx left  |  Cost: $0.04 cloud or 120s local",
        "Mio:    600 tok response,  11 sec, 0.19 GB cache, 750K ctx left  |  Cost: $0 and 10.7x faster",
    ], size=7.5, color=DIM, line_h=6)

    # ==========================================================
    # SLIDE 17: FULL COMPARISON TABLE AS BARS
    # ==========================================================
    pdf.slide("10K Task: Full Speedup Breakdown", "Stacked improvements: system prompt + DFlash + TurboQuant")

    pdf.bar_chart(15, 42, 265, 95, [
        ("Vanilla 27B (no opt)", 120, RED),
        ("+ Caveman ultra only", 45, ORANGE),
        ("+ Caveman + DFlash", 11.2, GREEN),
        ("+ Caveman + DFlash + TQ4", 11.2, CYAN),
        ("Mio 9B (all opt)", 18, (100, 220, 200)),
        ("Mio 35B-A3B MoE (all opt)", 11, ORANGE),
        ("Cloud: GPT-4o", 5.0, PURPLE),
        ("Cloud: Claude Opus", 7.0, PURPLE),
    ], unit="sec", max_val=140,
       title="Time to Respond to 10K Coding Prompt (seconds, lower is better)")

    # Speedup multipliers
    pdf.bold_text(15, 142, "Cumulative Speedup (vs Vanilla Ollama 27B @ 120s):", 10, WHITE)
    pdf.bar_chart(15, 152, 265, 40, [
        ("Caveman ultra (4x fewer tokens)", 2.7, ORANGE),
        ("+ DFlash (8.5x decode)", 10.7, GREEN),
        ("+ TurboQuant (context headroom)", 10.7, CYAN),
        ("Mio 35B-A3B full stack", 10.7, (100, 220, 200)),
    ], unit="x faster", max_val=13,
       title="Speedup Multiplier (X times faster)")

    pdf.set_fill_color(*CARD)
    pdf.rect(15, 193, 267, 5, "F")
    pdf.text_block(17, 193, [
        "Vanilla: 2min  |  +Caveman: 45s  |  +DFlash: 11s  |  Mio 9B: 18s  |  35B-A3B MoE: 11s  |  GPT-4o: 5.0s"
    ], size=7, color=GREEN)

    # ==========================================================
    # SLIDE 18: NATIVE AGENT
    # ==========================================================
    pdf.slide("Native Coding Agent", "Just type mio. Pure Python. No Node.js.")

    pdf.set_fill_color(*CARD)
    pdf.rect(15, 40, 267, 48, "F")
    pdf.set_font("Courier", "B", 11)
    pdf.set_text_color(*GREEN)
    pdf.text(22, 50, "$ mio")
    pdf.set_font("Courier", "", 9)
    pdf.set_text_color(*DIM)
    pdf.text(22, 60, "+---------------------------+")
    pdf.text(22, 67, "| Mio Agent              |")
    pdf.text(22, 74, "| Tier: large | Caveman: ultra |")
    pdf.text(22, 81, "+---------------------------+")

    pdf.bold_text(15, 96, "Slash Commands:", 11, WHITE)

    cmds = [
        ("/model", "Show current model, tier, context, TQ config"),
        ("/tier large|medium|small", "Switch model tier on the fly"),
        ("/caveman off|lite|full|ultra", "Toggle token-saving mode"),
        ("/tq", "Show TurboQuant cache settings"),
        ("/status", "Engine status: loaded tiers, VRAM, tok/s"),
        ("/models", "List all available model keys"),
        ("/configure", "Run config wizard inline"),
        ("/clear", "Clear conversation history"),
    ]
    y = 108
    for cmd, desc in cmds:
        pdf.set_font("Courier", "B", 8)
        pdf.set_text_color(*CYAN)
        pdf.text(22, y, cmd)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(*DIM)
        pdf.text(120, y, desc)
        y += 9

    pdf.bold_text(15, 188, "Streaming output | Caveman ultra default | No external dependencies", 9, GREEN)

    # ==========================================================
    # SLIDE 19: v0.2 FEATURES
    # ==========================================================
    pdf.slide("v0.2 Features", "7 new capabilities beyond core inference")

    features = [
        ("TurboQuant Wired", "KV cache compression now actually active (was defined but not called)", CYAN),
        ("Code Validation", "Auto ruff + syntax check, retry with error context (max 2 retries)", GREEN),
        ("Web Dashboard", "Live metrics at /dashboard - tok/s, VRAM, acceptance rate", ORANGE),
        ("Model Pull", "mio pull qwen3.5-27b - downloads target + DFlash draft in one command", PURPLE),
        ("Batch Inference", "mio batch --input prompts.jsonl - process N prompts with DFlash", CYAN),
        ("KV Cache Persist", "Save/restore cache to disk - resume sessions without re-prefill", GREEN),
        ("Prefix Caching", "Reuse KV for shared system prompts - saves ~1s per request", ORANGE),
    ]

    y = 42
    for name, desc, color in features:
        pdf.set_fill_color(*CARD)
        pdf.rect(15, y, 267, 16, "F")
        pdf.set_fill_color(*color)
        pdf.rect(15, y, 3, 16, "F")  # Color accent bar
        pdf.bold_text(22, y + 2, name, 10, color)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*DIM)
        pdf.text(22, y + 11, desc)
        y += 19

    pdf.bold_text(15, y + 5, "All opt-in except TQ wiring and prefix caching (always active).", 9, DIM)

    # ==========================================================
    # SLIDE 18b: v0.3 AGENT-LOOP FEATURES
    # ==========================================================
    pdf.slide("v0.3 Agent-Loop Features", "Multi-turn tool calling with Kilo / Cline / OpenAI SDKs")

    v3_features = [
        ("Relaxed Prefix Cache", "Longest-common-prefix + KV truncation. 94% hit rate on 50-turn agent session.", CYAN),
        ("OpenAI Function Calling", "tools field -> Qwen native <tool_call> XML -> parsed back to tool_calls JSON", GREEN),
        ("Unsloth MLX Q4_K_XL", "Brooooooklyn UD-Q4_K_XL re-quants fix mlx-community tool-call degradation", ORANGE),
        ("Conditional EOS Suppress", "Mask EOS first 40 tokens; keeps acceptance ~0.90 while preventing narrate-then-stop", PURPLE),
        ("Live Serve TUI", "rich.Live panel: per-req decode/prefill tok/s, accept, cache hits, tool-call detection", CYAN),
        ("tool_calls Round-Trip", "Preserve tool_calls + tool_call_id + role=tool so cache prefix matches cross-turn", GREEN),
        ("GPU Lock + SSE Keepalive", "Serial Metal encoder + 10s keepalives; parallel Kilo requests don't crash/timeout", ORANGE),
    ]

    y = 42
    for name, desc, color in v3_features:
        pdf.set_fill_color(*CARD)
        pdf.rect(15, y, 267, 16, "F")
        pdf.set_fill_color(*color)
        pdf.rect(15, y, 3, 16, "F")
        pdf.bold_text(22, y + 2, name, 10, color)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*DIM)
        pdf.text(22, y + 11, desc)
        y += 19

    pdf.bold_text(15, y + 5, "Drop-in backend for Kilo Code, Cline, and any OpenAI-spec coding agent.", 9, DIM)

    # ==========================================================
    # SLIDE 18c: PRODUCTION BENCHMARK — 50-TURN KILO AGENT
    # ==========================================================
    pdf.slide("Production Benchmark: 50-Turn Kilo Agent",
              "Backend code-audit task, Qwen3.5-35B-A3B large-moe, 88K-token context")

    # Left column: big numbers
    pdf.set_fill_color(*CARD)
    pdf.rect(15, 40, 130, 120, "F")
    pdf.set_fill_color(*CYAN)
    pdf.rect(15, 40, 3, 120, "F")

    pdf.bold_text(22, 46, "HEADLINE METRICS", 10, CYAN)

    big_stats = [
        ("94%", "cache hit rate (47 / 50)", CYAN),
        ("1.68M", "tokens saved from prefill", GREEN),
        ("173K", "peak prefill tok/s (burst)", ORANGE),
        ("168 tok/s", "peak decode (DFlash 0.90 accept)", PURPLE),
        ("~2.5 s", "wall per warm turn (vs ~50 s cold)", CYAN),
    ]
    sy = 58
    for val, desc, color in big_stats:
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(*color)
        pdf.text(22, sy, val)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*DIM)
        pdf.text(22, sy + 6, desc)
        sy += 20

    # Right column: comparison table v0.2 vs v0.3
    pdf.set_fill_color(*CARD)
    pdf.rect(152, 40, 130, 120, "F")
    pdf.set_fill_color(*GREEN)
    pdf.rect(152, 40, 3, 120, "F")

    pdf.bold_text(159, 46, "v0.2 vs v0.3 (same load)", 10, GREEN)

    pdf.set_font("Helvetica", "B", 7)
    pdf.set_text_color(*WHITE)
    pdf.text(159, 58, "Metric")
    pdf.text(215, 58, "v0.2")
    pdf.text(245, 58, "v0.3")

    rows = [
        ("Cache hit rate", "0%", "94%"),
        ("Tokens saved", "0", "1.68M"),
        ("Peak prefill tok/s", "~1,100", "173,799"),
        ("Avg decode tok/s", "53.2", "134.5"),
        ("Max decode tok/s", "54.6", "168.8"),
        ("Accept (long gens)", "0.67", "0.80-0.90"),
        ("Tool-call XML", "~50%", "100%"),
        ("Wall per turn", "~50 s", "~2.5 s"),
    ]
    ry = 66
    for metric, v2, v3 in rows:
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(*DIM)
        pdf.text(159, ry, metric)
        pdf.set_text_color(*RED)
        pdf.text(215, ry, v2)
        pdf.set_text_color(*GREEN)
        pdf.bold_text(245, ry - 1, v3, 8, GREEN)
        ry += 9

    pdf.bold_text(15, 168, "Without v0.3: 88K-ctx agent unusable (50s per turn). With v0.3: ~2.5s warm turn, fully local.", 9, WHITE)

    # ==========================================================
    # SLIDE 18d: PREFILL SPEEDUP STACK
    # ==========================================================
    pdf.slide("Prefill Speedup Stack",
              "Five optimizations stacked. End-to-end: 158x peak prefill, 20x warm TTFT.")

    pdf.bold_text(15, 36, "FIVE STAGES (in order of compound effect)", 10, CYAN)

    stages = [
        ("1. Chunked prefill (2K tok chunks)", "prevents Metal 30 GB single-buffer crash on 100K+ ctx", CYAN),
        ("2. LM-head slicing (only_last_logit)", "+13-15% prefill tok/s; skips wasted vocab matmuls", GREEN),
        ("3. Prefix cache (strict prefix)", "4.7-8.6x warm speedup on unchanged system prompt", ORANGE),
        ("4. Relaxed cache + KV truncation", "LCS lookup + trim survives Qwen <think> divergence", PURPLE),
        ("5. Cache-budget sizing (per-tier)", "max entries scaled to ctx window, prevents OOM", CYAN),
    ]

    y = 52
    for name, desc, color in stages:
        pdf.set_fill_color(*CARD)
        pdf.rect(15, y, 267, 16, "F")
        pdf.set_fill_color(*color)
        pdf.rect(15, y, 3, 16, "F")
        pdf.bold_text(22, y + 2, name, 10, color)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*DIM)
        pdf.text(22, y + 11, desc)
        y += 19

    # End-to-end summary box
    pdf.set_fill_color(*CARD)
    pdf.rect(15, 155, 267, 35, "F")
    pdf.set_fill_color(*GREEN)
    pdf.rect(15, 155, 3, 35, "F")
    pdf.bold_text(22, 161, "COMBINED EFFECT ON WARM TURN", 10, GREEN)
    summary_lines = [
        "Peak prefill tok/s: ~1,100 (cold)  ->  173,799 (warm hit)  [~158x]",
        "Warm turn at 88K ctx: ~50 s  ->  ~2.5 s  [~20x]",
        "Cache hit rate (Kilo 50-turn session): 0%  ->  94%",
    ]
    sy = 171
    pdf.set_font("Helvetica", "", 8.5)
    for line in summary_lines:
        pdf.set_text_color(*WHITE)
        pdf.text(22, sy, line)
        sy += 6

    # ==========================================================
    # SLIDE 18e: CONTEXT AUTO-COMPACTION (v0.4)
    # ==========================================================
    pdf.slide("v0.4: Context Auto-Compaction",
              "Two-stage compaction at 75% ctx window. Fixes Metal OOM on long agent sessions.")

    pdf.bold_text(15, 36, "TRIGGER", 10, CYAN)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*WHITE)
    pdf.text(15, 44, "When rendered prompt > 75% of context window, reduce to ~50% in place before prefill.")

    # Two-column stage layout
    pdf.set_fill_color(*CARD)
    pdf.rect(15, 55, 130, 105, "F")
    pdf.set_fill_color(*GREEN)
    pdf.rect(15, 55, 3, 105, "F")
    pdf.bold_text(22, 61, "STAGE 1: Tool-Result Truncation", 10, GREEN)
    s1 = [
        "Walk messages oldest -> newest",
        "Protect last 4 atomic groups",
        "role=tool content > 500 chars ->",
        "  [compacted: tool=read call_id=c1,",
        "   original was 11612 chars - elided]",
        "",
        "No LLM call. <5 ms typical.",
        "Reclaims 80-95% of prompt tokens",
        "in Kilo-style sessions.",
    ]
    sy = 71
    pdf.set_font("Helvetica", "", 8)
    for line in s1:
        pdf.set_text_color(*DIM if line.startswith(" ") or line == "" else WHITE)
        pdf.text(22, sy, line)
        sy += 9

    pdf.set_fill_color(*CARD)
    pdf.rect(152, 55, 130, 105, "F")
    pdf.set_fill_color(*PURPLE)
    pdf.rect(152, 55, 3, 105, "F")
    pdf.bold_text(159, 61, "STAGE 2: LLM Summarization", 10, PURPLE)
    s2 = [
        "Only if stage 1 insufficient.",
        "",
        "Take middle span (except system",
        "+ last 4 groups), ask current engine:",
        "  summarize in <300 words,",
        "  preserve files/commands read,",
        "  decisions, open questions",
        "",
        "Middle collapses to 1 synthetic",
        "user turn. ~1-2 s added latency.",
    ]
    sy = 71
    pdf.set_font("Helvetica", "", 8)
    for line in s2:
        pdf.set_text_color(*DIM if line.startswith(" ") or line == "" else WHITE)
        pdf.text(159, sy, line)
        sy += 9

    # Bottom: atomic-group invariant
    pdf.set_fill_color(*CARD)
    pdf.rect(15, 165, 267, 28, "F")
    pdf.set_fill_color(*ORANGE)
    pdf.rect(15, 165, 3, 28, "F")
    pdf.bold_text(22, 171, "ATOMIC-GROUP INVARIANT", 9, ORANGE)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*WHITE)
    pdf.text(22, 180, "[assistant with tool_calls] + [all matching role=tool messages] = ONE group.")
    pdf.text(22, 186, "Never split. Qwen's template requires every tool_call_id have a matching tool response.")

    # ==========================================================
    # SLIDE 19: GET STARTED
    # ==========================================================
    pdf.slide("Get Started")

    steps = [
        ("1", "pip install -e .", "Install Mio"),
        ("2", "mio download", "Download models (~22 GB)"),
        ("3", "mio configure", "Select model + DFlash + TQ bits + context"),
        ("4", "mio serve", "Start API server on :9090"),
    ]

    y = 42
    for num, cmd, desc in steps:
        pdf.set_fill_color(*CARD)
        pdf.rect(30, y, 237, 18, "F")
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(*CYAN)
        pdf.text(38, y + 12, num)
        pdf.set_font("Courier", "B", 12)
        pdf.set_text_color(*GREEN)
        pdf.text(52, y + 12, cmd)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*DIM)
        pdf.text(175, y + 12, desc)
        y += 24

    pdf.bold_text(30, y + 8, "Then connect any agent:", 11, WHITE)

    pdf.set_fill_color(*CARD)
    pdf.rect(30, y + 18, 237, 30, "F")
    pdf.set_font("Courier", "", 8.5)
    pdf.set_text_color(*GREEN)
    cy = y + 24
    for line in [
        'from openai import OpenAI',
        'client = OpenAI(base_url="http://localhost:9090/v1", api_key="local")',
        'r = client.chat.completions.create(model="mio-large", messages=[...], stream=True)',
    ]:
        pdf.text(38, cy, line)
        cy += 8

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*DIM)
    pdf.set_xy(30, 185)
    pdf.cell(237, 6, "MIT License  |  macOS Apple Silicon  |  OpenAI-Compatible API", align="C")

    # Save
    out = "docs/Mio-Presentation.pdf"
    pdf.output(out)
    print(f"PDF saved to: {out}")


if __name__ == "__main__":
    build()
