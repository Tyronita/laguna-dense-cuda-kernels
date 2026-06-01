"""Manim: Laguna MoE vs Dense — geometric layer-by-layer with MSE loss lines.

Shows the actual 40-layer structure with:
- Teacher: attention + MoE (256 experts + shared) per layer
- Student: attention + dense FFN per layer
- MSE loss lines connecting corresponding MoE/dense blocks
- Colour: blue=trainable, grey=frozen, orange=experts, green=shared

Usage: manim -qh laguna_arch_v2.py LagunaLayerComparison
"""
from manim import *
import numpy as np

BLUE = "#4A90D9"
GREY = "#666666"
ORANGE = "#E8913A"
GREEN = "#5CB85C"
RED = "#D94A4A"
CYAN = "#4AD9D9"
BG = "#0f0f1a"

N_LAYERS = 40
LAYER_H = 0.12
GAP = 0.02
TOTAL_H = N_LAYERS * (LAYER_H + GAP)

# Layer types: 0=GA (global attn), 1=SWA (sliding window), ratio 3:1
# Pattern: GA, SWA, SWA, SWA, GA, SWA, SWA, SWA, ...
# Layer 0 is dense (no MoE) for stability
LAYER_TYPES = []
for i in range(N_LAYERS):
    if i % 4 == 0:
        LAYER_TYPES.append("GA")
    else:
        LAYER_TYPES.append("SWA")


class LagunaLayerComparison(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("Laguna-XS.2: Layer-by-Layer Architecture", font_size=28, color=WHITE)
        title.to_edge(UP, buff=0.25)
        self.add(title)

        # ── Column positions ──
        teacher_x = -4.0
        student_x = 4.0
        mid_x = 0.0

        # ── Labels ──
        t_label = Text("Teacher (33B MoE)", font_size=18, color=WHITE)
        t_label.move_to([teacher_x, 3.0, 0])
        s_label = Text("Student (3B Dense)", font_size=18, color=WHITE)
        s_label.move_to([student_x, 3.0, 0])
        self.add(t_label, s_label)

        # Sub-labels
        t_sub = Text("256 experts, top-8 + shared", font_size=11, color=GREY_B)
        t_sub.next_to(t_label, DOWN, buff=0.08)
        s_sub = Text("1 dense SwiGLU FFN (K=8 width)", font_size=11, color=GREY_B)
        s_sub.next_to(s_label, DOWN, buff=0.08)
        self.add(t_sub, s_sub)

        # ── Draw layers ──
        y_start = 2.4
        teacher_layers = VGroup()
        student_layers = VGroup()
        mse_lines = VGroup()

        for i in range(N_LAYERS):
            y = y_start - i * (LAYER_H + GAP)
            lt = LAYER_TYPES[i]

            # ── Teacher layer ──
            # Attention (frozen, grey)
            attn_w = 1.2
            attn_t = Rectangle(width=attn_w, height=LAYER_H, color=GREY,
                              fill_opacity=0.3, stroke_width=0.5)
            attn_t.move_to([teacher_x - 1.0, y, 0])

            # MoE block (orange experts)
            if i == 0:
                # Layer 0 is dense for stability
                moe_t = Rectangle(width=1.8, height=LAYER_H, color=ORANGE,
                                 fill_opacity=0.25, stroke_width=0.8)
                moe_t.move_to([teacher_x + 0.5, y, 0])
            else:
                # Expert grid (small squares)
                moe_t = VGroup()
                moe_bg = Rectangle(width=1.8, height=LAYER_H, color=ORANGE,
                                  fill_opacity=0.1, stroke_width=0.5)
                moe_bg.move_to([teacher_x + 0.5, y, 0])
                moe_t.add(moe_bg)
                # Show a few expert dots
                for e in range(8):
                    dot = Dot(
                        point=[teacher_x + 0.5 - 0.7 + e * 0.2, y, 0],
                        radius=0.02, color=ORANGE
                    )
                    moe_t.add(dot)

            # Shared expert (green, small)
            shared_t = Rectangle(width=0.3, height=LAYER_H, color=GREEN,
                                fill_opacity=0.4, stroke_width=0.5)
            shared_t.move_to([teacher_x + 1.7, y, 0])

            teacher_layers.add(attn_t, moe_t, shared_t)

            # ── Student layer ──
            # Attention (frozen, grey) - identical
            attn_s = Rectangle(width=attn_w, height=LAYER_H, color=GREY,
                              fill_opacity=0.3, stroke_width=0.5)
            attn_s.move_to([student_x - 1.0, y, 0])

            # Dense FFN (blue, trainable)
            dense_s = Rectangle(width=1.8, height=LAYER_H, color=BLUE,
                               fill_opacity=0.5, stroke_width=0.8)
            dense_s.move_to([student_x + 0.5, y, 0])

            # Shared expert (green, frozen)
            shared_s = Rectangle(width=0.3, height=LAYER_H, color=GREEN,
                                fill_opacity=0.4, stroke_width=0.5)
            shared_s.move_to([student_x + 1.7, y, 0])

            student_layers.add(attn_s, dense_s, shared_s)

            # ── MSE loss line (connecting teacher MoE output to student dense) ──
            # Simulate MSE magnitude: deep layers have higher MSE
            # From the training data: shallow ~1e-4, mid ~2e-3, deep ~2e-2
            depth_frac = i / N_LAYERS
            if depth_frac < 0.33:
                mse_val = 8e-5  # shallow
                line_color = GREEN
                line_width = 0.5
            elif depth_frac < 0.66:
                mse_val = 2e-3  # mid
                line_color = ORANGE
                line_width = 1.0
            else:
                mse_val = 1.8e-2  # deep
                line_color = RED
                line_width = 1.5

            # Line from teacher MoE right edge to student dense left edge
            mse_line = Line(
                start=[teacher_x + 1.7 + 0.15, y, 0],
                end=[student_x - 1.0 - attn_w/2 - 0.1, y, 0],
                color=line_color,
                stroke_width=line_width,
                stroke_opacity=0.6,
            )
            mse_lines.add(mse_line)

        # ── Layer type indicators (left side) ──
        for i in range(N_LAYERS):
            y = y_start - i * (LAYER_H + GAP)
            lt = LAYER_TYPES[i]
            if i % 10 == 0:
                label = Text("L%d %s" % (i, lt), font_size=7, color=GREY_B)
                label.move_to([teacher_x - 2.3, y, 0])
                self.add(label)

        # ── Animate ──
        self.play(FadeIn(teacher_layers), run_time=1.5)
        self.play(FadeIn(student_layers), run_time=1.5)
        self.wait(0.5)

        # MSE lines appear
        mse_label = Text("MSE loss (per layer)", font_size=14, color=WHITE)
        mse_label.move_to([mid_x, 3.0, 0])
        self.play(Write(mse_label))
        self.play(LaggedStart(*[Create(line) for line in mse_lines], lag_ratio=0.03), run_time=2)

        # ── Legend ──
        legend_y = -3.2
        items = [
            (GREY, "Attention (frozen, 1.43B)"),
            (ORANGE, "MoE experts (256, teacher only)"),
            (BLUE, "Dense FFN (trainable, 0.98B)"),
            (GREEN, "Shared expert (frozen, 0.12B)"),
        ]
        for j, (col, txt) in enumerate(items):
            sq = Square(side_length=0.15, color=col, fill_opacity=0.6, stroke_width=0)
            sq.move_to([teacher_x + j * 2.8, legend_y, 0])
            lb = Text(txt, font_size=9, color=GREY_B)
            lb.next_to(sq, RIGHT, buff=0.1)
            self.add(sq, lb)

        # MSE legend
        for j, (col, txt) in enumerate([
            (GREEN, "shallow MSE ~1e-4"),
            (ORANGE, "mid MSE ~2e-3"),
            (RED, "deep MSE ~2e-2"),
        ]):
            ln = Line(LEFT * 0.2, RIGHT * 0.2, color=col, stroke_width=2)
            ln.move_to([teacher_x + 1.0 + j * 2.5, legend_y - 0.3, 0])
            lb = Text(txt, font_size=9, color=GREY_B)
            lb.next_to(ln, RIGHT, buff=0.1)
            self.add(ln, lb)

        self.wait(2)
