"""Manim scene: Laguna-XS.2 MoE → Dense student architecture comparison.
Shows expert routing, shared expert, frozen/trainable params.

Usage: manim -pql laguna_arch_manim.py LagunaArchitecture
"""
from manim import *

BLUE_TRAIN = "#4A90D9"
GREY_FROZEN = "#888888"
ORANGE_EXPERT = "#E8913A"
GREEN_SHARED = "#5CB85C"
RED_HIGHLIGHT = "#D94A4A"
BG = "#1a1a2e"


class LagunaArchitecture(Scene):
    def construct(self):
        self.camera.background_color = BG

        # ── Title ──────────────────────────────────────────────
        title = Text("Laguna-XS.2 → Dense: Architecture", font_size=36, color=WHITE)
        subtitle = Text("33B MoE (256 experts) → 3B Dense (K=8 width)", font_size=22, color=GREY_B)
        title.to_edge(UP, buff=0.3)
        subtitle.next_to(title, DOWN, buff=0.15)
        self.play(Write(title), FadeIn(subtitle))
        self.wait(1)

        # ── TEACHER (left side) ────────────────────────────────
        teacher_label = Text("Teacher: Laguna-XS.2", font_size=24, color=WHITE)
        teacher_label.move_to(LEFT * 3.5 + UP * 2.2)

        # Embedding
        embed_t = Rectangle(width=2.5, height=0.4, color=GREY_FROZEN, fill_opacity=0.3)
        embed_t_text = Text("Embeddings (0.41B)", font_size=12, color=GREY_FROZEN)
        embed_t.move_to(LEFT * 3.5 + UP * 1.5)
        embed_t_text.move_to(embed_t)

        # Attention block
        attn_t = Rectangle(width=2.5, height=0.5, color=GREY_FROZEN, fill_opacity=0.3)
        attn_t_text = Text("Attention 48/8 GQA (1.43B)", font_size=11, color=GREY_FROZEN)
        attn_t.next_to(embed_t, DOWN, buff=0.15)
        attn_t_text.move_to(attn_t)

        # MoE block - the star
        moe_box = Rectangle(width=2.5, height=1.8, color=ORANGE_EXPERT, fill_opacity=0.1, stroke_width=2)
        moe_box.next_to(attn_t, DOWN, buff=0.15)
        moe_label = Text("MoE Block (×39 layers)", font_size=13, color=ORANGE_EXPERT)
        moe_label.move_to(moe_box.get_top() + DOWN * 0.2)

        # Expert grid (8x4 = 32 shown, representing 256)
        expert_grid = VGroup()
        for row in range(4):
            for col in range(8):
                sq = Square(side_length=0.18, color=ORANGE_EXPERT, fill_opacity=0.6, stroke_width=0.5)
                sq.move_to(moe_box.get_center() + LEFT * 0.7 + RIGHT * col * 0.22 + UP * 0.3 + DOWN * row * 0.22)
                expert_grid.add(sq)

        expert_count = Text("256 experts (top-8 routed)", font_size=10, color=ORANGE_EXPERT)
        expert_count.move_to(moe_box.get_center() + DOWN * 0.55)

        # Shared expert
        shared_t = Rectangle(width=1.0, height=0.3, color=GREEN_SHARED, fill_opacity=0.4, stroke_width=1.5)
        shared_t_text = Text("shared", font_size=10, color=GREEN_SHARED)
        shared_t.move_to(moe_box.get_bottom() + UP * 0.25 + RIGHT * 0.6)
        shared_t_text.move_to(shared_t)

        # LM head
        head_t = Rectangle(width=2.5, height=0.4, color=GREY_FROZEN, fill_opacity=0.3)
        head_t_text = Text("LM Head", font_size=12, color=GREY_FROZEN)
        head_t.next_to(moe_box, DOWN, buff=0.15)
        head_t_text.move_to(head_t)

        # Total
        total_t = Text("33.4B total / 3B active", font_size=14, color=WHITE)
        total_t.next_to(head_t, DOWN, buff=0.2)
        vram_t = Text("67 GB VRAM", font_size=12, color=RED_HIGHLIGHT)
        vram_t.next_to(total_t, DOWN, buff=0.1)

        teacher_group = VGroup(teacher_label, embed_t, embed_t_text, attn_t, attn_t_text,
                               moe_box, moe_label, expert_grid, expert_count, shared_t,
                               shared_t_text, head_t, head_t_text, total_t, vram_t)

        self.play(FadeIn(teacher_label))
        self.play(FadeIn(embed_t), Write(embed_t_text))
        self.play(FadeIn(attn_t), Write(attn_t_text))
        self.play(FadeIn(moe_box), Write(moe_label))
        self.play(LaggedStart(*[FadeIn(sq, scale=0.5) for sq in expert_grid], lag_ratio=0.02))
        self.play(Write(expert_count))
        self.play(FadeIn(shared_t), Write(shared_t_text))
        self.play(FadeIn(head_t), Write(head_t_text))
        self.play(Write(total_t), Write(vram_t))
        self.wait(1)

        # ── Arrow ──────────────────────────────────────────────
        arrow = Arrow(LEFT * 0.8, RIGHT * 0.8, color=WHITE, stroke_width=3)
        arrow.move_to(ORIGIN + UP * 0)
        arrow_text = Text("densify\nDO-ACP", font_size=14, color=WHITE)
        arrow_text.next_to(arrow, UP, buff=0.1)
        self.play(GrowArrow(arrow), Write(arrow_text))
        self.wait(0.5)

        # ── STUDENT (right side) ───────────────────────────────
        student_label = Text("Student: Dense (3B)", font_size=24, color=WHITE)
        student_label.move_to(RIGHT * 3.5 + UP * 2.2)

        # Embedding - trainable (lm_head only)
        embed_s = Rectangle(width=2.5, height=0.4, color=GREY_FROZEN, fill_opacity=0.3)
        embed_s_text = Text("Embeddings (0.41B)", font_size=12, color=GREY_FROZEN)
        embed_s.move_to(RIGHT * 3.5 + UP * 1.5)
        embed_s_text.move_to(embed_s)

        # Attention - frozen
        attn_s = Rectangle(width=2.5, height=0.5, color=GREY_FROZEN, fill_opacity=0.3)
        attn_s_text = Text("Attention 48/8 GQA (1.43B)", font_size=11, color=GREY_FROZEN)
        attn_s.next_to(embed_s, DOWN, buff=0.15)
        attn_s_text.move_to(attn_s)

        # Dense FFN block - TRAINABLE
        dense_box = Rectangle(width=2.5, height=1.8, color=BLUE_TRAIN, fill_opacity=0.15, stroke_width=2.5)
        dense_box.next_to(attn_s, DOWN, buff=0.15)
        dense_label = Text("Dense FFN (×39 layers)", font_size=13, color=BLUE_TRAIN)
        dense_label.move_to(dense_box.get_top() + DOWN * 0.2)

        # Single wide FFN bar
        ffn_bar = Rectangle(width=2.0, height=0.5, color=BLUE_TRAIN, fill_opacity=0.5, stroke_width=2)
        ffn_bar.move_to(dense_box.get_center() + UP * 0.1)
        ffn_text = Text("1 SwiGLU FFN\nwidth K8×512 = 4096\n0.98B trainable", font_size=10, color=WHITE)
        ffn_text.move_to(ffn_bar)

        # Shared expert - frozen
        shared_s = Rectangle(width=1.0, height=0.3, color=GREEN_SHARED, fill_opacity=0.4, stroke_width=1.5)
        shared_s_text = Text("shared", font_size=10, color=GREEN_SHARED)
        shared_s.move_to(dense_box.get_bottom() + UP * 0.25 + RIGHT * 0.6)
        shared_s_text.move_to(shared_s)

        frozen_label = Text("frozen", font_size=9, color=GREY_FROZEN)
        frozen_label.next_to(shared_s, LEFT, buff=0.3)

        # LM head - trainable
        head_s = Rectangle(width=2.5, height=0.4, color=BLUE_TRAIN, fill_opacity=0.3, stroke_width=2)
        head_s_text = Text("LM Head (trainable)", font_size=12, color=BLUE_TRAIN)
        head_s.next_to(dense_box, DOWN, buff=0.15)
        head_s_text.move_to(head_s)

        # Total
        total_s = Text("3.0B total / 3B active", font_size=14, color=WHITE)
        total_s.next_to(head_s, DOWN, buff=0.2)
        vram_s = Text("6 GB VRAM (11× less)", font_size=12, color=GREEN_SHARED)
        vram_s.next_to(total_s, DOWN, buff=0.1)

        self.play(FadeIn(student_label))
        self.play(FadeIn(embed_s), Write(embed_s_text))
        self.play(FadeIn(attn_s), Write(attn_s_text))
        self.play(FadeIn(dense_box), Write(dense_label))
        self.play(FadeIn(ffn_bar), Write(ffn_text))
        self.play(FadeIn(shared_s), Write(shared_s_text), Write(frozen_label))
        self.play(FadeIn(head_s), Write(head_s_text))
        self.play(Write(total_s), Write(vram_s))
        self.wait(1)

        # ── Trainable params bar ───────────────────────────────
        bar_title = Text("Trainable parameters", font_size=18, color=WHITE)
        bar_title.move_to(DOWN * 2.8)
        self.play(Write(bar_title))

        # Full bar (3B)
        bar_bg = Rectangle(width=10, height=0.35, color=GREY_FROZEN, fill_opacity=0.2, stroke_width=1)
        bar_bg.move_to(DOWN * 3.3)

        # Trainable portion (1.39B / 3B = 46%)
        bar_train = Rectangle(width=4.6, height=0.35, color=BLUE_TRAIN, fill_opacity=0.6, stroke_width=0)
        bar_train.align_to(bar_bg, LEFT)
        bar_train.move_to(bar_bg.get_left() + RIGHT * 2.3 + UP * 0)

        # Frozen portion
        bar_frozen = Rectangle(width=5.4, height=0.35, color=GREY_FROZEN, fill_opacity=0.3, stroke_width=0)
        bar_frozen.align_to(bar_bg, RIGHT)
        bar_frozen.move_to(bar_bg.get_right() + LEFT * 2.7)

        train_label = Text("1.39B trainable (46%)", font_size=11, color=BLUE_TRAIN)
        train_label.move_to(bar_train)
        frozen_bar_label = Text("1.61B frozen (54%)", font_size=11, color=GREY_FROZEN)
        frozen_bar_label.move_to(bar_frozen)

        # Component labels below
        comp_labels = Text(
            "routed_dense (0.98B) + lm_head (0.41B)          attn (1.43B) + shared (0.12B)",
            font_size=9, color=GREY_B
        )
        comp_labels.next_to(bar_bg, DOWN, buff=0.15)

        self.play(FadeIn(bar_bg))
        self.play(GrowFromEdge(bar_train, LEFT), GrowFromEdge(bar_frozen, RIGHT))
        self.play(Write(train_label), Write(frozen_bar_label))
        self.play(Write(comp_labels))
        self.wait(2)

        # Final hold
        self.wait(1)
