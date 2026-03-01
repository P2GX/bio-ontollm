import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def draw_encoder_comparison(tokens):
    fig, ax = plt.subplots(figsize=(14, 4))

    spacing = 2.2
    box_width = 1.4
    token_height = 0.6
    rnn_height = 1.3

    margin_left = 1.5
    margin_top = 3.5

    # -------------------
    # RNN Section Title
    # -------------------
    ax.text(margin_left + (len(tokens)-1)*spacing/2,
            margin_top,
            "RNN based Encoder",
            ha="center",
            fontsize=14,
            weight="bold")

    previous_rnn_center = None

    for i, token in enumerate(tokens):
        x = margin_left + i * spacing

        # Token box
        token_box = FancyBboxPatch(
            (x, 0.8),
            box_width,
            token_height,
            boxstyle="round,pad=0.02",
            fc="#c9f7c9"
        )
        ax.add_patch(token_box)
        ax.text(x + box_width/2, 1.1, token, ha='center', va='center')

        # RNN box
        rnn_box = FancyBboxPatch(
            (x, 2.0),
            box_width,
            rnn_height,
            boxstyle="round,pad=0.05",
            fc="#f4a7a1"
        )
        ax.add_patch(rnn_box)
        ax.text(x + box_width/2, 2.65, "RNN", ha='center', va='center')

        # Arrow token -> RNN
        ax.annotate("",
                    xy=(x + box_width/2, 2.0),
                    xytext=(x + box_width/2, 1.4),
                    arrowprops=dict(arrowstyle="->"))

        # Arrow RNN -> next RNN
        current_center = (x + box_width, 2.65)
        if previous_rnn_center:
            ax.annotate("",
                        xy=(x, 2.65),
                        xytext=previous_rnn_center,
                        arrowprops=dict(arrowstyle="->"))
        previous_rnn_center = current_center

    # -------------------
    # Transformer Section
    # -------------------

    transformer_x = margin_left + len(tokens) * spacing + 2

    ax.text(transformer_x + box_width,
            margin_top,
            "Transformer Encoder",
            ha="center",
            fontsize=14,
            weight="bold")

    # Transformer big box
    transformer_box = FancyBboxPatch(
        (transformer_x, 2.0),
        box_width * 3,
        rnn_height,
        boxstyle="round,pad=0.1",
        fc="#f4a7a1"
    )
    ax.add_patch(transformer_box)

    ax.text(transformer_x + box_width * 1.5,
            2.65,
            "Encoder\n(Transformer)",
            ha='center',
            va='center')

    # Tokens under transformer
    for i, token in enumerate(tokens):
        x = transformer_x + i * (box_width * 0.8)

        token_box = FancyBboxPatch(
            (x, 0.8),
            box_width * 0.7,
            token_height,
            boxstyle="round,pad=0.02",
            fc="#c9f7c9"
        )
        ax.add_patch(token_box)
        ax.text(x + box_width*0.35, 1.1, token, ha='center', va='center')

        # Arrow token -> transformer
        ax.annotate("",
                    xy=(x + box_width*0.35, 1.95),
                    xytext=(x + box_width*0.35, 1.4),
                    arrowprops=dict(arrowstyle="->"))

    # Final styling
    ax.set_xlim(0, transformer_x + 6)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig("rnn_vs_transformer.svg", format="svg")
    plt.show()


# Example
tokens = ["The", "Cat", "Is", "Black"]
draw_encoder_comparison(tokens)