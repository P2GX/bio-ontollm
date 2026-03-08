import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_activations(show_gelu=False):
    """
    Plot ReLU and GELU activations
    """
    x = np.linspace(-3, 3, 200)
    relu = np.maximum(0, x)
    leaky = np.where(x > 0, x, 0.1 * x)
    # Approximation of GELU
    gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    
    ax.plot(x, relu, label='ReLU', color='#0000ff', lw=3)
    ax.plot(x, leaky, label='Leaky ReLU', color='#ff0000', ls='--', lw=2)
    if show_gelu:
        ax.plot(x, gelu, label='GELU', color='#00aa00', lw=2)

    ax.set_aspect('equal') # Prevents the "squeezed" look
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()

def draw_gelu_block():
    """
    Plot the FFN block of BERT with its GELU unit fan out
    """
    fig, ax = plt.subplots(figsize=(4, 6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Vertices for trapezoids
    top_trap = [[20, 80], [80, 80], [100, 60], [0, 60]]
    bot_trap = [[0, 30], [100, 30], [80, 10], [20, 10]]

    ax.add_patch(patches.Polygon(top_trap, fc="#F06292", ec="black"))
    ax.add_patch(patches.Rectangle((0, 35), 100, 20, fc="#B2EBF2", ec="black"))
    ax.add_patch(patches.Polygon(bot_trap, fc="#F06292", ec="black"))

    # Labels
    ax.text(50, 45, "GELU", ha='center', weight='bold', size=14)
    
    # Arrows
    arrow_data = [(50, 92, 50, 80), (50, 60, 50, 54), (50, 35, 50, 29), (50, 10, 50, 0)]
    for x1, y1, x2, y2 in arrow_data:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=2))

    plt.show()


def draw_encoder_compact(highlight=None):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10) 
    ax.axis('off')

    colors = {
        'mha': '#fdba74', 'ffn': '#93c5fd', 'norm': '#e2e8f0',
        'text': '#333333', 'highlight': '#1D9A73', 'residual': '#64748b'
    }

    def draw_box(y, text, color, key):
        is_hl = (highlight == key)
        rect = patches.FancyBboxPatch((2.5, y), 5, 0.8, boxstyle="round,pad=0.1", 
                                      linewidth=3 if is_hl else 1.5, 
                                      edgecolor=colors['highlight'] if is_hl else '#333333', 
                                      facecolor=color)
        ax.add_patch(rect)
        ax.text(5, y + 0.4, text, ha='center', va='center', fontsize=10, fontweight='bold')

    ax.plot([5, 5], [0.2, 0.8], color='#333', lw=1.5)
    ax.plot([3.5, 6.5], [0.8, 0.8], color='#333', lw=1.5)
    for x, label in zip([3.5, 5, 6.5], ['K', 'Q', 'V']):
        ax.plot([x, x], [0.8, 1.5], color='#333', lw=1.5)
        ax.text(x, 1.1, label, ha='center', va='center', fontsize=9, fontweight='bold', backgroundcolor='white')

    # --- MHA ---
    draw_box(1.5, "Multi-Head Attention", colors['mha'], 'mha')
    
    # Residual 1
    ax.plot([5, 1, 1], [0.6, 0.6, 3.2], color=colors['residual'], lw=1.2, ls='--')
    ax.annotate('', xy=(2.5, 3.2), xytext=(1, 3.2), arrowprops=dict(arrowstyle='->', color=colors['residual'], ls='--'))

    # --- Add & Norm 1 ---
    draw_box(2.8, "Add & Norm", colors['norm'], 'norm1')
    ax.annotate('', xy=(5, 2.8), xytext=(5, 2.3), arrowprops=dict(arrowstyle='->', lw=1.5))

    # --- FFN ---
    draw_box(4.5, "Feed Forward (FFN)", colors['ffn'], 'ffn')
    
    # Residual 2
    ax.plot([5, 1, 1], [4.0, 4.0, 6.2], color=colors['residual'], lw=1.2, ls='--')
    ax.annotate('', xy=(2.5, 6.2), xytext=(1, 6.2), arrowprops=dict(arrowstyle='->', color=colors['residual'], ls='--'))

    # --- Add & Norm 2 ---
    draw_box(5.8, "Add & Norm", colors['norm'], 'norm2')
    ax.annotate('', xy=(5, 5.8), xytext=(5, 5.3), arrowprops=dict(arrowstyle='->', lw=1.5))

    # Output
    ax.annotate('', xy=(5, 7.5), xytext=(5, 6.6), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.text(5, 7.8, "Output to Next Layer", ha='center', fontsize=9, fontweight='bold')

    plt.show()