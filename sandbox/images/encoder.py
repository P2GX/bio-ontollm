import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_encoder_final(highlight=None):
    fig, ax = plt.subplots(figsize=(7, 11), dpi=100)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis('off')

    colors = {
        'mha': '#fdba74', 'ffn': '#93c5fd', 'norm': '#e2e8f0',
        'text': '#333333', 'highlight': '#1D9A73', 'residual': '#64748b'
    }

    def draw_box(y, text, color, key):
        is_hl = (highlight == key)
        rect = patches.FancyBboxPatch((2.5, y), 5, 1, boxstyle="round,pad=0.1", 
                                      linewidth=3 if is_hl else 1.5, 
                                      edgecolor=colors['highlight'] if is_hl else '#333333', 
                                      facecolor=color)
        ax.add_patch(rect)
        ax.text(5, y + 0.5, text, ha='center', va='center', fontsize=11, fontweight='bold')

    # --- 1. INPUT TRUNK ---
    ax.plot([5, 5], [0.5, 1.5], color='#333', lw=2) 
    
    # Fork to K, Q, V
    ax.plot([3, 7], [1.5, 1.5], color='#333', lw=2) 
    for x, label in zip([3, 5, 7], ['K', 'Q', 'V']):
        ax.plot([x, x], [1.5, 3], color='#333', lw=2)
        ax.text(x, 2.2, label, ha='center', va='center', fontsize=12, fontweight='bold', 
                bbox=dict(facecolor='white', edgecolor='none', pad=1))

    # --- 2. MULTI-HEAD ATTENTION ---
    draw_box(3, "Multi-Head Attention", colors['mha'], 'mha')
    
    # Residual 1: Forks at y=1.2, goes to middle of Norm 1 (y=5)
    # Line goes: Right at trunk -> Left to x=1 -> Up to y=5 -> Right into box
    ax.plot([5, 1, 1], [1.2, 1.2, 5], color=colors['residual'], lw=1.5, ls='--')
    ax.annotate('', xy=(2.5, 5), xytext=(1, 5), 
                arrowprops=dict(arrowstyle='->', color=colors['residual'], lw=1.5, ls='--'))

    # Add & Norm 1
    draw_box(4.5, "Add & Norm", colors['norm'], 'norm1')
    ax.annotate('', xy=(5, 4.5), xytext=(5, 4), arrowprops=dict(arrowstyle='->', lw=2))

    # --- 3. FEED FORWARD (FFN) ---
    # Residual 2: Forks at y=5.8, goes to middle of Norm 2 (y=10)
    ax.plot([5, 1, 1], [5.8, 5.8, 10], color=colors['residual'], lw=1.5, ls='--')
    ax.annotate('', xy=(2.5, 10), xytext=(1, 10), 
                arrowprops=dict(arrowstyle='->', color=colors['residual'], lw=1.5, ls='--'))
    
    draw_box(7.5, "Feed Forward (FFN)", colors['ffn'], 'ffn')
    ax.annotate('', xy=(5, 7.5), xytext=(5, 5.5), arrowprops=dict(arrowstyle='->', lw=2))

    # Add & Norm 2
    draw_box(9.5, "Add & Norm", colors['norm'], 'norm2')
    ax.annotate('', xy=(5, 9.5), xytext=(5, 8.5), arrowprops=dict(arrowstyle='->', lw=2))

    # Output
    ax.annotate('', xy=(5, 11.9), xytext=(5, 10.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(5, 12, "Output to Next Layer", ha='center', fontsize=10, fontweight='bold')

    return fig

fig = draw_encoder_final(highlight='norm1')
plt.show()