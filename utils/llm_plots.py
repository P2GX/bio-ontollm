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