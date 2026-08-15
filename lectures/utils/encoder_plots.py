import matplotlib.pyplot as plt
import numpy as np



def frequency_plot(figsize=(10, 5)):
    # Setup
    n = 100
    d_model = 8  # Supporting i=0, 1, 2, 3 (each i has sin/cos)
    i_values = [0, 1, 2, 3]
    colors = ['#1f77b4', '#a52a2a', '#2ca02c', '#9467bd']
    pos = np.linspace(0, 10, 500) # Smooth line
    _fig, ax = plt.subplots(figsize=figsize)

    for idx, i in enumerate(i_values):
        denom = n**(2 * i / d_model)
        y_sin = np.sin(pos / denom)
        
        # Plot Sin as solid, Cos as dashed
        ax.plot(pos, y_sin, color=colors[idx], lw=2, 
                label=f"i={i} (λ={denom:.1f})")
        # Optional: uncomment below if you want to show the cosine pairs too
        # ax.plot(pos, np.cos(pos / denom), color=colors[idx], lw=1.5, ls='--', alpha=0.4)

    # Formatting
    ax.set_title(f"Positional Encoding Frequencies (n={n}, d=8)", fontsize=14, pad=15)
    ax.set_xlabel("Position Index (k)", fontsize=12)
    ax.set_ylabel("Encoding Value", fontsize=12)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.axhline(0, color='black', lw=1, alpha=0.3)

    # Add vertical markers for the first few integer positions (tokens)
    for k in range(5):
        ax.axvline(x=k, color='gray', lw=1, ls='-', alpha=0.1)
        ax.text(k, -1.05, f"k={k}", ha='center', fontsize=9, color='gray')

    ax.legend(title="Dimension Index", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
