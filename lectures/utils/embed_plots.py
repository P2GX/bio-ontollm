import matplotlib.pyplot as plt
import numpy as np





def cars_pca(figsize=(10, 8)):
    np.random.seed(42)

    # Number of car data points
    n_cars = 60

    # Generate synthetic data for dimensions
    # X-axis: Size (1 = Small/Compact, 5 = Large/SUV)
    size = np.random.uniform(1.0, 5.0, n_cars)

    # Y-axis: Color spectrum (1 = Warm tones/Red, 5 = Cool tones/Blue)
    color_spectrum = np.random.uniform(1.0, 5.0, n_cars)

    # Z-axis: Price (in thousands, influenced somewhat by size for realism)
    price = size * 15 + np.random.normal(0, 8, n_cars)
    price = np.clip(price, 10, 100) # Keep within a realistic range

    # Create the 3D figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')

    # Scatter plot: map the 'color_spectrum' variable to actual marker colors 
    # using a colormap (e.g., 'coolwarm' matches our red-to-blue axis intent)
    scatter = ax.scatter(
        xs=size, 
        ys=color_spectrum, 
        zs=price, 
        c=color_spectrum, 
        cmap='coolwarm', 
        s=70, 
        edgecolors='k', 
        alpha=0.85
    )

    # Set informative labels for the semantic vector space
    ax.set_xlabel('Size (Compact $\rightarrow$ Large)', labelpad=10, fontsize=11)
    ax.set_ylabel('Color Spectrum (Warm $\rightarrow$ Cool)', labelpad=10, fontsize=11)
    ax.set_zlabel('Price ($k)', labelpad=10, fontsize=11)

    ax.set_title('Semantic Vector Space of Cars (3D Embedding Illustration)', fontsize=13, pad=15)

    # Add a color bar representing the color dimension
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label('Color Metric', fontsize=10)

    # Adjust viewing angle for an optimal 3D perspective
    ax.view_init(elev=25, azim=135)

    plt.tight_layout()
    plt.show()