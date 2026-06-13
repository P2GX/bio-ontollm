import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np




def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_plot(figsize=(4, 3)):
    z = np.linspace(-10, 10, 100)
    phi_z = sigmoid(z)
    plt.figure(figsize=figsize)
    plt.plot(z, phi_z, color='#003366', linewidth=3) # FU Berlin Blue
    plt.axvline(0.0, color='k', linewidth=1, alpha=0.5)
    plt.axhline(y=0.5, color='k', linestyle='--', linewidth=1, alpha=0.5)
    plt.yticks([0.0, 0.5, 1.0])
    plt.xlabel('z')
    plt.ylabel(r'$\sigma(z)$')
    plt.title('Sigmoid (Logistic) Function')
    plt.grid(True, alpha=0.3)


def gdexample(figsize=(8,6)):
    # Define the Loss function (a simple parabola)
    def J(w):
        return (w - 3)**2 + 1

    # Define the derivative (slope)
    def dJ(w):
        return 2 * (w - 3)

    # Generate data for the curve
    w_range = np.linspace(0.5, 5, 100)
    loss_vals = J(w_range)

    # Setup the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the main loss curve
    ax.plot(w_range, loss_vals, color='#003366', lw=3, label='Loss Function')

    # Define points w1 and w2
    w1 = 1.2
    w2 = 1.8
    y1, y2 = J(w1), J(w2)

    # Plot the tangent line at w1
    slope = dJ(w1)
    x_tangent = np.linspace(w1 - 0.5, w1 + 0.5, 10)
    y_tangent = slope * (x_tangent - w1) + y1
    ax.plot(x_tangent, y_tangent, '--', color='green', lw=1.5)

    # Draw points
    ax.scatter([w1, w2], [y1, y2], color=['black', 'gray'], zorder=5)

    # Draw the step arrow
    ax.annotate('', xy=(w2, y2), xytext=(w1, y1),
                arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.3", 
                                color='red', lw=2))

    # Annotations
    ax.annotate('slope of loss', xy=(w1, y1), xytext=(w1-1.2, y1+2),
                arrowprops=dict(arrowstyle='->', color='green'),
                color='green', fontweight='bold', ha='center')

    ax.annotate('one step\nof gradient\ndescent', xy=(1.5, 3.2), xytext=(2.2, 6),
                arrowprops=dict(arrowstyle='->', color='red'),
                color='red', fontweight='bold')

    # Labels and Ticks
    ax.set_xlabel('w', loc='right', fontsize=12)
    ax.set_ylabel('Loss', loc='top', rotation=0, fontsize=12)
    ax.set_xticks([w1, 3])
    ax.set_xticklabels(['$w^1$\n0', '$w^{min}$\n(goal)'])
    ax.set_yticks([]) # Hide y-ticks for a clean "conceptual" look

    # Clean up the spines (axis lines)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    plt.tight_layout()


def plot_loss(figsize=(7, 4.5)):
    p = np.linspace(0.001, 1.0, 500)
    loss = -np.log(p)

    # 2. Define our target point at sigma(z) = 0.95
    target_p = 0.95
    target_loss = -np.log(target_p) # approx 0.0513

    # 3. Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(p, loss, color='#2c3e50', linewidth=2, label=r'$-\log \sigma(z)$')

    # 4. Add the highlight box (rectangle) around our target point
    # Rectangle arguments: (xmin, ymin), width, height
    box_width = 0.06
    box_height = 0.3
    '''
    rect = patches.Rectangle(
        (target_p - 0.03, target_loss - 0.15), 
        box_width, box_height, 
        linewidth=1.5, edgecolor='#e74c3c', facecolor='#e74c3c', alpha=0.2
    )
    ax.add_patch(rect)

    # 5. Add a text label pointing to the box
    ax.text(
        target_p - 0.05, target_loss + 0.4, 
        f"$\sigma(z) = {target_p}$\nLoss = {target_loss:.4f}", 
        color='#c0392b', fontsize=10, weight='bold', ha='right'
    )
    '''

    # 6. Styling and Polish for your presentation
    ax.set_xlabel(r"Output: $\sigma(z)$", fontsize=11)
    ax.set_ylabel(r"Loss: $-\log \sigma(z)$", fontsize=11)

    # Clean up axes and grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='both', linestyle='--', alpha=0.5)

    plt.tight_layout()
    