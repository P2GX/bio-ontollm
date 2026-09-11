import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from sklearn.linear_model import LogisticRegression



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


def logreg_example(figsize=(7, 5)):
    """
    Logistic regression example: probability of passing an exam as a function
    of hours studied.

    Data source: the classic Wikipedia "Logistic regression" example dataset.
    """

   

    # --- Data -------------------------------------------------------------
    hours = np.array([
        0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
        2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50
    ])
    passed = np.array([
        0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 1, 1, 1, 1, 1
    ])

    X = hours.reshape(-1, 1)
    y = passed

    # --- Fit logistic regression -------------------------------------------
    # Use a very small L2 penalty (large C) so the fit is close to an
    # unregularized MLE fit, matching the classic textbook coefficients.
    model = LogisticRegression(C=1e6)
    model.fit(X, y)

    beta0 = model.intercept_[0]
    beta1 = model.coef_[0][0]
    #print(f"Fitted model: logit(p) = {beta0:.4f} + {beta1:.4f} * hours")

    # Hours at which predicted probability of passing = 0.5
    x_50 = -beta0 / beta1
    #print(f"Hours needed for 50% pass probability: {x_50:.3f}")

    # --- Build smooth curve for plotting -----------------------------------
    x_smooth = np.linspace(0, 6, 300).reshape(-1, 1)
    y_prob = model.predict_proba(x_smooth)[:, 1]

    # --- Plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter of raw observations, jittered slightly in y for visibility
    # where points overlap (e.g., the two 1.75-hour observations).
    ax.scatter(
        hours, passed,
        facecolors="none", edgecolors="black", s=60, zorder=3,
        label="Observed outcome"
    )

    # Fitted logistic curve
    ax.plot(
        x_smooth, y_prob,
        color="#1f77b4", linewidth=2, zorder=2,
        label="Fitted logistic curve"
    )

    # Reference lines at p = 0.5
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.axvline(x_50, color="gray", linestyle="--", linewidth=1)
    ax.annotate(
        f"{x_50:.2f} h",
        xy=(x_50, 0.5), xytext=(x_50 + 0.15, 0.08),
        fontsize=9, color="gray"
    )

    ax.set_xlabel("Hours studied ($x_k$)")
    ax.set_ylabel("Probability of $y_k$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 6)
    ax.legend(loc="lower right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
