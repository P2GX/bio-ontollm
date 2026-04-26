import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np



def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_plot(figsize=(4, 3)):
    z = np.linspace(-10, 10, 100)
    phi_z = sigmoid(z)
    plt.figure(figsize=(8, 4))
    plt.plot(z, phi_z, color='#003366', linewidth=3) # FU Berlin Blue
    plt.axvline(0.0, color='k', linewidth=1, alpha=0.5)
    plt.axhline(y=0.5, color='k', linestyle='--', linewidth=1, alpha=0.5)
    plt.yticks([0.0, 0.5, 1.0])
    plt.xlabel('z')
    plt.ylabel(r'$\sigma(z)$')
    plt.title('Sigmoid (Logistic) Function')
    plt.grid(True, alpha=0.3)

    plt.show()
