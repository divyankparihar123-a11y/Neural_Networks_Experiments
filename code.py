import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import time

# AND gate dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

T = np.array([0, 0, 0, 1])   # Targets for AND gate

# Add bias input (x0 = 1)
X_bias = np.c_[np.ones(X.shape[0]), X]

# Initialize weights to zero (bias + 2 inputs)
W = np.zeros(X_bias.shape[1])

learning_rate = 0.01
epoch = 0
total_errors = []

def step_function(net):
    return 1 if net >= 0 else 0

# Training loop
while True:
    epoch += 1
    errors = 0
    rows = []

    print(f"\nEpoch {epoch}")
    print("-" * 60)

    for i in range(len(X_bias)):
        net = np.dot(X_bias[i], W)
        y = step_function(net)
        error = T[i] - y

        # Weight update
        W = W + learning_rate * error * X_bias[i]

        errors += abs(error)

        rows.append([
            X[i][0], X[i][1],
            T[i], y, error,
            W.copy()
        ])

    # Print table (avoid pandas to bypass DLL C-extension issues)
    print("{:>3} {:>3} {:>6} {:>9} {:>6} {}".format("x1", "x2", "Target", "Predicted", "Error", "Weights"))
    for r in rows:
        x1, x2, targ, pred, err, weights = r
        print(f"{int(x1):>3} {int(x2):>3} {targ:>6} {pred:>9} {err:>6} {np.array2string(weights, precision=2)})")

    total_errors.append(errors)

    # ---- Plot decision boundary ----
    plt.figure(figsize=(6, 5))

    for i in range(len(X)):
        if T[i] == step_function(np.dot(X_bias[i], W)):
            plt.scatter(X[i][0], X[i][1], color="green", s=100, label="Correct" if i == 0 else "")
        else:
            plt.scatter(X[i][0], X[i][1], color="red", s=100, label="Wrong" if i == 0 else "")

    # Decision boundary: w0 + w1*x + w2*y = 0
    if W[2] != 0:
        x_vals = np.array([-0.5, 1.5])
        y_vals = -(W[0] + W[1] * x_vals) / W[2]
        plt.plot(x_vals, y_vals, 'b--', label="Decision Boundary")

    plt.title(f"Epoch {epoch}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid()
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.savefig(f"epoch_{epoch}.png")
    plt.close()

    time.sleep(2)
    # Stop if no error
    if errors == 0:
        print(f"\n Converged at epoch {epoch}")
        break

# ---- Error vs Epoch plot ----
plt.figure(figsize=(6, 4))
plt.plot(range(1, epoch + 1), total_errors, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Total Error")
plt.title("Error vs Epoch")
plt.grid()
plt.savefig("errors_vs_epoch.png")
plt.close()
