import math
import random
import time

# ========== Utility & Activation Functions ==========
def sigmoid(x): return [[1 / (1 + math.exp(-elem)) for elem in row] for row in x]
def sigmoid_derivative(x): return [[sig * (1 - sig) for sig in row] for row in sigmoid(x)]
def tanh(x): return [[math.tanh(elem) for elem in row] for row in x]
def tanh_derivative(x): return [[1 - math.tanh(elem) ** 2 for elem in row] for row in x]
def matmul(A, B):
    return [[sum(a * b for a, b in zip(A[i], [B[k][j] for k in range(len(B))]))
             for j in range(len(B[0]))] for i in range(len(A))]
def add(A, B): return [[a + b for a, b in zip(rowA, rowB)] for rowA, rowB in zip(A, B)]
def add_bias(mat, bias): return [[elem + bias[0][j] for j, elem in enumerate(row)] for row in mat]
def elemwise_mul(A, B): return [[a * b for a, b in zip(rowA, rowB)] for rowA, rowB in zip(A, B)]
def scalar_sub(scalar, A): return [[scalar - elem for elem in row] for row in A]
def zeros(shape): return [[0.0] * shape[1] for _ in range(shape[0])]

# ========== GRU Layer (Forward Only for Demo) ==========
def gru_forward(X, weights):
    Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh, Wo, bo = weights
    batch_size = len(X)
    seq_len = len(X[0])
    input_dim = len(X[0][0])
    hidden_dim = len(Uz)
    output_dim = len(Wo[0])

    h_prev = zeros((batch_size, hidden_dim))
    for t in range(seq_len):
        x_t = [seq[t] for seq in X]
        z = sigmoid(add_bias(add(matmul(x_t, Wz), matmul(h_prev, Uz)), bz))
        r = sigmoid(add_bias(add(matmul(x_t, Wr), matmul(h_prev, Ur)), br))
        h_tilde = tanh(add_bias(add(matmul(x_t, Wh), matmul(elemwise_mul(r, h_prev), Uh)), bh))
        h_t = add(elemwise_mul(scalar_sub(1, z), h_prev), elemwise_mul(z, h_tilde))
        h_prev = h_t
    y_pred = add_bias(matmul(h_prev, Wo), bo)
    return y_pred

# ========== Loss Function ==========
def mse_loss(y_pred, y_true):
    total = 0.0
    for yp, yt in zip(y_pred, y_true):
        for pi, ti in zip(yp, yt):
            total += (pi - ti) ** 2
    return total / len(y_pred)

# ========== Data Generator ==========
def generate_dummy_data(samples=20, seq_len=10, input_dim=3, output_dim=1):
    X = [[[random.random() for _ in range(input_dim)] for _ in range(seq_len)] for _ in range(samples)]
    y = [[sum(seq[-1]) / input_dim for _ in range(output_dim)] for seq in X]  # target: avg of last input
    return X, y

# ========== Weight Initialization ==========
def init_weights(input_dim, hidden_dim, output_dim):
    def rand_mat(x, y): return [[random.uniform(-0.1, 0.1) for _ in range(y)] for _ in range(x)]
    def zero_bias(d): return [[0.0 for _ in range(d)]]
    return (
        rand_mat(input_dim, hidden_dim), rand_mat(hidden_dim, hidden_dim), zero_bias(hidden_dim),  # Wz, Uz, bz
        rand_mat(input_dim, hidden_dim), rand_mat(hidden_dim, hidden_dim), zero_bias(hidden_dim),  # Wr, Ur, br
        rand_mat(input_dim, hidden_dim), rand_mat(hidden_dim, hidden_dim), zero_bias(hidden_dim),  # Wh, Uh, bh
        rand_mat(hidden_dim, output_dim), zero_bias(output_dim)                                     # Wo, bo
    )

# ========== Main Training Loop ==========
def train_gru():
    # Hyperparameters
    input_dim = 3
    hidden_dim = 8
    output_dim = 1
    epochs = 5
    lr = 0.01

    print("Generating data...")
    X_train, y_train = generate_dummy_data(samples=20, seq_len=10, input_dim=input_dim, output_dim=output_dim)
    weights = init_weights(input_dim, hidden_dim, output_dim)

    print("Starting training (pure Python GRU)...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        y_pred = gru_forward(X_train, weights)
        loss = mse_loss(y_pred, y_train)

        print(f"Epoch {epoch}/{epochs}, Loss: {loss:.6f}")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    train_gru()
