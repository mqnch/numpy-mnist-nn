import numpy as np
import struct
from array import array
from os.path import join

# mnist data loader class
class MnistDataloader:
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        # store file paths
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        # read labels
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('magic number mismatch for labels')
            labels = np.frombuffer(file.read(), dtype=np.uint8)

        # read images
        with open(images_filepath, 'rb') as file:
            magic, size2, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('magic number mismatch for images')
            images = np.frombuffer(file.read(), dtype=np.uint8).reshape(size2, rows * cols)

        # normalize pixel values
        return images / 255.0, labels

    def load_data(self):
        # load train and test sets
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)
    
# initialize weights and biases
def init_params(hidden=128, seed=42):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, np.sqrt(2/784), size=(hidden, 784)).astype(np.float32)
    b1 = np.zeros((hidden, 1), dtype=np.float32)
    W2 = rng.normal(0, np.sqrt(2/hidden), size=(10, hidden)).astype(np.float32)
    b2 = np.zeros((10, 1), dtype=np.float32)
    return W1, b1, W2, b2

LEAKY_ALPHA = 0.01 

# leaky ReLU activation
def ReLU(Z):
    # leaky relu
    return np.where(Z > 0, Z, LEAKY_ALPHA * Z)

# softmax activation
def softmax(Z):
    Zs = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Zs)
    return expZ / np.sum(expZ, axis=0, keepdims=True)

# forward pass
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1 @ X + b1           
    A1 = ReLU(Z1)               
    Z2 = W2 @ A1 + b2           
    A2 = softmax(Z2)            
    return Z1, A1, Z2, A2

# convert labels to one-hot
def one_hot(Y):
    m = Y.size
    K = int(Y.max()) + 1
    one_hot_Y = np.zeros((K, m), dtype=np.float32)
    one_hot_Y[Y, np.arange(m)] = 1.0
    return one_hot_Y

# derivative of leaky relu
def deriv_ReLU(Z):
    grad = np.ones_like(Z, dtype=np.float32)
    grad[Z < 0] = LEAKY_ALPHA
    return grad

# backward pass
def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

# update weights and biases
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

# get predictions
def get_predictions(A2):
    return np.argmax(A2, 0)

# compute accuracy
def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

# cross entropy loss
def cross_entropy(A2, Y):
    Yoh = one_hot(Y)
    eps = 1e-12
    return -np.sum(Yoh * np.log(A2 + eps)) / Y.size

# adam state init
def init_adam(W1, b1, W2, b2):
    # first moment
    m = {
        "W1": np.zeros_like(W1),
        "b1": np.zeros_like(b1),
        "W2": np.zeros_like(W2),
        "b2": np.zeros_like(b2),
    }
    # second moment
    v = {
        "W1": np.zeros_like(W1),
        "b1": np.zeros_like(b1),
        "W2": np.zeros_like(W2),
        "b2": np.zeros_like(b2),
    }
    return m, v

# adam update step
def adam_update(W1, b1, W2, b2, grads, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    # unpack grads
    dW1, db1, dW2, db2 = grads

    # update biased first moments
    m["W1"] = beta1 * m["W1"] + (1 - beta1) * dW1
    m["b1"] = beta1 * m["b1"] + (1 - beta1) * db1
    m["W2"] = beta1 * m["W2"] + (1 - beta1) * dW2
    m["b2"] = beta1 * m["b2"] + (1 - beta1) * db2

    # update biased second moments
    v["W1"] = beta2 * v["W1"] + (1 - beta2) * (dW1 * dW1)
    v["b1"] = beta2 * v["b1"] + (1 - beta2) * (db1 * db1)
    v["W2"] = beta2 * v["W2"] + (1 - beta2) * (dW2 * dW2)
    v["b2"] = beta2 * v["b2"] + (1 - beta2) * (db2 * db2)

    # bias correction
    mhat_W1 = m["W1"] / (1 - beta1**t); vhat_W1 = v["W1"] / (1 - beta2**t)
    mhat_b1 = m["b1"] / (1 - beta1**t); vhat_b1 = v["b1"] / (1 - beta2**t)
    mhat_W2 = m["W2"] / (1 - beta1**t); vhat_W2 = v["W2"] / (1 - beta2**t)
    mhat_b2 = m["b2"] / (1 - beta1**t); vhat_b2 = v["b2"] / (1 - beta2**t)

    # parameter step
    W1 = W1 - lr * mhat_W1 / (np.sqrt(vhat_W1) + eps)
    b1 = b1 - lr * mhat_b1 / (np.sqrt(vhat_b1) + eps)
    W2 = W2 - lr * mhat_W2 / (np.sqrt(vhat_W2) + eps)
    b2 = b2 - lr * mhat_b2 / (np.sqrt(vhat_b2) + eps)

    return W1, b1, W2, b2, m, v

# training loop (sgd or adam)
def gradient_descent(X, Y, epochs, alpha, batch_size, hidden, seed,
                     optimizer="adam", beta1=0.9, beta2=0.999, eps=1e-8):
    # init params
    W1, b1, W2, b2 = init_params(hidden=hidden, seed=seed)
    m = X.shape[1]
    rng = np.random.default_rng(seed)

    # init adam state if used
    if optimizer == "adam":
        M, V = init_adam(W1, b1, W2, b2)
        t = 0  # time step

    for epoch in range(1, epochs + 1):
        # shuffle data
        idx = rng.permutation(m)
        Xs, Ys = X[:, idx], Y[idx]

        # iterate mini-batches
        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            Xb = Xs[:, start:end]
            Yb = Ys[start:end]

            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, Xb)
            dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, Xb, Yb)

            if optimizer == "adam":
                t += 1
                W1, b1, W2, b2, M, V = adam_update(
                    W1, b1, W2, b2,
                    grads=(dW1, db1, dW2, db2),
                    m=M, v=V, t=t,
                    lr=alpha, beta1=beta1, beta2=beta2, eps=eps
                )
            else:
                W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        # monitor epoch metrics
        _, _, _, A2_full = forward_prop(W1, b1, W2, b2, X)
        train_acc = get_accuracy(get_predictions(A2_full), Y)
        loss = cross_entropy(A2_full, Y)
        print(f"epoch {epoch:02d} | loss = {loss:.4f} | train_acc = {train_acc*100:.2f}%")

    return W1, b1, W2, b2

# main entry
def main():
    # load mnist data
    loader = MnistDataloader(
        'train-images.idx3-ubyte', 'train-labels.idx1-ubyte',
        't10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte'
    )
    (x_train, y_train), (x_test, y_test) = loader.load_data()
    X = x_train.T
    Y = y_train

    # train network
    print("training...")
    W1, b1, W2, b2 = gradient_descent(X, Y, epochs=10, alpha=0.01, batch_size=1024, hidden=512, seed=1,  optimizer="adam")

    # test network
    print("evaluating on test set...")
    _, _, _, A2_test = forward_prop(W1, b1, W2, b2, x_test.T)
    test_preds = get_predictions(A2_test)
    print(f"test accuracy: {get_accuracy(test_preds, y_test):.4f}")

# run main
if __name__ == "__main__":
    main()