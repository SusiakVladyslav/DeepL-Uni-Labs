import numpy as np
import sklearn
import torch
import torch.nn as nn

iris = sklearn.datasets.load_iris()
x = iris.data
y = iris.target

rng = np.random.default_rng(0)

train_indices = []
test_indices = []


for iris_class in [0, 1, 2]:

    class_indices = np.where(y == iris_class)[0]

    shuffled_indices = rng.permutation(class_indices)

    train_indices.extend(shuffled_indices[:35])
    test_indices.extend(shuffled_indices[35:])

x_train, y_train = x[train_indices], y[train_indices]
x_test, y_test = x[test_indices], y[test_indices]

x_mean = np.mean(x_train, axis=0)
x_std = np.std(x_train, axis=0, ddof=0)

x_train = (x_train - x_mean) / x_std
x_test = (x_test - x_mean) / x_std

rng_init = np.random.default_rng(0)

W1 = rng_init.normal(loc=0.0, scale=np.sqrt(2 / 4), size=(4, 8))
W2 = rng_init.normal(loc=0.0, scale=np.sqrt(2 / (8 + 3)), size=(8, 3))

b1 = np.zeros(8)
b2 = np.zeros(3)

def forward(X, W1, W2, b1, b2):
    Z1 = X @ W1 + b1
    A1 = np.maximum(0, Z1)
    Z2 = A1 @ W2 + b2

    cache = {
        "X": X,
        'Z1': Z1,
        'A1': A1,
        'Z2': Z2,
    }

    return Z2, cache


def cross_entropy_loss(logits, y):
    m = y.shape[0]
    max_logits = np.max(logits, axis=-1, keepdims=True)

    log_sum_exp = np.log(np.sum(np.exp(logits - max_logits), axis=-1, keepdims=True))
    log_probs = (logits - max_logits) - log_sum_exp

    loss = -np.mean(log_probs[np.arange(m), y])

    probs = np.exp(log_probs)
    return loss, probs


def backward(cache, probs, y, W2):
    m = y.shape[0]
    X = cache["X"]
    Z1 = cache["Z1"]
    A1 = cache["A1"]

    dZ2 = probs.copy()
    dZ2[np.arange(m), y] -= 1.0
    dZ2 /= m

    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    return dW1, db1, dW2, db2


Z2_np, cache_np = forward(x_train, W1, W2, b1, b2)
loss_np, probs_np = cross_entropy_loss(Z2_np, y_train)
dW1_np, db1_np, dW2_np, db2_np = backward(cache_np, probs_np, y_train, W2)

torch.set_default_dtype(torch.float64)


class PyTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


torch_model = PyTorchModel()

with torch.no_grad():
    torch_model.fc1.weight.copy_(torch.from_numpy(W1.T))
    torch_model.fc1.bias.copy_(torch.from_numpy(b1))
    torch_model.fc2.weight.copy_(torch.from_numpy(W2.T))
    torch_model.fc2.bias.copy_(torch.from_numpy(b2))

X_train_t = torch.from_numpy(x_train)
y_train_t = torch.from_numpy(y_train).long()

criterion = nn.CrossEntropyLoss()
logits_t = torch_model(X_train_t)
loss_t = criterion(logits_t, y_train_t)

loss_t.backward()

dW1_pt = torch_model.fc1.weight.grad.numpy().T
db1_pt = torch_model.fc1.bias.grad.numpy()
dW2_pt = torch_model.fc2.weight.grad.numpy().T
db2_pt = torch_model.fc2.bias.grad.numpy()
loss_pt = loss_t.item()

tolerance = 1e-12

diff_loss = np.abs(loss_np - loss_pt)
diff_W1 = np.max(np.abs(dW1_np - dW1_pt))
diff_b1 = np.max(np.abs(db1_np - db1_pt))
diff_W2 = np.max(np.abs(dW2_np - dW2_pt))
diff_b2 = np.max(np.abs(db2_np - db2_pt))

results = [
    ("Втрата", diff_loss, diff_loss <= tolerance),
    ("Градієнт W1", diff_W1, diff_W1 <= tolerance),
    ("Градієнт b1", diff_b1, diff_b1 <= tolerance),
    ("Градієнт W2", diff_W2, diff_W2 <= tolerance),
    ("Градієнт b2", diff_b2, diff_b2 <= tolerance),
]

print(f"Значення втрати NumPy:   {loss_np:.15f}")
print(f"Значення втрати PyTorch: {loss_pt:.15f}\n")

header = f"{'Величина':<15} | {'Макс. різниця NumPy/PyTorch':<30} | {'Перевірку пройдено':<20}"
print(header)
print("-" * len(header))
for name, diff, passed in results:
    status = "Так" if passed else "Ні"
    print(f"{name:<15} | {diff:<30.2e} | {status:<20}")


def backward_with_mode(cache, probs, y, W2, divide_by_m=True):
    m = y.shape[0]
    X = cache["X"]
    Z1 = cache["Z1"]
    A1 = cache["A1"]

    dZ2 = probs.copy()
    dZ2[np.arange(m), y] -= 1.0

    if divide_by_m:
        dZ2 /= m

    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    return dW1, db1, dW2, db2


print("\n" + "=" * 60)
print("ЗАВДАННЯ 3: Чисельне диференціювання")
print("=" * 60)

epsilon = 1e-6
tol_num = 1e-7

Z2_np, cache_np = forward(x_train, W1, W2, b1, b2)
loss_np, probs_np = cross_entropy_loss(Z2_np, y_train)
dW1_manual, db1_manual, dW2_manual, db2_manual = backward_with_mode(
    cache_np, probs_np, y_train, W2, divide_by_m=True
)

params_to_check = [
    ("W1[0, 0]", W1, (0, 0), dW1_manual[0, 0]),
    ("b1[0]",    b1, (0,),    db1_manual[0]),
    ("W2[0, 0]", W2, (0, 0), dW2_manual[0, 0]),
    ("b2[0]",    b2, (0,),    db2_manual[0]),
]

print(f"{'Параметр':<12} | {'Градієнт backward()':<20} | {'Чисельна похідна':<20} | {'Абсолютна різниця':<20} | {'Перевірку пройдено':<20}")
print("-" * 102)

for name, param_arr, idx, g_manual in params_to_check:
    orig_val = param_arr[idx]

    param_arr[idx] = orig_val + epsilon
    z2_plus, _ = forward(x_train, W1, W2, b1, b2)
    loss_plus, _ = cross_entropy_loss(z2_plus, y_train)

    param_arr[idx] = orig_val - epsilon
    z2_minus, _ = forward(x_train, W1, W2, b1, b2)
    loss_minus, _ = cross_entropy_loss(z2_minus, y_train)

    param_arr[idx] = orig_val

    g_num = (loss_plus - loss_minus) / (2 * epsilon)
    diff = np.abs(g_num - g_manual)
    passed = diff <= tol_num

    status = "Так" if passed else "Ні"
    print(f"{name:<12} | {g_manual:<20.8f} | {g_num:<20.8f} | {diff:<20.2e} | {status:<20}")


print("\n" + "=" * 60)
print("ЗАВДАННЯ 4: Дослід із навмисною помилкою (без ділення на m)")
print("=" * 60)

m_samples = x_train.shape[0]

dW1_bug, db1_bug, dW2_bug, db2_bug = backward_with_mode(
    cache_np, probs_np, y_train, W2, divide_by_m=False
)

diff_W1_bug = np.max(np.abs(dW1_bug - dW1_pt))
diff_b1_bug = np.max(np.abs(db1_bug - db1_pt))
diff_W2_bug = np.max(np.abs(dW2_bug - dW2_pt))
diff_b2_bug = np.max(np.abs(db2_bug - db2_pt))

print("\n1. Звірка з PyTorch (помилкова версія):")
print(f"Макс. різниця W1: {diff_W1_bug:.4e} (Перевірку пройдено: {diff_W1_bug <= 1e-12})")
print(f"Макс. різниця b1: {diff_b1_bug:.4e} (Перевірку пройдено: {diff_b1_bug <= 1e-12})")
print(f"Макс. різниця W2: {diff_W2_bug:.4e} (Перевірку пройдено: {diff_W2_bug <= 1e-12})")
print(f"Макс. різниця b2: {diff_b2_bug:.4e} (Перевірку пройдено: {diff_b2_bug <= 1e-12})")

params_bug_check = [
    ("W1[0, 0]", W1, (0, 0), dW1_bug[0, 0]),
    ("b1[0]", b1, (0,), db1_bug[0]),
    ("W2[0, 0]", W2, (0, 0), dW2_bug[0, 0]),
    ("b2[0]", b2, (0,), db2_bug[0]),
]

print("\n2. Чисельна перевірка (помилкова версія проти еталонної чисельної похідної):")
print(
    f"{'Параметр':<12} | {'Градієнт (bug)':<20} | {'Чисельна похідна':<20} | {'Абсолютна різниця':<20} | {'Перевірку пройдено':<20}")
print("-" * 102)

for name, param_arr, idx, g_bug in params_bug_check:
    orig_val = param_arr[idx]

    param_arr[idx] = orig_val + epsilon
    z2_plus, _ = forward(x_train, W1, W2, b1, b2)
    loss_plus, _ = cross_entropy_loss(z2_plus, y_train)

    param_arr[idx] = orig_val - epsilon
    z2_minus, _ = forward(x_train, W1, W2, b1, b2)
    loss_minus, _ = cross_entropy_loss(z2_minus, y_train)

    param_arr[idx] = orig_val
    g_num = (loss_plus - loss_minus) / (2 * epsilon)

    diff = np.abs(g_bug - g_num)
    ratio = g_bug / g_num if abs(g_num) > 1e-15 else np.nan
    passed = diff <= tol_num
    status = "Так" if passed else "Ні"

    print(f"{name:<12} | {g_bug:<20.8f} | {g_num:<20.8f} | {diff:<20.2e} | {status:<20}")

print(f"\nКоефіцієнт відхилення помилкового градієнта від чисельного: ≈ {ratio:.1f} (очікувано m = {m_samples})")