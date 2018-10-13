import matplotlib.pyplot as plt
import numpy as np
import torch
from .optimizers import AdaShift

np.random.seed(0)
torch.manual_seed(0)

N = 50

x = np.random.uniform(-5, 5, size=N)
x.sort()
y = np.square(x) + np.random.randn(N)

X = torch.Tensor(np.dstack([x, np.square(x), np.sqrt(np.abs(x))]))
y = torch.Tensor(y)

W = torch.tensor(0.01 * np.random.randn(3).astype(np.float32),
                 requires_grad=True)
optimizer = AdaShift([W])

losses = []
for i in range(1000):
  loss = (torch.matmul(X, W) - y).pow(2).mean()
  losses.append(float(loss))
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()

plt.subplot(121)
plt.semilogy(losses)
plt.subplot(122)
plt.plot(x, y.data.numpy(), label="true")
plt.plot(x, torch.matmul(X, W).data.numpy().squeeze(), label="preds")
print(W.data.numpy())
plt.legend()
plt.show()
plt.waitforbuttonpress()
