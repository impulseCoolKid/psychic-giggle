import numpy as np

M = 200

W = np.random.normal(0, 1, size=(M, M))
print("mean",np.mean(W))
print("SD",np.std(W))
np.save('wmodel2.npy', W)

print(f"Random matrix W of shape ({M},{M}) has been saved to 'w.npy'.")