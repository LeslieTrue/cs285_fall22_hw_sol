import numpy as np
a = np.array([1, 2.3, 3, 4.5])
for i in range(a.size):
    a[i] = int(a[i])
print(a)