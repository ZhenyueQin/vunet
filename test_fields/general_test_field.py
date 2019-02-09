import numpy as np

x = np.ones(shape=[2, 3, 4, 3])

x[:, :, :, 0] = 0
x[:, :, :, 1] = 1
x[:, :, :, 2] = 2

x[:, 1, :, :] += 1

print(x)
print("\n")
print(x[:, ::-1, :, :])

# x = x.reshape((1, 1, 1, 3))
#
# print(x)
