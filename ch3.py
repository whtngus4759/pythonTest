# def step_function(x):
# 	y = x>0
# 	return y.astype(np.int)

# vex = np.array([-1.0,1.0,2.0])

# y= vex>0
# print(y)
# print(type(y))

# y=y.astype(np.int)
# print(y)

import numpy as np
import matplotlib.pylab as plt

def step_function(x):
	return np.array(x>0, dtype=np.int)

x = np.arange(-5.0, 5.0 , 0.1)
y=step_function(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
# plt.show()

def sigmoid(x):
	return 1/(1+np.exp(-x))
vec=np.array([-1.0,1.0,2.0])
print(sigmoid(vec))

def ReLU(x):
	return np.maximum(0,x)

print(ReLU(vec))



