import numpy as np

class Perceptron:
    def __init__(self, inputs, bias = 1.0):
        self.weights = (np.random.rand(inputs + 1) * 2) - 1 # number of inputs + 1 for the bias input
        self.bias = bias

    def run(self, x):
        return self.sigmoid(np.dot(np.append(x, self.bias), self.weights))

    def set_weights(self, w_init):
        self.weights = np.array(w_init)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))


neuron = Perceptron(inputs=2)
neuron.set_weights([30, 30, -15])

print(f"{neuron.run([0, 0]):.10f}")
print(f"{neuron.run([0, 1]):.10f}")
print(neuron.run([1, 0]))
print(neuron.run([1, 1]))
#%%
