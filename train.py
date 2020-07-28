import numpy as np
from perceptron import Perceptron

training_inputs = []
training_inputs.append(np.array([1,1]))
training_inputs.append(np.array([1,0]))
training_inputs.append(np.array([0,1]))
training_inputs.append(np.array([0,0]))

labels = np.array([1,0,0,0])

perceptron = Perceptron(2)
perceptron.train(training_inputs , labels)

inputs = np.array([0,0])
print(perceptron.predict(inputs))

print("Weights & Bias:")
print(perceptron.weights)
print("Threshold:")
print(perceptron.threshold)
print("Learning_Rate:")
print(perceptron.learning_rate)
//This Clearly shows weights& bias depends on Learning rate 