# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 16:24:32 2020

@author: Shail Thakkar
"""

import numpy as np
from perceptron import Perceptron

training_inputs = []
training_inputs.append(np.array([1,0,0,0]))
training_inputs.append(np.array([0,1,0,0]))
training_inputs.append(np.array([0,0,1,0]))
training_inputs.append(np.array([0,0,0,1]))
training_inputs.append(np.array([1,1,0,0]))
training_inputs.append(np.array([0,0,1,1]))
training_inputs.append(np.array([1,1,1,0]))
training_inputs.append(np.array([0,1,1,1]))
training_inputs.append(np.array([1,0,0,1]))
training_inputs.append(np.array([0,1,1,0]))

labels = np.array([0,1,1,0,1,1,1,1,0,1])

perceptron = Perceptron(4)
perceptron.train(training_inputs , labels)

inputs = np.array([0,1,1,1])
print(perceptron.predict(inputs))

inputs = np.array([1,0,0,1])
print(perceptron.predict(inputs))
