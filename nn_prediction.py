# USAGE
# python3 nn_prediction.py

# import the necessary packages
from modules.nn import NeuralNetwork
import numpy as np

# construct the function dataset
X = np.array([[0], [0.05], [0.1], [0.15], [0.2], [0.25], [0.3], [0.325], [0.35], [0.4], [0.45], [0.5], [0.55], [0.6], [0.65], [0.675], [0.7], [0.75], [0.8], [0.85], [0.9], [0.925], [0.95], [1]])

y = np.array([[0.1], [0.2], [0.4], [0.55], [0.6], [0.55], [0.45], [0.3], [0.2], [0.1], [0.05], [0.02], [0.05], [0.1], [0.2], [0.3], [0.4], [0.55], [0.6], [0.55], [0.45], [0.3], [0.2], [0.1]])

# define our 1-4-1 neural network and train it
nn = NeuralNetwork([1, 4, 1], alpha=1)
nn.fit(X, y, epochs=20000)

# now that our network is trained, loop over the input data points
for (x, target) in zip(X, y):
	# make a prediction on the data point and display the result
	# to our console
	pred = nn.predict(x)[0][0]
	#step = 1 if pred > 0.5 else 0
	print("[INFO] data={}, ground-truth={}, pred={:.4f}".format(
		x, target[0], pred))

# print the weights vector
W = nn.weights()
print(W)
