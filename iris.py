import numpy as np
from sklearn import datasets
from stdp_snn import Classifier
import time

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train = np.concatenate((X[0:30], X[50:80], X[100:130]))
X_test = np.concatenate((X[30:50], X[80:100], X[130:150]))
y_train = np.concatenate((y[0:30], y[50:80], y[100:130]))
y_test = np.concatenate((y[30:50], y[80:100], y[130:150]))

NUM_INPUT_NEURONS = 4
NUM_HIDDEN_NEURONS = [12, 6]
NUM_OUTPUT_NEURONS = 3

TIME_STEPS = 100

TAU = 4
THRESHOLD = 1
OUTPUT_THRESHOLD = 2

BP_EPSILON = 8
BP_LR = 0.005

classifier = Classifier("lif", "bpstdp", NUM_INPUT_NEURONS, NUM_HIDDEN_NEURONS, NUM_OUTPUT_NEURONS, time_steps=TIME_STEPS, tau=TAU, threshold=THRESHOLD, output_threshold=OUTPUT_THRESHOLD, 
                        bp_epsilon=BP_EPSILON, bp_lr=BP_LR)

classifier.train(X_train, y_train, 10, shuffle_dataset=True)

start = time.time()
print(classifier.test(X_test, y_test, True))
end = time.time()
print(end - start)

classifier.plot_potential_history(classifier.output_layer)

input()