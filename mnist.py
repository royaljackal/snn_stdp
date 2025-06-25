import numpy as np
from tensorflow.keras.datasets import mnist
from stdp_snn import Classifier
import time

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

X_test_small = X_test[:200]
y_test_small = y_test[:200]

NUM_INPUT_NEURONS = 784  # 784 пикселя в MNIST
NUM_OUTPUT_NEURONS = 10  # 10 кассов в MNIST
NUM_HIDDEN_NEURONS = [750, 150]

TAU = 4  # Временная константа мембраны
THRESHOLD = 10  # Порог срабатывания
OUTPUT_THRESHOLD = 20

BP_EPSILON = 2
BP_LR = 0.0005

classifier = Classifier("if", "bpstdp", NUM_INPUT_NEURONS, NUM_HIDDEN_NEURONS, NUM_OUTPUT_NEURONS, tau=TAU, threshold=THRESHOLD, output_threshold=OUTPUT_THRESHOLD, 
                        bp_epsilon=BP_EPSILON, bp_lr=BP_LR, weights_type="randn")

classifier.train(X_test_small, y_test_small, 1, shuffle_dataset=True, print_progress=True, ignore_test=True)

start = time.time()
print(f"Accuracy: {classifier.test(X_test_small, y_test_small)}")
end = time.time()
print(f"Time elapsed: {end - start}")

classifier.plot_potential_history(layer=classifier.output_layer)

input()