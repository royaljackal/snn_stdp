import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from stdp_snn import Classifier
import time

NUM_INPUT_NEURONS = 4  # 4 признака в Iris dataset
NUM_HIDDEN_NEURONS = [10, 10]
NUM_OUTPUT_NEURONS = 3  # 3 класса в Iris

# Загружаем датасет Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Масштабируем данные
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Разделим данные на обучающую и тестовую выборки
X_train = np.concatenate((X[0:30], X[50:80], X[100:130]))
X_test = np.concatenate((X[30:50], X[80:100], X[130:150]))
y_train = np.concatenate((y[0:30], y[50:80], y[100:130]))
y_test = np.concatenate((y[30:50], y[80:100], y[130:150]))

classifier = Classifier("lif", "bpstdp", NUM_INPUT_NEURONS, NUM_HIDDEN_NEURONS, NUM_OUTPUT_NEURONS)
classifier.reset_weights()

classifier.train(X_train, y_train, 5)

#start = time.time()
#print(classifier.test(X_test, y_test, True))
#end = time.time()
#print(end - start)