import numpy as np
import matplotlib.pyplot as plt
import itertools
import random
from neurons.LIF import LIF
from neurons.IF import IF
from neurons.Izhkevich import Izhikevich
from stdp_types import stdp, sstdp, bpstdp
from IPython.display import clear_output

TIME_STEPS = 300  # Количество временных шагов для симуляции
TIME_PER_STEP = 1
TAU = 4  # Временная константа мембраны
THRESHOLD = 0.4  # Порог срабатывания
OUTPUT_THRESHOLD = 0.8

STDP_A_PLUS = 0.16  # Параметры STDP
STDP_A_MINUS = STDP_A_PLUS * 0.9
STDP_TAU = 1
STDP_WINDOW = 10

BP_EPSILON = 2
BP_LR = 0.001

class Classifier:
    def __init__(self, neuron_type, stdp_type, input_classes, hidden_sizes, output_clases, time_steps = TIME_STEPS, time_per_step = TIME_PER_STEP, 
                 tau = TAU, threshold = THRESHOLD, output_threshold = OUTPUT_THRESHOLD, rest_potential = 0.0,
                 A_p = STDP_A_PLUS, A_m = STDP_A_MINUS, stdp_tau = STDP_TAU, stdp_window = STDP_WINDOW,
                 bp_epsilon = BP_EPSILON, bp_lr = BP_EPSILON):
        self.neuron_type = neuron_type
        self.stdp_type = stdp_type
        self.time_steps = time_steps
        self.time_per_step = time_per_step
        self.input_classes = input_classes
        self.hidden_layers = []
        for i, hidden_size in enumerate(hidden_sizes):
            if (i == 0):
                self.hidden_layers.append([self.create_neuron(input_classes, self.time_steps, tau, rest_potential, threshold, 0) for _ in range(hidden_size)])
            else:
                self.hidden_layers.append([self.create_neuron(hidden_sizes[i - 1], self.time_steps, tau, rest_potential, threshold, 0) for _ in range(hidden_size)])
        self.output_layer = [self.create_neuron(hidden_sizes[len(hidden_sizes) - 1], self.time_steps, tau, rest_potential, output_threshold, 0) for _ in range(output_clases)]
        
        self.A_p = A_p
        self.A_m = A_m
        self.stdp_tau = stdp_tau
        self.stdp_window = stdp_window

        self.bp_epsilon = bp_epsilon
        self.bp_lr = bp_lr

        self.reset_weights()

    def create_neuron(self, input_count, time_steps, tau, v_reset, v_threshold, relax_time, print_input = False):
        match self.neuron_type:
            case "lif":
                return LIF(input_count, time_steps, tau, v_reset, v_threshold, relax_time, print_input)
            case "if":
                return IF(input_count, time_steps, tau, v_reset, v_threshold, relax_time, print_input)
            case "izh":
                return Izhikevich(input_count, time_steps, a=0.02, b=0.2, c=-65, d=8, v_threshold=30, print_input=print_input)

    def encode_sample(self, X, sample_id):
        spikes = np.zeros((self.input_classes, self.time_steps))
        for feature_id in range(self.input_classes):
            min = np.min(X[:, feature_id])
            max = np.max(X[:, feature_id])
            
            rate = (X[sample_id, feature_id] - min) / (max - min + 1e-9)
            spikes[feature_id, :] = np.random.rand(self.time_steps) < rate
        return spikes

    def train(self, X_train, y_train, epochs=10, shuffle_dataset = False, plot_history = False):
        if (shuffle_dataset):
            p = np.random.permutation(len(X_train))
            X_train, y_train = X_train[p], y_train[p]

        #self.print_weights()
        operations = 0
        multiplications = 0
        spike_count = 0
        for epoch in range(epochs):
            self.clear_potential_history()
            accuracy, epoch_operations, epoch_multiplications, epoch_spike_count = self.forward(X_train, y_train, epoch, True)
            print(f"Epoch: {epoch}; Accuracy: {accuracy}")
            operations += epoch_operations
            multiplications += epoch_multiplications
            spike_count += epoch_spike_count
            #self.print_weights()

        #for hidden_layer in self.hidden_layers:
        #    self.plot_potential_history(hidden_layer)
        print(f"""
              Операций: {operations}; 
              Доля умножений: {multiplications/operations}; 
              Операций за проход: {operations/(epochs * len(X_train) * self.time_steps * (sum([len(hidden_layer) for hidden_layer in self.hidden_layers]) + len(self.output_layer)))}; 
              Доля спайковых операций: {spike_count/(epochs * len(X_train) * self.time_steps * (sum([len(hidden_layer) for hidden_layer in self.hidden_layers]) + len(self.output_layer)))}""")
        
        if (plot_history):
            self.plot_potential_history(self.output_layer)
    
    def test(self, X_test, y_test, print_per_sample=False, shuffle_dataset = False, plot_history = False):
        if (shuffle_dataset):
            p = np.random.permutation(len(X_test))
            X_test, y_test = X_test[p], y_test[p]

        accuracy, _, _, _ = self.forward(X_test, y_test, -1, False, print_per_sample=print_per_sample)

        if (plot_history):
            self.plot_potential_history(self.output_layer)
        return accuracy

    def reset_weights(self):
        for neuron in (itertools.chain.from_iterable(self.hidden_layers + [self.output_layer])):
            neuron.weights = np.random.rand(len(neuron.weights))

    def forward(self, X, y, epoch, use_stdp=False, print_per_sample=False):
        correct = 0
        results = np.zeros((len(self.output_layer) + 1, len(self.output_layer)))
        operations = 0
        multiplications = 0
        spike_count = 0
        for sample_idx in range(len(X)):
            #clear_output(wait=True)
            #print(f"Epoch: {epoch}; Sample: {sample_idx}")

            # Сброс нейронов
            for neuron in (itertools.chain.from_iterable(self.hidden_layers + [self.output_layer])):
                neuron.reset()

            X_sample_spikes = self.encode_sample(X, sample_idx)
            output_spike_times = np.zeros(len(self.output_layer))

            current_time = 0
            # Для каждого временного промежутка
            
            for current_time_step in range(self.time_steps):
                # Получаем входные спайки для этого временного шага
                input_spikes = X_sample_spikes[:, current_time_step]
                
                # Обновляем скрытые нейроны
                for hidden_layer in self.hidden_layers:
                    hidden_spikes = np.zeros(len(hidden_layer))
                    for i, neuron in enumerate(hidden_layer):
                        hidden_spikes[i], neuron_operations, neuron_multiplications = neuron.update(input_spikes.nonzero()[0], current_time, self.time_per_step)
                        operations += neuron_operations
                        multiplications += neuron_multiplications
                    input_spikes = hidden_spikes
                    spike_count += np.count_nonzero(hidden_spikes == 1)

                # Обновляем выходные нейроны
                output_spikes = np.zeros(len(self.output_layer))
                for i, neuron in enumerate(self.output_layer):
                    output_spikes[i], neuron_operations, neuron_multiplications = neuron.update(input_spikes.nonzero()[0], current_time, self.time_per_step)
                    output_spike_times[i] = neuron.first_spike_time
                    operations += neuron_operations
                    multiplications += neuron_multiplications
                current_time += self.time_per_step
                spike_count += np.count_nonzero(output_spikes == 1)
            
                #stdp в цикле
                if (use_stdp):
                    self.stdp_train(X_sample_spikes, y, sample_idx, current_time_step)

            if (not use_stdp):
                #Считаем точность
                if np.all(~np.isfinite(output_spike_times)):
                    predicted = -1
                else:
                    predicted = np.argmin(output_spike_times)

                for i, neuron in enumerate(self.output_layer):
                    results[predicted + 1, y[sample_idx]] += 1

                if (print_per_sample):
                    print(f"Predicted: {predicted}, Correct: {y[sample_idx]}")
                
                if predicted == y[sample_idx]:
                    correct += 1

        #stdp вне цикла
        #if (use_stdp):
        #    rand_X_spikes = list(range(len(X_spikes)))
        #    random.shuffle(rand_X_spikes)
        #    for sample_idx in rand_X_spikes:    
        #        self.stdp_train(X_spikes, y, sample_idx)

        #for i in range(len(self.output_layer) + 1):
        #    print(f"neuron {i - 1}: 0 = {(results[i, 0] / np.sum(results[i, :]) * 100):2f}; 1 = {(results[i, 1] / np.sum(results[i, :]) * 100):2f}; 2 = {(results[i, 2] / np.sum(results[i, :]) * 100):2f}; count = {np.sum(results[i, :])}")
        #print()

        if (use_stdp):
            return self.forward(X, y, -1)
        else:
            return correct / len(X) * 100, operations, multiplications, spike_count

    def stdp_train(self, X_sample_spikes, y, sample_idx, current_time_step):
        match self.stdp_type:
            case "stdp":
                stdp.train(X_sample_spikes, y, sample_idx, self.time_per_step, self.stdp_tau, self.A_p, self.A_m, self.stdp_window, self.hidden_layers, self.output_layer)
            case "sstdp":
                sstdp.train(X_sample_spikes, y, sample_idx, self.time_per_step, self.stdp_tau, self.A_p, self.A_m, self.stdp_window, self.hidden_layers, self.output_layer)
            case "bpstdp":
                bpstdp.train(X_sample_spikes, y, sample_idx, current_time_step, self.input_classes, self.hidden_layers, self.output_layer, self.bp_epsilon, self.bp_lr)

    def clear_potential_history(self):
        for neuron in (itertools.chain.from_iterable(self.hidden_layers + [self.output_layer])):
            neuron.v_trace.clear()
            neuron.spike_trace.clear()

    def plot_potential_history(self, neurons):
        for i, neuron in enumerate(neurons):
            plt.figure(figsize=(25, 10))
            plt.plot(neuron.v_trace, label='Membrane potential', linewidth=1, zorder=1)

            # Отметим спайки крестиками
            spike_times = [t for t, spike in enumerate(neuron.spike_trace) if spike]
            spike_values = [neuron.v_threshold] * len(spike_times)  # по уровню порога
            plt.scatter(spike_times, spike_values, color='red', marker='x', label='Spike', zorder=2)

            plt.xlabel('Time step')
            plt.ylabel('Membrane potential')
            plt.title(f'Output Neuron {i} Potential & Spikes')
            plt.axhline(neuron.v_threshold, color='gray', linestyle='--', label='Threshold')
            plt.ylim(min(neuron.v_trace) - 0.5, neuron.v_threshold + 1)

            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

    def print_weights(self):
        #print("Hidden weights:")
        #for hidden_layer in self.hidden_layers:
        #    for neuron in hidden_layer:
        #        print(neuron.weights)

        print("Output weights:")
        for neuron in self.output_layer:
            print(neuron.weights)