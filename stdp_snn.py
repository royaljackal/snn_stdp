import numpy as np
import matplotlib.pyplot as plt
import itertools
from neurons.LIF import LIF
from neurons.IF import IF
from neurons.Izhkevich import Izhikevich
from stdp_types import stdp, sstdp

TIME_STEPS = 300  # Количество временных шагов для симуляции
TIME_PER_STEP = 1
TAU = 8  # Временная константа мембраны
THRESHOLD = 0.4  # Порог срабатывания
OUTPUT_THRESHOLD = 0.2
REST_POTENTIAL = 0.0  # Потенциал покоя
STDP_A_PLUS = 0.16  # Параметры STDP
STDP_A_MINUS = STDP_A_PLUS * 0.9
STDP_TAU = 1

class Classifier:
    def __init__(self, neuron_type, stdp_type, input_classes, hidden_sizes, output_clases, time_steps = TIME_STEPS, time_per_step = TIME_PER_STEP):
        self.neuron_type = neuron_type
        self.stdp_type = stdp_type
        self.input_classes = input_classes
        self.hidden_layers = []
        for i, hidden_size in enumerate(hidden_sizes):
            if (i == 0):
                self.hidden_layers.append([self.create_neuron(input_classes, TAU, REST_POTENTIAL, THRESHOLD, 0) for _ in range(hidden_size)])
            else:
                self.hidden_layers.append([self.create_neuron(hidden_sizes[i - 1], TAU, REST_POTENTIAL, THRESHOLD, 0) for _ in range(hidden_size)])
        self.output_layer = [self.create_neuron(hidden_sizes[len(hidden_sizes) - 1], TAU, REST_POTENTIAL, OUTPUT_THRESHOLD, 0) for _ in range(output_clases)]
        self.time_steps = time_steps
        self.time_per_step = time_per_step
        self.A_p = STDP_A_PLUS
        self.A_m = STDP_A_MINUS
        self.stdp_tau = STDP_TAU

    def create_neuron(self, input_count, tau, v_reset, v_threshold, relax_time, print_input = False):
        match self.neuron_type:
            case "lif":
                return LIF(input_count, tau, v_reset, v_threshold, relax_time, print_input)
            case "if":
                return IF(input_count, tau, v_reset, v_threshold, relax_time, print_input)
            case "izh":
                return Izhikevich(input_count, a=0.02, b=0.2, c=-65, d=8, v_threshold=30, print_input=print_input)

    def encode_input(self, X):
        spikes = np.zeros((len(X), self.input_classes, self.time_steps))
        for j in range(self.input_classes):
            min = np.min(X[:, j])
            max = np.max(X[:, j])
            for i in range(len(X)):
                rate = (X[i, j] - min) / (max - min + 1e-9)
                spikes[i, j, :] = np.random.rand(self.time_steps) < rate
        return spikes

    def train(self, X_train, y_train, epochs=10):
        X_train_spikes = self.encode_input(X_train)
        #self.print_weights()
        operations = 0
        multiplications = 0
        spike_count = 0
        for epoch in range(epochs):
            self.clear_potential_history()
            if (epoch % 2 == 0):
                accuracy, epoch_operations, epoch_multiplications, epoch_spike_count = self.forward(X_train_spikes, y_train, True)
            else:
                accuracy, epoch_operations, epoch_multiplications, epoch_spike_count = self.forward(X_train_spikes, y_train, False)
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
        self.plot_potential_history(self.output_layer)
    
    def test(self, X_test, y_test, print_per_sample=False):
        X_test_spikes = self.encode_input(X_test)
        accuracy, operations, multiplications, spike_count = self.forward(X_test_spikes, y_test, True, print_per_sample=print_per_sample)
        self.plot_potential_history(self.output_layer)
        return accuracy

    def reset_weights(self):
        for neuron in (itertools.chain.from_iterable(self.hidden_layers + [self.output_layer])):
            neuron.weights = np.random.rand(len(neuron.weights))

    def forward(self, X_spikes, y, use_stdp=False, print_per_sample=False):
        correct = 0
        results = np.zeros((len(self.output_layer) + 1, len(self.output_layer)))
        operations = 0
        multiplications = 0
        spike_count = 0
        for sample_idx in range(len(X_spikes)):
            # Сброс нейронов
            for neuron in (itertools.chain.from_iterable(self.hidden_layers + [self.output_layer])):
                neuron.reset()

            output_spike_times = np.zeros(len(self.output_layer))

            current_time = 0
            # Для каждого временного промежутка
            for current_time_step in range(self.time_steps):
                # Получаем входные спайки для этого временного шага
                input_spikes = X_spikes[sample_idx, :, current_time_step]
                
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

            if (use_stdp):
                self.stdp_train(X_spikes, y, sample_idx)
            
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

        #for i in range(len(self.output_layer) + 1):
        #    print(f"neuron {i - 1}: 0 = {(results[i, 0] / np.sum(results[i, :]) * 100):2f}; 1 = {(results[i, 1] / np.sum(results[i, :]) * 100):2f}; 2 = {(results[i, 2] / np.sum(results[i, :]) * 100):2f}; count = {np.sum(results[i, :])}")
        print()

        return correct / len(X_spikes) * 100, operations, multiplications, spike_count

    def stdp_train(self, X_spikes, y, sample_idx):
        match self.stdp_type:
            case "stdp":
                stdp.train(X_spikes, y, sample_idx, self.time_per_step, self.stdp_tau, self.A_p, self.A_m, self.hidden_layers, self.output_layer)
            case "sstdp":
                sstdp.train(X_spikes, y, sample_idx, self.time_per_step, self.stdp_tau, self.A_p, self.A_m, self.hidden_layers, self.output_layer)
            case "bpstdp":
                return

    def stdp_layer(self, layer, input_spikes):
        dw_inc = 0
        dw_dec = 0
        for i, neuron in enumerate(layer):
            for t_post, spike_post in enumerate(neuron.spikes):
                if (spike_post <= 0):
                    continue
                for j, input_neuron_spikes in enumerate(input_spikes):
                    for t_pre, spike_pre in enumerate(input_neuron_spikes):
                        if (spike_pre <= 0):
                            continue

                        delta_t = (t_post - t_pre) * self.time_per_step

                        if delta_t >= 0:
                            dw = self.A_p * np.exp(delta_t / self.stdp_tau)
                            dw_inc += 1
                            #print(f"dw_inc = {dw}")
                        else:
                            dw = -self.A_m * np.exp(-delta_t / self.stdp_tau)
                            dw_dec += 1
                            #print(f"dw_dec = {dw}")

                        neuron.weights[j] += dw
                        neuron.weights[j] = max(neuron.weights[j], 1e-3)

                neuron.weights /= np.linalg.norm(neuron.weights) + 1e-6

        #print (f"dw_inc: {dw_inc}; dw_dec: {dw_dec}")

    def stdp_supervised_output_layer(self, correct_class, input_spikes):
        for i, neuron in enumerate(self.output_layer):
            target = (i == correct_class)
            for t_post, spike_post in enumerate(neuron.spikes):
                if not spike_post:
                    continue
                for j, input_neuron_spikes in enumerate(input_spikes):
                    for t_pre, spike_pre in enumerate(input_neuron_spikes):
                        if not spike_pre:
                            continue

                        delta_t = (t_post - t_pre) * self.time_per_step

                        if target:
                            # усиливаем, если правильный нейрон
                            if delta_t >= 0:
                                dw = self.A_p * np.exp(delta_t / self.stdp_tau)
                            else:
                                dw = -self.A_m * np.exp(-delta_t / self.stdp_tau)
                        else:
                            # подавляем, если неправильный нейрон
                            if delta_t >= 0:
                                dw = -self.A_p * np.exp(delta_t / self.stdp_tau)
                            else:
                                dw = self.A_m * np.exp(-delta_t / self.stdp_tau)

                        neuron.weights[j] += dw
                        neuron.weights[j] = max(neuron.weights[j], 1e-3)

                # нормализация весов
                neuron.weights /= np.linalg.norm(neuron.weights) + 1e-6

    def clear_potential_history(self):
        for neuron in (itertools.chain.from_iterable(self.hidden_layers + [self.output_layer])):
            neuron.v_trace = []
            neuron.spike_trace = []

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