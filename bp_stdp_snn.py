import numpy as np
import matplotlib.pyplot as plt
import itertools

TIME_STEPS = 100  # Количество временных шагов для симуляции
TIME_PER_STEP = 1
TAU = 16  # Временная константа мембраны
THRESHOLD = 0.2  # Порог срабатывания
OUTPUT_THRESHOLD = 0.2
REST_POTENTIAL = 0.0  # Потенциал покоя
STDP_A_PLUS = 0.08  # Параметры STDP
STDP_A_MINUS = STDP_A_PLUS * 1.1
STDP_TAU = 32

class LIF:
    def __init__(self, input_count, tau, v_reset, v_threshold, relax_time):
        self.weights = np.zeros(input_count)
        self.tau = tau
        self.v_reset = v_reset
        self.v_threshold = v_threshold
        self.v = self.v_reset
        self.v_trace = []
        self.relax_time = relax_time
        self.relax_time_left = 0
        self.last_spike_time = -np.inf
        self.first_spike_time = np.inf
        self.spikes = []
        self.spike_trace = []

    def update(self, input_spikes, current_time, dt):
        spike = False
        
        if self.relax_time_left > 0:
            self.relax_time_left -= dt
            self.v = self.v_reset
        else:
            current_input = input_spikes @ self.weights
            self.v += (-self.v + current_input) * dt / self.tau
            self.v_trace.append(self.v)

            if self.v >= self.v_threshold:
                spike = True
                if (self.first_spike_time > current_time):
                    self.first_spike_time = current_time
                self.last_spike_time = current_time
                self.relax_time_left = self.relax_time
                self.v = self.v_reset

        if (spike):
            self.spikes.append(1)
            self.spike_trace.append(True)
        else:
            self.spikes.append(0)
            self.spike_trace.append(False)
            
        return spike
    
    def reset(self):
        self.v = self.v_reset
        self.last_spike_time = -np.inf
        self.first_spike_time = np.inf
        self.spikes = []
        self.relax_time_left = 0

class Classifier:
    def __init__(self, input_classes, hidden_sizes, output_clases, time_steps = TIME_STEPS, time_per_step = TIME_PER_STEP):
        self.input_classes = input_classes
        self.hidden_layers = []
        for i, hidden_size in enumerate(hidden_sizes):
            if (i == 0):
                self.hidden_layers.append([LIF(input_classes, TAU, REST_POTENTIAL, THRESHOLD, 0) for _ in range(hidden_size)])
            else:
                self.hidden_layers.append([LIF(hidden_sizes[i - 1], TAU, REST_POTENTIAL, THRESHOLD, 0) for _ in range(hidden_size)])
        self.output_layer = [LIF(hidden_sizes[len(hidden_sizes) - 1], TAU, REST_POTENTIAL, OUTPUT_THRESHOLD, 0) for _ in range(output_clases)]
        self.time_steps = time_steps
        self.time_per_step = time_per_step
        self.A_p = STDP_A_PLUS
        self.A_m = STDP_A_MINUS
        self.stdp_tau = STDP_TAU

    def encode_input(self, X):
        spikes = np.zeros((len(X), self.input_classes, self.time_steps))
        for j in range(self.input_classes):
            min = np.min(X[:, j])
            max = np.max(X[:, j])
            for i in range(len(X)):
                rate = (X[i, j] - min) / (max - min + 1e-9)
                spikes[i, j, :] = np.random.rand(self.time_steps) < rate
        return spikes

    def test(self, X_test, y_test, print_per_sample=False):
        self.clear_potential_history()
        X_test_spikes = self.encode_input(X_test)
        result = self.forward(X_test_spikes, y_test, print_per_sample=print_per_sample)

        for hidden_layer in self.hidden_layers:
            self.plot_potential_history(hidden_layer)
        self.plot_potential_history(self.output_layer)

        return result

    def reset_weights(self):
        for neuron in (itertools.chain.from_iterable(self.hidden_layers + [self.output_layer])):
            neuron.weights = np.random.rand(len(neuron.weights))
            #neuron.weights = np.full(len(neuron.weights), 1)

    def forward(self, X_spikes, y, stdp=False, print_per_sample=False):
        correct = 0
        results = np.zeros((len(self.output_layer) + 1, len(self.output_layer)))
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
                        hidden_spikes[i] = neuron.update(input_spikes, current_time, self.time_per_step)
                    input_spikes = hidden_spikes

                # Обновляем выходные нейроны
                output_spikes = np.zeros(len(self.output_layer))
                for i, neuron in enumerate(self.output_layer):
                    output_spikes[i] = neuron.update(input_spikes, current_time, self.time_per_step)
                    output_spike_times[i] = neuron.first_spike_time

                current_time += self.time_per_step

            if (stdp):
                self.stdp(X_spikes, y, sample_idx)

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

        for i in range(len(self.output_layer) + 1):
            print(f"neuron {i - 1}: 0 = {(results[i, 0] / np.sum(results[i, :]) * 100):2f}; 1 = {(results[i, 1] / np.sum(results[i, :]) * 100):2f}; 2 = {(results[i, 2] / np.sum(results[i, :]) * 100):2f}; count = {np.sum(results[i, :])}")
        print()

        return correct / len(X_spikes) * 100

    def stdp(self, X_spikes, y, sample_idx):
        # Получаем входные спайки для этого временного шага
        input_spikes = X_spikes[sample_idx, :, :]

        '''
        # Расчёт ошибки на выходе
        errors = np.zeros(len(self.output_layer))
        for i, neuron in enumerate(self.output_layer):
            if i == y[sample_idx]:
                errors[i] = 1 - neuron.spikes[sample_idx]  # должен спайкнуть
            else:
                errors[i] = -neuron.spikes[sample_idx]     # не должен

        # Обучение последнего слоя по ошибке
        
        self.bp_stdp_layer(self.output_layer, [neuron.spikes for neuron in self.hidden_layers[-1]], errors)

        prev_layer = self.output_layer
        for i, hidden_layer in enumerate(reversed(self.hidden_layers)):
            if i == 0:
                error = self.backpropagate_error(hidden_layer, self.output_layer, output_errors)
            else:
                error = self.backpropagate_error(hidden_layer, self.output_layer, output_errors)


        #self.stdp_supervised_output_layer(y[sample_idx], [neuron.spikes for neuron in self.hidden_layers[-1]])

    def backpropagate_error(self, presyn_layer, postsyn_layer, postsyn_errors):
        # Ошибка на выходном слое уже есть: output_errors
        # Теперь считаем ошибку на последнем скрытом слое
        presyn_errors = np.zeros(len(presyn_layer))
        for i, neuron in enumerate(presyn_layer):
            for j, postsyn_neuron in enumerate(postsyn_layer):
                presyn_errors[i] += postsyn_errors[j] * postsyn_neuron.weights[i]
        return presyn_errors
        '''
        desired_output = np.zeros(len(self.output_layer))
        desired_output[y[sample_idx]] = 1

        # --- Выходной слой ---
        output_errors = np.zeros(len(self.output_layer))
        for i, neuron in enumerate(self.output_layer):
            actual_output = 1 if np.isfinite(neuron.first_spike_time) else 0
            error = desired_output[i] - actual_output
            output_errors[i] = error

            #for j, input_neuron_spikes in enumerate(self.hidden_layers[-1]):
            #    for t_pre, spike_pre in enumerate(input_neuron_spikes.spikes):
            for j, input_neuron_spikes in enumerate(input_spikes):
                for t_pre, spike_pre in enumerate(input_neuron_spikes):
                    if spike_pre:
                        delta_t = neuron.first_spike_time - t_pre * self.time_per_step
                        if delta_t < 0:
                            dw = self.A_p * np.exp(delta_t / self.stdp_tau)
                        else:
                            dw = -self.A_m * np.exp(-delta_t / self.stdp_tau)
                        neuron.weights[j] += error * dw
                        neuron.weights[j] = max(neuron.weights[j], 1e-3)
            neuron.weights = np.clip(neuron.weights, 1e-3, max(neuron.weights))
            #neuron.weights /= np.linalg.norm(neuron.weights) + 1e-6

        # --- Скрытый слой ---
        hidden_errors = self.backpropagate_error(output_errors)
        for i, neuron in enumerate(self.hidden_layers[-1]):
            error = hidden_errors[i]
            for j in range(len(X_spikes[sample_idx])):
                for t_pre, spike_pre in enumerate(X_spikes[sample_idx, j]):
                    if spike_pre:
                        delta_t = neuron.first_spike_time - t_pre * self.time_per_step
                        if delta_t < 0:
                            dw = self.A_p * np.exp(delta_t / self.stdp_tau)
                        else:
                            dw = -self.A_m * np.exp(-delta_t / self.stdp_tau)
                        neuron.weights[j] += error * dw
                        neuron.weights[j] = max(neuron.weights[j], 1e-3)
            neuron.weights = np.clip(neuron.weights, 1e-3, max(neuron.weights))
            #neuron.weights /= np.linalg.norm(neuron.weights) + 1e-6
        
        self.print_weights()

    def backpropagate_error(self, output_errors):
        # Ошибка на выходном слое уже есть: output_errors
        # Теперь считаем ошибку на последнем скрытом слое
        hidden_errors = np.zeros(len(self.hidden_layers[-1]))
        for i, hidden_neuron in enumerate(self.hidden_layers[-1]):
            for j, output_neuron in enumerate(self.output_layer):
                hidden_errors[i] += output_errors[j] * output_neuron.weights[i]
        return hidden_errors

    def bp_stdp_layer(self, layer, input_spikes, errors):
        for i, neuron in enumerate(layer):
            error_mod = errors[i]  # модификатор от ошибки
            for t_post, spike_post in enumerate(neuron.spikes):
                if spike_post <= 0:
                    continue
                for j, input_neuron_spikes in enumerate(input_spikes):
                    for t_pre, spike_pre in enumerate(input_neuron_spikes):
                        if spike_pre <= 0:
                            continue
                        
                        delta_t = (t_post - t_pre) * self.time_per_step
                        
                        if delta_t > 0:
                            dw = self.A_p * np.exp(delta_t / self.stdp_tau)
                        else:
                            dw = -self.A_m * np.exp(-delta_t / self.stdp_tau)
                        
                        #print(error_mod * dw)
                        neuron.weights[j] += error_mod * dw
                        neuron.weights[j] = max(neuron.weights[j], 1e-3)
            
            print("Unnormalized")
            self.print_weights()
            
            # Нормализация весов
            neuron.weights = np.clip(neuron.weights, 1e-3, max(neuron.weights))
            neuron.weights /= np.linalg.norm(neuron.weights) + 1e-6

            print("Normalized")
            self.print_weights()

    def train(self, X_train, y_train, epochs=10):
        X_train_spikes = self.encode_input(X_train)
        self.print_weights()
        for epoch in range(epochs):
            self.clear_potential_history()
            accuracy = self.forward(X_train_spikes, y_train, True)
            print(f"Epoch: {epoch}; Accuracy: {accuracy}")

        for hidden_layer in self.hidden_layers:
            self.plot_potential_history(hidden_layer)
        self.plot_potential_history(self.output_layer)

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
            plt.ylim(min(neuron.v_trace), max(neuron.v_trace))

            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

    def print_weights(self):
        print("Hidden weights:")
        for hidden_layer in self.hidden_layers:
            for neuron in hidden_layer:
                print(neuron.weights)

        print("Output weights:")
        for neuron in self.output_layer:
            print(neuron.weights)
