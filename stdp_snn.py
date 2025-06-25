import numpy as np
import matplotlib.pyplot as plt
from neurons.LIF import LIF
from neurons.IF import IF
from neurons.Izhkevich import Izhikevich
from stdp_types import stdp, sstdp, bpstdp
from IPython.display import clear_output
from io import BytesIO
from PIL import Image

TIME_STEPS = 50
TIME_PER_STEP = 1
TAU = 4
THRESHOLD = 0.4
OUTPUT_THRESHOLD = 0.8

STDP_A_PLUS = 0.16
STDP_A_MINUS = STDP_A_PLUS * 0.9
STDP_TAU = 1
STDP_WINDOW = 10

BP_EPSILON = 2
BP_LR = 0.001

class Classifier:
    def __init__(self, neuron_type, stdp_type, input_classes, hidden_sizes, output_clases, time_steps = TIME_STEPS, time_per_step = TIME_PER_STEP,
                 tau = TAU, threshold = THRESHOLD, output_threshold = OUTPUT_THRESHOLD, rest_potential = 0.0,
                 A_p = STDP_A_PLUS, A_m = STDP_A_MINUS, stdp_tau = STDP_TAU, stdp_window = STDP_WINDOW,
                 bp_epsilon = BP_EPSILON, bp_lr = BP_EPSILON,
                 weights_type = "rand"):
        self.neuron_type = neuron_type
        self.stdp_type = stdp_type
        self.time_steps = time_steps
        self.time_per_step = time_per_step
        self.input_classes = input_classes
        self.hidden_layers = []
        for i, hidden_size in enumerate(hidden_sizes):
            if (i == 0):
                self.hidden_layers.append(self.create_neuron(input_classes, hidden_size, self.time_steps, tau, rest_potential, threshold, 0))
            else:
                self.hidden_layers.append(self.create_neuron(hidden_sizes[i - 1], hidden_size, self.time_steps, tau, rest_potential, threshold, 0))
        self.output_layer = self.create_neuron(hidden_sizes[len(hidden_sizes) - 1], output_clases, self.time_steps, tau, rest_potential, output_threshold, 0)
        
        self.A_p = A_p
        self.A_m = A_m
        self.stdp_tau = stdp_tau
        self.stdp_window = stdp_window

        self.bp_epsilon = bp_epsilon
        self.bp_lr = bp_lr

        self.best_accuracy = -1
        self.error_count = 0
        self.error_margin = 1
        self.weights_type = weights_type

        self.reset_weights()

    def create_neuron(self, size, input_count, time_steps, tau, v_reset, v_threshold, relax_time, print_input = False):
        match self.neuron_type:
            case "lif":
                return LIF(input_count, size, time_steps, tau, v_reset, v_threshold, relax_time, print_input)
            case "if":
                return IF(input_count, size, time_steps, tau, v_reset, v_threshold, relax_time, print_input)
            case "izh":
                return Izhikevich(input_count, size, time_steps, a=0.02, b=0.2, c=-65.0, d=8.0, v_threshold=v_threshold, print_input=print_input)

    def encode_sample(self, X, sample_id):
        if not hasattr(self, 'mins'):
            self.mins = X.min(axis=0)
            self.maxs = X.max(axis=0)

        X_sample = X[sample_id]
        rates = (X_sample - self.mins) / (self.maxs - self.mins + 1e-9)
        randoms = np.random.rand(self.input_classes, self.time_steps)
        return (randoms < rates[:, None]).astype(np.uint8)

    def train(self, X_train, y_train, epochs=10, shuffle_dataset = False, plot_history = False, print_per_sample = False, print_progress = False, ignore_test = False):
        self.error_count = 0
        self.best_accuracy = -1
        self.epoch_log = []

        operations = 0
        multiplications = 0
        spike_count = 0
        for epoch in range(epochs):
            if (shuffle_dataset):
                p = np.random.permutation(len(X_train))
                X_train, y_train = X_train[p], y_train[p]

            self.clear_potential_history()
            accuracy, epoch_operations, epoch_multiplications, epoch_spike_count = self.forward(X_train, y_train, epoch, True, print_per_sample, print_progress, ignore_test)
            operations += epoch_operations
            multiplications += epoch_multiplications
            spike_count += epoch_spike_count

            if (accuracy >= self.best_accuracy):
                self.best_accuracy = accuracy
                self.save_weights()
                self.error_count = 0
            """
            else:
                self.error_count += 1
                if (self.error_count >= self.error_margin):
                    self.load_weights()
                    self.error_count = 0
            """
            epoch_result = f"Epoch: {epoch}; Accuracy: {self.best_accuracy}"
            if (print_progress):
                self.epoch_log.append(f"Epoch: {epoch}; Accuracy: {accuracy}")
                for log_item in self.epoch_log:
                    print(log_item)
            else:
                print(epoch_result)


        if (accuracy < self.best_accuracy):
            self.load_weights()

        print(f"""
              Операций: {operations}; 
              Доля умножений: {multiplications/float(operations)}; 
              Операций за проход: {operations/(epochs * len(X_train) * self.time_steps * (sum([hidden_layer.size for hidden_layer in self.hidden_layers]) + self.output_layer.size))};
              Доля спайковых операций: {spike_count/(epochs * len(X_train) * self.time_steps * (sum([hidden_layer.size for hidden_layer in self.hidden_layers]) + self.output_layer.size))}""")
        
        if (plot_history):
            self.plot_potential_history(self.output_layer)
    
    def test(self, X_test, y_test, print_per_sample=False, shuffle_dataset = False, plot_history = False):
        if (shuffle_dataset):
            p = np.random.permutation(len(X_test))
            X_test, y_test = X_test[p], y_test[p]

        self.clear_potential_history()
        accuracy, _, _, _ = self.forward(X_test, y_test, -1, False, print_per_sample=print_per_sample)

        if (plot_history):
            self.plot_potential_history(self.output_layer)
        return accuracy
    
    def test_sample(self, sample):
        self.clear_potential_history()

        sample_spikes = self.encode_sample([sample], 0)
        for current_time_step in range(self.time_steps):
            input_spikes = sample_spikes[:, current_time_step]
            
            for hidden_layer in self.hidden_layers:
                input_spikes, _, _ = hidden_layer.update(input_spikes, current_time_step, self.time_per_step)

            self.output_layer.update(input_spikes, current_time_step, self.time_per_step)
        
        spike_counts = np.sum(self.output_layer.spikes, axis=1)
        return np.argmax(spike_counts)

    def reset_weights(self):
        for layer in self.hidden_layers + [self.output_layer]:
            if (self.weights_type == "rand"):
                layer.weights = np.random.rand(*layer.weights.shape)
            elif (self.weights_type == "randn"):
                layer.weights = np.random.randn(*layer.weights.shape)

    def save_weights(self):
        self.saved_weights = [0] * (len(self.hidden_layers) + 1)
        for i, layer in enumerate(self.hidden_layers + [self.output_layer]):
            self.saved_weights[i] = layer.weights

    def load_weights(self):
        if hasattr(self, 'saved_weights'):
            for i, layer in enumerate(self.hidden_layers + [self.output_layer]):
                layer.weights = self.saved_weights[i]

    def forward(self, X, y, epoch, use_stdp=False, print_per_sample=False, print_progress = False, ignore_test=False):
        correct = 0
        operations = 0
        multiplications = 0
        spike_count = 0
        for sample_idx in range(len(X)):
            if (print_progress):
                clear_output(wait=True)
                print(f"Epoch: {epoch}; Sample: {sample_idx}")
                for log_item in self.epoch_log:
                    print(log_item)

            for layer in self.hidden_layers + [self.output_layer]:
                layer.reset()

            X_sample_spikes = self.encode_sample(X, sample_idx)

            for current_time_step in range(self.time_steps):
                input_spikes = X_sample_spikes[:, current_time_step]
                
                for hidden_layer in self.hidden_layers:
                    hidden_spikes, layer_operations, layer_multiplications = hidden_layer.update(input_spikes, current_time_step, self.time_per_step)
                    operations += layer_operations
                    multiplications += layer_multiplications
                    input_spikes = hidden_spikes
                    spike_count += np.count_nonzero(hidden_spikes == 1)

                output_spikes, layer_operations, layer_multiplications = self.output_layer.update(input_spikes, current_time_step, self.time_per_step)
                operations += layer_operations
                multiplications += layer_multiplications
                spike_count += np.count_nonzero(output_spikes == 1)
                
                if (use_stdp and self.stdp_type == "bpstdp"):
                    self.stdp_train(X_sample_spikes, y, sample_idx, current_time_step)

            if (use_stdp and self.stdp_type != "bpstdp"):
                self.stdp_train(X_sample_spikes, y, sample_idx, current_time_step)
            
            if (not use_stdp or self.stdp_type != "bpstdp"):
                spike_counts = np.sum(self.output_layer.spikes, axis=1)
                predicted = np.argmax(spike_counts)
                if (spike_counts[predicted] == 0):
                    predicted = -1

                if (print_per_sample):
                    print(f"Predicted: {predicted}, Correct: {y[sample_idx]}")
                
                if predicted == y[sample_idx]:
                    correct += 1

        if (ignore_test):
            return 0, operations, multiplications, spike_count
        elif (use_stdp and (self.stdp_type == "bpstdp" or len(self.hidden_layers) > 1)):
            return self.forward(X, y, -1)
        else:
            return correct / len(X) * 100, operations, multiplications, spike_count

    def stdp_train(self, X_sample_spikes, y, sample_idx, current_time_step):
        y_sample = y[sample_idx]
        match self.stdp_type:
            case "stdp":
                stdp.train(X_sample_spikes, y_sample, self.time_per_step, 
                           self.hidden_layers, self.output_layer,
                           self.stdp_tau, self.A_p, self.A_m, self.stdp_window)
            case "sstdp":
                sstdp.train(X_sample_spikes, y_sample, self.time_per_step, 
                            self.hidden_layers, self.output_layer,
                            self.stdp_tau, self.A_p, self.A_m, self.stdp_window)
            case "bpstdp":
                bpstdp.train(X_sample_spikes, y_sample, current_time_step, 
                             self.input_classes, self.hidden_layers, self.output_layer, 
                             self.bp_epsilon, self.bp_lr)

    def clear_potential_history(self):
        for layer in self.hidden_layers + [self.output_layer]:
            layer.v_trace.clear()
            layer.spike_trace.clear()

    def plot_potential_history(self, layer, return_images = False):
        v_trace = np.array(layer.v_trace)
        spike_trace = np.array(layer.spike_trace)

        images = []
        
        for i in range(layer.size):
            neuron_v_trace = v_trace[:, i]
            neuron_spike_trace = spike_trace[:, i]

            plt.figure(figsize=(25, 10))
            plt.plot(neuron_v_trace, label='Membrane potential', linewidth=1, zorder=1)

            spike_times = np.where(neuron_spike_trace)[0]
            spike_values = np.full_like(spike_times, fill_value=layer.v_threshold, dtype="float32")
            plt.scatter(spike_times, spike_values, color='red', marker='x', label='Spike', zorder=2)

            plt.xlabel('Time step')
            plt.ylabel('Membrane potential')
            plt.title(f'Output Neuron {i} Potential & Spikes')
            plt.axhline(layer.v_threshold, color='gray', linestyle='--', label='Threshold')
            plt.ylim(min(neuron_v_trace) - 0.5, layer.v_threshold + 1)

            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            if (return_images):
                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img = Image.open(buf)
                images.append(img)
                plt.close()

        if (return_images):
            return images
        else:
            plt.show()

    def print_weights(self):
        #print("Hidden weights:")
        #for hidden_layer in self.hidden_layers:
        #    for neuron in hidden_layer:
        #        print(neuron.weights)

        print("Output weights:")
        print(self.output_layer.weights)