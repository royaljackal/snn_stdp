import numpy as np

class IF:
    def __init__(self, input_count, tau, v_reset, v_threshold, relax_time, print_input = False):
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
        self.print_input = print_input

    def update(self, spike_indicies, current_time, dt):
        operations = 0
        multiplications = 0
        current_input = 0

        for spike_index in spike_indicies:
            current_input += self.weights[spike_index]
            operations += 1

        if self.print_input:
            print(current_input)

        self.v += current_input * dt / self.tau
        self.v_trace.append(self.v)
        operations += 3
        multiplications += 2

        spike = False
        if self.v >= self.v_threshold:
            spike = True
            if (self.first_spike_time > current_time):
                self.first_spike_time = current_time
            self.last_spike_time = current_time
            self.relax_time_left = self.relax_time
            self.v = self.v_reset
        elif self.relax_time_left > 0:
            self.relax_time_left -= dt
            self.v = self.v_reset

        if (spike):
            self.spikes.append(1)
            self.spike_trace.append(True)
        else:
            self.spikes.append(0)
            self.spike_trace.append(False)
        return spike, operations, multiplications
    
    def reset(self):
        self.v = self.v_reset
        self.last_spike_time = -np.inf
        self.first_spike_time = np.inf
        self.spikes = []
        self.relax_time_left = 0