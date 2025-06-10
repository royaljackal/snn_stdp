import numpy as np

class Izhikevich:
    def __init__(self, input_count, time_steps, a, b, c, d, v_threshold, print_input=False):
        self.weights = np.zeros(input_count)
        self.a = a
        self.b = b
        self.c = c  # reset voltage
        self.d = d  # recovery increment
        self.v = -65.0  # initial membrane potential
        self.u = self.b * self.v  # initial recovery variable
        self.v_threshold = v_threshold
        self.v_trace = []
        self.u_trace = []
        self.spikes = [0] * time_steps
        self.spike_trace = []
        self.last_spike_time = -np.inf
        self.first_spike_time = np.inf
        self.print_input = print_input

    def update(self, spike_indices, current_time, dt):
        operations = 0
        multiplications = 0
        current_input = 0
        for spike_index in spike_indices:
            current_input += self.weights[spike_index] * 20
            operations += 2
            multiplications += 1

        if self.print_input:
            print(current_input)

        # Izhikevich model update
        dv = (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + current_input) * dt
        du = self.a * (self.b * self.v - self.u) * dt
        multiplications += 6
        operations += 14

        self.v += dv
        self.u += du

        self.v_trace.append(self.v)
        self.u_trace.append(self.u)

        spike = False
        if self.v >= self.v_threshold:
            spike = True
            self.v = self.c
            self.u += self.d
            operations += 1
            if self.first_spike_time > current_time:
                self.first_spike_time = current_time
            self.last_spike_time = current_time

        if spike:
            self.spikes[current_time] = 1
            self.spike_trace.append(True)
        else:
            self.spike_trace.append(False)

        return spike, operations, multiplications

    def reset(self):
        self.v = -65.0
        self.u = self.b * self.v
        self.spikes = [0] * len(self.spikes)
        self.last_spike_time = -np.inf
        self.first_spike_time = np.inf