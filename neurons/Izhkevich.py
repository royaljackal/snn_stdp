import numpy as np

class Izhikevich:
    def __init__(self, size, input_count, time_steps, a, b, c, d, v_threshold = 30, print_input=False):
        self.size = size
        self.input_count = input_count
        self.weights = np.zeros((size, input_count))

        self.a = a
        self.b = b
        self.c = c  # reset voltage
        self.d = d  # recovery increment
        self.v = np.full(size, self.c)  # initial membrane potential
        self.u = self.b * self.v  # initial recovery variable
        self.v_threshold = v_threshold

        self.spikes = np.zeros((size, time_steps), dtype=np.uint8)

        self.print_input = print_input
        self.v_trace = []
        self.u_trace = []
        self.spike_trace = []

    def update(self, input_spikes, current_time, dt):
        operations = 0
        multiplications = 0

        current_input = self.weights @ (input_spikes * 20)
        operations += self.size * len(input_spikes.nonzero())

        if self.print_input:
            print(current_input)

        dv = (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + current_input) * dt
        du = self.a * (self.b * self.v - self.u) * dt
        multiplications += 6 * self.size
        operations += 14 * self.size

        self.v += dv
        self.u += du

        self.v_trace.append(self.v.copy())
        self.u_trace.append(self.u.copy())

        spike_mask = self.v >= self.v_threshold
        self.spikes[:, current_time] = spike_mask.astype(np.uint8)
        self.spike_trace.append(spike_mask.copy())

        self.v[spike_mask] = self.c
        self.u[spike_mask] += self.d

        return spike_mask, operations, multiplications

    def reset(self):
        self.v = np.full(self.size, self.c)
        self.u = self.b * self.v
        self.spikes.fill(0)