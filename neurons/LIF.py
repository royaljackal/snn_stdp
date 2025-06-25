import numpy as np

class LIF:
    def __init__(self, size, input_count, time_steps, tau, v_reset, v_threshold, relax_time, print_input = False):
        self.size = size
        self.input_count = input_count
        self.weights = np.zeros((size, input_count))

        self.tau = tau
        self.v_reset = v_reset
        self.v_threshold = v_threshold
        self.relax_time = relax_time
        
        self.v = np.full(size, v_reset, dtype=np.float32)
        self.relax_time_left = np.zeros(size, dtype=np.float32)
        
        self.spikes = np.zeros((size, time_steps), dtype=np.uint8)
        
        self.print_input = print_input
        self.v_trace = []
        self.spike_trace = []

    def update(self, input_spikes, current_time, dt):
        operations = 0
        multiplications = 0

        current_input = self.weights @ input_spikes
        operations += self.size * np.count_nonzero(input_spikes)

        if self.print_input:
            print(current_input)
            
        dv = (-self.v + current_input) * (dt / self.tau)
        self.v += dv
        operations += 4 * self.size
        multiplications += 2 * self.size

        self.v_trace.append(self.v.copy())

        refractory_mask = self.relax_time_left > 0
        self.v[refractory_mask] = self.v_reset
        self.relax_time_left[refractory_mask] -= dt

        spike_mask = self.v >= self.v_threshold
        self.spikes[:, current_time] = spike_mask.astype(np.uint8)
        self.spike_trace.append(spike_mask.copy())

        self.v[spike_mask] = self.v_reset
        self.relax_time_left[spike_mask] = self.relax_time

        return spike_mask, operations, multiplications
    
    def reset(self):
        self.v.fill(self.v_reset)
        self.spikes.fill(0)
        self.relax_time_left.fill(0)