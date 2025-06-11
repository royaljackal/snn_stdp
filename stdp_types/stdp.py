import numpy as np

def train(X_sample_spikes, y, sample_idx, time_per_step, tau, A_p, A_m, window, hidden_layers, output_layer):
    
    for hidden_layer in hidden_layers:
        stdp_layer(hidden_layer, X_sample_spikes)
        input_spikes = [neuron.spikes for neuron in hidden_layer]

    stdp_layer(time_per_step, tau, A_p, A_m, window, output_layer, input_spikes)

def stdp_layer(time_per_step, tau, A_p, A_m, window, layer, input_spikes):
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

                        delta_t = (t_post - t_pre) * time_per_step

                        if delta_t >= 0:
                            dw = A_p * np.exp(delta_t / tau)
                            dw_inc += 1
                            #print(f"dw_inc = {dw}")
                        else:
                            dw = -A_m * np.exp(-delta_t / tau)
                            dw_dec += 1
                            #print(f"dw_dec = {dw}")

                        neuron.weights[j] += dw
                        neuron.weights[j] = max(neuron.weights[j], 1e-3)

                neuron.weights /= np.linalg.norm(neuron.weights) + 1e-6