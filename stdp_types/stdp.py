import numpy as np

def train(X_sample_spikes, y_sample, time_per_step, 
          hidden_layers, output_layer,
          tau, A_p, A_m, window):
    
    input_spikes = X_sample_spikes
    for hidden_layer in hidden_layers:
        stdp_layer(hidden_layer, input_spikes, time_per_step, tau, A_p, A_m, window)
        input_spikes = hidden_layer.spikes

    stdp_layer(output_layer, input_spikes, time_per_step, tau, A_p, A_m, window)

def stdp_layer(layer, input_spikes, time_per_step, tau, A_p, A_m, window):
    for i, neuron_spikes_post in enumerate(layer.spikes):
        for t_post, spike_post in enumerate(neuron_spikes_post):
            if (spike_post <= 0):
                continue
            for j, neuron_spikes_pre in enumerate(input_spikes):
                for t_pre in range(max(0, t_post - window), min(t_post + window, len(neuron_spikes_pre))):
                    spike_pre = neuron_spikes_pre[t_pre]
                    if not spike_pre:
                            continue

                    delta_t = (t_post - t_pre) * time_per_step

                    if delta_t >= 0:
                        dw = A_p * np.exp(delta_t / tau)
                    else:
                        dw = -A_m * np.exp(-delta_t / tau)

                    layer.weights[i][j] += dw
                    layer.weights[i][j] = max(layer.weights[i][j], 1e-3)

            layer.weights[i] /= np.linalg.norm(layer.weights[i]) + 1e-6