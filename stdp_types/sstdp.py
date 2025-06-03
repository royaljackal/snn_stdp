import numpy as np

def train(X_spikes, y, sample_idx, time_per_step, tau, A_p, A_m, hidden_layers, output_layer):
    input_spikes = X_spikes[sample_idx, :, :]
    
    stdp_supervised_output_layer(time_per_step, tau, A_p, A_m, y[sample_idx], output_layer, [neuron.spikes for neuron in hidden_layers[len(hidden_layers) - 1]])

def stdp_layer(time_per_step, tau, A_p, A_m, layer, input_spikes):
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

def stdp_supervised_output_layer(time_per_step, tau, A_p, A_m, correct_class, output_layer, input_spikes):
    for i, neuron in enumerate(output_layer):
        target = (i == correct_class)

        #print(f"neuron {i}")
        #for neuron in output_layer:
        #    print(neuron.weights)

        for t_post, spike_post in enumerate(neuron.spikes):
            if not spike_post:
                continue
            for j, input_neuron_spikes in enumerate(input_spikes):
                for t_pre, spike_pre in enumerate(input_neuron_spikes):
                    if not spike_pre:
                        continue

                    delta_t = (t_post - t_pre) * time_per_step

                    if target:
                        # усиливаем, если правильный нейрон
                        if delta_t >= 0:
                            dw = A_p * np.exp(delta_t / tau)
                        else:
                            dw = -A_m * np.exp(-delta_t / tau)
                    else:
                        # подавляем, если неправильный нейрон
                        if delta_t >= 0:
                            dw = -A_p * np.exp(delta_t / tau)
                        else:
                            dw = A_m * np.exp(-delta_t / tau)

                    #print(f"weight_before:{neuron.weights[j]}")
                    neuron.weights[j] += dw
                    neuron.weights[j] = max(neuron.weights[j], 1e-3)
                    #print(f"weight_after:{neuron.weights[j]}")
                    #print(f"neuron_post:{i}; t_post:{t_post};\nneuron_pre:{j}; t_pre:{t_pre}\ndw:{dw}\n")

            # нормализация весов
            neuron.weights /= np.linalg.norm(neuron.weights) + 1e-6

        #    print(f"spike {t_post}")
        #    for neuron in output_layer:
        #        print(neuron.weights)
        #print()