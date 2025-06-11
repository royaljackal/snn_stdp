import numpy as np

def train(X_sample_spikes, y, sample_idx, current_time_step, input_classes, hidden_layers, output_layer, epsilon, lr):
    if (current_time_step <= epsilon or current_time_step % epsilon != 0): 
        return
    
    input_spikes = np.array(X_sample_spikes)
    class_count = len(output_layer)
    hidden_count = len(hidden_layers)
    delta = np.zeros([class_count, 1])
    correct_class = y[sample_idx]
    energy = 0.0

    output_weights = np.array([output_neuron.weights for output_neuron in output_layer]).T
    hidden_weights = []
    for hidden_layer in hidden_layers:
        hidden_weights.append(np.array([hidden_neuron.weights for hidden_neuron in hidden_layer]).T)

    output_spikes = np.array([output_neuron.spikes for output_neuron in output_layer])
    hidden_spikes = []
    for hidden_layer in hidden_layers:
        hidden_spikes.append(np.array([hidden_neuron.spikes for hidden_neuron in hidden_layer]))

    if (sum(output_spikes[correct_class, current_time_step - epsilon + 1 : current_time_step + 1]) < 1):
        delta[y, 0] = 1
        energy += 1
    for output_class in range(class_count):
        if (output_class != correct_class):
            if (sum(output_spikes[output_class, current_time_step - epsilon + 1 : current_time_step + 1]) >= 1):
                delta[output_class, 0] =- 1
                energy += 1

    delta_h = [1] * len(hidden_layers)
    delta_h[len(hidden_layers) - 1] = np.dot(output_weights, delta)
    output_weights += np.dot((sum(hidden_spikes[hidden_count - 1][:, current_time_step - epsilon + 1 : current_time_step + 1].T).T).reshape([len(hidden_layers[hidden_count - 1]), 1]), delta.T) * lr
    
    for der in range(hidden_count - 1, 0, -1):
        derivative = (sum(hidden_spikes[der][:, current_time_step - epsilon + 1 : current_time_step + 1].T).T) > 0 # or >= 0
        delta_h[der] = delta_h[der] * derivative.reshape([len(hidden_layers[der]), 1])
        delta_h[der-1] = np.dot(hidden_weights[der], delta_h[der])                
        hidden_weights[der] += np.dot((sum(hidden_spikes[der - 1][:, current_time_step - epsilon + 1 : current_time_step + 1].T).T).reshape([len(hidden_layers[der - 1]), 1]), delta_h[der].T) * lr
        
    derivative = (sum(hidden_spikes[0][:, current_time_step - epsilon + 1 : current_time_step + 1].T).T) > 0 # or >=0
    delta_h[0] = delta_h[0] * derivative.reshape([len(hidden_layers[0]), 1])
    hidden_weights[0] += np.dot((sum(input_spikes[:, current_time_step - epsilon + 1 : current_time_step + 1].T).T).reshape([input_classes, 1]), delta_h[0].T) * lr

    output_weights = output_weights.T
    for i, output_neuron in enumerate(output_layer):
        output_neuron.weights = output_weights[i]
    for l_i, hidden_layer in enumerate(hidden_layers):
        hidden_weights[l_i] = hidden_weights[l_i].T
        for i, hidden_neuron in enumerate(hidden_layer):
            hidden_neuron.weights = hidden_weights[l_i][i]