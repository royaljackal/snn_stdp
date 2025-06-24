import numpy as np

def train(X_sample_spikes, y_sample, current_time_step, 
          input_classes, hidden_layers, output_layer, 
          epsilon, lr):
    if (current_time_step <= epsilon or current_time_step % epsilon != 0): 
        return
    
    input_spikes = np.array(X_sample_spikes)
    class_count = output_layer.size
    hidden_count = len(hidden_layers)
    hidden_sizes = [layer.size for layer in hidden_layers]

    output_weights = output_layer.weights.T
    hidden_weights = []
    for hidden_layer in hidden_layers:
        hidden_weights.append(hidden_layer.weights.T)

    output_spikes = output_layer.spikes
    hidden_spikes = []
    for hidden_layer in hidden_layers:
        hidden_spikes.append(hidden_layer.spikes)

    output_weights, hidden_weights = train_internal(current_time_step, epsilon, lr,
                   input_spikes, input_classes, class_count, 
                   hidden_count, hidden_sizes, y_sample, 
                   output_weights, hidden_weights, output_spikes, hidden_spikes)

    output_layer.weights = output_weights.T
    for i, hidden_layer in enumerate(hidden_layers):
        hidden_layer.weights = hidden_weights[i].T

def train_internal(current_time_step, epsilon, lr,
                   input_spikes, input_classes, class_count, 
                   hidden_count, hidden_sizes, correct_class, 
                   output_weights, hidden_weights, output_spikes, hidden_spikes):
    
    delta = np.zeros([class_count, 1])

    correct_spike_count = sum(output_spikes[correct_class, current_time_step - epsilon + 1 : current_time_step + 1])
    if (correct_spike_count < 1):
        delta[correct_class, 0] = 1
    for output_class in range(class_count):
        if (output_class != correct_class):
            incorrect_spike_count = sum(output_spikes[output_class, current_time_step - epsilon + 1 : current_time_step + 1])
            if (incorrect_spike_count >= 1):
                delta[output_class, 0] =- 1

    delta_h = [1] * hidden_count
    delta_h[hidden_count - 1] = np.dot(output_weights, delta)
    output_weights += np.dot((sum(hidden_spikes[hidden_count - 1][:, current_time_step - epsilon + 1 : current_time_step + 1].T).T).reshape([hidden_sizes[hidden_count - 1], 1]), delta.T) * lr
    
    for der in range(hidden_count - 1, 0, -1):
        derivative = (sum(hidden_spikes[der][:, current_time_step - epsilon + 1 : current_time_step + 1].T).T) > 0
        delta_h[der] = delta_h[der] * derivative.reshape([hidden_sizes[der], 1])
        delta_h[der-1] = np.dot(hidden_weights[der], delta_h[der])                
        hidden_weights[der] += np.dot((sum(hidden_spikes[der - 1][:, current_time_step - epsilon + 1 : current_time_step + 1].T).T).reshape([hidden_sizes[der - 1], 1]), delta_h[der].T) * lr
        
    derivative = (sum(hidden_spikes[0][:, current_time_step - epsilon + 1 : current_time_step + 1].T).T) > 0
    delta_h[0] = delta_h[0] * derivative.reshape([hidden_sizes[0], 1])
    hidden_weights[0] += np.dot((sum(input_spikes[:, current_time_step - epsilon + 1 : current_time_step + 1].T).T).reshape([input_classes, 1]), delta_h[0].T) * lr

    return output_weights, hidden_weights