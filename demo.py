import gradio as gr
import io
import sys
from stdp_snn import Classifier
from sklearn import datasets
import numpy as np
from tensorflow.keras.datasets import mnist

clf_instance = None

X_train = None
X_test = None
y_train = None
y_test = None

PRESETS = {
    ("lif", "stdp", "iris"): {
        "hidden_sizes": "10", "time_steps": 100, "time_per_step": 1, "tau": 4,
        "threshold": 0.5, "output_threshold": 0.5, "rest_potential": 0.0,
        "A_p": 0.16, "A_m": 0.16 * 0.9, "stdp_tau": 1, "stdp_window": 10,
        "num_epochs": 10
    },
    ("lif", "sstdp", "iris"): {
        "hidden_sizes": "10", "time_steps": 100, "time_per_step": 1, "tau": 8,
        "threshold": 0.3, "output_threshold": 0.4, "rest_potential": 0.0,
        "A_p": 0.16, "A_m": 0.16 * 0.9, "stdp_tau": 0.8, "stdp_window": 5,
        "num_epochs": 10
    },
    ("lif", "bpstdp", "iris"): {
        "hidden_sizes": "12,6", "time_steps": 100, "time_per_step": 1, "tau": 4,
        "threshold": 1, "output_threshold": 2, "rest_potential": 0.0,
        "bp_epsilon": 8, "bp_lr": 0.005,
        "num_epochs": 10
    },
    ("if", "bpstdp", "iris"): {
        "hidden_sizes": "12,6", "time_steps": 100, "time_per_step": 1, "tau": 4,
        "threshold": 1, "output_threshold": 2, "rest_potential": 0.0,
        "bp_epsilon": 8, "bp_lr": 0.005,
        "num_epochs": 10
    },
    ("izh", "bpstdp", "iris"): {
        "hidden_sizes": "12,6", "time_steps": 100, "time_per_step": 1, "tau": 4,
        "threshold": 30, "output_threshold": 30, "rest_potential": 0.0,
        "bp_epsilon": 8, "bp_lr": 0.001,
        "num_epochs": 10
    },
    ("lif", "bpstdp", "iris"): {
        "hidden_sizes": "750,150", "time_steps": 50, "time_per_step": 1, "tau": 4,
        "threshold": 10, "output_threshold": 20, "rest_potential": 0.0,
        "bp_epsilon": 2, "bp_lr": 0.0005,
        "num_epochs": 1
    },
    ("if", "bpstdp", "iris"): {
        "hidden_sizes": "750,150", "time_steps": 50, "time_per_step": 1, "tau": 4,
        "threshold": 10, "output_threshold": 20, "rest_potential": 0.0,
        "bp_epsilon": 2, "bp_lr": 0.0005,
        "num_epochs": 1
    },
    ("izh", "bpstdp", "iris"): {
        "hidden_sizes": "750,150", "time_steps": 50, "time_per_step": 1, "tau": 4,
        "threshold": 30, "output_threshold": 30, "rest_potential": 0.0,
        "bp_epsilon": 2, "bp_lr": 0.00001,
        "num_epochs": 1
    },
}

def load_dataset(dataset_type):
    global X_train
    global X_test
    global y_train
    global y_test

    if (dataset_type == "iris"):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

        X_train = np.concatenate((X[0:30], X[50:80], X[100:130]))
        X_test = np.concatenate((X[30:50], X[80:100], X[130:150]))
        y_train = np.concatenate((y[0:30], y[50:80], y[100:130]))
        y_test = np.concatenate((y[30:50], y[80:100], y[130:150]))
    elif (dataset_type == "mnist"):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        X_train = X_train / 255.0
        X_test = X_test / 255.0

        X_train = X_train.reshape(-1, 784)
        X_test = X_test.reshape(-1, 784)

def capture_output(func, *args, **kwargs):
    buffer = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = buffer
    try:
        result = func(*args, **kwargs)
    finally:
        sys.stdout = sys_stdout
    return buffer.getvalue(), result

def handle_all_updates(neuron_type, stdp_type, dataset):
    key = (neuron_type, stdp_type, dataset)
    preset = PRESETS.get(key, {})

    if stdp_type in ["stdp", "sstdp"]:
        vis = [gr.update(visible=True)] * 4 + [gr.update(visible=False)] * 2
    elif stdp_type == "bpstdp":
        vis = [gr.update(visible=False)] * 4 + [gr.update(visible=True)] * 2
    else:
        vis = [gr.update(visible=False)] * 6

    return [
        gr.update(value=preset.get("hidden_sizes", "")),
        gr.update(value=preset.get("num_epochs", 10)),
        gr.update(value=preset.get("time_steps", 0)),
        gr.update(value=preset.get("time_per_step", 0.0)),
        gr.update(value=preset.get("tau", 0.0)),
        gr.update(value=preset.get("threshold", 0.0)),
        gr.update(value=preset.get("output_threshold", 0.0)),
        gr.update(value=preset.get("rest_potential", 0.0)),
        gr.update(value=preset.get("A_p", None)),
        gr.update(value=preset.get("A_m", None)),
        gr.update(value=preset.get("stdp_tau", None)),
        gr.update(value=preset.get("stdp_window", None)),
        gr.update(value=preset.get("bp_epsilon", None)),
        gr.update(value=preset.get("bp_lr", None)),
    ] + vis

def handle_train(neuron_type, stdp_type, hidden_sizes, num_epochs,
                 time_steps, time_per_step, tau, threshold, output_threshold,
                 rest_potential, dataset, test_values,
                 A_p, A_m, stdp_tau, stdp_window,
                 bp_epsilon, bp_lr):
    global clf_instance

    dataset_io = {
        "mnist": (784, 10),
        "iris": (4, 3)
    }
    input_classes, output_clases = dataset_io[dataset]
    load_dataset(dataset)

    params = dict(
        neuron_type=neuron_type,
        stdp_type=stdp_type,
        input_classes=input_classes,
        hidden_sizes=[int(x) for x in hidden_sizes.split(",") if x.strip()],
        output_clases=output_clases,
        time_steps=int(time_steps),
        time_per_step=float(time_per_step),
        tau=float(tau),
        threshold=float(threshold),
        output_threshold=float(output_threshold),
        rest_potential=float(rest_potential),
    )

    if stdp_type in ["stdp", "sstdp"]:
        params.update(dict(A_p=float(A_p), A_m=float(A_m), stdp_tau=float(stdp_tau), stdp_window=float(stdp_window)))
    elif stdp_type == "bpstdp":
        params.update(dict(bp_epsilon=int(bp_epsilon), bp_lr=float(bp_lr)))
        

    clf_instance = Classifier(**params)
    log_output, _ = capture_output(clf_instance.train, X_train, y_train, int(num_epochs))
    return "Обучение завершено.", log_output

def handle_test(test_values):
    if clf_instance is None:
        return "Сначала обучите модель", ""
    values = [float(x) for x in test_values.split(",") if x.strip()]
    log_output, result = capture_output(clf_instance.test_sample, values)

    images = clf_instance.plot_potential_history(clf_instance.output_layer, True)

    return result, log_output, images

with gr.Blocks() as demo:
    with gr.Row():
        neuron_type = gr.Dropdown(["lif", "if", "izh"], label="Тип нейрона", value="lif")
        stdp_type = gr.Dropdown(["stdp", "sstdp", "bpstdp"], label="Тип STDP", value="stdp")
        dataset = gr.Dropdown(["iris", "mnist"], label="Датасет", value="iris")

    with gr.Row():
        num_epochs = gr.Number(label="Эпохи", precision=0)
        hidden_sizes = gr.Textbox(label="Скрытые размеры")
        time_steps = gr.Number(label="Шаги", precision=0)
        time_per_step = gr.Number(label="Время на шаг", precision=4)
        tau = gr.Number(label="Tau", precision=4)
        threshold = gr.Number(label="Порог", precision=4)
        output_threshold = gr.Number(label="Порог выхода", precision=4)
        rest_potential = gr.Number(label="Потенциал покоя", precision=4)

    with gr.Row():
        A_p = gr.Number(label="A+", visible=True)
        A_m = gr.Number(label="A-", visible=True)
        stdp_tau = gr.Number(label="STDP Tau", visible=True)
        stdp_window = gr.Number(label="STDP Window", visible=True)

        bp_epsilon = gr.Number(label="BP epsilon", visible=False, precision=0)
        bp_lr = gr.Number(label="BP learning rate", visible=False)

    test_values = gr.Textbox(label="Значения для теста (через запятую)")

    
    stdp_type.change(fn=handle_all_updates,
        inputs=[neuron_type, stdp_type, dataset],
        outputs=[
            hidden_sizes, num_epochs, time_steps, time_per_step, tau, threshold,
            output_threshold, rest_potential, A_p, A_m, stdp_tau, stdp_window,
            bp_epsilon, bp_lr,
            A_p, A_m, stdp_tau, stdp_window, bp_epsilon, bp_lr
        ])

    neuron_type.change(fn=handle_all_updates,
        inputs=[neuron_type, stdp_type, dataset],
        outputs=[
            hidden_sizes, num_epochs, time_steps, time_per_step, tau, threshold,
            output_threshold, rest_potential, A_p, A_m, stdp_tau, stdp_window,
            bp_epsilon, bp_lr,
            A_p, A_m, stdp_tau, stdp_window, bp_epsilon, bp_lr
        ])

    dataset.change(fn=handle_all_updates,
        inputs=[neuron_type, stdp_type, dataset],
        outputs=[
            hidden_sizes, num_epochs, time_steps, time_per_step, tau, threshold,
            output_threshold, rest_potential, A_p, A_m, stdp_tau, stdp_window,
            bp_epsilon, bp_lr,
            A_p, A_m, stdp_tau, stdp_window, bp_epsilon, bp_lr
        ])

    with gr.Row():
        train_btn = gr.Button("Обучение")
        test_btn = gr.Button("Тест")

    output_text = gr.Textbox(label="Результат", interactive=False)
    log_textbox = gr.Textbox(label="Лог", interactive=False)
    gallery = gr.Gallery(label="Графики", show_label=True, columns=3, height=250)

    train_btn.click(handle_train,
        inputs=[neuron_type, stdp_type, hidden_sizes, num_epochs,
                time_steps, time_per_step, tau, threshold, output_threshold,
                rest_potential, dataset, test_values,
                A_p, A_m, stdp_tau, stdp_window,
                bp_epsilon, bp_lr],
        outputs=[output_text, log_textbox]
    )

    test_btn.click(handle_test, inputs=[test_values], outputs=[output_text, log_textbox, gallery])

demo.launch(debug=True)
