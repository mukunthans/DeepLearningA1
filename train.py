import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist, mnist
import wandb

parameters_dict = {
    "wandb_project": "SWEEP_PROJECT01",
    "wandb_entity": "your_entity_name",
    "dataset": "mnist",
    "epochs": 5,
    "batch_size": 64,
    "loss": "cross_entropy",
    "optimizer": "adam",
    "learning_rate": 0.001,
    "momentum": 0.9,
    "beta": 0.9,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "weight_decay": 0,
    "weight_init": "xavier",
    "hidden_layers": 3,
    "neurons": 64
}
parameters_dict["activation"] = "sigmoid"

parser = argparse.ArgumentParser()
parser.add_argument("--wandb_project", type=str, default=default_params["wandb_project"], help="WandB project name")
parser.add_argument("--wandb_entity", type=str, default=default_params["wandb_entity"], help="WandB entity name")
parser.add_argument("--dataset", type=str, default=default_params["dataset"], help="Dataset to use: mnist or fashion_mnist")
parser.add_argument("--epochs", type=int, default=default_params["epochs"], help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=default_params["batch_size"], help="Batch size")
parser.add_argument("--loss", type=str, default=default_params["loss"], help="Loss function: cross_entropy or mean_squared_error")
parser.add_argument("--optimizer", type=str, default=default_params["optimizer"], help="Optimizer: sgd, momentum, nag, rmsprop, adam, or nadam")
parser.add_argument("--learning_rate", type=float, default=default_params["learning_rate"], help="Learning rate")
parser.add_argument("--momentum", type=float, default=default_params["momentum"], help="Momentum")
parser.add_argument("--beta", type=float, default=default_params["beta"], help="Beta (for RMSProp)")
parser.add_argument("--beta1", type=float, default=default_params["beta1"], help="Beta1 (for Adam/Nadam)")
parser.add_argument("--beta2", type=float, default=default_params["beta2"], help="Beta2 (for Adam/Nadam)")
parser.add_argument("--epsilon", type=float, default=default_params["epsilon"], help="Epsilon (for Adam/Nadam)")
parser.add_argument("--weight_decay", type=float, default=default_params["weight_decay"], help="Weight decay")
parser.add_argument("--weight_init", type=str, default=default_params["weight_init"], help="Weight initialization: random or xavier")
parser.add_argument("--hidden_layers", type=int, default=default_params["hidden_layers"], help="Number of hidden layers")
parser.add_argument("--neurons", type=int, default=default_params["neurons"], help="Number of neurons per hidden layer")
parser.add_argument("--activation", type=str, default=default_params["activation"], help="Activation function: sigmoid, tanh, or relu")
parser.add_argument("--sweep", action="store_true", help="Run W&B sweep agent")
args = parser.parse_args()
parameters_dict = vars(args)
parameters_dict["input_size"] = 784
parameters_dict["output_size"] = 10
parameters_dict["output_activation"] = "softmax"
parameters_dict["hidden_size"] = parameters_dict["neurons"]
parameters_dict["num_layers"] = parameters_dict["hidden_layers"]
INPUT_SIZE = parameters_dict["input_size"]
OUTPUT_SIZE = parameters_dict["output_size"]

class FFNeuralNetwork:
    def __init__(self, hidden_units, num_hidden_layers, input_dim, output_dim, weight_init="random", output_activation="softmax", initialize=True):
        self.hidden_units = hidden_units
        self.num_hidden_layers = num_hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_init_type = weight_init
        self.output_activation_type = output_activation
        if initialize:
            self.initialize_parameters()
    def initialize_parameters(self):
        layer_dims = [self.input_dim] + [self.hidden_units] * self.num_hidden_layers + [self.output_dim]
        self.weights = [np.random.randn(layer_dims[i], layer_dims[i+1]) for i in range(len(layer_dims) - 1)]
        self.biases = [np.zeros(layer_dims[i+1]) for i in range(len(layer_dims) - 1)]
        if self.weight_init_type.lower() == "xavier":
            self.weights = [w * np.sqrt(1 / w.shape[0]) for w in self.weights]
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    def forward(self, x):
        self.pre_activation, self.post_activation = [x], [x]
        for i in range(self.num_hidden_layers):
            x = self.activation(np.dot(x, self.weights[i]) + self.biases[i])
            self.pre_activation.append(x)
            self.post_activation.append(x)
        z = np.dot(x, self.weights[-1]) + self.biases[-1]
        self.pre_activation.append(z)
        a_out = self.softmax(z)
        self.post_activation.append(a_out)
        return a_out
    def activation(self, x):
        if parameters_dict["activation"] == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif parameters_dict["activation"] == "tanh":
            return np.tanh(x)
        elif parameters_dict["activation"] == "relu":
            return np.maximum(0, x)
        else:
            return x

class ActivationFunctions:
    functions = {"identity": lambda x: x, "sigmoid": lambda x: 1 / (1 + np.exp(-x)), "tanh": np.tanh, "relu": lambda x: np.maximum(0, x)}
    derivatives = {"identity": lambda a: np.ones_like(a), "sigmoid": lambda a: a * (1 - a), "tanh": lambda a: 1 - a ** 2, "relu": lambda a: (a > 0).astype(int)}

def get_activation_derivative(a, activation_name):
    return ActivationFunctions.derivatives[activation_name](a)

class Optimizer:
    def __init__(self, nn, lr, optimizer, momentum, epsilon, beta, beta1, beta2, t, decay):
        self.nn = nn
        self.lr = lr
        self.optimizer = optimizer
        self.momentum = momentum
        self.epsilon = epsilon
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = t
        self.decay = decay
        self.v_w = [np.zeros_like(w) for w in nn.weights]
        self.v_b = [np.zeros_like(b) for b in nn.biases]
        self.s_w = [np.zeros_like(w) for w in nn.weights]
        self.s_b = [np.zeros_like(b) for b in nn.biases]
    def SGD(self, d_weights, d_biases):
        for i in range(len(self.nn.weights)):
            self.nn.weights[i] -= self.lr * (d_weights[i] + self.decay * self.nn.weights[i])
            self.nn.biases[i] -= self.lr * (d_biases[i] + self.decay * self.nn.biases[i])
    def MomentumGD(self, d_weights, d_biases):
        for i in range(len(self.nn.weights)):
            self.v_w[i] = self.momentum * self.v_w[i] + d_weights[i]
            self.v_b[i] = self.momentum * self.v_b[i] + d_biases[i]
            self.nn.weights[i] -= self.lr * (self.v_w[i] + self.decay * self.nn.weights[i])
            self.nn.biases[i] -= self.lr * (self.v_b[i] + self.decay * self.nn.biases[i])
    def NAG(self, d_weights, d_biases):
        for i in range(len(self.nn.weights)):
            self.v_w[i] = self.momentum * self.v_w[i] + d_weights[i]
            self.v_b[i] = self.momentum * self.v_b[i] + d_biases[i]
            self.nn.weights[i] -= self.lr * (self.momentum * self.v_w[i] + d_weights[i] + self.decay * self.nn.weights[i])
            self.nn.biases[i] -= self.lr * (self.momentum * self.v_b[i] + d_biases[i] + self.decay * self.nn.biases[i])
    def RMSProp(self, d_weights, d_biases):
        for i in range(len(self.nn.weights)):
            self.s_w[i] = self.beta * self.s_w[i] + (1 - self.beta) * (d_weights[i] ** 2)
            self.s_b[i] = self.beta * self.s_b[i] + (1 - self.beta) * (d_biases[i] ** 2)
            self.nn.weights[i] -= self.lr * (d_weights[i] / (np.sqrt(self.s_w[i]) + self.epsilon) + self.decay * self.nn.weights[i])
            self.nn.biases[i] -= self.lr * (d_biases[i] / (np.sqrt(self.s_b[i]) + self.epsilon) + self.decay * self.nn.biases[i])
    def Adam(self, d_weights, d_biases):
        self.t += 1
        for i in range(len(self.nn.weights)):
            self.v_w[i] = self.beta1 * self.v_w[i] + (1 - self.beta1) * d_weights[i]
            self.v_b[i] = self.beta1 * self.v_b[i] + (1 - self.beta1) * d_biases[i]
            self.s_w[i] = self.beta2 * self.s_w[i] + (1 - self.beta2) * (d_weights[i] ** 2)
            self.s_b[i] = self.beta2 * self.s_b[i] + (1 - self.beta2) * (d_biases[i] ** 2)
            v_w_hat = self.v_w[i] / (1 - self.beta1 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta1 ** self.t)
            s_w_hat = self.s_w[i] / (1 - self.beta2 ** self.t)
            s_b_hat = self.s_b[i] / (1 - self.beta2 ** self.t)
            self.nn.weights[i] -= self.lr * (v_w_hat / (np.sqrt(s_w_hat) + self.epsilon) + self.decay * self.nn.weights[i])
            self.nn.biases[i] -= self.lr * (v_b_hat / (np.sqrt(s_b_hat) + self.epsilon) + self.decay * self.nn.biases[i])
    def optimize(self, d_weights, d_biases):
        opt_key = self.optimizer
        if opt_key == "nesterov":
            opt_key = "nag"
        opts = {"sgd": self.SGD, "momentum": self.MomentumGD, "nag": self.NAG, "rmsprop": self.RMSProp, "adam": self.Adam}
        opts[opt_key](d_weights, d_biases)

class Backpropagation:
    def __init__(self, nn, loss="cross_entropy", act_func="sigmoid"):
        self.nn = nn
        self.loss = loss
        self.activation_function = act_func
    def output_activation_derivative(self, y_pred):
        return np.diag(y_pred) - np.outer(y_pred, y_pred) if self.nn.output_activation_type == "softmax" else None
    def activation_derivative(self, a):
        return get_activation_derivative(a, self.activation_function)
    def backward(self, y, y_pred):
        d_weights, d_biases, d_h, d_a = [], [], [], []
        d_h.append(y_pred - y)
        d_a.append(np.array([np.matmul((y_pred[i] - y[i]), self.output_activation_derivative(y_pred[i])) for i in range(y_pred.shape[0])]))
        for i in range(self.nn.num_hidden_layers, 0, -1):
            d_weights.append(np.matmul(self.nn.post_activation[i].T, d_a[-1]))
            d_biases.append(np.sum(d_a[-1], axis=0))
            d_h.append(np.matmul(d_a[-1], self.nn.weights[i].T))
            d_a.append(d_h[-1] * self.activation_derivative(self.nn.post_activation[i]))
        d_weights.append(np.matmul(self.nn.post_activation[0].T, d_a[-1]))
        d_biases.append(np.sum(d_a[-1], axis=0))
        d_weights.reverse()
        d_biases.reverse()
        for i in range(len(d_weights)):
            d_weights[i] /= y.shape[0]
            d_biases[i] /= y.shape[0]
        return d_weights, d_biases

def loss(loss_type, y, y_pred):
    if loss_type == "cross_entropy":
        return -np.sum(y * np.log(y_pred + 1e-8))
    elif loss_type == "mean_squared_error":
        return np.sum((y - y_pred)**2) / 2
    else:
        raise Exception("Invalid loss function")

def load_data(type, dataset='mnist'):
    if dataset == 'mnist':
        (x, y), (x_test, y_test) = mnist.load_data()
    elif dataset == 'fashion_mnist':
        (x, y), (x_test, y_test) = fashion_mnist.load_data()
    if type == 'train':
        x = x.reshape(x.shape[0], 784) / 255.0
        y = np.eye(10)[y]
        return x, y
    elif type == 'test':
        x_test = x_test.reshape(x_test.shape[0], 784) / 255.0
        y_test = np.eye(10)[y_test]
        return x_test, y_test

run = wandb.init(project=parameters_dict["wandb_project"], entity=parameters_dict["wandb_entity"], config=parameters_dict)
run.name = f"opt_{parameters_dict['optimizer']}_lr_{parameters_dict['learning_rate']}_act_{parameters_dict['activation']}_hid_{parameters_dict['num_layers']}_nrns_{parameters_dict['hidden_size']}_randn_" + str(np.random.randint(1000))
nn = FFNeuralNetwork(neurons=parameters_dict["hidden_size"], num_hidden_layers=parameters_dict["num_layers"], input_dim=INPUT_SIZE, output_dim=OUTPUT_SIZE, act_func=parameters_dict["activation"], weight_init=parameters_dict["weight_init"], out_act_func="softmax", init_toggle=True)
bp = Backpropagation(nn=nn, loss=parameters_dict["loss"], act_func=parameters_dict["activation"])
opt = Optimizer(nn=nn, bp=bp, lr=parameters_dict["learning_rate"], optimizer=parameters_dict["optimizer"], momentum=parameters_dict["momentum"], beta=parameters_dict["beta"], beta1=parameters_dict["beta1"], beta2=parameters_dict["beta2"], epsilon=parameters_dict["epsilon"], t=0, decay=parameters_dict["weight_decay"])
x_train, y_train = load_data(type="train", dataset=parameters_dict["dataset"])
x_test, y_test = load_data(type="test", dataset=parameters_dict["dataset"])
x_train_act, x_val, y_train_act, y_val = train_test_split(x_train, y_train, test_size=0.1)
for epoch in range(parameters_dict["epochs"]):
    for i in range(0, x_train_act.shape[0], parameters_dict["batch_size"]):
        x_batch = x_train_act[i:i+parameters_dict["batch_size"]]
        y_batch = y_train_act[i:i+parameters_dict["batch_size"]]
        y_pred = nn.forward(x_batch)
        d_weights, d_biases = bp.backward(y_batch, y_pred)
        opt.run(d_weights, d_biases)
    opt.t += 1
    y_pred = nn.forward(x_train_act)
    train_loss = loss(parameters_dict["loss"], y_train_act, y_pred)
    train_accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_train_act, axis=1)) / y_train_act.shape[0]
    val_loss = loss(parameters_dict["loss"], y_val, nn.forward(x_val))
    val_accuracy = np.sum(np.argmax(nn.forward(x_val), axis=1) == np.argmax(y_val, axis=1)) / y_val.shape[0]
    print("Epoch: {}".format(epoch+1))
    print("Train Accuracy: {}".format(train_accuracy))
    print("Validation Accuracy: {}".format(val_accuracy))
    wandb.log({"epoch": epoch+1, "train_loss": train_loss, "train_accuracy": train_accuracy, "val_loss": val_loss, "val_accuracy": val_accuracy})
y_pred_test = nn.forward(x_test)
test_loss = loss(parameters_dict["loss"], y_test, y_pred_test)
test_accuracy = np.sum(np.argmax(y_pred_test, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
print("Test Accuracy: {}".format(test_accuracy))
wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy})
wandb.finish()
