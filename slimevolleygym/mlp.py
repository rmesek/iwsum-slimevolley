"""
Simple MLP policy for loading pre-trained estool models (saved in /zoo).

Code based on https://github.com/hardmaru/estool
"""

import json
from collections import namedtuple

import numpy as np

# ── Activation helpers ───────────────────────────────────────────────────────


def relu(x):
    return np.maximum(x, 0.0)


def passthru(x):
    return x


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    e = np.exp(x - np.max(x))  # subtract max for numerical stability
    return e / e.sum()


def sample(p):
    """Draw one index from a categorical distribution given probabilities p."""
    return np.random.choice(len(p), p=p)


# ── Game spec ────────────────────────────────────────────────────────────────

Game = namedtuple(
    "Game",
    [
        "env_name",
        "time_factor",
        "input_size",
        "output_size",
        "layers",
        "activation",
        "noise_bias",
        "output_noise",
        "rnn_mode",
    ],
)

games = {}

games["slimevolley"] = Game(
    env_name="SlimeVolley",
    input_size=12,
    output_size=3,
    time_factor=0,
    layers=[20, 20],
    activation="tanh",
    noise_bias=0.0,
    output_noise=[False, False, False],
    rnn_mode=False,
)

games["slimevolleylite"] = Game(
    env_name="SlimeVolley",
    input_size=12,
    output_size=3,
    time_factor=0,
    layers=[10, 10],
    activation="tanh",
    noise_bias=0.0,
    output_noise=[False, False, False],
    rnn_mode=False,
)


def makeSlimePolicy(filename):
    model = Model(games["slimevolley"])
    model.load_model(filename)
    return model


def makeSlimePolicyLite(filename):
    model = Model(games["slimevolleylite"])
    model.load_model(filename)
    return model


# ── Model ────────────────────────────────────────────────────────────────────


class Model:
    """Simple feedforward MLP, compatible with weights saved by estool."""

    def __init__(self, game):
        self.output_noise = game.output_noise
        self.env_name = game.env_name
        self.layer_1 = game.layers[0]
        self.layer_2 = game.layers[1]
        self.rnn_mode = False
        self.time_input = 0
        self.sigma_bias = game.noise_bias
        self.sigma_factor = 0.5
        if game.time_factor > 0:
            self.time_factor = float(game.time_factor)
            self.time_input = 1
        self.input_size = game.input_size
        self.output_size = game.output_size
        if self.layer_2 > 0:
            self.shapes = [
                (self.input_size + self.time_input, self.layer_1),
                (self.layer_1, self.layer_2),
                (self.layer_2, self.output_size),
            ]
        elif self.layer_2 == 0:
            self.shapes = [
                (self.input_size + self.time_input, self.layer_1),
                (self.layer_1, self.output_size),
            ]
        else:
            raise ValueError("layer_2 must be >= 0")

        self.sample_output = False
        if game.activation == "relu":
            self.activations = [relu, relu, passthru]
        elif game.activation == "sigmoid":
            self.activations = [np.tanh, np.tanh, sigmoid]
        elif game.activation == "softmax":
            self.activations = [np.tanh, np.tanh, softmax]
            self.sample_output = True
        elif game.activation == "passthru":
            self.activations = [np.tanh, np.tanh, passthru]
        else:
            self.activations = [np.tanh, np.tanh, np.tanh]

        self.weight = []
        self.bias = []
        self.bias_log_std = []
        self.bias_std = []
        self.param_count = 0

        for idx, shape in enumerate(self.shapes):
            self.weight.append(np.zeros(shape=shape))
            self.bias.append(np.zeros(shape=shape[1]))
            self.param_count += np.prod(shape) + shape[1]
            if self.output_noise[idx]:
                self.param_count += shape[1]
            log_std = np.zeros(shape=shape[1])
            self.bias_log_std.append(log_std)
            self.bias_std.append(np.exp(self.sigma_factor * log_std + self.sigma_bias))

        self.render_mode = False

    def predict(self, x, t=0, mean_mode=False):
        h = np.array(x).flatten()
        if self.time_input == 1:
            h = np.concatenate([h, [float(t) / self.time_factor]])
        for i, (w, b) in enumerate(zip(self.weight, self.bias)):
            h = np.matmul(h, w) + b
            if self.output_noise[i] and not mean_mode:
                h += np.random.randn(self.shapes[i][1]) * self.bias_std[i]
            h = self.activations[i](h)
        if self.sample_output:
            h = sample(h)
        return h

    def set_model_params(self, model_params):
        pointer = 0
        for i, shape in enumerate(self.shapes):
            s_w = np.prod(shape)
            s_b = shape[1]
            chunk = np.array(model_params[pointer : pointer + s_w + s_b])
            self.weight[i] = chunk[:s_w].reshape(shape)
            self.bias[i] = chunk[s_w:]
            pointer += s_w + s_b
            if self.output_noise[i]:
                self.bias_log_std[i] = np.array(model_params[pointer : pointer + s_b])
                self.bias_std[i] = np.exp(
                    self.sigma_factor * self.bias_log_std[i] + self.sigma_bias
                )
                pointer += s_b

    def load_model(self, filename):
        with open(filename) as f:
            data = json.load(f)
        print(f"loading file {filename}")
        self.set_model_params(np.array(data[0]))

    def get_random_model_params(self, stdev=0.1):
        return np.random.randn(self.param_count) * stdev
