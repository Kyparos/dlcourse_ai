import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.layers = [FullyConnectedLayer(n_input, hidden_layer_size),
                       ReLULayer(),
                       FullyConnectedLayer(hidden_layer_size, n_output)]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        for layer in self.layers:
            layer.reset_grad()

        X = X.copy()
        for layer in self.layers:
            X = layer.forward(X)
        loss, dpred = softmax_with_cross_entropy(X, y)
        d_out = dpred.copy()
        for layer in self.layers[::-1]:
            d_out = layer.backward(d_out)
        if self.reg:
            l2_loss = 0
            for (layer_key, param_name), param in self.params().items():
                if param_name == 'W':
                    l2_step_loss, d_l2 = l2_regularization(param.value, self.reg)
                    l2_loss += l2_step_loss
                    param.grad += d_l2
            loss += l2_loss
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        X = X.copy()
        pred = np.zeros(X.shape[0], np.int)
        for layer in self.layers:
            X = layer.forward(X)
        return X.argmax(axis=1)

    def params(self):
        result = {}
        for num_layer, layer in enumerate(self.layers):
            for key, value in layer.params().items():
                result[(num_layer, key)] = value

        return result
