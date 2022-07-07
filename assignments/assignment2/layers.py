import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W
    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''

    predictions_copy = predictions.copy()
    max = np.max(predictions_copy, axis=predictions_copy.ndim - 1)
    max = max if max.ndim == 0 else max.reshape((predictions_copy.shape[0], 1))
    predictions_copy -= max
    if predictions.ndim > 1:
        probs = (np.exp(predictions_copy) / np.exp(predictions_copy).sum(axis=1).reshape((predictions_copy.shape[0], -1)))
        return probs
    probs = np.exp(predictions_copy) / np.exp(predictions_copy).sum()
    return probs


def cross_entropy_loss(probs, target_index):
    """
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    """
    if probs.shape == (len(probs),):
        loss = - np.log(probs[target_index])
    else:
        loss = - np.log(probs[np.arange(len(probs)), target_index]).mean()

    return loss


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    probs = softmax(preds)
    loss = cross_entropy_loss(probs, target_index)
    mask = np.zeros_like(preds)
    if preds.shape == (len(preds),):
        mask[target_index] = 1
        dprediction = - (mask - probs)

    else:
        mask[np.arange(len(mask)), target_index] = 1
        dprediction = - (mask - probs) / (len(mask))
    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.derivative = 0

    def reset_grad(self):
        pass

    def forward(self, X):
        self.derivative = X > 0

        return X * self.derivative

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        d_result = d_out * self.derivative
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def reset_grad(self):
        self.W.grad = np.zeros_like(self.W.value)
        self.B.grad = np.zeros_like(self.B.value)

    def forward(self, X):
        self.X = X.copy()
        return X.dot(self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        self.W.grad += self.X.T.dot(d_out)
        self.B.grad += np.ones((1, self.X.shape[0])).dot(d_out)
        d_input = d_out.dot(self.W.value.transpose())

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
