import numpy as np


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
        probs = (np.exp(predictions_copy) / np.exp(predictions_copy).sum(axis=1).reshape(
            (predictions_copy.shape[0], -1)))
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


def softmax_with_cross_entropy(predictions, target_index):
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
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    mask = np.zeros_like(predictions)
    if predictions.shape == (len(predictions),):
        mask[target_index] = 1
        dprediction = - (mask - probs)

    else:
        mask[np.arange(len(mask)), target_index] = 1
        dprediction = - (mask - probs) / (len(mask))
    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W
    return loss, grad


def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)
    loss, dpred = softmax_with_cross_entropy(predictions, target_index)
    dW = np.dot(X.T, dpred)
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        W = self.W.copy()
        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            batch_number = np.random.randint(len(batches_indices))
            loss_pred, dpred = linear_softmax(X[batches_indices[batch_number], :], W, y[batches_indices[batch_number]])
            loss_reg, dreg = l2_regularization(W, reg)
            loss = loss_reg + loss_pred
            W -= learning_rate * (dreg + dpred)
            print("Epoch %i, loss: %f" % (epoch, loss))
            loss_history.append(loss)
        self.W = W
        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.dot(X, self.W).argmax(axis=1)

        return y_pred
