from builtins import range
import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength (lambda essentially)

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]

     #############################################################################
        # TODO:                                                                     #
        # Compute the gradient of the loss function and store it dW.                #
        # Rather that first computing the loss and then computing the derivative,   #
        # it may be simpler to compute the derivative at the same time that the     #
        # loss is being computed. As a result you may need to modify some of the    #
        # code above to compute the gradient.                                       #
        #############################################################################

    for i in range(num_train):
        # (3073,) dot (3073, 10) -> (10,)
        # raw scores for all 10 classes for this specific training example
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        # prevent exploding of W*x so once we subtract max largest shifted
        # score is 0 and other terms = exp(negative)
        scores -= np.max(scores)

        # exponentiate all 10 values and divide by their sum
        # if some class had a high score, exponentiating will make it bigger
        # so after normalizing, it will dominate probability
        p = np.exp(scores)
        p /= p.sum()  # normalize

        logp = np.log(p)

        '''
        loss for one image: -log(p_correct_class)

        want 0 when model is perfect (p_correct=1) and big when model bad (p_correct ~ 0)
        sum of all per image loss at the end averaged later on

        lets say C = 3 so we mite have scores = [s_0, s_1, s_2]
        then p_0 = e^{s_0} / (e^{s_0} + e^{s_1} + e^{s_2})

        lets say class 0 is correct so y would be [1, 0, 0] (one hot encoded)
        loss = -log(p_0) which ends up being p_0 - 1 (i.e p - y and this carries for other classes also)
        '''

        # model outputs p_y and we want it close to 1 for correct class
        # "badness" number thats 0 when p_y = 1 (perfect), grows as p_y -> 0
        loss -= logp[y[i]]  # negative log probability is the loss: -log(p correct class)

        # computing dL/dw for this specific image

        # ONE HOT ENCODING HERE IS IMPORTANT AND PREVIOUSLY JUST y[i] was being subtracted
        # which would've caused nudges to be incorrect
        Y = np.zeros(num_classes)
        Y[y[i]] = 1.0

        dl_dscores = p - Y
        dscores_dw = X[i]

        dW += dscores_dw[:, np.newaxis].dot(dl_dscores[np.newaxis, :])

    # normalized hinge loss plus regularization
    # fit the data but balance it with keeping weights small
    loss = (loss / num_train) + (reg * np.sum(W * W))

    # derivative of (lambda * W^2 -> 2 * lambda * W)
    dW = (dW / num_train) + (2 * reg * W)

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################


    return loss, dW
