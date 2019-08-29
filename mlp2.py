"""TODO."""

import abc

import torch
import numpy as np


class Module(object):
    """An abstract module representing a Neural Network layer.

    Attributes:
        params: A list of parameters that the module uses to compute the
            forward and backward passes. Each subclassing module must add its
            parameters to self.params.
        gradients: A list of the same size as `params` which contains the
            gradients for the parameters. The gradients have the same ordering
            as the params, e.g. gradients[i] contains the gradient for
            params[i]. The gradients attribute is only set once `backward` has
            been called and is unset when `zero_grad` is called.
    """

    __meta__ = abc.ABCMeta

    def __init__(self):
        self.params = None
        self.gradients = None
        self._inputs = None

    def __call__(self, inputs):
        """Computes the forward pass.

        Must be overridden by the subclass. Make sure to call super().__call__
        in order to track the inputs.

        Args:
            inputs: A numpy.ndarray of the input features.
        """
        self._inputs = inputs

    @abc.abstractmethod
    def backward(self, grad_outputs):
        """Computes the backward pass.

        Args:
            grad_outputs: The gradient of the output returned by `__call__`.
        Returns:
            The gradient of the inputs.
        """

    def zero_grad(self):
        """Clears out the Module's gradients."""
        self._inputs = None
        self.gradients = None


class Linear(Module):
    """A linear Neural Network layer."""

    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self._w = np.random.normal(shape=(num_inputs, num_outputs))
        self._b = np.random.normal(shape=(num_outputs,))
        self.parameters = [self._w, self._b]

    def __call__(self, inputs):
        super().__call__(self, inputs)
        return np.dot(inputs, self._w) + self._b

    def backward(self, grad_outputs):
        grad_inputs = np.dot(grad_outputs, self._w.T)
        dw, db = self.gradients if self.gradients else 0, 0
        dw += np.sum(
            np.expand_dims(grad_outputs,  axis=1) *
            np.expand_dims(self._inputs, axis=-1),
            axis=0)
        db += np.sum(grad_outputs, axis=0)
        self.gradients = [dw, db]
        return grad_inputs


class Relu(Module):
    """A Rectified Linear Unit (ReLU) layer."""

    def __call__(self, inputs):
        super().__call__(self, inputs)
        return np.clip(inputs, 0, inputs)

    def backward(self, grad_outputs):
        return grad_outputs * (self._inputs > 0).astype(np.float32)


# TODO(ehotaj): Create a new Loss module type which takes both inputs and
# targets in __call__ and take no grad_outputs in backward.
class MSELoss(Module):
    """Mean Squared Error (MSE) loss."""

    def __call__(self, inputs, targets):
        super().__call__(self, inputs)
        self._targets = targets
        return np.mean((np.squeeze(inputs) - targets)**2)

    def backward(self):
        N = self._targets.shape[0]
        err = np.expand_dims(np.squeeze(self._inputs) - self._targets, axis=-1)
        return 2 * err / N

    def zero_grad(self):
        super().zero_grad()
        self._targets = None
