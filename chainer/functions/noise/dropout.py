import numpy

from chainer import configuration
from chainer import cuda
from chainer import function
from chainer.utils import argument
from chainer.utils import type_check


class Dropout(function.Function):

    """Dropout regularization."""

    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        self.retain_inputs(())
        if not hasattr(self, 'mask'):
            scale = x[0].dtype.type(1. / (1 - self.dropout_ratio))
            xp = cuda.get_array_module(*x)
            if xp == numpy:
                flag = xp.random.rand(*x[0].shape) >= self.dropout_ratio
            else:
                flag = (xp.random.rand(*x[0].shape, dtype=numpy.float32) >=
                        self.dropout_ratio)
            self.mask = scale * flag
        return x[0] * self.mask,

    def backward(self, x, gy):
        return gy[0] * self.mask,


def dropout(x, ratio=.5, **kwargs):
    """Drops elements of input variable randomly.

    This function drops input elements randomly with probability ``ratio`` and
    scales the remaining elements by factor ``1 / (1 - ratio)``. In testing
    mode, it does nothing and just returns ``x``.

    .. warning::

       ``train`` argument is not supported anymore since v2.
       Instead, use ``chainer.using_config('train', train)``.
       See :func:`chainer.using_config`.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable.
        ratio (float): Dropout ratio. It should be ``(0, 1]``.

    Returns:
        ~chainer.Variable: Output variable.

    See the paper by G. Hinton: `Improving neural networks by preventing \
    co-adaptation of feature detectors <https://arxiv.org/abs/1207.0580>`_.

    .. admonition:: Example

        >>> x = np.array([-2.6, -1, 0, 1, 2.6])
        >>> x
        array([-2.6, -1. ,  0. ,  1. ,  2.6])
        >>> F.hard_sigmoid(x).data
        array([ 0. ,  0.3,  0.5,  0.7,  1. ])

    """
    argument.check_unexpected_kwargs(
        kwargs, train='train argument is not supported anymore. '
        'Use chainer.using_config')
    argument.assert_kwargs_empty(kwargs)

    if configuration.config.train:
        return Dropout(ratio)(x)
    return x
