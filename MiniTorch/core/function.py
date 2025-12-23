import weakref

from MiniTorch.core.config import Config
from MiniTorch.core.variable import Variable
from MiniTorch.utils.type_check import as_array, as_variable


class Function:
    """
    using it as a base virtual table
    """

    def __init__(self):
        """
        we save input and output of the function
        """
        self.inputs = None
        self.output = None

    def __call__(self: "Function", *inputs):
        inputs = [as_variable(x) for x in inputs]
        # we save the input variable
        xs = [x.data for x in inputs]
        # forward()
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)

        outputs = [Variable(as_array(y)) for y in ys]

        # if enable_backprob, the config default is true
        if Config.enable_backprob:
            self.generation = max([x.generation for x in inputs])
            # let the outputs save the creator function
            for output in outputs:
                output.set_creator(self)
            # save the input and output
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError

    def backward(self, gys):
        raise NotImplementedError
