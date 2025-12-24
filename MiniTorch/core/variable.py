import numpy as np

from MiniTorch.core.config import Config, using_config


class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None) -> None:
        # I use type check to avoid input problem
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported")

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        """
        len(Variable)
        """
        return len(self.data)

    def __repr__(self) -> str:
        """print(Variable)"""
        if self.data is None:
            return "variable(None)"
        else:
            p = str(self.data).replace("\n", "\n" + " " * 9)
            return "variable(" + p + ")"

    def __mul__(self, other):
        """
        Variable * Variable
        """
        from MiniTorch.ops.mul import mul  # just for avoiding depending recycle

        return mul(self, other)

    def __rmul__(self, other):
        from MiniTorch.ops.mul import mul

        return mul(self, other)

    def __add__(self, other):
        """
        Variable + Variable
        """
        from MiniTorch.ops.add import add  # just for avoiding depending recycle

        return add(self, other)

    def __radd__(self, other):
        from MiniTorch.ops.add import add

        return add(self, other)

    def __neg__(self):
        from MiniTorch.ops.neg import neg

        return neg(self)

    def __sub__(self, other):
        from MiniTorch.ops.sub import sub

        return sub(self, other)

    def __rsub__(self, other):
        from MiniTorch.ops.sub import rsub

        return rsub(self, other)

    def __truediv__(self, other):
        from MiniTorch.ops.div import div

        return div(self, other)

    def __rtruediv__(self, other):
        from MiniTorch.ops.div import rdiv

        return rdiv(self, other)

    def __pow__(self, other):
        from MiniTorch.ops.pow import pow

        return pow(self, other)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        from MiniTorch.ops.reshape import reshape

        return reshape(self, shape)

    @property
    def T(self):
        from MiniTorch.ops.transpose import transpose

        return transpose(self)

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def clear_grad(self):
        self.grad = None

    def backward(self, retain_grad: bool = False, create_graph: bool = False):
        if not Config.enable_backprob:
            raise RuntimeError("backward() is not allowed in no_grad mode")

        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        # bfs
        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]  # type: ignore

            with using_config("enable_backprob", create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)

            if (not retain_grad) and (not create_graph):
                for y in f.outputs:
                    y().grad = None  # type:ignore
