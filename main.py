import numpy as np
from MiniTorch import Variable,

def main():
    x = np.array([1.0])
    y = x**2
    print(type(x), x.ndim)
    print(type(y))


if __name__ == "__main__":
    main()
