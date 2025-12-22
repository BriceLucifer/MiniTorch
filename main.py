import numpy as np

from MiniTorch import Variable


def main():
    x = Variable(np.array(2.0))
    y = x**2
    print(y)


if __name__ == "__main__":
    main()
