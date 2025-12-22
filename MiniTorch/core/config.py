import contextlib


class Config:
    enable_backprob = True


@contextlib.contextmanager
def using_config(name: str, value: bool):
    """
    using_config:
        name: "Config"
        value: bool (if you are truing to using backprobagation)
    example:
        '''python
            wiht using_config("Config", false):
                x = Variable(np.array(2.0))
                y = square(x)
        '''
    """
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config("enable_backprob", False)


def with_grad():
    return using_config("enable_backprob", True)
