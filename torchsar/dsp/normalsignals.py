import torch as th
from torchsar.utils.const import PI


def rect(x):
    r"""
    Rectangle function:
        rect(x) = {1, if |x|<= 0.5; 0, otherwise}
    """

    # return hs(x + 0.5) * ihs(x - 0.5)
    # return th.where(th.abs(x) > 0.5, 0., 1.0)
    y = th.ones_like(x)
    y[x < -0.5] = 0.
    y[x > 0.5] = 0.
    return y


def chirp(t, T, Kr):
    r"""
    Create a chirp signal :
        S_{tx}(t) = rect(t/T) * exp(1j*pi*Kr*t^2)
    """

    return rect(t / T) * th.exp(1j * PI * Kr * t**2)


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    Ts = 3
    Fs = 100
    Ns = int(Ts * Fs)
    x = th.linspace(-Ts / 2., Ts / 2., Ns)

    y = rect(x)

    plt.figure()
    plt.plot(y)
    plt.show()
