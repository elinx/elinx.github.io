import torch
from torch.autograd.functional import jacobian
from torch import tensor
from torch.autograd import grad


x = tensor([1., 2, 3, 4], requires_grad=True)
w = torch.arange(12, dtype=torch.float32, requires_grad=True).reshape(3, 4)


def torch_method():
    y = w.matmul(x)
    g = torch.ones_like(y) * 2

    dx = grad(outputs=y, inputs=x, grad_outputs=g, retain_graph=True)
    dw = grad(outputs=y, inputs=w, grad_outputs=g)
    return y, dx[0], dw[0]


def my_method():
    def fc(x):
        def vjp(v):
            return w.T.matmul(v), v.outer(x)  # x, w
        return vjp, w.matmul(x)

    vjp, y = fc(x)
    g = torch.ones_like(y) * 2
    dx, dw = vjp(g)
    return y, dx, dw


t = torch_method()
m = my_method()
torch.allclose(t[0], m[0])
torch.allclose(t[1], m[1])
torch.allclose(t[2], m[2])
