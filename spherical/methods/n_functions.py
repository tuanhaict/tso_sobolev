import numpy as np
import torch


class NFunction:
    """Base class for N-functions used in Orlicz geometry"""

    def __call__(self, t):
        """Evaluate Phi(t)"""
        raise NotImplementedError

    def derivative(self, t):
        """First derivative Phi'(t)"""
        raise NotImplementedError

    def second_derivative(self, t):
        """Second derivative Phi''(t)"""
        raise NotImplementedError


class PowerNFunction(NFunction):
    """Phi(t) = ((p-1)^(p-1) / p^p) * |t|^p"""

    def __init__(self, p):
        assert p > 1, "p must be greater than 1"
        self.p = p
        self.coeff = ((p - 1) ** (p - 1)) / (p ** p)

    def __call__(self, t):
        if isinstance(t, torch.Tensor):
            return self.coeff * torch.abs(t) ** self.p
        else:
            return self.coeff * np.abs(t) ** self.p

    def derivative(self, t):
        if isinstance(t, torch.Tensor):
            return (
                self.coeff
                * self.p
                * torch.abs(t) ** (self.p - 1)
                * torch.sign(t)
            )
        else:
            return (
                self.coeff
                * self.p
                * np.abs(t) ** (self.p - 1)
                * np.sign(t)
            )

    def second_derivative(self, t):
        # Defined a.e. for t != 0
        if isinstance(t, torch.Tensor):
            return (
                self.coeff
                * self.p
                * (self.p - 1)
                * torch.abs(t) ** (self.p - 2)
            )
        else:
            return (
                self.coeff
                * self.p
                * (self.p - 1)
                * np.abs(t) ** (self.p - 2)
            )


class ExpNFunction(NFunction):
    """Phi(t) = exp(t) - t - 1"""

    def __call__(self, t):
        if isinstance(t, torch.Tensor):
            return torch.exp(t) - t - 1
        else:
            return np.exp(t) - t - 1

    def derivative(self, t):
        if isinstance(t, torch.Tensor):
            return torch.exp(t) - 1
        else:
            return np.exp(t) - 1

    def second_derivative(self, t):
        if isinstance(t, torch.Tensor):
            return torch.exp(t)
        else:
            return np.exp(t)


class ExpSquaredNFunction(NFunction):
    """Phi(t) = exp(t^2) - 1"""

    def __call__(self, t):
        if isinstance(t, torch.Tensor):
            return torch.exp(t ** 2) - 1
        else:
            return np.exp(t ** 2) - 1

    def derivative(self, t):
        if isinstance(t, torch.Tensor):
            return 2 * t * torch.exp(t ** 2)
        else:
            return 2 * t * np.exp(t ** 2)

    def second_derivative(self, t):
        if isinstance(t, torch.Tensor):
            return 2 * torch.exp(t ** 2) * (1 + 2 * t ** 2)
        else:
            return 2 * np.exp(t ** 2) * (1 + 2 * t ** 2)
class EntropyLogNFunction(NFunction):
    """Phi(t) = (1+t) * log(1+t) - t"""

    def __call__(self, t):
        if isinstance(t, torch.Tensor):
            return (1+t) * torch.log(1+t) - t
        else:
            return (1+t) * np.log(1+t) - t

    def derivative(self, t):
        if isinstance(t, torch.Tensor):
            return torch.log(1+t)
        else:
            return np.log(1+t)

    def second_derivative(self, t):
        if isinstance(t, torch.Tensor):
            return 1 / (1+t)
        else:
            return 1 / (1+t)
class LogNFunction(NFunction):
    """Phi(t) = t * log(1+t)"""

    def __call__(self, t):
        if isinstance(t, torch.Tensor):
            return t * torch.log(1+t)
        else:
            return t * np.log(1+t)

    def derivative(self, t):
        if isinstance(t, torch.Tensor):
            return torch.log(1+t) + t / (1+t)
        else:
            return np.log(1+t) + t / (1+t)

    def second_derivative(self, t):
        if isinstance(t, torch.Tensor):
            return 1 / (1+t) + 1/(1+t)**2
        else:
            return 1 / (1+t) + 1/(1+t)**2

class LinearNFunction(NFunction):
    """Phi(t) = |t| (W1 limit case)"""

    def __call__(self, t):
        if isinstance(t, torch.Tensor):
            return torch.abs(t)
        else:
            return np.abs(t)

    def derivative(self, t):
        if isinstance(t, torch.Tensor):
            return torch.sign(t)
        else:
            return np.sign(t)

    def second_derivative(self, t):
        raise NotImplementedError(
            "LinearNFunction is not C^2 and cannot be used with Newton optimization."
        )
class ExpQuadraticQuarterNFunction(NFunction):
    """Phi(t) = exp(t^2 / 4) - 1"""

    def __call__(self, t):
        if isinstance(t, torch.Tensor):
            return torch.exp(t**2 / 4.0) - 1
        else:
            return np.exp(t**2 / 4.0) - 1

    def derivative(self, t):
        if isinstance(t, torch.Tensor):
            return (t / 2.0) * torch.exp(t**2 / 4.0)
        else:
            return (t / 2.0) * np.exp(t**2 / 4.0)

    def second_derivative(self, t):
        if isinstance(t, torch.Tensor):
            return torch.exp(t**2 / 4.0) * (0.5 + t**2 / 4.0)
        else:
            return np.exp(t**2 / 4.0) * (0.5 + t**2 / 4.0)
class ExpHalfLinearCorrectedNFunction(NFunction):
    """Phi(t) = exp(t/2) - 1 - t/2"""

    def __call__(self, t):
        if isinstance(t, torch.Tensor):
            return torch.exp(t / 2.0) - 1 - t / 2.0
        else:
            return np.exp(t / 2.0) - 1 - t / 2.0

    def derivative(self, t):
        if isinstance(t, torch.Tensor):
            return 0.5 * torch.exp(t / 2.0) - 0.5
        else:
            return 0.5 * np.exp(t / 2.0) - 0.5

    def second_derivative(self, t):
        if isinstance(t, torch.Tensor):
            return 0.25 * torch.exp(t / 2.0)
        else:
            return 0.25 * np.exp(t / 2.0)
class OrliczNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, d, n_function, max_iter=50, tol=1e-6):
        eps = 1e-12
        d_pos = d.detach().clamp(min=0.0)
        
        # Binary search
        hi = d_pos.max().clamp(min=1.0)
        for _ in range(60):
            if n_function(d_pos / hi).mean() <= 1.0:
                break
            hi = hi * 2.0
        
        lo = torch.tensor(eps, device=d.device, dtype=d.dtype)
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            if n_function(d_pos / mid).mean() > 1.0:
                lo = mid
            else:
                hi = mid
            if (hi - lo) / hi.clamp(min=eps) < tol:
                break
        
        lam = hi
        ctx.save_for_backward(d, lam)
        ctx.n_function = n_function
        return lam
    
    @staticmethod
    def backward(ctx, grad_output):
        d, lam = ctx.saved_tensors
        n_function = ctx.n_function
        eps = 1e-12
        
        # Φ'(d_i / λ*)
        x = d / lam
        phi_prime = n_function.derivative(x)   # (N,)
        
        # dλ*/d(d_i) = Φ'(d_i/λ*) / (λ* · n · E[d_j/λ* · Φ'(d_j/λ*)])
        denom = (x * phi_prime).mean() * lam    # scalar
        grad_d = grad_output * phi_prime / (d.shape[0] * denom.clamp(min=eps))
        
        return grad_d, None, None, None