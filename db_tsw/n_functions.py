import numpy as np
import torch
class NFunction:
    """Base class for N-functions used in Orlicz geometry"""
    
    def __call__(self, t):
        """Evaluate Phi(t)"""
        raise NotImplementedError
    
    def derivative(self, t):
        """Derivative of Phi(t) if needed"""
        raise NotImplementedError


class PowerNFunction(NFunction):
    """N-function: Phi(t) = ((p-1)^(p-1) / p^p) * t^p"""
    
    def __init__(self, p):
        assert p > 1, "p must be greater than 1"
        self.p = p
        self.coeff = ((p - 1) ** (p - 1)) / (p ** p)
    
    def __call__(self, t):
        if isinstance(t, torch.Tensor):
            return self.coeff * torch.pow(torch.abs(t), self.p)
        else:
            return self.coeff * np.abs(t) ** self.p
    
    def derivative(self, t):
        if isinstance(t, torch.Tensor):
            return self.coeff * self.p * torch.pow(torch.abs(t), self.p - 1) * torch.sign(t)
        else:
            return self.coeff * self.p * np.abs(t) ** (self.p - 1) * np.sign(t)


class ExpNFunction(NFunction):
    """N-function: Phi(t) = exp(t) - t - 1"""
    
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


class ExpSquaredNFunction(NFunction):
    """N-function: Phi(t) = exp(t^2) - 1"""
    
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


class LinearNFunction(NFunction):
    """N-function: Phi(t) = t (limit case, for compatibility with W1)"""
    
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