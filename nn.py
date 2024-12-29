import torch

class Linear:

    def __init__(self, in_features, out_features, bias=True, dtype=torch.float64, generator=None):
        self.weight = torch.randn(in_features, out_features, dtype=dtype, generator=generator) * (in_features)**-0.5
        self.bias = torch.zeros(out_features, dtype=dtype) * 0 if bias else None
        # grads
        self.weight_grad = None
        self.bias_grad = None

    def __repr__(self):
        return f'MyLinear(in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, bias={self.bias is not None})'

    def parameters(self):
        if self.bias is not None:
            return [self.weight, self.bias]
        else:
            return [self.weight]
    
    def grads(self):
        if self.bias is not None:
            return [self.weight_grad, self.bias_grad]
        else:
            return [self.weight_grad]
    
    def __call__(self, x):
        if self.bias is not None:
            out = x @ self.weight + self.bias
        else:
            out = x @ self.weight
        # backward buffer
        self.x = x
        return out
    
    def backward(self, grad):
        """
            Input:
                x: input of current layer
                out: output of current layer
                grad: grad from next layer
            Output:
                x_grad: grad back to previous layer
        """
        x_grad = grad @ self.weight.T
        self.weight_grad = self.x.T @ grad
        if self.bias is not None:
            self.bias_grad = grad.sum(dim=0)
        return x_grad

class BatchNorm1d:
    def __init__(self, in_features, eps=1e-5, momentum=0.001, dtype=torch.float64): # manual bn need fp64
        self.weight = torch.ones(in_features, dtype=dtype)
        self.bias = torch.zeros(in_features, dtype=dtype)
        self.running_mean = torch.zeros(in_features, dtype=dtype)
        self.running_var = torch.ones(in_features, dtype=dtype)
        self.eps = eps
        self.momentum = momentum
        self._training = True # internal flag
        self.dtype = dtype
        # grads
        self.weight_grad = None
        self.bias_grad = None
    
    def __repr__(self):
        return f'MyBatchNorm1d(in_features={self.weight.shape[0]}, eps={self.eps}, momentum={self.momentum})'

    def parameters(self):
        return [self.weight, self.bias]
    
    def grads(self):
        return [self.weight_grad, self.bias_grad]
    
    def __call__(self, x):
        if self._training:
            mean = x.mean(dim=0)
            var = ((x - mean) ** 2).mean(dim=0) # as torch, we don't use Bessel correction
            
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
            
        inv_std = (var + self.eps).pow(-0.5)
        x_normalized = (x - mean) * inv_std
        out = self.weight * x_normalized + self.bias
        
        # backward buffer
        self.inv_std = inv_std
        self.x_normalized = x_normalized
        return out

    def backward(self, grad):
        inv_std, x_normalized = self.inv_std, self.x_normalized
        b, d = x_normalized.shape
        
        self.weight_grad = (x_normalized * grad).sum(dim=0)
        self.bias_grad = grad.sum(dim=0)
        
        dx = ((grad - self.bias_grad / b) - x_normalized * (self.weight_grad / b)) * inv_std * self.weight
        return dx

class LayerNorm:
    def __init__(self, in_features, eps=1e-5, dtype=torch.float64):
        self.weight = torch.ones(in_features, dtype=dtype)
        self.bias = torch.zeros(in_features, dtype=dtype)
        self.eps = eps
        self.dtype = dtype
        # grads
        self.weight_grad = None
        self.bias_grad = None
    
    def __repr__(self):
        return f'MyLayerNorm(in_features={self.weight.shape[0]}, eps={self.eps})'

    def parameters(self):
        return [self.weight, self.bias]
    
    def grads(self):
        return [self.weight_grad, self.bias_grad]
    
    def __call__(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
            
        inv_std = (var + self.eps).pow(-0.5)
        x_normalized = (x - mean) * inv_std
        out = self.weight * x_normalized + self.bias
        
        # backward buffer
        self.inv_std = inv_std
        self.x_normalized = x_normalized
        return out

    def backward(self, grad):
        inv_std, x_normalized = self.inv_std, self.x_normalized
        
        self.weight_grad = (x_normalized * grad).sum(dim=tuple(range(x_normalized.ndim-1))) # all dims except last
        self.bias_grad = grad.sum(dim=tuple(range(x_normalized.ndim-1)))
        
        dx = ((grad - grad.mean(dim=-1, keepdim=True)) - x_normalized * (x_normalized * grad).mean(dim=-1, keepdim=True)) * inv_std * self.weight
        return dx

class Func:

    def parameters(self):
        return []
    
    def grads(self):
        return []

class Tanh(Func):

    def __repr__(self):
        return f'MyTanh()'
    
    def __call__(self, x):
        out = x.tanh()
        # backward buffer
        self.out = out
        return out
    
    def backward(self, grad):
        x_grad = grad * (1 - self.out**2)
        return x_grad

class ReLU(Func):

    def __repr__(self):
        return f'MyReLU()'
    
    def __call__(self, x):
        out = x.relu()
        # backward buffer
        self.out = out
        return out

    def backward(self, grad):
        x_grad = grad * (self.out > 0).float()
        return x_grad

class CrossEntropyLoss(Func):

    def __repr__(self):
        return f'MyCrossEntropyLoss()'

    def __call__(self, x, y):
        xmax = x.max(dim=-1, keepdim=True)[0]
        exp_l = (x - xmax).exp()
        count = exp_l.sum(dim=-1, keepdim=True)
        probs = exp_l / count
        loss = -probs[range(y.shape[0]), y].log().mean()
        # backward buffer
        self.probs = probs
        self.y = y
        return loss
        
    
    def backward(self, grad):
        b = self.y.shape[0]
        x_grad = self.probs
        x_grad[range(b), self.y] -= 1
        x_grad = x_grad / b * grad
        return x_grad
        
