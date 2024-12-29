import torch

class Module:

    def parameters(self):
        return []
    
    def grads(self):
        return []
    
    def zero_grad(self):
        for p in self.grads():
            if p is not None:
                p.zero_()

class Sequential(Module):
    def __init__(self, layers):
        self.layers = layers
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def grads(self):
        return [g for layer in self.layers for g in layer.grads()]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

class Embedding(Module):

    def __init__(self, num_embeddings, embedding_dim, dtype=torch.float64, generator=None):
        self.weight = torch.randn(num_embeddings, embedding_dim, dtype=dtype, generator=generator) * (num_embeddings)**-0.5
        self.dtype = dtype
        # grads
        self.weight_grad = None

    def __repr__(self):
        return f'MyEmbedding(num_embeddings={self.weight.shape[0]}, embedding_dim={self.weight.shape[1]})'

    def parameters(self):
        return [self.weight]
    
    def grads(self):
        return [self.weight_grad]
    
    def __call__(self, x):
        out = self.weight[x]
        # backward buffer
        self.x = x
        return out
    
    def backward(self, grad):
        self.weight_grad = torch.zeros_like(self.weight)
        self.weight_grad.index_add_(dim=0, index=self.x.view(-1), source=grad.view(-1, self.weight.shape[1]))
        return None

class Linear(Module):

    def __init__(self, in_features, out_features, bias=True, dtype=torch.float64, generator=None):
        self.weight = torch.randn(in_features, out_features, dtype=dtype, generator=generator) * (in_features)**-0.5
        self.bias = torch.zeros(out_features, dtype=dtype) * 0 if bias else None
        self.dtype = dtype
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

class BatchNorm1d(Module):
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

class LayerNorm(Module):
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

class Tanh(Module):

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

class ReLU(Module):

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

class CrossEntropyLoss(Module):

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
        
class Flatten(Module):

    def __repr__(self):
        return f'MyFlatten()'
    
    def __call__(self, x):
        out = x.view(x.shape[0], -1)
        # backward buffer
        self.x_shape = x.shape
        return out
    
    def backward(self, grad):
        return grad.view(*self.x_shape)

class Optimizer:

    def __init__(self, model, lr):
        self.model = model
        self.lr = lr
    
    def zero_grad(self):
        for g in self.model.grads():
            if g is not None:
                g.zero_()

class SGD(Optimizer):

    def step(self):
        for p, g in zip(self.model.parameters(), self.model.grads()):
            p.data -= self.lr * g