import math
import random
class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._op = _op
        self._prev = set(_children)
        self.label = label
    
    def __repr__(self) -> str:
        return f'Value(data={self.data}, grad={self.grad})'
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=(self, other), _op='+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other), _op='*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        x = self.data
        out = Value(x ** other, _children=(self,), _op=f'**{other}')

        def _backward():
            self.grad += other * x ** (other - 1) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), _children=(self,), _op='exp')
 
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out
    
    def log(self):
        x = self.data
        assert x > 0, 'logarithm of negative number is undefined'
        out = Value(math.log(x), _children=(self,), _op='log')

        def _backward():
            self.grad += (1 / x) * out.grad
        out._backward = _backward

        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, _children=(self,), _op='tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        x = self.data
        r = x if x > 0 else 0
        out = Value(r, _children=(self,), _op='relu')

        def _backward():
            self.grad += (r > 0) * out.grad
        out._backward = _backward

        return out
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return -1 * self
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return (-self) + other

    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __rtruediv__(self, other):
        return (self ** -1) * other
    
    def backward(self):

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

class Module:

    def __init__(self) -> None:
        self.params = []
    
    def parameters(self):
        return self.params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

class Neuron(Module):

    def __init__(self, input_dim, activation=None) -> None:
        self.weights = [Value(random.uniform(-1,1)) for _ in range(input_dim)]
        self.bias = Value(random.uniform(-1,1))
        self.activation = activation
        self.params = self.weights + [self.bias]
    
    def __call__(self, x):
        act = sum([x*w for x, w in zip(x, self.weights)], self.bias)
        if self.activation == 'relu':
            out = act.relu()
        elif self.activation == 'tanh':
            out = act.tanh()
        else:
            out = act
        return out

    def __repr__(self):
        return f"Neuron({len(self.weights)})"

class Layer(Module):

    def __init__(self, input_dim, output_dim, activation=None) -> None:
        self.neurons = [Neuron(input_dim, activation) for _ in range(output_dim)]
        self.params = [p for n in self.neurons for p in n.params]
    
    def __call__(self, x):
        return [n(x) for n in self.neurons]
    
    def __repr__(self):
        return f"Layer({[n for n in self.neurons]})"
    

class MLP(Module):

    def __init__(self, dims, activation=None) -> None:
        # 最后一层不需要激活
        self.layers = [Layer(dims[i], dims[i+1], activation) if i < len(dims)-2 else Layer(dims[i], dims[i+1]) for i in range(len(dims)-1)]
        self.params = [p for layer in self.layers for p in layer.params]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x
    
    def __repr__(self):
        return f"MLP({[layer for layer in self.layers]})"


if __name__ == '__main__':
    import random
    # my mlp
    random.seed(43)
    dims = [10,5,5,1]
    mlp = MLP(dims, activation='relu')
    x = [Value(random.uniform(0,1)) for _ in range(10)]
    ot = mlp(x)
    ot.backward()

    # torch mlp
    import torch
    import torch.nn.functional as F

    tls = []
    bls = []
    for i in range(len(dims)-1):
        l = torch.tensor([[w.data for w in n.weights] for n in mlp.layers[i].neurons]).double()
        l.requires_grad = True
        tls.append(l)
        b = torch.tensor([n.bias.data for n in mlp.layers[i].neurons]).double()
        b.requires_grad = True
        bls.append(b)

    xs = torch.tensor([xi.data for xi in x]).double()
    xs.requires_grad = True
    tot = xs
    for tl, bl in zip(tls, bls):
        tot = F.relu(tl @ tot + bl)
    tot.backward()


    print(tot.item(), abs(tot.item() - ot.data), (xs.grad - torch.tensor([xi.grad for xi in x])).abs())

    print(f'num of params: {len(mlp.parameters())}')
    print(mlp)