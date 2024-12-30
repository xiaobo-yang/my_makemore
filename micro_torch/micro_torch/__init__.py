import torch

class MicroTensor:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.grad_fn = None
        self.requires_grad = False
        self.shape = self.data.shape if hasattr(self.data, 'shape') else None
        self.dtype = self.data.dtype if hasattr(self.data, 'dtype') else None
        self.device = self.data.device if hasattr(self.data, 'device') else None
        self.ndim = self.data.ndim if hasattr(self.data, 'ndim') else None

    def backward(self, grad=None):
        raise NotImplementedError('backward is not implemented')
        if grad is None:
            grad = torch.ones_like(self.data)
        if self.grad_fn is not None:
            self.grad_fn.backward(grad)
        else:
            raise ValueError('grad_fn is None')


    def __repr__(self):
        if self.grad_fn is not None:
            return f'MicroTensor(data={self.data}, grad_fn={self.grad_fn.__name__})'
        else:
            return f'MicroTensor(data={self.data})'
    
    def __getitem__(self, idx):
        if isinstance(idx, MicroTensor):
            return MicroTensor(self.data[idx.data])
        else:
            return MicroTensor(self.data[idx])
    
    def __setitem__(self, idx, value):
        self.data[idx] = value.data

    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        return iter(self.data)

    def __add__(self, other):
        if not isinstance(other, MicroTensor):
            other = MicroTensor(other)
        return MicroTensor(self.data + other.data)
    
    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return (-1) * self
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __mul__(self, other):
        if not isinstance(other, MicroTensor):
            other = MicroTensor(other)
        return MicroTensor(self.data * other.data)
    
    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, other):
        assert isinstance(other, (float, int)), 'other must be a float or int'
        return MicroTensor(self.data ** other)
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other):
        return other * self**-1
    
    def __matmul__(self, other):
        assert isinstance(other, MicroTensor), 'other must be a MicroTensor'
        return MicroTensor(self.data @ other.data)
    
    def __rmatmul__(self, other):
        assert isinstance(other, MicroTensor), 'other must be a MicroTensor'
        return MicroTensor(other.data @ self.data)

    def max(self, dim=None, keepdim=False):
        if dim is not None:
            data, idx = self.data.max(dim=dim, keepdim=keepdim)
            return MicroTensor(data), MicroTensor(idx)
        else:
            return MicroTensor(self.data.max())
    
    def min(self, dim=None, keepdim=False):
        if dim is not None:
            data, idx = self.data.min(dim=dim, keepdim=keepdim)
            return MicroTensor(data), MicroTensor(idx)
        else:
            return MicroTensor(self.data.min())
    
    def sum(self, dim=None, keepdim=False):
        return MicroTensor(self.data.sum(dim=dim, keepdim=keepdim))
    
    def mean(self, dim=None, keepdim=False):
        return MicroTensor(self.data.mean(dim=dim, keepdim=keepdim))
    
    def pow(self, other):
        return self ** other
    
    def exp(self):
        return MicroTensor(self.data.exp())
    
    def log(self):
        return MicroTensor(self.data.log())
    
    def log10(self):
        return MicroTensor(self.data.log10())
    
    def tanh(self):
        return MicroTensor(self.data.tanh())
    
    def softmax(self, dim):
        return MicroTensor(self.data.softmax(dim=dim))
    
    def view(self, *shape):
        return MicroTensor(self.data.view(*shape))
    
    def reshape(self, *shape):
        return self.view(*shape)
    
    def item(self):
        return self.data.item()
    
    def size(self):
        return self.data.size()
    
    def dim(self):
        return self.data.dim()
    
    def tolist(self):
        return self.data.tolist()
    
    def to(self, device_or_dtype):
        return MicroTensor(self.data.to(device_or_dtype))

    def float(self):
        return MicroTensor(self.data.float())

    @property
    def T(self):
        if self.dim() == 1:
            return self
        elif self.dim() == 2:
            return self.transpose(0, 1)
        else:
            dims = list(range(self.dim()))
            return self.permute(*dims[::-1])
    
    def transpose(self, dim0, dim1):
        return MicroTensor(self.data.transpose(dim0, dim1))
    
    def zero_(self):
        self.data.zero_()
    
    def index_add_(self, dim, index, source):
        self.data.index_add_(dim, index.data, source.data)
    
    def split(self, split_size, dim=0):
        return self.data.split(split_size, dim)
    
    def tensor_split(self, indices, dim=0):
        tensors = self.data.tensor_split(indices, dim)
        return [MicroTensor(t) for t in tensors]
    

def tensor(data, dtype=None, device=None):
    return MicroTensor(torch.tensor(data, dtype=dtype, device=device))

def manual_seed(seed):
    torch.manual_seed(seed)

def zeros(*shape, dtype=None, device=None):
    return MicroTensor(torch.zeros(shape, dtype=dtype, device=device))

def ones(*shape, dtype=None, device=None):
    return MicroTensor(torch.ones(shape, dtype=dtype, device=device))

def randn(*shape, dtype=None, device=None):
    return MicroTensor(torch.randn(shape, dtype=dtype, device=device))

def randint(low, high, size, dtype=None, device=None):
    return MicroTensor(torch.randint(low, high, size, dtype=dtype, device=device))

def multinomial(tensor, num_samples, replacement=False):
    return MicroTensor(torch.multinomial(tensor.data, num_samples, replacement=replacement))

def zeros_like(tensor, dtype=None, device=None):
    return MicroTensor(torch.zeros_like(tensor.data, dtype=dtype, device=device))

def ones_like(tensor, dtype=None, device=None):
    return MicroTensor(torch.ones_like(tensor.data, dtype=dtype, device=device))

def randn_like(tensor, dtype=None, device=None):
    return MicroTensor(torch.randn_like(tensor.data, dtype=dtype, device=device))

def cat(tensors, dim=0):
    torch_tensors = [t.data for t in tensors]
    return MicroTensor(torch.cat(torch_tensors, dim))
    
def stack(tensors, dim=0):
    torch_tensors = [t.data for t in tensors]
    return MicroTensor(torch.stack(torch_tensors, dim))

float16 = torch.float16
float32 = torch.float32
float64 = torch.float64
int8 = torch.int8
int16 = torch.int16
int32 = torch.int32
int64 = torch.int64
uint8 = torch.uint8
bool = torch.bool

class Generator(torch.Generator):
    pass
