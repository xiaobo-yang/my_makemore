import os
import random

import micro_torch
from micro_torch.nn import Module, Embedding, Flatten, Linear, BatchNorm1d, Tanh, Sequential, CrossEntropyLoss
from micro_torch.optim import SGD


# data
file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'names.txt')
words = open(file_path, 'r').read().splitlines()
words = sorted(list(set(words))) # set cause uncontrollable randomness， sorted for reproducibility
random.seed(42)
random.shuffle(words)

chs = list(set(''.join(words + ['.'])))
chs = sorted(chs, reverse=False)
stoi = {ch: i for i, ch in enumerate(chs)}
itos = {i: ch for i, ch in enumerate(chs)}

# predict next token use previous tokens
block_size = 2
X, Y = [], []

for w in words:
    context = '.' * block_size
    for ch in w + '.':
        x = [stoi[c] for c in context]
        y = stoi[ch]
        X.append(x)
        Y.append(y)
        context = context[1:] + ch

X = micro_torch.tensor(X)
Y = micro_torch.tensor(Y)
n1, n2  = int(0.8 * len(X)), int(0.9 * len(X))

X_train, X_val, X_test = X.tensor_split([n1, n2])
Y_train, Y_val, Y_test = Y.tensor_split([n1, n2])

X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape


# args
class MLP(Module):
    def __init__(self, vocab_size, block_size, n_embd, n_hidden, n_layer, dtype=micro_torch.float64):
        layers = [Embedding(vocab_size, n_embd, dtype=dtype), Flatten(), Linear(n_embd * block_size, n_hidden, bias=False, dtype=dtype), BatchNorm1d(n_hidden, dtype=dtype), Tanh()]
        for _ in range(n_layer-2):
            layers.extend([Linear(n_hidden, n_hidden, bias=False, dtype=dtype), BatchNorm1d(n_hidden, dtype=dtype), Tanh()])
        layers.extend([Linear(n_hidden, vocab_size, bias=False, dtype=dtype),])
        layers[-1].weight.data *= 0.1
        self.net = Sequential(layers)
        self.block_size = block_size

    def parameters(self):
        return self.net.parameters()
    
    def grads(self):
        return self.net.grads()

    def __call__(self, x):
        return self.net(x)

    def backward(self, grad):
        grad = self.net.backward(grad)
        return grad # None
    
    def eval(self):
        for l in self.net.layers:
            l._training = False

    def train(self):
        for l in self.net.layers:
            l._training = True

    def generate(self, s, max_new_tokens, do_sample=True, temperature=1.0):
        assert isinstance(s, str), 'str in, str out'
        assert len(s) == self.block_size, 'input string length must be equal to block size'
        x = micro_torch.tensor([[stoi[ch] for ch in s]])
        for _ in range(max_new_tokens):
            cond = x[:, -self.block_size:]
            logits = self(cond) * (1 / temperature)
            probs = logits.softmax(dim=-1)
            if do_sample:
                next_x = micro_torch.multinomial(probs, num_samples=1)
            else:
                next_x = probs.argmax(dim=-1, keepdim=True)
            x = micro_torch.cat([x, next_x], dim=-1)
            if next_x.item() == 0:
                break
        s = ''.join([itos[idx.item()] for idx in x[0]])
        return s


n_embd = 10
n_hidden = 100
vocab_size = 27
n_layer = 2
dtype = micro_torch.float64
eval_interval = 100
bs = 32
n_steps = 1000
ini_lr = 0.1
lossis = []

# model
micro_torch.manual_seed(42)
model = MLP(vocab_size, block_size, n_embd, n_hidden, n_layer, dtype)
loss_fn = CrossEntropyLoss()
optimizer = SGD(model, ini_lr)

# train
model.train()
for step in range(n_steps):
    lr = ini_lr if step < int(n_steps * 0.75) else ini_lr / 10
    optimizer.lr = lr
    idx = micro_torch.randint(0, X_train.shape[0], (bs,)) 
    x, y = X_train[idx], Y_train[idx]

    # forward
    logits = model(x)
    loss = loss_fn(logits, y)

    # backward
    # since grad buffer is stored in model class, we need to call backward imediately after forward
    # otherwise, grad buffer will be overwritten by next forward
    h_grad = loss_fn.backward(grad=1.0) # last layer, dloss=1.0
    model.backward(h_grad)
    
    # update
    optimizer.step()
    optimizer.zero_grad()

    # eval
    if step % eval_interval == 0: 
        model.eval()
        x, y = X_val, Y_val
        logits = model(x)
        val_loss = loss_fn(logits, y) # val loss is actually one step later than train loss
        print(f'step: {step}, train loss: {loss.item()}, val loss: {val_loss.item()}')
        model.train()
    lossis.append(loss.log10().item())
    
# inference
model.eval()
x, y = X_test, Y_test
logits = model(x)
test_loss = loss_fn(logits, y).item()
print(f'test loss: {test_loss}')

micro_torch.manual_seed(42)
for _ in range(10):
    out = model.generate('.' * block_size, max_new_tokens=10, do_sample=True, temperature=0.5)
    print(out)





