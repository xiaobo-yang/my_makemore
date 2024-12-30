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
            p.data -= self.lr * g.data