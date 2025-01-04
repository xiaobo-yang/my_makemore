import torch
from tiny_torch import *

"""
Implementation of backward pass of the models from scratch.
Forward pass of the models are adapted from https://github.com/karpathy/makemore
"""

# ---------------------------------------- Bigram ------------------------------------------------------
class Bigram(Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.wte = Embedding(config.vocab_size + 1, config.n_embd, dtype=config.dtype) # token embeddings table
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False, dtype=config.dtype)
        self.lm_head.weight.data *= 0.1
        n_params = sum(p.numel() for p in self.parameters())
        print("number of bigram parameters: %d" % (n_params,))
    
    def parameters(self):
        return list(self.wte.parameters()) + list(self.lm_head.parameters())
    
    def grads(self):
        return list(self.wte.grads()) + list(self.lm_head.grads())
    
    def get_block_size(self):
        return self.block_size
    
    def forward(self, idx):
        tok_emb = self.wte(idx)
        logits = self.lm_head(tok_emb)
        return logits
    
    def backward(self, dlogits):
        dwte = self.lm_head.backward(dlogits)
        self.wte.backward(dwte)

# ---------------------------------------- MLP ------------------------------------------------------
class MLP(Module):

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.wte = Embedding(config.vocab_size + 1, config.n_embd, dtype=config.dtype) # token embeddings table
        # +1 in the line above for a special <BLANK> token that gets inserted if encoding a token
        # before the beginning of the input sequence
        self.mlp = Sequential([
            Linear(self.block_size * config.n_embd, config.n_embd2, dtype=config.dtype),
            Tanh(),
            Linear(config.n_embd2, self.vocab_size, dtype=config.dtype)
        ])
        self.mlp[-1].weight.data *= 0.1
        self.mlp[-1].bias.data *= 0.01
        n_params = sum(p.numel() for p in self.parameters())
        print("number of mlp parameters: %d" % (n_params,))
        self.config = config
    
    def parameters(self):
        return list(self.wte.parameters()) + list(self.mlp.parameters())
    
    def grads(self):
        return list(self.wte.grads()) + list(self.mlp.grads())

    def get_block_size(self):
        return self.block_size

    def forward(self, idx):
        # gather the word embeddings of the previous 3 words
        idx_buf = []
        embs = []
        for k in range(self.block_size):
            tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)
            idx_buf.append(idx.unsqueeze(-1))
            embs.append(tok_emb)
            idx = torch.roll(idx, 1, 1)
            idx[:, 0] = self.vocab_size # special <BLANK> token
        # concat all of the embeddings together and pass through an MLP
        x = torch.cat(embs, -1) # (b, t, n_embd * block_size)
        logits = self.mlp(x)
        # backward buffer
        self.idx_buf = torch.cat(idx_buf, -1) # (b, t, t)


        return logits
    
    def backward(self, grad):
        grad = self.mlp.backward(grad)
        # mlp backprop to wte
        b, t, _ = grad.shape # (b, t, n_embd * block_size)
        grad = grad.view(b * t * self.config.block_size, self.config.n_embd) # (b*t*block_size, n_embd)
        wte_weight = self.wte.weight
        wte_grad = torch.zeros_like(wte_weight)
        wte_grad.index_add_(dim=0, index=self.idx_buf.view(-1), source=grad)
        self.wte.weight_grad = wte_grad

# ---------------------------------------- RNN ------------------------------------------------------
class RNN(Module):

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.n_embd = config.n_embd
        self.n_embd2 = config.n_embd2
        self.start = torch.zeros(1, config.n_embd2, dtype=config.dtype) # the starting hidden state
        self.wte = Embedding(config.vocab_size, config.n_embd, dtype=config.dtype) # token embeddings table
        self.Cw = Linear(config.n_embd + config.n_embd2, config.n_embd2, dtype=config.dtype) # rnn cell weight
        self.lm_head = Linear(config.n_embd2, self.vocab_size, dtype=config.dtype)
        self.lm_head.weight.data *= 0.1
        num_params = sum(p.numel() for p in self.parameters())
        print("number of rnn parameters: %d" % (num_params,))
        # grads
        self.start_grad = None
    
    def parameters(self):
        return [self.start] + list(self.wte.parameters()) + list(self.Cw.parameters()) + list(self.lm_head.parameters())
    
    def grads(self):
        return [self.start_grad] + list(self.wte.grads()) + list(self.Cw.grads()) + list(self.lm_head.grads())

    def get_block_size(self):
        return self.block_size

    def forward(self, x):
        b, t = x.size()
        emb = self.wte(x) # (b, t, n_embd)
        # sequentially iterate over the inputs and update the RNN state each tick
        hprev = self.start.expand((b, -1)) # expand out the batch dimension
        hiddens = []
        emb_cat_hprevs = []
        for i in range(t):
            xt = emb[:, i, :] # (b, n_embd)
            emb_i_cat_hprev = torch.cat([xt, hprev], dim=1)
            # --- rnn cell ---
            hi = self.Cw(emb_i_cat_hprev)
            hi = hi.tanh()
            # --------------
            hprev = hi
            hiddens.append(hi)
            emb_cat_hprevs.append(emb_i_cat_hprev)
        # decode the outputs
        hidden = torch.stack(hiddens, 1) # (b, t, n_embd2)
        logits = self.lm_head(hidden)
        # backward buffer
        self.hidden = hidden
        self.emb_cat_hprevs = emb_cat_hprevs
        return logits

    def backward(self, grad):
        hidden, emb_cat_hprevs = self.hidden, self.emb_cat_hprevs
        t = hidden.size(1)
        dhidden = self.lm_head.backward(grad)
        # logits grad to start, wte, Cw grad
        dembs = []
        dCw, dhprev = 0., 0.
        if self.Cw.bias is not None:
            dCw_bias = 0.
        for i in range(t-1, -1, -1):
            # hidden state grad, emb grad
            dhi = dhidden[:, i, :] + dhprev # grad from logits + grad from prev hidden state
            hi = hidden[:, i, :]
            dhi = (1 - hi**2) * dhi # grad of tanh
            demb_i_cat_dhi = dhi @ self.Cw.weight.T
            demb_i, dhprev = demb_i_cat_dhi.tensor_split([self.n_embd,], dim=1)
            dembs.append(demb_i)
            # cell weight grad
            emb_i_cat_hprev = emb_cat_hprevs[i]
            dCw += emb_i_cat_hprev.T @ dhi
            if self.Cw.bias is not None:
                dCw_bias += dhi.sum(dim=0)
        dstart = dhprev.sum(dim=0, keepdim=True)
        demb = torch.stack(dembs[::-1], 1)
        self.wte.backward(demb)
        self.start_grad = dstart
        self.Cw.weight_grad = dCw
        if self.Cw.bias is not None:
            self.Cw.bias_grad = dCw_bias

# ---------------------------------------- GRU ------------------------------------------------------
class GRU(Module):

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.n_embd = config.n_embd
        self.n_embd2 = config.n_embd2
        self.start = torch.zeros(1, config.n_embd2, dtype=config.dtype) # the starting hidden state
        self.wte = Embedding(config.vocab_size, config.n_embd, dtype=config.dtype) # token embeddings table
        self.Cr = Linear(config.n_embd + config.n_embd2, config.n_embd2, dtype=config.dtype)
        self.Cbar = Linear(config.n_embd + config.n_embd2, config.n_embd2, dtype=config.dtype)
        self.Cz = Linear(config.n_embd + config.n_embd2, config.n_embd2, dtype=config.dtype)
        self.lm_head = Linear(config.n_embd2, self.vocab_size, dtype=config.dtype)
        self.lm_head.weight.data *= 0.1
        num_params = sum(p.numel() for p in self.parameters())
        print("number of gru parameters: %d" % (num_params,))
        # grads
        self.start_grad = None
    
    def parameters(self):
        return [self.start] + list(self.wte.parameters()) + list(self.Cr.parameters()) + list(self.Cbar.parameters()) + list(self.Cz.parameters()) + list(self.lm_head.parameters())
    
    def grads(self):
        return [self.start_grad] + list(self.wte.grads()) + list(self.Cr.grads()) + list(self.Cbar.grads()) + list(self.Cz.grads()) + list(self.lm_head.grads())

    def get_block_size(self):
        return self.block_size

    def forward(self, x):
        b, t = x.size()
        emb = self.wte(x) # (b, t, n_embd)
        # sequentially iterate over the inputs and update the RNN state each tick
        hprev = self.start.expand((b, -1)) # expand out the batch dimension
        hiddens = []
        emb_cat_hprevs, emb_cat_hprev_resets, hprevs, hbars, zs, rs = [], [], [], [], [], []
        for i in range(t):
            emb_i = emb[:, i, :] # (b, n_embd)
            # --- gru cell ---
            emb_i_cat_hprev = torch.cat([emb_i, hprev], dim=1)
            ri = self.Cr(emb_i_cat_hprev)
            ri = ri.sigmoid()
            hprev_reset = ri * hprev
            emb_i_cat_hprev_reset = torch.cat([emb_i, hprev_reset], dim=1)
            hbar = self.Cbar(emb_i_cat_hprev_reset)
            hbar = hbar.tanh()
            zi = self.Cz(emb_i_cat_hprev)
            zi = zi.sigmoid()
            hi = (1 - zi) * hprev + zi * hbar
            # backward buffer
            hiddens.append(hi)
            emb_cat_hprevs.append(emb_i_cat_hprev)
            emb_cat_hprev_resets.append(emb_i_cat_hprev_reset)
            hprevs.append(hprev)
            hbars.append(hbar)
            zs.append(zi)
            rs.append(ri)
            # update hprev
            hprev = hi
        # decode the outputs
        hidden = torch.stack(hiddens, 1) # (b, t, n_embd2)
        logits = self.lm_head(hidden)
        # backward buffer
        self.hidden = hidden
        self.emb_cat_hprevs = emb_cat_hprevs
        self.emb_cat_hprev_resets = emb_cat_hprev_resets
        self.hprevs = hprevs
        self.hbars = hbars
        self.zs = zs
        self.rs = rs
        return logits

    def backward(self, dlogits):
        (hidden, emb_cat_hprevs, emb_cat_hprev_resets, hprevs, hbars, zs, rs) = (
            self.hidden, self.emb_cat_hprevs, self.emb_cat_hprev_resets, self.hprevs, self.hbars, self.zs, self.rs
        )
        t = hidden.size(1)
        dhidden = self.lm_head.backward(dlogits)
        # logits grad to start, wte, Cw grad
        dembs = []
        dCr, dCbar, dCz, dhprev = 0., 0., 0., 0.
        if self.Cr.bias is not None:
            dCr_bias = 0.
        if self.Cbar.bias is not None:
            dCbar_bias = 0.
        if self.Cz.bias is not None:
            dCz_bias = 0.
        for i in range(t-1, -1, -1):
            # hidden state grad, emb grad
            dhi = dhidden[:, i, :] + dhprev # grad from logits + grad from prev hidden state
            dhbar = dhi * zs[i]
            dhprev = dhi * (1 - zs[i])
            dzi = dhi * (hbars[i] - hprevs[i])
            dzi = dzi * (1 - zs[i]) * zs[i]
            demb_i_cat_hprev = dzi @ self.Cz.weight.T
            dCz += emb_cat_hprevs[i].T @ dzi
            if self.Cz.bias is not None:
                dCz_bias += dzi.sum(dim=0)

            dhbar = dhbar * (1 - hbars[i]**2)
            demb_i_cat_hprev_reset = dhbar @ self.Cbar.weight.T
            dCbar += emb_cat_hprev_resets[i].T @ dhbar
            if self.Cbar.bias is not None:
                dCbar_bias += dhbar.sum(dim=0)

            demb_i, dhprev_reset = demb_i_cat_hprev_reset.tensor_split([self.n_embd,], dim=1)
            dri = dhprev_reset * hprevs[i]
            dhprev += dhprev_reset * rs[i]
            dri = dri * (1 - rs[i]) * rs[i]
            demb_i_cat_hprev += dri @ self.Cr.weight.T
            dCr += emb_cat_hprevs[i].T @ dri
            if self.Cr.bias is not None:
                dCr_bias += dri.sum(dim=0)
            demb_more, dhprev_more = demb_i_cat_hprev.tensor_split([self.n_embd,], dim=1)
            demb_i += demb_more
            dhprev += dhprev_more
            dembs.append(demb_i)
        dstart = dhprev.sum(dim=0, keepdim=True)
        demb = torch.stack(dembs[::-1], 1)
        self.wte.backward(demb)
        self.start_grad = dstart
        self.Cr.weight_grad = dCr
        if self.Cr.bias is not None:
            self.Cr.bias_grad = dCr_bias
        self.Cbar.weight_grad = dCbar
        if self.Cbar.bias is not None:
            self.Cbar.bias_grad = dCbar_bias
        self.Cz.weight_grad = dCz
        if self.Cz.bias is not None:
            self.Cz.bias_grad = dCz_bias

# ---------------------------------------- Transformer ------------------------------------------------------
class CausalSelfAttention(Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = Linear(config.n_embd, 3 * config.n_embd, dtype=config.dtype)
        # output projection
        self.c_proj = Linear(config.n_embd, config.n_embd, dtype=config.dtype)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.bias = torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
    
    def parameters(self):
        return list(self.c_attn.parameters()) + list(self.c_proj.parameters())
    
    def grads(self):
        return list(self.c_attn.grads()) + list(self.c_proj.grads())

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        qkv = self.c_attn(x)
        q, k ,v  = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att_dot = (q @ k.transpose(-2, -1)) * (1.0 / k.size(-1)**0.5)
        att_mask = att_dot.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = att_mask.softmax(dim=-1)
        y_trans = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y_preproj = y_trans.transpose(1, 2).contiguous().view(B, T, C) # (B, nh, T, hs) -> (B, T, nh*hs)
        y = self.c_proj(y_preproj) # (B, T, C) -> (B, T, C)
        # backward buffer
        self.y_preproj = y_preproj
        self.att = att
        self.q = q
        self.k = k
        self.v = v
        self.x = x
        return y

    def backward(self, dy):
        B, T, C = dy.size()
        (y_preproj, att, q, k, v, x) = (
            self.y_preproj, self.att, self.q, self.k, self.v, self.x
        )
        dC_proj = (y_preproj.transpose(-2, -1) @ dy).sum(dim=0)
        self.c_proj.weight_grad = dC_proj
        if self.c_proj.bias is not None:
            dC_proj_bias = dy.sum(dim=[0, 1])
            self.c_proj.bias_grad = dC_proj_bias
        dy_preproj = dy @ self.c_proj.weight.T
        dy_trans = dy_preproj.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        datt = dy_trans @ v.transpose(-2, -1)
        dv = att.transpose(-2, -1) @ dy_trans
        datt_mask = att * (datt - (att * datt).sum(dim=-1, keepdim=True))
        datt_dot = datt_mask.masked_fill(self.bias[:,:,:T,:T] == 0, 0)
        dq = (datt_dot @ k) * (1.0 / k.size(-1)**0.5)
        dk = (datt_dot.transpose(-2, -1) @ q) * (1.0 / k.size(-1)**0.5)
        dq = dq.transpose(1, 2).reshape(B, T, C)
        dk = dk.transpose(1, 2).reshape(B, T, C)
        dv = dv.transpose(1, 2).reshape(B, T, C)
        dqkv = torch.cat([dq, dk, dv], dim=2)
        dC_atten = (x.transpose(-2, -1) @ dqkv).sum(dim=0)
        self.c_attn.weight_grad = dC_atten
        if self.c_attn.bias is not None:
            dC_atten_bias = dqkv.sum(dim=[0, 1])
            self.c_attn.bias_grad = dC_atten_bias
        dx = dqkv @ self.c_attn.weight.T
        return dx

class Block(Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, dtype=config.dtype)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, dtype=config.dtype)
        self.mlp = Sequential([
            Linear(config.n_embd, 4 * config.n_embd, dtype=config.dtype),
            GELU(),
            Linear(4 * config.n_embd, config.n_embd, dtype=config.dtype),
        ])
    
    def parameters(self):
        return list(self.ln_1.parameters()) + list(self.attn.parameters()) + list(self.ln_2.parameters()) + list(self.mlp.parameters())
    
    def grads(self):
        return list(self.ln_1.grads()) + list(self.attn.grads()) + list(self.ln_2.grads()) + list(self.mlp.grads())

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
    def backward(self, dx):
        dx = dx + self.ln_2.backward(self.mlp.backward(dx))
        dx = dx + self.ln_1.backward(self.attn.backward(dx))
        return dx

class Transformer(Module):
    """ Transformer Language Model, exactly as seen in GPT-2 """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.wte = Embedding(config.vocab_size, config.n_embd, dtype=config.dtype)
        self.wpe = Embedding(config.block_size, config.n_embd, dtype=config.dtype)
        self.transformer = Sequential([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, dtype=config.dtype)
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False, dtype=config.dtype)
        self.lm_head.weight.data *= 0.1

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"number of transformer parameters: {n_params}")
    
    def parameters(self):
        return list(self.wte.parameters()) + list(self.wpe.parameters()) + list(self.transformer.parameters()) + list(self.ln_f.parameters()) + list(self.lm_head.parameters())
    
    def grads(self):
        return list(self.wte.grads()) + list(self.wpe.grads()) + list(self.transformer.grads()) + list(self.ln_f.grads()) + list(self.lm_head.grads())

    def get_block_size(self):
        return self.block_size

    def forward(self, idx):
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer:
            x = block(x)
        h = self.ln_f(x)
        logits = self.lm_head(h)

        return logits
    
    def backward(self, dlogits):
        dh = self.lm_head.backward(dlogits)
        dh = self.ln_f.backward(dh)
        dx = self.transformer.backward(dh)
        dtok_emb = dx
        self.wte.backward(dtok_emb)
        dpos_emb = dx.sum(dim=0, keepdim=True)
        self.wpe.backward(dpos_emb)
