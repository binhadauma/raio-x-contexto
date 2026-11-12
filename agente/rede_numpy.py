"""MLP escrita à mão em NumPy: forward, backward (backprop manual) e Adam.

NumPy entra só como álgebra de vetores/matrizes (permitido pela Fase 3).
Não há autograd: os gradientes são derivados à mão e verificados por
diferenças finitas em tests/test_rede.py.

Arquitetura:  entrada -> [Linear -> ReLU] x L -> Linear (saída linear, valores-Q)

Derivadas (para a camada l):
    z_l = a_{l-1} W_l + b_l        a_l = relu(z_l)
    dL/dz_L = dL/dsaida                         (saída linear)
    dL/dW_l = a_{l-1}^T dL/dz_l                 dL/db_l = soma_batch(dL/dz_l)
    dL/da_{l-1} = dL/dz_l W_l^T                 dL/dz_{l-1} = dL/da_{l-1} * 1[z_{l-1} > 0]
"""
import numpy as np


class MLP:
    def __init__(self, dims, seed=0):
        """dims = [entrada, oculta1, ..., saida]. Inicialização He."""
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, np.sqrt(2.0 / dims[i]), (dims[i], dims[i + 1]))
                  for i in range(len(dims) - 1)]
        self.b = [np.zeros(dims[i + 1]) for i in range(len(dims) - 1)]
        # estado do Adam
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    # ---------------------------------------------------------------- forward
    def forward(self, x, guardar=False):
        """x: (batch, entrada) -> (batch, saida). Com guardar=True mantém cache."""
        a = np.atleast_2d(x)
        cache = [a]  # ativações a_0..a_L ; z's guardados à parte
        zs = []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            zs.append(z)
            a = np.maximum(z, 0.0) if i < len(self.W) - 1 else z
            cache.append(a)
        if guardar:
            self._cache, self._zs = cache, zs
        return a

    # --------------------------------------------------------------- backward
    def backward(self, grad_saida):
        """grad_saida = dL/dsaida (batch, saida). Devolve (dW, db) por camada."""
        dW = [None] * len(self.W)
        db = [None] * len(self.b)
        dz = grad_saida
        for l in reversed(range(len(self.W))):
            dW[l] = self._cache[l].T @ dz
            db[l] = dz.sum(axis=0)
            if l > 0:
                da = dz @ self.W[l].T
                dz = da * (self._zs[l - 1] > 0)
        return dW, db

    # ------------------------------------------------------------------- Adam
    def passo_adam(self, dW, db, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, clip=5.0):
        """Atualiza os pesos com Adam (com recorte de norma global do gradiente)."""
        norma = np.sqrt(sum((g ** 2).sum() for g in dW) + sum((g ** 2).sum() for g in db))
        escala = min(1.0, clip / (norma + 1e-12))
        self.t += 1
        for i in range(len(self.W)):
            for p, g, m, v in ((self.W, dW, self.mW, self.vW), (self.b, db, self.mb, self.vb)):
                gi = g[i] * escala
                m[i] = b1 * m[i] + (1 - b1) * gi
                v[i] = b2 * v[i] + (1 - b2) * gi * gi
                mh = m[i] / (1 - b1 ** self.t)
                vh = v[i] / (1 - b2 ** self.t)
                p[i] -= lr * mh / (np.sqrt(vh) + eps)

    # ------------------------------------------------------- cópia / persistência
    def copiar_de(self, outra, tau=1.0):
        """Atualização da rede-alvo: tau=1 cópia dura; tau<1 média móvel (soft)."""
        for i in range(len(self.W)):
            self.W[i] = tau * outra.W[i] + (1 - tau) * self.W[i]
            self.b[i] = tau * outra.b[i] + (1 - tau) * self.b[i]

    def para_dict(self):
        return {"W": [w.tolist() for w in self.W], "b": [b.tolist() for b in self.b]}

    def de_dict(self, d):
        self.W = [np.array(w) for w in d["W"]]
        self.b = [np.array(b) for b in d["b"]]
