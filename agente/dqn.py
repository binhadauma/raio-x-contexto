"""Deep Q-Network do zero (NumPy): replay buffer, rede-alvo, ε-greedy, Huber.

Algoritmo (Mnih et al., 2015), por passo de treino:
    1. amostra um minibatch (s, a, r, s', fim) do replay buffer
    2. alvo  y = r + γ · max_a' Q_alvo(s', a') · (1 - fim)
    3. perda Huber( Q(s,a) , y )  — só a ação tomada recebe gradiente
    4. backprop manual + Adam; a cada `sincronizar` passos Q_alvo ← Q

Ações = tipos de distrator (6). Estado = vetor de 17 números (ver ambiente).
"""
import json
import random
from collections import deque

import numpy as np

from agente.rede_numpy import MLP


class BufferReplay:
    def __init__(self, capacidade=20000, seed=0):
        self.dados = deque(maxlen=capacidade)
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.dados)

    def adicionar(self, s, a, r, s2, fim):
        self.dados.append((s, a, r, s2, fim))

    def amostrar(self, n):
        lote = self.rng.sample(self.dados, n)
        s, a, r, s2, f = zip(*lote)
        return (np.array(s, float), np.array(a, int), np.array(r, float),
                np.array(s2, float), np.array(f, float))


class AgenteDQN:
    nome = "dqn"

    def __init__(self, dim_obs, n_acoes, ocultas=(64, 64), gamma=0.95, lr=1e-3,
                 batch=64, sincronizar=200, eps_ini=1.0, eps_fim=0.05,
                 eps_passos=9000, seed=0):
        dims = [dim_obs, *ocultas, n_acoes]
        self.q = MLP(dims, seed)
        self.alvo = MLP(dims, seed)
        self.alvo.copiar_de(self.q)
        self.n_acoes, self.gamma, self.lr = n_acoes, gamma, lr
        self.batch, self.sincronizar = batch, sincronizar
        self.eps_ini, self.eps_fim, self.eps_passos = eps_ini, eps_fim, eps_passos
        self.buffer = BufferReplay(seed=seed)
        self.rng = random.Random(seed)
        self.passos = 0
        self.ultima_perda = None

    # ------------------------------------------------------------ política
    @property
    def epsilon(self):
        f = min(1.0, self.passos / self.eps_passos)
        return self.eps_ini + f * (self.eps_fim - self.eps_ini)

    def valores_q(self, obs):
        return self.q.forward(np.array(obs, float))[0]

    def agir(self, obs, explorar=True):
        """ε-greedy (explorar=True) ou gulosa (explorar=False, usada na avaliação)."""
        if explorar and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_acoes)
        return int(np.argmax(self.valores_q(obs)))

    # ------------------------------------------------------------ aprendizado
    def observar(self, s, a, r, s2, fim, treinar=True):
        self.buffer.adicionar(s, a, r, s2, fim)
        self.passos += 1
        if treinar and len(self.buffer) >= self.batch:
            self._treinar_passo()
        if self.passos % self.sincronizar == 0:
            self.alvo.copiar_de(self.q)

    def _treinar_passo(self):
        s, a, r, s2, fim = self.buffer.amostrar(self.batch)
        y = r + self.gamma * self.alvo.forward(s2).max(axis=1) * (1.0 - fim)
        q_todos = self.q.forward(s, guardar=True)
        idx = np.arange(len(a))
        erro = q_todos[idx, a] - y
        # derivada da Huber(δ=1): erro dentro de [-1,1], senão sinal(erro)
        d = np.clip(erro, -1.0, 1.0) / len(a)
        grad = np.zeros_like(q_todos)
        grad[idx, a] = d
        dW, db = self.q.backward(grad)
        self.q.passo_adam(dW, db, lr=self.lr)
        self.ultima_perda = float(np.mean(np.where(np.abs(erro) <= 1, 0.5 * erro ** 2,
                                                   np.abs(erro) - 0.5)))

    # ------------------------------------------------------------ persistência
    def salvar(self, caminho):
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump({"q": self.q.para_dict(), "passos": self.passos}, f)

    def carregar(self, caminho):
        with open(caminho, encoding="utf-8") as f:
            d = json.load(f)
        self.q.de_dict(d["q"])
        self.alvo.copiar_de(self.q)
        self.passos = d.get("passos", 0)
