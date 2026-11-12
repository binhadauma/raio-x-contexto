"""Q-learning tabular (linha de base de aprendizado).

Discretiza o percepto em poucas faixas -> chave de dicionário Python.
Serve para mostrar, na defesa, POR QUE a aproximação de função (DQN) ajuda:
a tabela precisa de muitos episódios para cobrir o espaço de estados.

    Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') - Q(s,a) ]
"""
import random


def discretizar(obs, n_acoes=6):
    """(faixa de distância, última ação, faixa do último CTS) — 3 faixas cada."""
    dist, _, ultimo_cts = obs[0], obs[1], obs[2]
    f_dist = 0 if dist < 0.15 else (1 if dist < 0.35 else 2)
    f_cts = 0 if ultimo_cts < 0.4 else (1 if ultimo_cts < 0.75 else 2)
    ultima = next((i for i in range(n_acoes) if obs[5 + i] == 1.0), -1)
    return (f_dist, ultima, f_cts)


class AgenteQTabular:
    nome = "q_tabular"

    def __init__(self, n_acoes, alpha=0.2, gamma=0.95, eps_ini=1.0, eps_fim=0.05,
                 eps_passos=9000, seed=0):
        self.n_acoes, self.alpha, self.gamma = n_acoes, alpha, gamma
        self.eps_ini, self.eps_fim, self.eps_passos = eps_ini, eps_fim, eps_passos
        self.tabela = {}  # estado -> lista de Q por ação
        self.rng = random.Random(seed)
        self.passos = 0

    @property
    def epsilon(self):
        f = min(1.0, self.passos / self.eps_passos)
        return self.eps_ini + f * (self.eps_fim - self.eps_ini)

    def _q(self, chave):
        return self.tabela.setdefault(chave, [0.0] * self.n_acoes)

    def valores_q(self, obs):
        return self._q(discretizar(obs, self.n_acoes))

    def agir(self, obs, explorar=True):
        if explorar and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_acoes)
        q = self.valores_q(obs)
        melhor = max(q)
        return self.rng.choice([i for i, v in enumerate(q) if v == melhor])  # desempate

    def observar(self, s, a, r, s2, fim, treinar=True):
        self.passos += 1
        if not treinar:
            return
        q = self._q(discretizar(s, self.n_acoes))
        alvo = r + (0.0 if fim else self.gamma * max(self._q(discretizar(s2, self.n_acoes))))
        q[a] += self.alpha * (alvo - q[a])
