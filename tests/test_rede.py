import random
import unittest

import numpy as np

from agente.dqn import AgenteDQN
from agente.rede_numpy import MLP


class TestRede(unittest.TestCase):
    def test_gradiente_vs_diferencas_finitas(self):
        """Valida o backprop manual: gradiente analítico ≈ numérico."""
        rng = np.random.default_rng(1)
        net = MLP([5, 7, 6, 3], seed=2)
        x = rng.normal(size=(4, 5))
        alvo = rng.normal(size=(4, 3))

        def perda():
            return 0.5 * np.sum((net.forward(x) - alvo) ** 2)

        saida = net.forward(x, guardar=True)
        dW, db = net.backward(saida - alvo)
        eps = 1e-6
        for l in range(len(net.W)):
            for params, grads in ((net.W[l], dW[l]), (net.b[l], db[l])):
                it = np.nditer(params, flags=["multi_index"])
                for _ in it:
                    idx = it.multi_index
                    orig = params[idx]
                    params[idx] = orig + eps; mais = perda()
                    params[idx] = orig - eps; menos = perda()
                    params[idx] = orig
                    num = (mais - menos) / (2 * eps)
                    self.assertAlmostEqual(grads[idx], num, places=4)

    def test_adam_reduz_perda(self):
        rng = np.random.default_rng(0)
        net = MLP([3, 16, 1], seed=0)
        x = rng.normal(size=(64, 3)); y = (x[:, :1] * 2 - x[:, 1:2])
        perdas = []
        for _ in range(300):
            s = net.forward(x, guardar=True)
            perdas.append(float(np.mean((s - y) ** 2)))
            dW, db = net.backward(2 * (s - y) / len(x))
            net.passo_adam(dW, db, lr=1e-2)
        self.assertLess(perdas[-1], perdas[0] * 0.2)

    def test_dqn_aprende_bandit(self):
        """Num problema trivial (só a ação 2 rende), o DQN deve escolhê-la."""
        ag = AgenteDQN(4, 4, ocultas=(16,), eps_passos=300, batch=16, seed=3, sincronizar=20)
        r = random.Random(0)
        for _ in range(1200):
            s = [r.random() for _ in range(4)]
            a = ag.agir(s)
            ag.observar(s, a, 1.0 if a == 2 else 0.0, s, True)
        escolhas = [ag.agir([r.random() for _ in range(4)], explorar=False) for _ in range(30)]
        self.assertGreaterEqual(escolhas.count(2), 28)

    def test_salvar_carregar(self):
        import os, tempfile
        a = AgenteDQN(5, 3, seed=1); b = AgenteDQN(5, 3, seed=9)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "w.json"); a.salvar(p); b.carregar(p)
        x = [0.1, 0.2, 0.3, 0.4, 0.5]
        np.testing.assert_allclose(a.valores_q(x), b.valores_q(x))


if __name__ == "__main__":
    unittest.main()
