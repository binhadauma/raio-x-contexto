import unittest

from avaliador import metricas as M
from avaliador.tokens import contar_tokens


class TestAvaliador(unittest.TestCase):
    def test_factual(self):
        self.assertEqual(M.avaliar_factual("O código secreto é 7431.", "7431"), 1.0)
        self.assertEqual(M.avaliar_factual("Era 7431, ou talvez 9999?", "7431"), 0.5)
        self.assertEqual(M.avaliar_factual("Acho que era 1234.", "7431"), 0.0)
        self.assertEqual(M.avaliar_factual("Não lembro.", "7431"), 0.0)
        # 5 dígitos não conta como o código de 4
        self.assertEqual(M.avaliar_factual("O número 74310 foi citado.", "7431"), 0.0)

    def test_restricao(self):
        ok = "O céu tem a cor do mar. Isso vem da luz."
        self.assertEqual(M.avaliar_restricao(ok, "azul", 2), 1.0)
        self.assertEqual(M.avaliar_restricao("O céu é azul. Sim.", "azul", 2), 0.5)
        self.assertEqual(M.avaliar_restricao("A. B. C.", "azul", 2), 0.5)
        self.assertEqual(M.avaliar_restricao("É AZUL. Muito. Mesmo.", "azul", 2), 0.0)
        # acento e caixa são normalizados
        self.assertEqual(M.avaliar_restricao("Muito RÁPIDO.", "rápido", 2), 0.5)

    def test_contar_frases(self):
        self.assertEqual(M.contar_frases("Um. Dois! Três?"), 3)
        self.assertEqual(M.contar_frases("Sem ponto final"), 1)
        self.assertEqual(M.contar_frases(""), 0)

    def test_cosseno(self):
        a = M.vetor_tf("ajudar a planejar a horta comunitária")
        self.assertAlmostEqual(M.cosseno(a, a), 1.0)
        b = M.vetor_tf("futebol e campeonato brasileiro")
        self.assertEqual(M.cosseno(a, b), 0.0)
        self.assertEqual(M.cosseno({}, a), 0.0)

    def test_coerencia_satura_em_1(self):
        obj = "ajudar a planejar a horta comunitária do bairro"
        self.assertEqual(M.avaliar_coerencia("Meu objetivo é " + obj + ".", obj), 1.0)
        self.assertEqual(M.avaliar_coerencia("Posso falar de qualquer coisa.", obj), 0.0)

    def test_cts_pesos_somam_um(self):
        self.assertAlmostEqual(M.calcular_cts(1, 1, 1), 1.0)
        self.assertAlmostEqual(M.calcular_cts(0, 0, 0), 0.0)
        self.assertAlmostEqual(M.calcular_cts(1, 0, 0), 0.4)

    def test_ruptura(self):
        self.assertEqual(M.ponto_ruptura([0.9, 0.4, 0.8, 0.3, 0.2]), 3)  # queda isolada ignorada
        self.assertIsNone(M.ponto_ruptura([0.9, 0.8, 0.7]))
        self.assertEqual(M.ponto_ruptura([0.3, 0.2], consecutivas=2), 0)
        self.assertEqual(M.ponto_ruptura([0.9, 0.4], consecutivas=1), 1)

    def test_tokens(self):
        self.assertEqual(contar_tokens(""), 0)
        self.assertGreater(contar_tokens("uma frase com cinco palavras."), 5)
        self.assertGreater(contar_tokens("a " * 100), contar_tokens("a " * 10))

    def test_ic_bootstrap_contem_media(self):
        xs = [4, 5, 5, 6, 7, 5, 4, 6]
        lo, hi = M.ic_bootstrap(xs)
        self.assertLessEqual(lo, M.media(xs))
        self.assertGreaterEqual(hi, M.media(xs))


if __name__ == "__main__":
    unittest.main()
