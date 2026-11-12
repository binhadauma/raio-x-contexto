import random
import unittest

from agente.baselines import criar_politica
from agente.laco import rodar_episodio
from agente.q_learning import AgenteQTabular
from ambiente.ambiente_teste import DIM_OBS, AmbienteTeste
from ambiente.geradores import ACOES, N_ACOES, gerar_ancora, gerar_distrator
from ambiente.llm_simulada import LLMSimulada
from avaliador import metricas as M


class TestAmbiente(unittest.TestCase):
    def test_geradores_para_todas_as_acoes(self):
        rng = random.Random(0)
        for a in range(N_ACOES):
            self.assertGreater(len(gerar_distrator(a, rng)), 20)
        anc = gerar_ancora(rng)
        self.assertIn(anc.codigo, anc.texto())
        self.assertIn(anc.palavra_proibida, anc.texto())

    def test_obs_e_fim_do_episodio(self):
        env = AmbienteTeste(LLMSimulada(seed=1), n_fases=5, seed=1)
        obs = env.reset()
        self.assertEqual(len(obs), DIM_OBS)
        for i in range(5):
            obs, r, feito, ev = env.step(i % N_ACOES)
            self.assertEqual(len(obs), DIM_OBS)
            self.assertEqual(feito, i == 4)
            self.assertTrue(0.0 <= ev["cts"] <= 1.0)

    def test_sonda_nao_entra_no_historico(self):
        env = AmbienteTeste(LLMSimulada(seed=1), n_fases=3, seed=1)
        env.reset()
        n0 = len(env.historico)
        env.step(0)
        self.assertEqual(len(env.historico), n0 + 2)  # só distrator + resposta

    def test_reprodutivel_com_semente(self):
        def roda():
            env = AmbienteTeste(LLMSimulada(seed=5), n_fases=6, seed=5)
            r, _ = rodar_episodio(env, criar_politica("ciclica"), seed=11)
            return r["cts_por_fase"]
        self.assertEqual(roda(), roda())

    def test_controle_neutro_nao_rompe_e_conflito_rompe(self):
        def media_ruptura(nome):
            env = AmbienteTeste(LLMSimulada(seed=3), n_fases=12, seed=3)
            pol = criar_politica(nome)
            return [rodar_episodio(env, pol, seed=100 + i)[0] for i in range(40)]
        neutro = media_ruptura("fixa_neutro")
        conflito = media_ruptura("fixa_conflito")
        self.assertLessEqual(sum(r["ruptura_fase"] is not None for r in neutro), 4)
        self.assertGreaterEqual(sum(r["ruptura_fase"] is not None for r in conflito), 38)
        self.assertGreater(M.media(r["cts_medio"] for r in neutro),
                           M.media(r["cts_medio"] for r in conflito) + 0.3)

    def test_politicas_rodam(self):
        env = AmbienteTeste(LLMSimulada(seed=2), n_fases=4, seed=2)
        for pol in (criar_politica("aleatoria"), AgenteQTabular(N_ACOES)):
            r, total = rodar_episodio(env, pol, treinar=True, explorar=True)
            self.assertEqual(len(r["cts_por_fase"]), 4)

    def test_hook_injetar_nao_exigido(self):
        class LLMMinima:                       # sem `injetar`: como uma LLM real
            nome = "minima"
            def reiniciar(self): pass
            def responder(self, m, temperatura=0.7): return "ok"
        env = AmbienteTeste(LLMMinima(), n_fases=2, seed=0)
        env.reset(); env.step(1); env.step(4)
        self.assertEqual(len(env.serie_cts), 2)


if __name__ == "__main__":
    unittest.main()
