"""Políticas de referência para comparar com o agente aprendiz.

Todas seguem a mesma interface do agente: `agir(obs, explorar)` e `observar(...)`.
"""
import random

from ambiente.geradores import ACOES, N_ACOES


class PoliticaAleatoria:
    nome = "aleatoria"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def agir(self, obs, explorar=True):
        return self.rng.randrange(N_ACOES)

    def observar(self, *args, **kw):
        pass


class PoliticaFixa:
    """Sempre o mesmo distrator (ex.: 'neutro' = controle sem estresse)."""

    def __init__(self, acao):
        self.acao = ACOES.index(acao) if isinstance(acao, str) else acao
        self.nome = f"fixa_{ACOES[self.acao]}"

    def agir(self, obs, explorar=True):
        return self.acao

    def observar(self, *args, **kw):
        pass


class PoliticaCiclica:
    """Percorre todos os tipos em ordem (cobertura sistemática, sem aprender)."""
    nome = "ciclica"

    def __init__(self):
        self.i = 0

    def agir(self, obs, explorar=True):
        a = self.i % N_ACOES
        self.i += 1
        return a

    def observar(self, *args, **kw):
        pass


def criar_politica(nome, agente=None, seed=0):
    """Fábrica usada pelo servidor/experimentos."""
    if nome == "dqn" or nome == "q_tabular":
        if agente is None:
            raise ValueError("política aprendida exige um agente treinado")
        return agente
    if nome == "aleatoria":
        return PoliticaAleatoria(seed)
    if nome == "ciclica":
        return PoliticaCiclica()
    if nome.startswith("fixa_"):
        return PoliticaFixa(nome[5:])
    raise ValueError(f"política desconhecida: {nome}")
