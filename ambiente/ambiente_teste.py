"""Ambiente do agente-testador (interface estilo Gym: reset / step).

Um episódio = uma conversa. Em cada passo (fase) o agente escolhe o TIPO de
distrator; o ambiente o injeta, a LLM responde, e então são feitas 3 sondas
"em bifurcação" (fork): histórico + sonda -> resposta, SEM gravar a sonda no
histórico. Isso evita que a própria sonda reforce a memória da âncora.

Percepto (vetor `obs`, só o que o agente PODE observar -> parcialmente observável):
    [0]      distância em tokens desde a âncora (normalizada)
    [1]      fração do episódio já percorrida
    [2]      último CTS
    [3]      média dos CTS até agora
    [4]      variação do CTS (último - penúltimo)
    [5:11]   one-hot da última ação
    [11:17]  one-hot da penúltima ação (o agente conhece as próprias ações)
    [17:23]  frequência de uso de cada ação neste episódio
O agente NUNCA vê o estado interno da LLM (nem a interferência da simulada).

Recompensa (agente adversarial, quer encontrar o ponto de ruptura):
    r = (1 - CTS) + bônus_ruptura   (bônus = 2 * fração de fases restantes,
                                     pago uma vez, na 1ª ruptura)
"""
import random

from ambiente.geradores import (ACOES, N_ACOES, gerar_ancora, gerar_distrator,
                                gerar_sondas)
from avaliador import metricas as M
from avaliador.tokens import contar_tokens, contar_tokens_mensagens

DIM_OBS = 5 + 3 * N_ACOES
MAX_TOKENS_NORM = 1500.0  # normalização da distância


class AmbienteTeste:
    def __init__(self, llm, n_fases=12, temperatura=0.7, seed=None):
        self.llm = llm
        self.n_fases = n_fases
        self.temperatura = temperatura
        self.rng = random.Random(seed)
        # sem reset() aqui: quem usa chama reset() (evita uma chamada extra à LLM real)

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        self.llm.reiniciar()
        self.ancora = gerar_ancora(self.rng)
        self.sondas = gerar_sondas(self.ancora)
        self.fase = 0
        self.historico = [{"role": "user", "content": self.ancora.texto()}]
        self.historico.append({"role": "assistant",
                               "content": self.llm.responder(self.historico, self.temperatura)})
        self.tokens_ancora_fim = contar_tokens_mensagens(self.historico)
        self.serie_cts = []
        self.eventos = []
        self.contagem = [0] * N_ACOES
        self.ultima_acao = None
        self.penultima_acao = None
        self.ruptura_paga = False
        return self._obs()

    # ------------------------------------------------------------------- obs
    @property
    def distancia_tokens(self):
        return contar_tokens_mensagens(self.historico) - self.tokens_ancora_fim

    def _obs(self):
        cts = self.serie_cts
        ultimo = cts[-1] if cts else 1.0
        penultimo = cts[-2] if len(cts) > 1 else ultimo
        onehot = [0.0] * N_ACOES
        if self.ultima_acao is not None:
            onehot[self.ultima_acao] = 1.0
        onehot2 = [0.0] * N_ACOES
        if self.penultima_acao is not None:
            onehot2[self.penultima_acao] = 1.0
        total = max(1, sum(self.contagem))
        return ([min(self.distancia_tokens / MAX_TOKENS_NORM, 2.0),
                 self.fase / self.n_fases,
                 ultimo,
                 M.media(cts) if cts else 1.0,
                 ultimo - penultimo]
                + onehot + onehot2 + [c / total for c in self.contagem])

    # ------------------------------------------------------------------ passo
    def _aplicar_distrator(self, acao):
        texto = gerar_distrator(acao, self.rng)
        n_tok = contar_tokens(texto)
        # gancho opcional: só a LLM simulada implementa `injetar`
        if hasattr(self.llm, "injetar"):
            self.llm.injetar(acao, n_tok)
        self.historico.append({"role": "user", "content": texto})
        resposta = self.llm.responder(self.historico, self.temperatura)
        self.historico.append({"role": "assistant", "content": resposta})
        return texto, resposta, n_tok

    def _sondar(self):
        """Faz as 3 sondas em fork e devolve (respostas, notas)."""
        respostas, notas = {}, {}
        for nome, pergunta in self.sondas.items():
            fork = self.historico + [{"role": "user", "content": pergunta}]
            respostas[nome] = self.llm.responder(fork, self.temperatura)
        a = self.ancora
        notas["factual"] = M.avaliar_factual(respostas["factual"], a.codigo)
        notas["restricao"] = M.avaliar_restricao(respostas["restricao"],
                                                 a.palavra_proibida, a.max_frases)
        notas["coerencia"] = M.avaliar_coerencia(respostas["coerencia"], a.objetivo)
        return respostas, notas

    def step(self, acao: int):
        assert 0 <= acao < N_ACOES, "ação inválida"
        texto, resposta, n_tok = self._aplicar_distrator(acao)
        respostas, notas = self._sondar()
        cts = M.calcular_cts(notas["factual"], notas["restricao"], notas["coerencia"])
        self.serie_cts.append(cts)
        self.contagem[acao] += 1
        self.penultima_acao = self.ultima_acao
        self.ultima_acao = acao
        self.fase += 1

        ruptura = M.ponto_ruptura(self.serie_cts)
        recompensa = 1.0 - cts
        if ruptura is not None and not self.ruptura_paga:
            self.ruptura_paga = True
            recompensa += 2.0 * (1.0 - self.fase / self.n_fases)

        evento = {
            "fase": self.fase, "acao": acao, "acao_nome": ACOES[acao],
            "distrator": texto, "resposta_distrator": resposta,
            "sondas": self.sondas, "respostas": respostas, "notas": notas,
            "cts": cts, "tokens": self.distancia_tokens, "tokens_distrator": n_tok,
            "ruptura_fase": (ruptura + 1) if ruptura is not None else None,
            "recompensa": recompensa,
        }
        self.eventos.append(evento)
        feito = self.fase >= self.n_fases
        return self._obs(), recompensa, feito, evento

    # ---------------------------------------------------------------- resumo
    def resumo(self):
        r = M.ponto_ruptura(self.serie_cts)
        return {
            "cts_por_fase": self.serie_cts,
            "ruptura_fase": (r + 1) if r is not None else None,
            "ruptura_tokens": self.eventos[r]["tokens"] if r is not None else None,
            "cts_medio": M.media(self.serie_cts),
            "tokens_finais": self.distancia_tokens,
        }
