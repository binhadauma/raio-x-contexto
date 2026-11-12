"""LLM simulada: ambiente paramétrico para treinar o agente e plano B da defesa.

ATENÇÃO (declarar na defesa): este NÃO é um modelo de linguagem real nem uma
medida empírica de nenhuma LLM. É um modelo paramétrico de esquecimento, com
parâmetros escolhidos por nós, usado para (1) treinar o agente muito rápido e
(2) validar a mecânica do pipeline. Conclusões sobre LLMs reais só valem para
as execuções com `llm_http.py`.

Como funciona
-------------
A "memória" da âncora decai com a interferência acumulada I:

    p_fato   = exp(-I / TAU_FATO)
    p_regra  = exp(-I / TAU_REGRA)      (regras de formato decaem mais rápido)
    p_obj    = exp(-I / TAU_OBJ)

Cada distrator soma interferência  ΔI = w[tipo] * habituação * sinergia * sqrt(tokens/50):
  * habituação  : repetir o mesmo tipo rende menos (0,75 por repetição recente)
  * sinergia    : certas sequências somam mais (ex.: anáfora -> conflito)
Assim a ORDEM das ações importa, e o agente tem algo real para aprender.

A resposta é TEXTO (não um booleano): o avaliador não enxerga o estado oculto,
apenas lê a saída, exatamente como faria com uma LLM de verdade.
"""
import math
import random
import re

from ambiente.geradores import ACOES, RESTRICOES

# peso base de interferência por tipo de distrator (ordem de ACOES)
PESOS = [0.12, 0.55, 0.65, 0.60, 1.10, 0.40]
HABITUACAO = 0.75
# SINERGIA[anterior][atual] = acréscimo multiplicativo
SINERGIA = {
    (1, 2): 0.40,  # garden-path -> aninhamento
    (2, 3): 0.40,  # aninhamento -> anáfora
    (3, 4): 0.60,  # anáfora -> conflito
    (5, 4): 0.50,  # fato longo -> conflito
}
TAU_FATO = 5.0
TAU_REGRA = 3.4
TAU_OBJ = 6.5

# textos de resposta à sonda de restrição: (cumpre as regras, viola as regras)
_RESPOSTAS_RESTRICAO = {
    "azul": ("O céu de dia limpo tem a cor do mar profundo. Isso ocorre pelo espalhamento da luz.",
             "O céu é azul porque a luz azul se espalha mais. Isso se chama espalhamento de Rayleigh. "
             "Ao entardecer o azul dá lugar ao laranja. Por isso o céu azul muda de cor."),
    "rápido": ("O guepardo acelera com passos longos e precisos. Em segundos alcança a presa.",
               "O guepardo é o animal mais rápido da terra. Ele corre rápido por poucos segundos. "
               "Depois precisa descansar bastante. Seu corpo é feito para correr rápido."),
    "bonito": ("O sol desce e pinta o horizonte de dourado. As ondas refletem o brilho até sumir.",
               "É um pôr do sol muito bonito. O céu fica laranja e rosa. "
               "As ondas refletem a luz do sol. É um momento bonito para ver com calma."),
    "grande": ("A baleia desliza devagar, com um corpo imenso e escuro. Seu sopro sobe até vários metros.",
               "A baleia-franca é um animal grande e calmo. Ela nada perto da costa para criar os filhotes. "
               "Seu corpo grande tem calosidades na cabeça. Vê-la é uma experiência incrível."),
}
_ACKS = [
    "Entendi, obrigado por explicar.",
    "Certo, ficou claro para mim.",
    "Interessante. Posso continuar quando você quiser.",
    "Ok, anotado.",
]


class LLMSimulada:
    nome = "simulada"

    def __init__(self, seed=None, tau_escala=1.0):
        self.rng = random.Random(seed)
        self.tau_escala = tau_escala
        self.reiniciar()

    # ----------------------------------------------------------- ciclo de vida
    def reiniciar(self):
        self.interferencia = 0.0
        self.historico_acoes = []
        self.contaminacao = 0.0  # quanto a LLM "acreditou" num distrator conflitante

    def injetar(self, acao: int, n_tokens: int):
        """Gancho exclusivo do simulador: o ambiente informa o tipo/tamanho do
        estímulo (numa LLM real esse efeito seria desconhecido ao agente)."""
        h = HABITUACAO ** sum(1 for a in self.historico_acoes[-2:] if a == acao)
        sin = 1.0
        if self.historico_acoes:
            sin += SINERGIA.get((self.historico_acoes[-1], acao), 0.0)
        delta = PESOS[acao] * h * sin * math.sqrt(max(n_tokens, 1) / 50.0)
        self.interferencia += delta
        if ACOES[acao] == "conflito":
            self.contaminacao = min(1.0, self.contaminacao + 0.35)
        self.historico_acoes.append(acao)

    # ----------------------------------------------------------------- helpers
    def _ruido(self, p, temperatura):
        sigma = 0.03 + 0.07 * max(0.0, temperatura)
        return min(1.0, max(0.0, p + self.rng.gauss(0.0, sigma)))

    def _lembra(self, tau, temperatura):
        p = math.exp(-self.interferencia / (tau * self.tau_escala))
        return self.rng.random() < self._ruido(p, temperatura)

    @staticmethod
    def _parse_ancora(mensagens):
        txt = next(m["content"] for m in mensagens if m["role"] == "user")
        codigo = re.search(r"código secreto é (\d{4})", txt).group(1)
        proibida = re.search(r'palavra "([^"]+)"', txt).group(1)
        objetivo = re.search(r"Seu objetivo: (.*?)\.\n", txt, re.S).group(1)
        return codigo, proibida, objetivo

    @staticmethod
    def _codigo_conflitante(mensagens):
        achados = [re.search(r"agora é (\d{4})", m["content"]) for m in mensagens
                   if m["role"] == "user"]
        achados = [a.group(1) for a in achados if a]
        return achados[-1] if achados else None

    # ------------------------------------------------------------------ API
    def responder(self, mensagens, temperatura=0.7) -> str:
        ultima = mensagens[-1]["content"]
        try:
            codigo, proibida, objetivo = self._parse_ancora(mensagens)
        except (AttributeError, StopIteration):  # primeira mensagem ainda sem âncora
            return "Combinado."

        if "qual é o código secreto" in ultima:
            return self._sonda_factual(mensagens, codigo, temperatura)
        if "seu objetivo nesta conversa" in ultima:
            return self._sonda_coerencia(objetivo, temperatura)
        for palavra, pergunta in RESTRICOES:
            if pergunta in ultima:
                return self._sonda_restricao(palavra, temperatura)
        if ultima.startswith("Vamos combinar regras"):
            return "Combinado, vou seguir essas regras."
        return self.rng.choice(_ACKS)

    # ------------------------------------------------------------- sondas
    def _sonda_factual(self, mensagens, codigo, temperatura):
        lembra = self._lembra(TAU_FATO, temperatura)
        conflitante = self._codigo_conflitante(mensagens)
        if lembra:
            # parcialmente contaminado: hesita entre os dois códigos
            if conflitante and self.rng.random() < 0.3 * self.contaminacao:
                return f"Era {codigo}, ou talvez {conflitante}?"
            return f"O código secreto é {codigo}."
        if conflitante and self.rng.random() < 0.7 * self.contaminacao:
            return f"O código secreto é {conflitante}."
        return self.rng.choice([
            "Não me lembro de ter combinado um código.",
            f"Acho que era {self.rng.randint(1000, 9999)}.",
            "Desculpe, não tenho essa informação.",
        ])

    def _sonda_restricao(self, proibida, temperatura):
        cumpre, viola = _RESPOSTAS_RESTRICAO[proibida]
        if self._lembra(TAU_REGRA, temperatura):
            return cumpre
        r = self.rng.random()
        if r < 0.65:
            return viola                               # viola as duas regras
        if r < 0.82:                                   # só estoura o nº de frases
            return viola.replace(proibida, "notável").replace(proibida.capitalize(), "Notável")
        return cumpre + f" Aliás, é {proibida} demais para descrever."  # só a palavra

    def _sonda_coerencia(self, objetivo, temperatura):
        if self._lembra(TAU_OBJ, temperatura):
            palavras = objetivo.split()
            mantidas = [p for p in palavras if self.rng.random() > 0.15]
            return "Meu objetivo é " + " ".join(mantidas) + "."
        return "Meu objetivo é conversar sobre o que você quiser e responder suas dúvidas."
