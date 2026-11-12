"""Avaliador do Raio-X de Contexto (medida de desempenho do PEAS).

Tudo aqui é Python puro + `math`/`re`: nenhuma biblioteca de PLN.

Para cada fase o avaliador recebe as 3 respostas às sondas e calcula:

    factual    : o código secreto aparece (e só ele)?               [0, 1]
    restricao  : respeitou "máx. N frases" e "palavra proibida"?    [0, 1]
    coerencia  : cosseno TF entre resposta e objetivo inicial       [0, 1]

    CTS = 0,4 * factual + 0,3 * restricao + 0,3 * coerencia
    (CTS = Context Tracking Score, a "medida de rastreamento de contexto")

Ponto de ruptura = primeira fase em que o CTS fica abaixo do limiar por
`consecutivas` fases seguidas (reduz falsos positivos por ruído de amostragem).
"""
import math
import re
import unicodedata

PESO_FACTUAL = 0.4
PESO_RESTRICAO = 0.3
PESO_COERENCIA = 0.3
LIMIAR_RUPTURA = 0.5
FASES_CONSECUTIVAS = 2
# cosseno que consideramos "resposta plenamente alinhada" (paráfrases curtas
# raramente passam de ~0,6 de cosseno com o texto original)
COSSENO_REFERENCIA = 0.5

STOPWORDS = frozenset(
    "a o as os um uma uns umas de do da dos das em no na nos nas por para com "
    "sem sobre e ou mas que se ao aos à às é são ser foi era seu sua seus suas "
    "meu minha como mais menos muito muita ja já nao não eu voce você nos nós "
    "isso isto esse essa este esta nesta neste desta deste ate até entre "
    "tambem também so só ha há".split()
)


# ---------------------------------------------------------------- texto base
def normalizar(texto: str) -> str:
    """minúsculas e sem acentos (NFD), para comparações robustas."""
    texto = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def tokenizar_palavras(texto: str):
    """Lista de palavras normalizadas, sem stopwords."""
    palavras = re.findall(r"[a-z0-9]+", normalizar(texto))
    stop = {normalizar(s) for s in STOPWORDS}
    return [p for p in palavras if p not in stop and len(p) > 1]


def vetor_tf(texto: str) -> dict:
    """Bag-of-words com frequência de termos (dicionário palavra -> contagem)."""
    tf = {}
    for p in tokenizar_palavras(texto):
        tf[p] = tf.get(p, 0) + 1
    return tf


def cosseno(a: dict, b: dict) -> float:
    """Similaridade de cosseno entre dois vetores esparsos (dicts)."""
    if not a or not b:
        return 0.0
    produto = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return produto / (na * nb) if na and nb else 0.0


def contar_frases(texto: str) -> int:
    """Conta frases por terminadores . ! ? (ignora segmentos vazios)."""
    partes = re.split(r"[.!?]+(?:\s|$)", texto.strip())
    return len([p for p in partes if p.strip()])


# ------------------------------------------------------------ três critérios
def avaliar_factual(resposta: str, codigo: str) -> float:
    """1,0 se só o código correto aparece; 0,5 se ele aparece junto de outro
    número de 4 dígitos (hesitação/contaminação); 0,0 caso contrário."""
    numeros = re.findall(r"(?<!\d)\d{4}(?!\d)", resposta)
    if codigo not in numeros:
        return 0.0
    return 1.0 if all(n == codigo for n in numeros) else 0.5


def avaliar_restricao(resposta: str, palavra_proibida: str, max_frases: int) -> float:
    """Cada regra violada tira 0,5. Regras: palavra proibida e nº de frases."""
    nota = 1.0
    if normalizar(palavra_proibida) in normalizar(resposta):
        nota -= 0.5
    if contar_frases(resposta) > max_frases:
        nota -= 0.5
    return nota


def avaliar_coerencia(resposta: str, objetivo: str) -> float:
    """Cosseno TF resposta x objetivo, reescalado por COSSENO_REFERENCIA."""
    c = cosseno(vetor_tf(resposta), vetor_tf(objetivo))
    return min(1.0, c / COSSENO_REFERENCIA)


def calcular_cts(factual: float, restricao: float, coerencia: float) -> float:
    return (PESO_FACTUAL * factual + PESO_RESTRICAO * restricao
            + PESO_COERENCIA * coerencia)


# ------------------------------------------------------------------ ruptura
def ponto_ruptura(serie_cts, limiar=LIMIAR_RUPTURA, consecutivas=FASES_CONSECUTIVAS):
    """Índice (0-based) da primeira fase da sequência que cruza o limiar.

    Retorna None se não houve ruptura. Exemplo: com consecutivas=2,
    [0.9, 0.4, 0.8, 0.3, 0.2] -> 3 (a queda isolada da fase 1 é ruído)."""
    abaixo = 0
    for i, v in enumerate(serie_cts):
        abaixo = abaixo + 1 if v < limiar else 0
        if abaixo >= consecutivas:
            return i - consecutivas + 1
    return None


# -------------------------------------------------------- estatística básica
def media(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def desvio(xs):
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = media(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def ic_bootstrap(xs, n=2000, alfa=0.05, rng=None):
    """Intervalo de confiança bootstrap (percentil) da média."""
    import random
    rng = rng or random.Random(0)
    xs = list(xs)
    if not xs:
        return (float("nan"), float("nan"))
    medias = sorted(media([rng.choice(xs) for _ in xs]) for _ in range(n))
    return medias[int(n * alfa / 2)], medias[int(n * (1 - alfa / 2)) - 1]
