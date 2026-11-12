"""Contador de tokens próprio (sem tokenizador de biblioteca).

Aproximação: ~1,3 token por palavra + 1 token por sinal de pontuação.
Para português, a razão tokens/palavra de tokenizadores BPE costuma ficar
entre 1,3 e 1,6; o valor é um parâmetro declarado, não uma medida exata.
"""
import re

TOKENS_POR_PALAVRA = 1.3
_RE_PALAVRA = re.compile(r"\w+", re.UNICODE)
_RE_PONTUACAO = re.compile(r"[^\w\s]", re.UNICODE)


def contar_tokens(texto: str) -> int:
    """Estimativa de tokens de um texto."""
    palavras = len(_RE_PALAVRA.findall(texto))
    pontuacao = len(_RE_PONTUACAO.findall(texto))
    return int(round(palavras * TOKENS_POR_PALAVRA + pontuacao))


def contar_tokens_mensagens(mensagens) -> int:
    """Soma de tokens de uma lista de mensagens [{'role':..., 'content':...}]."""
    return sum(contar_tokens(m["content"]) for m in mensagens)
