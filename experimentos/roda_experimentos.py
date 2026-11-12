"""Roda os experimentos do Critério 2 (PEAS) e grava resultados reproduzíveis.

Uso:
    python experimentos/roda_experimentos.py            # completo (~4 min)
    python experimentos/roda_experimentos.py --rapido   # smoke test (~15 s)

Saídas em experimentos/resultados/:
    resultados.json      tudo que o dashboard e os gráficos consomem
    resumo_politicas.csv tabela de comparação entre políticas
    dqn_pesos.json       pesos do DQN (semente 0) carregados pelo servidor
    *.svg                gráficos para o relatório (ver graficos_svg.py)

Desenho experimental
--------------------
* LLM = simulada (ver aviso em ambiente/llm_simulada.py): valida o pipeline e o
  agente; NÃO é evidência sobre LLMs reais.
* Para cada semente s ∈ {0..S-1}: treina DQN e Q-tabular; avalia TODAS as
  políticas nos mesmos 100 episódios (mesmas âncoras; temperatura 0,7).
* Medida de desempenho (PEAS): CTS por fase, fase/tokens do ponto de ruptura,
  taxa de ruptura e área sob a curva de CTS (menor = agente mais eficaz).
* Mapa de calor: efeito de cada distrator na queda de CTS, medido com política
  ALEATÓRIA (todas as ações cobrem todas as fases -> comparação sem viés).
"""
import argparse
import csv
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from agente.baselines import criar_politica  # noqa: E402
from agente.dqn import AgenteDQN  # noqa: E402
from agente.laco import rodar_episodio  # noqa: E402
from agente.q_learning import AgenteQTabular  # noqa: E402
from ambiente.ambiente_teste import DIM_OBS, AmbienteTeste  # noqa: E402
from ambiente.geradores import ACOES, N_ACOES  # noqa: E402
from ambiente.llm_simulada import LLMSimulada  # noqa: E402
from avaliador import metricas as M  # noqa: E402

PASTA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
POLITICAS = ["dqn", "q_tabular", "aleatoria", "ciclica", "fixa_conflito", "fixa_neutro"]


def treinar(agente, episodios, n_fases, seed):
    env = AmbienteTeste(LLMSimulada(seed=seed + 7), n_fases=n_fases, seed=seed + 7)
    recompensas = []
    for _ in range(episodios):
        _, r = rodar_episodio(env, agente, treinar=True, explorar=True)
        recompensas.append(r)
    return recompensas


def avaliar(politica, n_eps, n_fases, seed):
    env = AmbienteTeste(LLMSimulada(seed=100 + seed), n_fases=n_fases, seed=100 + seed)
    saidas, acoes_por_fase = [], [[0] * N_ACOES for _ in range(n_fases)]
    for ep in range(n_eps):
        resumo, _ = rodar_episodio(env, politica, explorar=False, seed=1000 + ep)
        saidas.append(resumo)
        for i, e in enumerate(env.eventos):
            acoes_por_fase[i][e["acao"]] += 1
    return saidas, acoes_por_fase


def suavizar(xs, janela=50):
    return [M.media(xs[max(0, i - janela + 1):i + 1]) for i in range(len(xs))]


def agregar(resumos, n_fases):
    cts_fase = [[r["cts_por_fase"][i] for r in resumos] for i in range(n_fases)]
    rup = [r["ruptura_fase"] for r in resumos]
    rup_ok = [x for x in rup if x is not None]
    tok = [r["ruptura_tokens"] for r in resumos if r["ruptura_tokens"] is not None]
    auc = [r["cts_medio"] for r in resumos]
    lo, hi = M.ic_bootstrap(rup_ok) if rup_ok else (None, None)
    return {
        "n_episodios": len(resumos),
        "cts_medio_por_fase": [M.media(c) for c in cts_fase],
        "cts_desvio_por_fase": [M.desvio(c) for c in cts_fase],
        "taxa_ruptura": len(rup_ok) / len(resumos),
        "ruptura_fase_media": M.media(rup_ok) if rup_ok else None,
        "ruptura_fase_ic95": [lo, hi],
        "ruptura_tokens_media": M.media(tok) if tok else None,
        "ruptura_fases": rup,
        "cts_medio_episodio": M.media(auc),
        "cts_medio_episodio_desvio": M.desvio(auc),
    }


def mapa_calor(n_eps, n_fases):
    """ΔCTS médio por (ação, fase) sob política aleatória."""
    env = AmbienteTeste(LLMSimulada(seed=555), n_fases=n_fases, seed=556)
    pol = criar_politica("aleatoria", seed=557)
    soma = [[0.0] * n_fases for _ in range(N_ACOES)]
    cont = [[0] * n_fases for _ in range(N_ACOES)]
    for ep in range(n_eps):
        rodar_episodio(env, pol, seed=5000 + ep)
        anterior = 1.0
        for i, e in enumerate(env.eventos):
            soma[e["acao"]][i] += anterior - e["cts"]
            cont[e["acao"]][i] += 1
            anterior = e["cts"]
    return [[(soma[a][i] / cont[a][i]) if cont[a][i] else None for i in range(n_fases)]
            for a in range(N_ACOES)], cont


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--sementes", type=int, default=5)
    ap.add_argument("--episodios", type=int, default=1500, help="episódios de treino")
    ap.add_argument("--avaliacao", type=int, default=100, help="episódios de avaliação/semente")
    ap.add_argument("--fases", type=int, default=12)
    a = ap.parse_args()
    if a.rapido:
        a.sementes, a.episodios, a.avaliacao = 2, 300, 30
    os.makedirs(PASTA, exist_ok=True)
    t0 = time.time()

    resumos = {p: [] for p in POLITICAS}
    acoes_dqn = [[0] * N_ACOES for _ in range(a.fases)]
    curvas = {"dqn": [], "q_tabular": []}

    for s in range(a.sementes):
        print(f"[semente {s}] treinando DQN e Q-tabular ({a.episodios} episódios)...", flush=True)
        eps_passos = int(a.episodios * a.fases * 0.6)
        dqn = AgenteDQN(DIM_OBS, N_ACOES, eps_passos=eps_passos, seed=s)
        curvas["dqn"].append(suavizar(treinar(dqn, a.episodios, a.fases, s)))
        qt = AgenteQTabular(N_ACOES, eps_passos=eps_passos, seed=s)
        curvas["q_tabular"].append(suavizar(treinar(qt, a.episodios, a.fases, s)))
        if s == 0:
            dqn.salvar(os.path.join(PASTA, "dqn_pesos.json"))

        for nome in POLITICAS:
            if nome == "dqn":
                pol = dqn
            elif nome == "q_tabular":
                pol = qt
            else:
                pol = criar_politica(nome, seed=s)
            saidas, acoes = avaliar(pol, a.avaliacao, a.fases, s)
            resumos[nome] += saidas
            if nome == "dqn":
                for i in range(a.fases):
                    for k in range(N_ACOES):
                        acoes_dqn[i][k] += acoes[i][k]
        print(f"   ok ({time.time() - t0:.0f}s)", flush=True)

    politicas = {n: agregar(r, a.fases) for n, r in resumos.items()}
    n_curva = a.episodios
    treino = {n: [M.media([c[i] for c in cs]) for i in range(n_curva)]
              for n, cs in curvas.items()}
    heat, cont = mapa_calor(max(200, a.avaliacao * 5), a.fases)
    total_dqn = a.sementes * a.avaliacao
    politica_aprendida = [[c / total_dqn for c in linha] for linha in acoes_dqn]

    resultado = {
        "config": {"fases": a.fases, "sementes": a.sementes, "episodios_treino": a.episodios,
                   "episodios_avaliacao_por_semente": a.avaliacao, "temperatura": 0.7,
                   "llm": "simulada (parametrica) - nao e medida empirica de LLM real",
                   "limiar_ruptura": M.LIMIAR_RUPTURA, "consecutivas": M.FASES_CONSECUTIVAS},
        "acoes": ACOES,
        "politicas": politicas,
        "treino": treino,
        "mapa_calor": {"delta_cts": heat, "contagem": cont},
        "politica_aprendida_freq": politica_aprendida,
    }
    with open(os.path.join(PASTA, "resultados.json"), "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False)

    with open(os.path.join(PASTA, "resumo_politicas.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["politica", "episodios", "taxa_ruptura", "fase_ruptura_media",
                    "ic95_inf", "ic95_sup", "tokens_ruptura_medio", "cts_medio", "cts_desvio"])
        for n, p in politicas.items():
            w.writerow([n, p["n_episodios"], f"{p['taxa_ruptura']:.3f}",
                        _f(p["ruptura_fase_media"]), _f(p["ruptura_fase_ic95"][0]),
                        _f(p["ruptura_fase_ic95"][1]), _f(p["ruptura_tokens_media"], 0),
                        f"{p['cts_medio_episodio']:.3f}", f"{p['cts_medio_episodio_desvio']:.3f}"])

    from experimentos.graficos_svg import gerar_todos
    gerar_todos(resultado, PASTA)

    print("\nResumo (fase média de ruptura; menor = estresse mais eficiente):")
    for n, p in politicas.items():
        print(f"  {n:15s} taxa={p['taxa_ruptura']:.2f}  fase={_f(p['ruptura_fase_media'])}  "
              f"CTS médio={p['cts_medio_episodio']:.3f}")
    print(f"\nconcluído em {time.time() - t0:.0f}s -> {PASTA}")


def _f(x, casas=2):
    return "" if x is None else f"{x:.{casas}f}"


if __name__ == "__main__":
    main()
