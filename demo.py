"""Demonstração em terminal (sem navegador): um episódio, fase a fase.

    python demo.py                       # DQN contra a LLM simulada
    python demo.py --politica aleatoria  # compara com a política aleatória
    python demo.py --ollama llama3.2     # contra um modelo local do Ollama

Útil como plano B na defesa e como caso de teste executável (Critério 3).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agente.baselines import criar_politica  # noqa: E402
from agente.dqn import AgenteDQN  # noqa: E402
from ambiente.ambiente_teste import DIM_OBS, AmbienteTeste  # noqa: E402
from ambiente.geradores import N_ACOES  # noqa: E402
from ambiente.llm_http import criar_llm  # noqa: E402

PESOS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "experimentos", "resultados", "dqn_pesos.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--politica", default="dqn")
    ap.add_argument("--fases", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ollama", metavar="MODELO", help="usa um modelo do Ollama local")
    a = ap.parse_args()

    llm = criar_llm("ollama", a.ollama) if a.ollama else criar_llm("simulada")
    if not a.ollama:
        llm.rng.seed(a.seed)
    env = AmbienteTeste(llm, n_fases=a.fases, seed=a.seed)
    agente = None
    if a.politica == "dqn":
        agente = AgenteDQN(DIM_OBS, N_ACOES)
        if not os.path.exists(PESOS):
            sys.exit("Sem pesos do DQN. Rode: python experimentos/roda_experimentos.py --rapido")
        agente.carregar(PESOS)
    pol = criar_politica(a.politica, agente=agente, seed=a.seed)

    obs, feito = env.reset(), False
    print(f"LLM: {llm.nome} | política: {a.politica} | código secreto: {env.ancora.codigo}\n")
    print("fase  distrator       tokens  factual  regras  objetivo   CTS")
    while not feito:
        obs, _, feito, ev = env.step(pol.agir(obs, explorar=False))
        n = ev["notas"]
        marca = "  <- RUPTURA" if ev["ruptura_fase"] == ev["fase"] else ""
        print(f"{ev['fase']:>4}  {ev['acao_nome']:<14} {ev['tokens']:>7}  {n['factual']:>7.2f}  "
              f"{n['restricao']:>6.2f}  {n['coerencia']:>8.2f}  {ev['cts']:.2f}{marca}")
    r = env.resumo()
    print(f"\nruptura na fase {r['ruptura_fase']} (~{r['ruptura_tokens']} tokens); CTS médio {r['cts_medio']:.2f}")


if __name__ == "__main__":
    main()
