"""Servidor do dashboard (biblioteca padrão: http.server + json + threading).

    python servidor.py [--porta 8765]   ->  http://127.0.0.1:8765

API JSON:
    GET  /api/config                 provedores, políticas, estado do DQN, chaves detectadas
    POST /api/executar               inicia uma execução (1 episódio) -> {id}
    POST /api/treinar                treina o DQN na LLM simulada       -> {id}
    GET  /api/estado?id=..&desde=N   eventos novos desde o índice N (polling)
    POST /api/parar                  interrompe a execução atual
    GET  /api/resultados             resultados.json dos experimentos
    GET  /                           arquivos estáticos de web/

Segurança: escuta só em 127.0.0.1; chaves de API vêm de variáveis de ambiente
e nunca são enviadas ao navegador.
"""
import argparse
import json
import mimetypes
import os
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from agente.baselines import criar_politica
from agente.dqn import AgenteDQN
from agente.laco import rodar_episodio
from ambiente.ambiente_teste import DIM_OBS, AmbienteTeste
from ambiente.geradores import ACOES, N_ACOES
from ambiente.llm_http import criar_llm
from ambiente.llm_simulada import LLMSimulada
from avaliador import metricas as M

RAIZ = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(RAIZ, "web")
RESULTADOS = os.path.join(RAIZ, "experimentos", "resultados")
PESOS = os.path.join(RESULTADOS, "dqn_pesos.json")


class Job:
    """Uma execução ou treino em andamento; o navegador faz polling dos eventos."""

    def __init__(self, tipo):
        self.id = uuid.uuid4().hex[:8]
        self.tipo = tipo
        self.eventos = []
        self.status = "rodando"  # rodando | concluido | erro | parado
        self.erro = None
        self.parar = False
        self.lock = threading.Lock()

    def emitir(self, ev):
        with self.lock:
            self.eventos.append(ev)

    def a_partir_de(self, n):
        with self.lock:
            return list(self.eventos[n:]), len(self.eventos)


class Aplicacao:
    def __init__(self):
        self.jobs = {}
        self.ativo = None
        self.dqn = AgenteDQN(DIM_OBS, N_ACOES, seed=0)
        self.dqn_treinado = False
        self._carregar_pesos()

    def _carregar_pesos(self):
        if os.path.exists(PESOS):
            try:
                self.dqn.carregar(PESOS)
                self.dqn_treinado = True
            except Exception:  # pesos de uma versão antiga/incompatível
                self.dqn_treinado = False

    def ocupado(self):
        return self.ativo is not None and self.ativo.status == "rodando"

    # ------------------------------------------------------------- execução
    def iniciar_execucao(self, p):
        job = Job("execucao")
        self.jobs[job.id] = job
        self.ativo = job
        threading.Thread(target=self._executar, args=(job, p), daemon=True).start()
        return job

    def _executar(self, job, p):
        try:
            fases = max(3, min(int(p.get("fases", 12)), 30))
            temp = float(p.get("temperatura", 0.7))
            seed = p.get("seed")
            seed = int(seed) if seed not in (None, "") else None
            if p.get("provedor", "simulada") == "simulada":
                llm = LLMSimulada(seed=seed)
            else:
                llm = criar_llm(p["provedor"], p.get("modelo", ""), p.get("url", ""))
            env = AmbienteTeste(llm, n_fases=fases, temperatura=temp, seed=seed)
            nome_pol = p.get("politica", "dqn")
            if nome_pol == "dqn" and not self.dqn_treinado:
                raise RuntimeError("DQN ainda não treinado: clique em 'Treinar DQN' "
                                   "ou rode experimentos/roda_experimentos.py")
            pol = criar_politica(nome_pol, agente=self.dqn, seed=seed or 0)
            online = bool(p.get("aprender_online")) and nome_pol == "dqn"

            env_obs = env.reset()  # gera âncora (já chama a LLM uma vez)
            a = env.ancora
            job.emitir({"tipo": "inicio", "llm": llm.nome, "politica": nome_pol, "fases": fases,
                        "ancora": a.texto(), "codigo": a.codigo,
                        "palavra_proibida": a.palavra_proibida, "objetivo": a.objetivo,
                        "resposta_ancora": env.historico[1]["content"]})
            obs, feito = env_obs, False
            while not feito:
                if job.parar:
                    job.status = "parado"
                    return
                acao = pol.agir(obs, explorar=online)
                obs2, r, feito, ev = env.step(acao)
                pol.observar(obs, acao, r, obs2, feito, treinar=online)
                obs = obs2
                ev = dict(ev, tipo="fase")
                job.emitir(ev)
            job.emitir({"tipo": "fim", "resumo": env.resumo()})
            job.status = "concluido"
        except Exception as e:  # erros de rede/LLM aparecem na interface
            job.erro = f"{type(e).__name__}: {e}"
            job.status = "erro"
            traceback.print_exc()

    # --------------------------------------------------------------- treino
    def iniciar_treino(self, p):
        job = Job("treino")
        self.jobs[job.id] = job
        self.ativo = job
        threading.Thread(target=self._treinar, args=(job, p), daemon=True).start()
        return job

    def _treinar(self, job, p):
        try:
            episodios = max(10, min(int(p.get("episodios", 600)), 5000))
            fases = max(3, min(int(p.get("fases", 12)), 30))
            self.dqn = AgenteDQN(DIM_OBS, N_ACOES, eps_passos=int(episodios * fases * 0.6), seed=0)
            env = AmbienteTeste(LLMSimulada(seed=7), n_fases=fases, seed=7)
            recompensas = []
            for ep in range(episodios):
                if job.parar:
                    job.status = "parado"
                    return
                _, r = rodar_episodio(env, self.dqn, treinar=True, explorar=True)
                recompensas.append(r)
                if ep % 5 == 0 or ep == episodios - 1:
                    job.emitir({"tipo": "treino", "episodio": ep + 1, "total": episodios,
                                "recompensa": r, "media50": M.media(recompensas[-50:]),
                                "epsilon": self.dqn.epsilon, "perda": self.dqn.ultima_perda})
            self.dqn_treinado = True
            self.dqn.salvar(PESOS)
            job.emitir({"tipo": "fim"})
            job.status = "concluido"
        except Exception as e:
            job.erro = f"{type(e).__name__}: {e}"
            job.status = "erro"
            traceback.print_exc()


APP = Aplicacao()


class Handler(BaseHTTPRequestHandler):
    server_version = "RaioX/1.0"

    def log_message(self, fmt, *args):  # silencia o log por requisição
        pass

    # --------------------------------------------------------------- helpers
    def _json(self, obj, status=200):
        corpo = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(corpo)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(corpo)

    def _corpo_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 1_000_000:
            raise ValueError("corpo grande demais")
        return json.loads(self.rfile.read(n) or b"{}")

    def _estatico(self, caminho):
        rel = "index.html" if caminho in ("", "/") else caminho.lstrip("/")
        alvo = os.path.realpath(os.path.join(WEB, rel))
        if not alvo.startswith(os.path.realpath(WEB) + os.sep) or not os.path.isfile(alvo):
            return self._json({"erro": "não encontrado"}, 404)
        tipo = mimetypes.guess_type(alvo)[0] or "application/octet-stream"
        with open(alvo, "rb") as f:
            dados = f.read()
        self.send_response(200)
        self.send_header("Content-Type", tipo + ("; charset=utf-8" if tipo.startswith("text") or "javascript" in tipo else ""))
        self.send_header("Content-Length", str(len(dados)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(dados)

    # ------------------------------------------------------------------- GET
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/api/config":
            return self._json({
                "acoes": ACOES,
                "provedores": ["simulada", "ollama", "openai", "gemini"],
                "politicas": ["dqn", "aleatoria", "ciclica", "fixa_neutro", "fixa_conflito"],
                "dqn_treinado": APP.dqn_treinado,
                "chaves": {"gemini": bool(os.environ.get("GEMINI_API_KEY")),
                           "openai": bool(os.environ.get("OPENAI_API_KEY"))},
                "ocupado": APP.ocupado(),
            })
        if u.path == "/api/estado":
            job = APP.jobs.get((q.get("id") or [""])[0])
            if not job:
                return self._json({"erro": "job desconhecido"}, 404)
            novos, total = job.a_partir_de(int(q.get("desde", ["0"])[0]))
            return self._json({"status": job.status, "erro": job.erro, "eventos": novos,
                               "total": total, "tipo": job.tipo})
        if u.path == "/api/resultados":
            caminho = os.path.join(RESULTADOS, "resultados.json")
            if not os.path.exists(caminho):
                return self._json({"erro": "rode experimentos/roda_experimentos.py"}, 404)
            with open(caminho, encoding="utf-8") as f:
                dados = f.read().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(dados)))
            self.end_headers()
            self.wfile.write(dados)
            return
        return self._estatico(u.path)

    # ------------------------------------------------------------------ POST
    def do_POST(self):
        u = urlparse(self.path)
        try:
            p = self._corpo_json()
        except (ValueError, json.JSONDecodeError) as e:
            return self._json({"erro": f"JSON inválido: {e}"}, 400)
        if u.path in ("/api/executar", "/api/treinar"):
            if APP.ocupado():
                return self._json({"erro": "já existe uma execução em andamento"}, 409)
            job = APP.iniciar_execucao(p) if u.path == "/api/executar" else APP.iniciar_treino(p)
            return self._json({"id": job.id})
        if u.path == "/api/parar":
            if APP.ativo:
                APP.ativo.parar = True
            return self._json({"ok": True})
        return self._json({"erro": "rota desconhecida"}, 404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--porta", type=int, default=8765)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.porta), Handler)
    print(f"Raio-X de Contexto em http://127.0.0.1:{a.porta}  (Ctrl+C para sair)")
    if not APP.dqn_treinado:
        print("Aviso: DQN sem pesos. Use 'Treinar DQN' no dashboard ou rode "
              "experimentos/roda_experimentos.py")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nencerrado")


if __name__ == "__main__":
    main()
