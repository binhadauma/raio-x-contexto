import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import servidor


def chamar(base, caminho, corpo=None):
    dados = None if corpo is None else json.dumps(corpo).encode()
    req = urllib.request.Request(base + caminho, data=dados, method="POST" if dados else "GET",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


class TestServidor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), servidor.Handler)
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_config_e_estatico(self):
        cfg = chamar(self.base, "/api/config")
        self.assertEqual(len(cfg["acoes"]), 6)
        with urllib.request.urlopen(self.base + "/", timeout=5) as r:
            self.assertIn("Raio-X", r.read().decode())

    def test_path_traversal_bloqueado(self):
        for caminho in ("/../servidor.py", "/%2e%2e/servidor.py", "/..%2fservidor.py"):
            with self.assertRaises(urllib.error.HTTPError) as c:
                urllib.request.urlopen(self.base + caminho, timeout=5)
            self.assertEqual(c.exception.code, 404)

    def test_execucao_completa_simulada(self):
        j = chamar(self.base, "/api/executar", {"provedor": "simulada", "politica": "aleatoria",
                                                "fases": 4, "seed": 1})
        for _ in range(100):
            est = chamar(self.base, f"/api/estado?id={j['id']}&desde=0")
            if est["status"] != "rodando":
                break
            time.sleep(0.1)
        self.assertEqual(est["status"], "concluido", est.get("erro"))
        tipos = [e["tipo"] for e in est["eventos"]]
        self.assertEqual(tipos, ["inicio", "fase", "fase", "fase", "fase", "fim"])

    def test_erro_de_llm_aparece_no_estado(self):
        j = chamar(self.base, "/api/executar", {"provedor": "ollama", "url": "http://127.0.0.1:9",
                                                "politica": "ciclica", "fases": 3})
        for _ in range(200):
            est = chamar(self.base, f"/api/estado?id={j['id']}&desde=0")
            if est["status"] != "rodando":
                break
            time.sleep(0.2)
        self.assertEqual(est["status"], "erro")
        self.assertIn("ErroLLM", est["erro"])

    def test_json_invalido(self):
        req = urllib.request.Request(self.base + "/api/executar", data=b"{ruim", method="POST")
        with self.assertRaises(urllib.error.HTTPError) as c:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(c.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
