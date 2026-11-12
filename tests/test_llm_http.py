import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from ambiente.llm_http import ErroLLM, Ollama, OpenAICompativel, criar_llm


class FalsoServidor(BaseHTTPRequestHandler):
    visto = []

    def log_message(self, *a): pass

    def do_POST(self):
        corpo = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FalsoServidor.visto.append((self.path, corpo))
        if self.path == "/api/chat":
            saida = {"message": {"role": "assistant", "content": "olá do ollama"}}
        elif self.path == "/v1/chat/completions":
            saida = {"choices": [{"message": {"content": "olá da openai"}}]}
        else:
            self.send_response(400); self.end_headers(); self.wfile.write(b"x"); return
        dados = json.dumps(saida).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(dados))); self.end_headers()
        self.wfile.write(dados)


class TestHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), FalsoServidor)
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_ollama(self):
        llm = Ollama("m", self.url)
        self.assertEqual(llm.responder([{"role": "user", "content": "oi"}], 0.3), "olá do ollama")
        caminho, corpo = FalsoServidor.visto[-1]
        self.assertEqual(corpo["options"]["temperature"], 0.3)
        self.assertFalse(corpo["stream"])

    def test_openai_compativel(self):
        llm = OpenAICompativel("m", self.url)
        self.assertEqual(llm.responder([{"role": "user", "content": "oi"}]), "olá da openai")

    def test_erro_http_vira_ErroLLM(self):
        llm = Ollama("m", self.url + "/inexistente")
        with self.assertRaises(ErroLLM):
            llm.responder([{"role": "user", "content": "oi"}])

    def test_servidor_fora_do_ar(self):
        with self.assertRaises(ErroLLM):
            import ambiente.llm_http as h
            h._post_json("http://127.0.0.1:9/x", {}, tentativas=1, timeout=1)

    def test_fabrica(self):
        self.assertEqual(criar_llm("simulada").nome, "simulada")
        self.assertTrue(criar_llm("ollama", "abc").nome.startswith("ollama:abc"))
        with self.assertRaises(ValueError):
            criar_llm("xyz")


if __name__ == "__main__":
    unittest.main()
