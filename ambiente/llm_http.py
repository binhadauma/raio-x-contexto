"""Clientes HTTP para LLMs reais usando SÓ a biblioteca padrão (urllib + json).

Interface comum (a mesma da LLM simulada):

    llm.reiniciar()                          # nada a fazer em LLMs reais
    llm.responder(mensagens, temperatura)    # -> str

`mensagens` = [{"role": "user"|"assistant", "content": "..."}].

Provedores:
  * Ollama  (local, grátis):        POST {url}/api/chat
  * OpenAI-compatível (LM Studio, vLLM, OpenAI...): POST {url}/v1/chat/completions
  * Gemini  (nuvem):                POST generativelanguage.googleapis.com/...:generateContent

Chaves de API ficam SÓ em variáveis de ambiente (GEMINI_API_KEY, OPENAI_API_KEY);
nunca são enviadas ao navegador nem gravadas em arquivo.
"""
import json
import os
import time
import urllib.error
import urllib.request


class ErroLLM(RuntimeError):
    pass


def _post_json(url, corpo, cabecalhos=None, timeout=120, tentativas=3):
    """POST JSON com reintento exponencial para erros transitórios (429/5xx)."""
    dados = json.dumps(corpo).encode("utf-8")
    cab = {"Content-Type": "application/json"}
    cab.update(cabecalhos or {})
    ultimo = None
    for i in range(tentativas):
        req = urllib.request.Request(url, data=dados, headers=cab, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            ultimo = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}"
            if e.code not in (429, 500, 502, 503, 504):
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            ultimo = f"conexão: {e}"
        time.sleep(1.5 * (2 ** i))
    raise ErroLLM(f"falha ao chamar {url}: {ultimo}")


class LLMBase:
    nome = "http"

    def reiniciar(self):
        pass


class Ollama(LLMBase):
    def __init__(self, modelo="llama3.2", url="http://localhost:11434", timeout=180):
        self.modelo, self.url, self.timeout = modelo, url.rstrip("/"), timeout
        self.nome = f"ollama:{modelo}"

    def responder(self, mensagens, temperatura=0.7) -> str:
        r = _post_json(f"{self.url}/api/chat", {
            "model": self.modelo, "messages": mensagens, "stream": False,
            "options": {"temperature": temperatura},
        }, timeout=self.timeout)
        try:
            return r["message"]["content"]
        except KeyError:
            raise ErroLLM(f"resposta inesperada do Ollama: {str(r)[:200]}")


class OpenAICompativel(LLMBase):
    def __init__(self, modelo="gpt-4o-mini", url="https://api.openai.com", timeout=120):
        self.modelo, self.url, self.timeout = modelo, url.rstrip("/"), timeout
        self.nome = f"openai:{modelo}"

    def responder(self, mensagens, temperatura=0.7) -> str:
        chave = os.environ.get("OPENAI_API_KEY", "")
        cab = {"Authorization": f"Bearer {chave}"} if chave else {}
        r = _post_json(f"{self.url}/v1/chat/completions", {
            "model": self.modelo, "messages": mensagens, "temperature": temperatura,
        }, cab, self.timeout)
        try:
            return r["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise ErroLLM(f"resposta inesperada: {str(r)[:200]}")


class Gemini(LLMBase):
    def __init__(self, modelo="gemini-2.0-flash", timeout=120):
        self.modelo, self.timeout = modelo, timeout
        self.nome = f"gemini:{modelo}"

    def responder(self, mensagens, temperatura=0.7) -> str:
        chave = os.environ.get("GEMINI_API_KEY")
        if not chave:
            raise ErroLLM("defina a variável de ambiente GEMINI_API_KEY")
        contents = [{"role": "user" if m["role"] == "user" else "model",
                     "parts": [{"text": m["content"]}]} for m in mensagens]
        url = ("https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.modelo}:generateContent")
        r = _post_json(url, {"contents": contents,
                             "generationConfig": {"temperature": temperatura}},
                       {"x-goog-api-key": chave}, self.timeout)
        try:
            return r["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise ErroLLM(f"resposta inesperada do Gemini: {str(r)[:200]}")


def criar_llm(provedor: str, modelo: str = "", url: str = ""):
    """Fábrica usada pelo servidor. `provedor` ∈ {simulada, ollama, openai, gemini}."""
    if provedor == "simulada":
        from ambiente.llm_simulada import LLMSimulada
        return LLMSimulada()
    if provedor == "ollama":
        return Ollama(modelo or "llama3.2", url or "http://localhost:11434")
    if provedor == "openai":
        return OpenAICompativel(modelo or "gpt-4o-mini", url or "https://api.openai.com")
    if provedor == "gemini":
        return Gemini(modelo or "gemini-2.0-flash")
    raise ValueError(f"provedor desconhecido: {provedor}")
