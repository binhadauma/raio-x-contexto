# Raio-X de Contexto

Agente-testador que mede **em que momento uma LLM perde o rastreamento do contexto inicial** de uma conversa (Projeto de IA, Fase 3 — Binha Ferraz Dauma).

O agente planta uma **âncora** (código secreto, regras de formato e objetivo), injeta **distratores linguísticos** fase a fase e, depois de cada um, faz **3 sondas** que só podem ser respondidas corretamente se a âncora ainda estiver ativa. A resposta é pontuada pelo **CTS** (*Context Tracking Score*) e o **ponto de ruptura** é registrado. Um **DQN escrito do zero** aprende qual sequência de distratores derruba o CTS mais cedo.

> **Aviso sobre a LLM simulada.** `ambiente/llm_simulada.py` é um modelo paramétrico de esquecimento, com parâmetros escolhidos pela autora. Serve para treinar o agente rápido e como plano B offline. Os resultados com ela **validam o pipeline e o agente**, não dizem nada sobre LLMs reais. Conclusões sobre modelos reais exigem execuções com Ollama/API (seção 5).

## 1. Instalação

Requer Python 3.10+. A única dependência é o NumPy (álgebra de matrizes; permitido pelo enunciado da Fase 3). Servidor, clientes HTTP, avaliador, DQN, gráficos SVG e dashboard usam só a biblioteca padrão.

```bash
pip install -r requirements.txt
```

## 2. Execução

| O que | Comando |
|---|---|
| Dashboard visual | `python servidor.py` → abra http://127.0.0.1:8765 |
| Demo no terminal (plano B) | `python demo.py` (ou `--politica aleatoria`, `--ollama llama3.2`) |
| Experimentos completos (~2,5 min) | `python experimentos/roda_experimentos.py` |
| Experimentos rápidos (~10 s) | `python experimentos/roda_experimentos.py --rapido` |
| Testes (30) | `python -m unittest discover -s tests -t .` |

O repositório já traz `experimentos/resultados/` com os resultados, os gráficos SVG e os pesos do DQN (`dqn_pesos.json`), então o dashboard funciona logo após a instalação. Sem esses pesos, use o botão **Treinar DQN** (~10 s).

## 3. Arquitetura

```
AGENTE (DQN)  --ação: tipo de distrator-->  AMBIENTE  --mensagens-->  LLM (simulada | HTTP)
     ^                                          |
     +---- percepto: distância, CTS, histórico -+<-- AVALIADOR (factual, regras, cosseno)
```

```
agente/       rede_numpy.py (MLP + backprop + Adam)  dqn.py  q_learning.py  baselines.py  laco.py
ambiente/     geradores.py  llm_simulada.py  llm_http.py  ambiente_teste.py
avaliador/    tokens.py  metricas.py
web/          index.html  style.css  app.js   (gráficos em <canvas>, sem bibliotecas)
experimentos/ roda_experimentos.py  graficos_svg.py  resultados/
tests/        avaliador, rede (gradient check), ambiente, HTTP (servidor falso), servidor
servidor.py   demo.py
```

### Matriz PEAS → código

| PEAS | Onde |
|---|---|
| **Performance** | `avaliador/metricas.py`: CTS = 0,4·factual + 0,3·regras + 0,3·coerência; ponto de ruptura = 1ª fase com CTS < 0,5 por 2 fases seguidas |
| **Environment** | `ambiente/`: âncora + distratores + LLM; parcialmente observável, estocástico, sequencial |
| **Actuators** | `ambiente_teste.step()`: injeta distrator, faz as sondas, registra métricas |
| **Sensors** | `ambiente_teste._obs()`: vetor de 23 números (tokens, CTS, últimas ações). O agente **não** vê o estado interno da LLM |

### Detalhes de projeto que importam na defesa

* **Sondas em bifurcação (fork):** a sonda é enviada com o histórico, mas **não** é gravada nele; senão a própria pergunta reforçaria a memória da âncora.
* **Avaliação só por texto:** o avaliador lê a resposta como faria com uma LLM real. Na simulada, o único atalho é o gancho `injetar()`, que informa tipo/tamanho do distrator, e não existe em LLMs reais.
* **Distratores (6 ações discretas):** neutro, garden-path, aninhamento recursivo, anáfora ambígua, instrução conflitante, fato longo irrelevante.
* **Recompensa:** `(1 − CTS) + 2·(fração de fases restantes)` pago uma vez, na 1ª ruptura.
* **Tokens:** contador próprio (~1,3 token/palavra + pontuação), uma aproximação declarada.
* **DQN do zero:** MLP 23→64→64→6, replay buffer, rede-alvo, ε-greedy, perda Huber, Adam, backprop manual verificado por diferenças finitas (`tests/test_rede.py`).

## 4. Resultados (LLM simulada)

5 sementes × 100 episódios de avaliação por política (500 por linha), 12 fases, temperatura 0,7. O treino do DQN e do Q-tabular usa 1.500 episódios por semente.

| Política | Taxa de ruptura | Fase média da ruptura (IC 95%) | CTS médio ± dp |
|---|---|---|---|
| **DQN (aprendiz)** | 100% | **4,56** [4,41; 4,72] | 0,370 ± 0,065 |
| Q-tabular | 98% | 5,78 [5,57; 5,99] | 0,458 ± 0,076 |
| Sempre conflito | 99% | 5,23 [5,05; 5,42] | 0,416 ± 0,069 |
| Cíclica | 98% | 6,21 [6,04; 6,37] | 0,493 ± 0,071 |
| Aleatória | 94% | 6,57 [6,37; 6,78] | 0,519 ± 0,096 |
| Controle (só neutro) | 0,2% | — | 0,916 ± 0,046 |

Gráficos: `experimentos/resultados/*.svg` (curva de CTS por política, ruptura, aprendizado, mapa de calor) e CSV em `resumo_politicas.csv`.

**Como ler:** o controle neutro quase nunca rompe, então a ruptura é causada pelos distratores e não pelo tamanho do contexto. O DQN rompe ~2 fases antes da política aleatória e ~0,7 fase antes da melhor política fixa. **Limites:** (i) a vantagem sobre "sempre conflito" depende de a simulada premiar sequências (sinergias e habituação definidas por nós); (ii) variações de semente e de ordem de sorteio deslocam o resultado em ±0,4 fase (execuções anteriores deram 4,96), então compare sempre com o IC; (iii) a coluna de tokens mede a simulada, não um modelo real.

## 5. Usando uma LLM real

```bash
# Ollama (local, grátis)
ollama pull llama3.2 && ollama serve
python demo.py --ollama llama3.2          # ou escolha "Ollama" no dashboard

# Gemini / OpenAI-compatível: chaves só por variável de ambiente
export GEMINI_API_KEY=...                 # ou OPENAI_API_KEY=...
python servidor.py
```

Uma execução usa N distratores + 3·N sondas (12 fases ≈ 48 chamadas). O DQN treinado na simulada é aplicado à LLM real (transferência); marque **aprender online** para que ele continue aprendendo com as recompensas reais.

## 6. Declaração de uso de IA generativa

> **Preencha antes de entregar.** Este texto descreve o que o histórico da conversa mostra; ajuste para refletir fielmente o que você fez e entendeu.

* **Gemini e Perplexity** (já declarados na Fase 2): pesquisa bibliográfica e rascunho de PEAS e justificativa técnica; tudo lido e verificado pela autora.
* **Claude (Anthropic)**, nesta Fase 3: gerou o **código-fonte inicial** de todos os módulos (avaliador, geradores, LLM simulada, cliente HTTP, ambiente, rede NumPy, DQN, Q-learning, servidor, dashboard, testes e gráficos SVG), rodou os testes e os experimentos e rascunhou este README.
* **Desenvolvido pela autora:** [PREENCHER: ideia e formulação do problema, pergunta de pesquisa, escolha dos algoritmos, modelagem PEAS (Fases 1 e 2); ajustes, revisão e testes feitos por você nesta fase].
* **Verificação:** o backprop foi validado por diferenças finitas; os 30 testes automatizados cobrem avaliador, rede, ambiente, clientes HTTP e servidor.
* **Dados:** todos os textos (âncoras, distratores, sondas) e a LLM simulada são **sintéticos**, escritos em templates; nenhum dado real foi usado.

## 7. Referências

Hahn, Futrell, Levy & Gibson (2022); Lakretz et al. (2021); Li, Feng, Narang et al. (2024); Murty et al. (2023); Zhou et al. (2026); Mnih et al. (2015), *Human-level control through deep reinforcement learning* (algoritmo DQN).
