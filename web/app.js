// Dashboard do Raio-X de Contexto — JavaScript puro, gráficos em <canvas> feitos à mão.
"use strict";

const $ = (id) => document.getElementById(id);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const NOMES_SONDA = { factual: "Código", restricao: "Regras", coerencia: "Objetivo" };
const ROT_POL = { dqn: "DQN (aprendiz)", q_tabular: "Q-tabular", aleatoria: "Aleatória",
  ciclica: "Cíclica", fixa_conflito: "Sempre conflito", fixa_neutro: "Controle (neutro)" };
const COR_POL = { dqn: "--a4", q_tabular: "--a5", aleatoria: "--a2", ciclica: "--a3",
  fixa_conflito: "--a1", fixa_neutro: "--a0" };

let ACOES = [];
let resultados = null;       // resultados.json dos experimentos
let run = { eventos: [], resumo: null };
let treinoAoVivo = [];       // pontos da curva de treino em andamento
let jobId = null, desde = 0, timer = null;

const corAcao = (i) => css("--a" + i);

// ---------------------------------------------------------------- utilidades
async function api(caminho, corpo) {
  const opt = corpo === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(corpo) };
  const r = await fetch(caminho, opt);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.erro || r.statusText);
  return j;
}

function el(tag, cls, texto) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (texto !== undefined) e.textContent = texto;   // textContent: saída da LLM nunca vira HTML
  return e;
}

function setStatus(txt, cls) { const s = $("status"); s.textContent = txt; s.className = "pill " + (cls || ""); }
function aviso(txt, erro) { const a = $("aviso"); a.textContent = txt || ""; a.className = "aviso" + (erro ? " erro" : ""); }

/** Prepara um canvas para alta densidade de pixels e devolve contexto + tamanho lógico. */
function prepara(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = +canvas.getAttribute("height");
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  canvas.style.height = h + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.font = "11px system-ui, sans-serif";
  return { ctx, w, h };
}

/** Eixos + grade. Retorna funções de escala. */
function eixos(c, { xmin, xmax, ymin, ymax, xlabel, ylabel, ml = 44, mr = 12, mt = 10, mb = 34, yfmt = (v) => v.toFixed(1), xticks = 6, yticks = 5, semXticks = false }) {
  const { ctx, w, h } = c, pw = w - ml - mr, ph = h - mt - mb;
  const X = (v) => ml + (pw * (v - xmin)) / (xmax - xmin || 1);
  const Y = (v) => mt + ph - (ph * (v - ymin)) / (ymax - ymin || 1);
  ctx.strokeStyle = css("--grid"); ctx.fillStyle = css("--axis"); ctx.lineWidth = 1;
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let i = 0; i <= yticks; i++) {
    const v = ymin + ((ymax - ymin) * i) / yticks, y = Y(v);
    ctx.beginPath(); ctx.moveTo(ml, y); ctx.lineTo(ml + pw, y); ctx.stroke();
    ctx.fillText(yfmt(v), ml - 6, y);
  }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let i = 0; i <= xticks && !semXticks; i++) {
    const v = xmin + ((xmax - xmin) * i) / xticks;
    ctx.fillText(Math.round(v), X(v), mt + ph + 5);
  }
  ctx.fillText(xlabel, ml + pw / 2, h - 13);
  if (ylabel) { ctx.save(); ctx.translate(11, mt + ph / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(ylabel, 0, 0); ctx.restore(); }
  return { X, Y, ml, mt, pw, ph };
}

function legenda(destino, itens) {
  destino.replaceChildren();
  for (const [cor, rot] of itens) {
    const s = el("span"); const i = el("i"); i.style.background = cor;
    s.append(i, rot); destino.append(s);
  }
}

// -------------------------------------------------------------- gráfico CTS
function desenhaCTS() {
  const c = prepara($("c-cts"));
  const fases = run.eventos.filter((e) => e.tipo === "fase");
  const tmax = Math.max(800, ...fases.map((f) => f.tokens)) * 1.05;
  const g = eixos(c, { xmin: 0, xmax: tmax, ymin: 0, ymax: 1, xlabel: "tokens de contexto desde a âncora", ylabel: "CTS" });
  const { ctx } = c;
  ctx.setLineDash([5, 4]); ctx.strokeStyle = css("--axis"); ctx.beginPath();
  ctx.moveTo(g.ml, g.Y(0.5)); ctx.lineTo(g.ml + g.pw, g.Y(0.5)); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = css("--axis"); ctx.textAlign = "left"; ctx.textBaseline = "bottom";
  ctx.fillText("limiar de ruptura (0,5)", g.ml + 6, g.Y(0.5) - 3);
  if (!fases.length) return;
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 2; ctx.beginPath();
  fases.forEach((f, i) => { const x = g.X(f.tokens), y = g.Y(f.cts); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();
  for (const f of fases) {           // ponto colorido = tipo de distrator aplicado
    ctx.fillStyle = corAcao(f.acao); ctx.beginPath(); ctx.arc(g.X(f.tokens), g.Y(f.cts), 5, 0, 7); ctx.fill();
    ctx.strokeStyle = css("--card"); ctx.lineWidth = 1.5; ctx.stroke();
  }
  const rup = fases.find((f) => f.ruptura_fase);
  if (rup) {
    const alvo = fases[rup.ruptura_fase - 1], x = g.X(alvo.tokens);
    ctx.strokeStyle = css("--bad"); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(x, g.mt); ctx.lineTo(x, g.mt + g.ph); ctx.stroke();
    ctx.fillStyle = css("--bad"); ctx.textAlign = x > g.ml + g.pw - 120 ? "right" : "left"; ctx.textBaseline = "top";
    ctx.fillText(`ruptura ≈ ${alvo.tokens} tokens (fase ${rup.ruptura_fase})`, x + (ctx.textAlign === "left" ? 6 : -6), g.mt + 2);
  }
}

// --------------------------------------------------- política aprendida (barras)
function desenhaPolitica() {
  const c = prepara($("c-politica"));
  if (!resultados) { c.ctx.fillStyle = css("--muted"); c.ctx.fillText("sem resultados de experimentos", 12, 20); return; }
  const f = resultados.politica_aprendida_freq, n = f.length;
  const g = eixos(c, { xmin: 0.5, xmax: n + 0.5, ymin: 0, ymax: 1, xlabel: "fase", ylabel: "frequência", ml: 52, semXticks: true, yfmt: (v) => Math.round(v * 100) + "%" });
  const bw = (g.pw / n) * 0.7, { ctx } = c;
  ctx.fillStyle = css("--axis"); ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let i = 1; i <= n; i++) ctx.fillText(i, g.X(i), g.mt + g.ph + 5);
  f.forEach((linha, i) => {
    let base = 0;
    linha.forEach((p, a) => {
      ctx.fillStyle = corAcao(a);
      const y1 = g.Y(base), y2 = g.Y(base + p);
      ctx.fillRect(g.X(i + 1) - bw / 2, y2, bw, y1 - y2); base += p;
    });
  });
  legenda($("leg-acoes"), ACOES.map((a, i) => [corAcao(i), a]));
}

// ------------------------------------------------------------- mapa de calor
function desenhaCalor() {
  const c = prepara($("c-calor"));
  if (!resultados) return;
  const heat = resultados.mapa_calor.delta_cts, n = heat[0].length, ml = 92, mt = 6, mb = 40;
  const cw = (c.w - ml - 8) / n, ch = (c.h - mt - mb) / heat.length, { ctx } = c;
  const vmax = Math.max(...heat.flat().filter((v) => v !== null).map(Math.abs)) || 1;
  const base = css("--card");
  heat.forEach((linha, a) => {
    ctx.fillStyle = css("--axis"); ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText(ACOES[a], ml - 8, mt + a * ch + ch / 2);
    linha.forEach((v, i) => {
      if (v === null) return;
      const t = Math.max(0, Math.min(1, v / vmax));
      ctx.globalAlpha = 0.12 + 0.88 * t; ctx.fillStyle = css("--a4");
      ctx.fillRect(ml + i * cw + 1, mt + a * ch + 1, cw - 2, ch - 2); ctx.globalAlpha = 1;
      if (cw > 18) { ctx.fillStyle = css("--ink"); ctx.textAlign = "center"; ctx.fillText(Math.round(v * 100), ml + i * cw + cw / 2, mt + a * ch + ch / 2); }
    });
  });
  ctx.fillStyle = css("--axis"); ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let i = 0; i < n; i++) ctx.fillText(i + 1, ml + i * cw + cw / 2, c.h - mb + 4);
  ctx.fillText("fase em que o distrator foi aplicado", ml + (cw * n) / 2, c.h - 13);
}

// ------------------------------------------------------ comparação de políticas
function desenhaComparacao() {
  const c = prepara($("c-comp"));
  if (!resultados) return;
  const pols = resultados.politicas, n = resultados.config.fases;
  const g = eixos(c, { xmin: 1, xmax: n, ymin: 0, ymax: 1, xlabel: "fase", ylabel: "CTS médio", xticks: n - 1, semXticks: true });
  c.ctx.fillStyle = css("--axis"); c.ctx.textAlign = "center"; c.ctx.textBaseline = "top";
  for (let i = 1; i <= n; i++) c.ctx.fillText(i, g.X(i), g.mt + g.ph + 5);
  const { ctx } = c, itens = [];
  ctx.setLineDash([5, 4]); ctx.strokeStyle = css("--axis"); ctx.beginPath();
  ctx.moveTo(g.ml, g.Y(0.5)); ctx.lineTo(g.ml + g.pw, g.Y(0.5)); ctx.stroke(); ctx.setLineDash([]);
  for (const [nome, p] of Object.entries(pols)) {
    ctx.strokeStyle = css(COR_POL[nome]); ctx.lineWidth = nome === "dqn" ? 3 : 2; ctx.beginPath();
    p.cts_medio_por_fase.forEach((v, i) => { const x = g.X(i + 1), y = g.Y(v); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke(); itens.push([css(COR_POL[nome]), ROT_POL[nome]]);
  }
  legenda($("leg-pol"), itens);
}

// -------------------------------------------------------------- curva de treino
function desenhaTreino() {
  const c = prepara($("c-treino"));
  const series = [];
  if (treinoAoVivo.length) series.push(["dqn", treinoAoVivo.map((p) => p.media50), "DQN (ao vivo)"]);
  else if (resultados) for (const [k, v] of Object.entries(resultados.treino)) series.push([k, v, ROT_POL[k]]);
  if (!series.length) return;
  const todos = series.flatMap((s) => s[1]);
  const ymin = Math.min(...todos), ymax = Math.max(...todos), nmax = Math.max(...series.map((s) => s[1].length));
  const g = eixos(c, { xmin: 0, xmax: nmax, ymin, ymax, xlabel: "episódio (média móvel de 50)", ylabel: "recompensa" });
  const { ctx } = c;
  for (const [k, ys] of series) {
    ctx.strokeStyle = css(COR_POL[k]); ctx.lineWidth = 2.5; ctx.beginPath();
    ys.forEach((v, i) => { const x = g.X(treinoAoVivo.length ? i * 5 : i), y = g.Y(v); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
  }
  legenda($("leg-treino"), series.map((s) => [css(COR_POL[s[0]]), s[2]]));
}

// ------------------------------------------------------------------- tabela
function montaTabela() {
  if (!resultados) return;
  const t = el("table"), cab = el("tr");
  ["Política", "Taxa de ruptura", "Fase média da ruptura (IC 95%)", "Tokens até a ruptura", "CTS médio ± dp"].forEach((h) => cab.append(el("th", "", h)));
  t.append(cab);
  for (const [nome, p] of Object.entries(resultados.politicas)) {
    const tr = el("tr"), ic = p.ruptura_fase_ic95;
    const fase = (p.ruptura_fase_media === null || p.taxa_ruptura < 0.05) ? "—" : `${p.ruptura_fase_media.toFixed(2)} [${ic[0].toFixed(2)}; ${ic[1].toFixed(2)}]`;
    [ROT_POL[nome], (p.taxa_ruptura * 100).toFixed(0) + "%", fase,
      (p.ruptura_tokens_media === null || p.taxa_ruptura < 0.05) ? "—" : Math.round(p.ruptura_tokens_media),
      `${p.cts_medio_episodio.toFixed(3)} ± ${p.cts_medio_episodio_desvio.toFixed(3)}`].forEach((v) => tr.append(el("td", "", String(v))));
    t.append(tr);
  }
  const caixa = $("tabela"); caixa.className = ""; caixa.replaceChildren(t);
  const cfg = resultados.config;
  caixa.append(el("p", "vazio", `n = ${cfg.sementes} sementes × ${cfg.episodios_avaliacao_por_semente} episódios por política; ` +
    `${cfg.fases} fases; temperatura ${cfg.temperatura}. LLM: ${cfg.llm}.`));
}

// ------------------------------------------------------- conversa ao vivo
function notaClasse(v, tipo) {
  const alto = tipo === "coerencia" ? 0.7 : 0.99, meio = tipo === "coerencia" ? 0.3 : 0.01;
  return v >= alto ? "ok" : v >= meio ? "meio" : "ruim";
}

function cartaoFase(ev) {
  const d = el("div", "fase" + (ev.ruptura_fase === ev.fase ? " ruptura" : ""));
  const h = el("header");
  h.append(el("b", "", `Fase ${ev.fase}`));
  const b = el("span", "badge", ev.acao_nome); b.style.background = corAcao(ev.acao); h.append(b);
  h.append(el("span", "quem", `${ev.tokens} tokens`));
  if (ev.ruptura_fase === ev.fase) h.append(el("span", "tag", "● início da ruptura"));
  h.append(el("span", "cts", "CTS " + ev.cts.toFixed(2)));
  d.append(h);
  d.append(el("p", "", "Agente: " + ev.distrator));
  d.append(el("p", "quem", "LLM: " + ev.resposta_distrator));
  const s = el("div", "sondas");
  for (const k of ["factual", "restricao", "coerencia"]) {
    const l = el("div", "sonda");
    l.append(el("span", "quem", NOMES_SONDA[k]), el("span", "", ev.respostas[k]));
    l.append(el("span", "nota " + notaClasse(ev.notas[k], k), ev.notas[k].toFixed(2)));
    s.append(l);
  }
  d.append(s);
  return d;
}

function processa(ev) {
  if (ev.tipo === "inicio") {
    run = { eventos: [ev], resumo: null };
    const a = $("ancora"); a.className = "ancora";
    a.textContent = ev.ancora + "\n\nLLM: " + ev.resposta_ancora;
    $("fases-lista").replaceChildren(); $("kpis").textContent = "";
  } else if (ev.tipo === "fase") {
    run.eventos.push(ev);
    const lista = $("fases-lista"); lista.append(cartaoFase(ev));
    lista.lastChild.scrollIntoView({ block: "nearest" });
  } else if (ev.tipo === "fim" && ev.resumo) {
    run.resumo = ev.resumo; mostraKpis();
  } else if (ev.tipo === "treino") {
    treinoAoVivo.push(ev);
    setStatus(`treinando ${ev.episodio}/${ev.total} · ε=${ev.epsilon.toFixed(2)}`, "rodando");
  }
}

function mostraKpis() {
  const r = run.resumo, k = $("kpis"); k.replaceChildren();
  const item = (rot, val) => { const s = el("span"); s.append(rot + " ", el("b", "", val)); k.append(s); };
  item("ruptura na fase", r.ruptura_fase ?? "não houve");
  item("tokens até a ruptura", r.ruptura_tokens ?? "—");
  item("CTS médio", r.cts_medio.toFixed(2));
  item("contexto final", r.tokens_finais + " tokens");
}

// ------------------------------------------------------------- ciclo de vida
function redesenha() { desenhaCTS(); desenhaPolitica(); desenhaCalor(); desenhaComparacao(); desenhaTreino(); }

function bloquear(ocupado) {
  $("btn-iniciar").disabled = ocupado; $("btn-treinar").disabled = ocupado; $("btn-parar").disabled = !ocupado;
}

async function poll() {
  try {
    const r = await api(`/api/estado?id=${jobId}&desde=${desde}`);
    desde = r.total; r.eventos.forEach(processa);
    if (r.tipo === "treino") desenhaTreino(); else { desenhaCTS(); }
    if (r.status !== "rodando") {
      clearInterval(timer); timer = null; bloquear(false);
      if (r.status === "erro") { setStatus("erro", "erro"); aviso(r.erro, true); }
      else { setStatus(r.status === "parado" ? "parado" : "concluído", r.status === "parado" ? "" : "concluido"); aviso(""); }
      if (r.tipo === "treino") carregaConfig();
    }
  } catch (e) { clearInterval(timer); timer = null; bloquear(false); setStatus("erro", "erro"); aviso(e.message, true); }
}

function comecar(j, rotulo) {
  jobId = j.id; desde = 0; bloquear(true); setStatus(rotulo, "rodando"); aviso("");
  timer = setInterval(poll, 450); poll();
}

async function iniciar() {
  const corpo = { provedor: $("provedor").value, modelo: $("modelo").value.trim(), url: $("url").value.trim(),
    politica: $("politica").value, fases: +$("fases").value, temperatura: +$("temp").value,
    seed: $("seed").value, aprender_online: $("online").checked };
  try { run = { eventos: [], resumo: null }; desenhaCTS(); comecar(await api("/api/executar", corpo), "executando"); }
  catch (e) { aviso(e.message, true); }
}

async function treinar() {
  treinoAoVivo = [];
  try { comecar(await api("/api/treinar", { episodios: 600, fases: +$("fases").value }), "treinando"); }
  catch (e) { aviso(e.message, true); }
}

function ajustaCampos() {
  const p = $("provedor").value;
  $("lbl-modelo").hidden = p === "simulada"; $("lbl-url").hidden = !(p === "ollama" || p === "openai");
  $("modelo").placeholder = { ollama: "llama3.2", openai: "gpt-4o-mini", gemini: "gemini-2.0-flash" }[p] || "";
  $("url").placeholder = { ollama: "http://localhost:11434", openai: "https://api.openai.com" }[p] || "";
}

async function carregaConfig() {
  const cfg = await api("/api/config");
  ACOES = cfg.acoes;
  if (!$("politica").options.length) cfg.politicas.forEach((p) => $("politica").add(new Option(ROT_POL[p] || p, p)));
  if (!cfg.dqn_treinado) aviso("DQN sem pesos: clique em “Treinar DQN” (≈10 s) ou use outra política.");
  else if ($("aviso").textContent.startsWith("DQN sem pesos")) aviso("");
}

async function init() {
  $("provedor").addEventListener("change", ajustaCampos); ajustaCampos();
  $("btn-iniciar").addEventListener("click", iniciar);
  $("btn-treinar").addEventListener("click", treinar);
  $("btn-parar").addEventListener("click", () => api("/api/parar", {}));
  await carregaConfig();
  try { resultados = await api("/api/resultados"); montaTabela(); } catch (_) { /* ainda sem experimentos */ }
  redesenha();
  let t; window.addEventListener("resize", () => { clearTimeout(t); t = setTimeout(redesenha, 120); });
}
init().catch((e) => aviso("Falha ao iniciar: " + e.message, true));
