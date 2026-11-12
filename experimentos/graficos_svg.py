"""Gráficos SVG feitos à mão (sem matplotlib) para colar no relatório da Fase 3."""
import os

CORES = {  # paleta Okabe-Ito (distinguível por daltônicos)
    "dqn": "#D55E00", "q_tabular": "#CC79A7", "aleatoria": "#56B4E9",
    "ciclica": "#009E73", "fixa_conflito": "#E69F00", "fixa_neutro": "#7F7F7F",
}
ROTULOS = {
    "dqn": "DQN (aprendiz)", "q_tabular": "Q-tabular", "aleatoria": "Aleatória",
    "ciclica": "Cíclica", "fixa_conflito": "Sempre conflito", "fixa_neutro": "Controle (neutro)",
}
W, H, ML, MR, MT, MB = 760, 420, 60, 170, 40, 55


def _svg(corpo, titulo):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
            f'font-family="sans-serif" font-size="12">'
            f'<rect width="{W}" height="{H}" fill="#fff"/>'
            f'<text x="{ML}" y="22" font-size="15" font-weight="bold">{titulo}</text>{corpo}</svg>')


def _eixos(xmax, ymin, ymax, xlabel, ylabel, passos_y=5):
    pw, ph = W - ML - MR, H - MT - MB
    s = [f'<line x1="{ML}" y1="{MT + ph}" x2="{ML + pw}" y2="{MT + ph}" stroke="#444"/>',
         f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT + ph}" stroke="#444"/>']
    for i in range(passos_y + 1):
        v = ymin + (ymax - ymin) * i / passos_y
        y = MT + ph - ph * i / passos_y
        s.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML + pw}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        s.append(f'<text x="{ML - 8}" y="{y + 4:.1f}" text-anchor="end">{v:.2f}</text>')
    s.append(f'<text x="{ML + pw / 2}" y="{H - 12}" text-anchor="middle">{xlabel}</text>')
    s.append(f'<text transform="translate(16 {MT + ph / 2}) rotate(-90)" text-anchor="middle">{ylabel}</text>')
    return "".join(s)


def _legenda(itens):
    x = W - MR + 16
    s = []
    for i, (cor, rot) in enumerate(itens):
        y = MT + 8 + i * 20
        s.append(f'<line x1="{x}" y1="{y}" x2="{x + 18}" y2="{y}" stroke="{cor}" stroke-width="3"/>'
                 f'<text x="{x + 24}" y="{y + 4}">{rot}</text>')
    return "".join(s)


def grafico_cts(res):
    n = res["config"]["fases"]
    pw, ph = W - ML - MR, H - MT - MB
    xs = lambda i: ML + pw * i / (n - 1)
    ys = lambda v: MT + ph * (1 - v)
    corpo = [_eixos(n, 0, 1, "fase (cada fase ≈ 50–70 tokens a mais de contexto)", "CTS médio")]
    lim = res["config"]["limiar_ruptura"]
    corpo.append(f'<line x1="{ML}" y1="{ys(lim):.1f}" x2="{ML + pw}" y2="{ys(lim):.1f}" '
                 f'stroke="#000" stroke-dasharray="5 4"/><text x="{ML + 4}" y="{ys(lim) - 5:.1f}">limiar de ruptura</text>')
    for i in range(n):
        corpo.append(f'<text x="{xs(i):.1f}" y="{MT + ph + 16}" text-anchor="middle">{i + 1}</text>')
    itens = []
    for nome, p in res["politicas"].items():
        pts = " ".join(f"{xs(i):.1f},{ys(v):.1f}" for i, v in enumerate(p["cts_medio_por_fase"]))
        corpo.append(f'<polyline points="{pts}" fill="none" stroke="{CORES[nome]}" stroke-width="2.5"/>')
        itens.append((CORES[nome], ROTULOS[nome]))
    corpo.append(_legenda(itens))
    return _svg("".join(corpo), "Curva de retenção de contexto (CTS) por política")


def grafico_ruptura(res):
    # políticas com <5% de rupturas não têm média informativa (ex.: controle)
    nomes = [n for n, p in res["politicas"].items() if p["taxa_ruptura"] >= 0.05]
    n_fases = res["config"]["fases"]
    pw, ph = W - ML - MR, H - MT - MB
    corpo = [_eixos(0, 0, n_fases, "", "fase média da ruptura (barra de erro = IC 95%)", 6)]
    largura = pw / max(len(nomes), 1)
    for i, nome in enumerate(nomes):
        p = res["politicas"][nome]
        v, (lo, hi) = p["ruptura_fase_media"], p["ruptura_fase_ic95"]
        x = ML + i * largura + largura * 0.2
        bw = largura * 0.6
        y = MT + ph * (1 - v / n_fases)
        corpo.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{MT + ph - y:.1f}" fill="{CORES[nome]}"/>')
        ylo, yhi = (MT + ph * (1 - lo / n_fases), MT + ph * (1 - hi / n_fases))
        cx = x + bw / 2
        corpo.append(f'<line x1="{cx:.1f}" y1="{ylo:.1f}" x2="{cx:.1f}" y2="{yhi:.1f}" stroke="#000" stroke-width="1.5"/>')
        corpo.append(f'<text x="{cx:.1f}" y="{yhi - 6:.1f}" text-anchor="middle">{v:.2f}</text>')
        corpo.append(f'<text x="{cx:.1f}" y="{MT + ph + 16}" text-anchor="middle" font-size="10">{ROTULOS[nome].split(" (")[0]}</text>')
    return _svg("".join(corpo), "Fase média do ponto de ruptura (menor = estresse mais eficiente)")


def grafico_treino(res):
    pw, ph = W - ML - MR, H - MT - MB
    todas = [v for c in res["treino"].values() for v in c]
    ymin, ymax = min(todas), max(todas)
    n = len(next(iter(res["treino"].values())))
    corpo = [_eixos(n, ymin, ymax, "episódio de treino", "recompensa média (janela 50)")]
    itens = []
    for nome, c in res["treino"].items():
        pts = " ".join(f"{ML + pw * i / (n - 1):.1f},{MT + ph * (1 - (v - ymin) / (ymax - ymin)):.1f}"
                       for i, v in enumerate(c) if i % max(1, n // 300) == 0)
        corpo.append(f'<polyline points="{pts}" fill="none" stroke="{CORES[nome]}" stroke-width="2.5"/>')
        itens.append((CORES[nome], ROTULOS[nome]))
    corpo.append(_legenda(itens))
    return _svg("".join(corpo), "Aprendizado: recompensa por episódio")


def grafico_calor(res):
    heat = res["mapa_calor"]["delta_cts"]
    acoes = res["acoes"]
    n = res["config"]["fases"]
    ml = 110  # margem esquerda maior para os nomes das ações
    pw, ph = W - ml - 20, H - MT - MB
    cw, ch = pw / n, ph / len(acoes)
    vmax = max(abs(v) for l in heat for v in l if v is not None) or 1
    corpo = []
    for a, linha in enumerate(heat):
        corpo.append(f'<text x="{ml - 8}" y="{MT + a * ch + ch / 2 + 4:.1f}" text-anchor="end">{acoes[a]}</text>')
        for i, v in enumerate(linha):
            if v is None:
                continue
            t = max(0.0, min(1.0, v / vmax))  # queda de CTS -> mais escuro
            r, g, b = int(255 - 70 * t), int(255 - 200 * t), int(255 - 255 * t)
            corpo.append(f'<rect x="{ml + i * cw:.1f}" y="{MT + a * ch:.1f}" width="{cw:.1f}" height="{ch:.1f}" '
                         f'fill="rgb({r},{g},{b})" stroke="#fff"/>'
                         f'<text x="{ml + i * cw + cw / 2:.1f}" y="{MT + a * ch + ch / 2 + 4:.1f}" '
                         f'text-anchor="middle" font-size="10">{v:+.2f}</text>')
    for i in range(n):
        corpo.append(f'<text x="{ml + i * cw + cw / 2:.1f}" y="{MT + ph + 16}" text-anchor="middle">{i + 1}</text>')
    corpo.append(f'<text x="{ml + pw / 2}" y="{H - 12}" text-anchor="middle">fase em que o distrator foi aplicado</text>')
    return _svg("".join(corpo), "Queda de CTS causada por cada distrator (política aleatória)")


def gerar_todos(res, pasta):
    for nome, fn in (("cts_por_politica", grafico_cts), ("ruptura_por_politica", grafico_ruptura),
                     ("curva_treino", grafico_treino), ("mapa_calor", grafico_calor)):
        with open(os.path.join(pasta, nome + ".svg"), "w", encoding="utf-8") as f:
            f.write(fn(res))
