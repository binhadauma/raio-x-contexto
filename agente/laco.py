"""Laço agente–ambiente compartilhado por servidor, treino e experimentos."""


def rodar_episodio(env, politica, treinar=False, explorar=False, callback=None,
                   seed=None, parar=None):
    """Executa um episódio completo.

    treinar : se True, chama politica.observar(...) a cada passo (aprendizado online)
    explorar: se True, usa ε-greedy; se False, política gulosa/determinística
    callback: função(evento) chamada após cada fase (alimenta o dashboard ao vivo)
    parar   : função() -> bool; se retornar True, interrompe o episódio
    Retorna (resumo_do_ambiente, recompensa_total).
    """
    obs = env.reset(seed=seed)
    total, feito = 0.0, False
    while not feito:
        if parar is not None and parar():
            break
        a = politica.agir(obs, explorar=explorar)
        obs2, r, feito, evento = env.step(a)
        politica.observar(obs, a, r, obs2, feito, treinar=treinar)
        total += r
        obs = obs2
        if callback:
            callback(evento)
    return env.resumo(), total
