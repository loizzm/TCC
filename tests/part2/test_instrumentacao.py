"""Guarda da INSTRUMENTAÇÃO do relatório da Parte 2 — não mede o modelo.

Motivo de existir (HANDOFF_P2_7 §35.3 e §35.7-2): o achado mais caro do bloco do
caso real foi um PONTO CEGO DE MEDIÇÃO — a suíte inteira passava verde por uma
regressão porque a linha que a denunciaria não estava lá. E o mesmo padrão
reincidiu DUAS vezes no mesmo arquivo no mesmo dia, nos dois casos por um
diagnóstico que SUMIA do relatório quando o `n` ficava baixo:

  1. um `return` antecipado sob `if aceitas < 100:` levava embora o bloco
     `2.6-adim` inteiro — cujo motivo de existir é justamente medir o caminho que
     NÃO depende da calibração, e que portanto desaparecia exatamente quando a
     calibração piorava;
  2. o `record_p2("2.6-adim-aceitas", ...)` ficava dentro do
     `if aceitas_adim >= 100:` sem par no `else`, e era a única linha do bloco a
     carregar o `n` e o tamanho do subconjunto sem calibração.

As duas foram corrigidas, e as duas eram desfazíveis sem que nada acusasse: com a
população real (`aceitas = 214`) o portão nunca dispara, então a suíte de produção
prova apenas não-regressão. Esta guarda fecha esse vão.

A invariante que ela sustenta, e é a única:

    NENHUM critério declarado desaparece do relatório por causa de um portão de
    `n`. Se o `n` for insuficiente, a linha APARECE dizendo isso, com o `n` real.

Um diagnóstico que some quando a população encolhe é pior que nenhum: quem
compara duas rodadas não vê que a base mudou, e a ausência lê-se como "não houve
problema" em vez de "não foi medido".

Desenho: o teste NÃO roda o pipeline nem carrega o modelo nem toca `data/`. Ele
recorta por AST a REGIÃO DE RELATÓRIO de `test_2_6_degradacao_vs_oraculo` — tudo
a partir do primeiro `record_p2` de topo, ou seja depois do laço de medição — e a
executa duas vezes com ACUMULADORES INJETADOS, uma com `n` alto e outra com `n`
baixo, coletando as chamadas num `record_p2` de mentira. É instrumentação sob
teste, não modelo. Custo: milissegundos.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

ALVO = Path(__file__).with_name("test_part2.py")
FUNCAO = "test_2_6_degradacao_vs_oraculo"

# Quantos `record_p2` com id LITERAL a região de relatório declara hoje. Ver o
# piso em `_regiao_de_relatorio`: sem acompanhar o número corrente, a guarda
# aceita silenciosamente que critérios sumam do arquivo.
N_CRITERIOS_LITERAIS = 12


def _regiao_de_relatorio() -> tuple[list[ast.stmt], list[str]]:
    """Statements do relatório de `FUNCAO` + os ids de critério FIXOS que ela emite.

    A fronteira é o primeiro `record_p2` no nível de topo da função: antes dele é
    o laço que mede, depois dele é o que reporta. Só o que reporta interessa aqui.

    Os ids devolvidos são os literais — `record_p2(f"2.6[{k}]", ...)` fica de fora
    de propósito: aquele é UMA LINHA POR PARÂMETRO medido, então sua ausência com
    `n` baixo é consequência de não haver o que medir, não de um portão engolindo
    um critério declarado. Os literais são os critérios que existem sempre.
    """
    arvore = ast.parse(ALVO.read_text(encoding="utf-8"), filename=str(ALVO))
    fn = next((n for n in ast.walk(arvore)
               if isinstance(n, ast.FunctionDef) and n.name == FUNCAO), None)
    assert fn is not None, f"{FUNCAO} não existe em {ALVO} — teste desatualizado"

    inicio = None
    for i, st in enumerate(fn.body):
        if (isinstance(st, ast.Expr) and isinstance(st.value, ast.Call)
                and isinstance(st.value.func, ast.Name)
                and st.value.func.id == "record_p2"):
            inicio = i
            break
    assert inicio is not None, (
        f"{FUNCAO} não tem nenhum `record_p2` no nível de topo — ou o relatório "
        "sumiu, ou a função foi reestruturada e esta guarda precisa acompanhar")

    regiao = fn.body[inicio:]
    ids: list[str] = []
    for st in regiao:
        for no in ast.walk(st):
            if (isinstance(no, ast.Call) and isinstance(no.func, ast.Name)
                    and no.func.id == "record_p2" and no.args
                    and isinstance(no.args[0], ast.Constant)
                    and isinstance(no.args[0].value, str)):
                if no.args[0].value not in ids:
                    ids.append(no.args[0].value)
    # PISO, e ele acompanha o número corrente (re-review, R-4). Com o piso em 4
    # e 12 ids declarados, OITO critérios poderiam ser apagados sem que esta
    # guarda reclamasse — ela deixaria de cobrir justamente o que promete. O
    # número é atualizado à mão quando um critério entra ou sai de propósito;
    # se cair sozinho, é apagamento acidental e tem de doer.
    assert len(ids) >= N_CRITERIOS_LITERAIS, (
        f"a região de relatório declara {len(ids)} critérios literais, menos "
        f"que os {N_CRITERIOS_LITERAIS} correntes: {ids}. Se algum foi removido "
        "de propósito, baixe N_CRITERIOS_LITERAIS junto e diga por quê.")
    return regiao, ids


def _executa(regiao: list[ast.stmt], acumuladores: dict) -> dict[str, str]:
    """Roda a região de relatório com os acumuladores injetados.

    A região vira o corpo de uma função sem argumentos: os nomes injetados são
    lidos do namespace global de execução, e os locais dela (`pior`, `d_adim`, …)
    continuam locais. É isso que faz um `return` reintroduzido ser DETECTÁVEL em
    vez de virar `SyntaxError`: ele simplesmente interrompe a função, as linhas
    seguintes não são registradas, e a asserção do teste acusa quais faltaram.
    """
    registrado: dict[str, str] = {}

    def record_p2(cid, name, target, measured, ok):   # noqa: ANN001 — dublê
        assert cid not in registrado, f"critério {cid!r} registrado duas vezes"
        registrado[cid] = measured

    fn = ast.FunctionDef(
        name="_relatorio", args=ast.arguments(
            posonlyargs=[], args=[], vararg=None, kwonlyargs=[],
            kw_defaults=[], kwarg=None, defaults=[]),
        body=[ast.fix_missing_locations(ast.copy_location(st, st)) for st in regiao],
        decorator_list=[], returns=None, type_params=[])
    mod = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))

    ns = {"record_p2": record_p2, "np": np, **acumuladores}
    exec(compile(mod, "<regiao-de-relatorio>", "exec"), ns)   # noqa: S102
    ns["_relatorio"]()
    return registrado


def _acumuladores(n_fisico: int, n_adim: int, n_wnT: int, n_sem_calib: int) -> dict:
    """Acumuladores plausíveis com as populações pedidas, sem rodar nada.

    Os erros são constantes e o do real fica ACIMA do oráculo por uma folga
    pequena, para que os `assert` de degradação da própria região passem: o que
    está sob teste é a presença das linhas, não o veredito delas.
    """
    erros = {k: [1.0] * max(n_fisico, 1) for k in ("K", "tau", "theta", "wn", "zeta")}
    return {
        "test_samples": [None] * 300,
        "aceitas": n_fisico,
        "err_oraculo": {k: list(v) for k, v in erros.items()},
        "err_real": {k: [x + 0.5 for x in v] for k, v in erros.items()},
        "aceitas_adim": n_adim,
        "sem_calib_adim": n_sem_calib,
        "err_orac_adim": [1.0] * max(n_adim, 1),
        "err_real_adim": [1.5] * max(n_adim, 1),
        "aceitas_wnT_adim": n_wnT,
        "err_orac_wnT_adim": [0.7] * max(n_wnT, 1),
        "err_real_wnT_adim": [1.2] * max(n_wnT, 1),
        "err_orac_wnT_sc": [0.7] * n_sem_calib,
        "err_real_wnT_sc": [1.2] * n_sem_calib,
        # θ/T adimensional e acerto de ordem (revisão final, §6). Reaproveitam
        # as populações de ωₙ e do subconjunto sem calibração de propósito: o
        # que este dublê precisa reproduzir é a FORMA dos acumuladores, não os
        # valores. Quando a região de relatório ganha um acumulador novo, ele
        # tem de aparecer aqui — a guarda acusa a falta com um `NameError` na
        # região recortada, que é a quebra ruidosa prevista na docstring.
        "aceitas_thT_adim": n_wnT,
        "err_orac_thT_adim": [0.7] * max(n_wnT, 1),
        "err_real_thT_adim": [1.2] * max(n_wnT, 1),
        "err_orac_thT_sc": [0.7] * n_sem_calib,
        "err_real_thT_sc": [1.2] * n_sem_calib,
        "ordem_ok": n_fisico,
        "ordem_ok_sc": n_sem_calib,
        "ordem_n_sc": n_sem_calib,
        "aceitas_Kyr_adim": n_wnT,
        "err_orac_Kyr_adim": [0.7] * max(n_wnT, 1),
        "err_real_Kyr_adim": [1.2] * max(n_wnT, 1),
        "err_orac_Kyr_sc": [0.7] * n_sem_calib,
        "err_real_Kyr_sc": [1.2] * n_sem_calib,
    }


def test_nenhum_criterio_some_por_portao_de_n():
    """Com `n` ALTO e com `n` BAIXO, o MESMO conjunto de critérios é registrado.

    Este é o teste que falha se alguém reintroduzir o `return` antecipado sob
    `if aceitas < 100:`, ou prender qualquer outra linha declarada dentro de um
    portão de `n` sem par no `else`.
    """
    regiao, esperados = _regiao_de_relatorio()

    alto = _executa(regiao, _acumuladores(214, 143, 143, 33))
    baixo = _executa(regiao, _acumuladores(3, 2, 1, 0))

    faltando_alto = [c for c in esperados if c not in alto]
    assert not faltando_alto, (
        f"com n alto, critérios declarados e não registrados: {faltando_alto}")

    faltando_baixo = [c for c in esperados if c not in baixo]
    assert not faltando_baixo, (
        "PORTÃO DE `n` ENGOLIU CRITÉRIO. Com n baixo, estes critérios são "
        f"declarados mas NÃO chegam ao relatório: {faltando_baixo}. "
        "Um diagnóstico que some quando a população encolhe some no pior "
        "momento possível — ver HANDOFF_P2_7 §35.3 e §35.7-2. O ramo de `n` "
        "insuficiente tem de REGISTRAR a linha dizendo isso, não pular.")

    # A diferença entre as duas rodadas é LEGÍTIMA só para as linhas por
    # parâmetro (`2.6[K]`, `2.6[tau]`, …): elas existem uma por grandeza que
    # HOUVE o que medir, e com n baixo não há. Qualquer outra ausência é o
    # defeito que esta guarda persegue.
    so_no_alto = set(alto) - set(baixo)
    indevidas = [c for c in so_no_alto
                 if not (c.startswith("2.6[") and c.endswith("]"))]
    assert not indevidas, (
        "critérios que existem com n alto e somem com n baixo, sem serem "
        f"linhas por parâmetro: {sorted(indevidas)}")
    assert not set(baixo) - set(alto), (
        f"critérios que só aparecem com n baixo: {sorted(set(baixo) - set(alto))}")


def test_linha_de_n_insuficiente_carrega_o_n_real():
    """Não basta a linha existir: ela tem de dizer QUAL era o `n`.

    Uma linha "não asseverável" sem o número não deixa comparar duas rodadas —
    era metade do defeito do `2.6-adim-aceitas`, a única linha do bloco que
    carregava a contagem e o tamanho do subconjunto sem calibração.
    """
    regiao, _ = _regiao_de_relatorio()
    baixo = _executa(regiao, _acumuladores(3, 2, 1, 0))

    # Todo id com portão de `n` entra aqui — não só os antigos. "Conferido à
    # mão, não virou teste" é o padrão que esta guarda existe para fechar
    # (re-review, R-3): as linhas novas de θ, K e ordem entraram no teste de
    # PRESENÇA e ficaram fora do de CONTEÚDO por um turno.
    for cid, marca in (("2.6", "n=3"), ("2.6-adim[zeta]", "n=2"),
                       ("2.6-adim[wn_T]", "n=1"),
                       ("2.6-adim[theta_T]", "n=1"),
                       ("2.6-adim[K_yrange]", "n=1"),
                       ("2.12-ordem", "n=300")):
        assert marca in baixo[cid], (
            f"{cid} não diz o `n` real com n baixo: {baixo[cid]!r} "
            f"(esperado conter {marca!r})")

    assert "2/300" in baixo["2.6-adim-aceitas"], (
        f"2.6-adim-aceitas perdeu a contagem: {baixo['2.6-adim-aceitas']!r}")
    assert "0 sem calibração" in baixo["2.6-adim-aceitas"], (
        "2.6-adim-aceitas perdeu o tamanho do subconjunto sem calibração: "
        f"{baixo['2.6-adim-aceitas']!r}")

    # E o subconjunto sem calibração, que não tem portão de `n` por construção
    # (é pequeno de propósito), registra `n=0` em vez de sumir.
    for cid in ("2.6-adim[wn_T/sem-calib]", "2.6-adim[theta_T/sem-calib]",
                "2.6-adim[K_yrange/sem-calib]", "2.12-ordem[sem-calib]"):
        assert "n=0" in baixo[cid], (
            f"linha sem-calib com população vazia não diz `n=0`: {baixo[cid]!r}")
