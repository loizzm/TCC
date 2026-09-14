# Os lotes de avaliação — o que está versionado, e como refazer cada um

**O que fica no git:** os JSON de cada lote (`verdade.json`, `resultado.json`,
`analise.json`) — que são a MEDIDA — e **dois PNGs por lote**, como amostra do
render. As demais figuras saíram do versionamento em 14/09/2026: pesavam
102 MB, e nenhuma conclusão do trabalho depende de reler o pixel. Os números
continuam auditáveis, porque os JSON continuam aqui.

**As figuras removidas não se perderam.** Elas seguem no histórico do git. Para
recuperar um lote inteiro, sem precisar saber a receita:

```bash
git checkout 943e2a1 -- reports/amostras_aleatorias/<lote>/
```

`943e2a1` é o último commit em que todas estavam versionadas. Isso vale para
QUALQUER lote desta tabela, inclusive os de proveniência não registrada.

**Ressalva sobre "regerar".** Mesmo com a semente certa, a figura só sai byte a
byte igual se as versões de matplotlib e numpy forem as mesmas. A física e a
verdade saem idênticas; o pixel, não necessariamente. Para auditar um número
publicado, prefira o `git checkout` acima.

## Os lotes

| lote | n | receita | confiança |
|---|---|---|---|
| `(raiz)` `rg_*`/`neg_*`/`multi_*` | 99 | `rg_aleatorio.py --n 33 --seed 20260908 --modo familia` | verificada (33×3, defaults do script) |
| `balanceado/` | 100 | `rg_aleatorio.py --modo balanceado --seed 20260908` | semente registrada no RELATORIO §205 |
| `balanceado2/` | 100 | `rg_aleatorio.py --modo balanceado --seed 20260909` | semente registrada no RELATORIO §206 |
| `lote100/` | 100 | `rg_aleatorio.py --modo balanceado --seed 20260909` | semente registrada no RELATORIO §1363 |
| `lote_ruido/` | 100 | `gera_lote_ruidoso.py` (defaults: `--por-celula 10 --seed 20260911`) | verificada (docstring do script) |
| `lote_misto2/` | 400 | `gera_lote_ruidoso.py --por-celula 20 --seed 20260914 --snrs 5 10 15 20 25 30 35 40 50 60 --p-escuro 0.5 --out reports/amostras_aleatorias/lote_misto2` | verificada |
| `sem_omite/` | 100 | `rg_aleatorio.py --entrada omite` | comando no RELATORIO §628; semente NÃO registrada |
| `sem_omite_fit/` | 100 | `rg_aleatorio.py --entrada omite_fit` | comando no RELATORIO §628; semente NÃO registrada |
| `lote_k_maior1/` | 100 | `rg_aleatorio.py --modo balanceado`, filtrado por \|K\| ≥ 1 | **não registrada** |
| `lote_k_menor1/` | 100 | `rg_aleatorio.py --modo balanceado`, filtrado por \|K\| < 1 | **não registrada** |
| `lote_selecao/` | 100 | `rg_aleatorio.py --modo balanceado`, semente nova, separado dos lotes de controle | **não registrada** |
| `lote100_v2/` | 100 | `rg_aleatorio.py --modo balanceado` | **não registrada** |
| `lote100_piso/` | 100 | `rg_aleatorio.py --modo balanceado` | **não registrada** |
| `lote100_1deg/` | 100 | `rg_aleatorio.py --modo balanceado` | **não registrada** |
| `fase_nao_minima/` | 40 | estrato `fase_nao_minima` do `dataset/generator.py` | **não registrada** |
| `nmp_render/` | 200 | `sonda_render_nmp.py` | **não registrada** |
| `nmp_ruido/` | 100 | variante ruidosa do estrato de fase não-mínima | **não registrada** |

**Os marcados "não registrada" são exatamente por que este arquivo existe.**
Nove lotes foram gerados sem que o comando e a semente ficassem escritos em
lugar nenhum, e isso só apareceu quando se tentou reduzir o versionamento. Para
esses, o `git checkout` acima é o único caminho de volta.

**Regra para lotes novos:** quem gerar um lote grava a linha de comando
completa — script, semente, `--out` — aqui nesta tabela, no mesmo commit que
traz o lote. É o que torna a redução acima segura de repetir.

## Os três lotes de CONTROLE

`lote_k_maior1`, `lote_k_menor1` e `lote_ruido` são os lotes contra os quais os
números de assertividade da Parte 2 são citados. Eles NÃO entram em escolha de
época nem em calibração de constante — para isso existe o `lote_selecao`, que é
de outra semente. Misturar os dois papéis faria o número final deixar de ser
estimativa de generalização.
