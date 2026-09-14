---
tags: [tcc, indice, home]
aliases: [Home, Índice]
---

# TCC — Identificação de plantas a partir de imagens

> Dado **um gráfico de resposta ao degrau como imagem** — um PNG qualquer, com resolução, cores, grade, legendas e ruído arbitrários — recuperar os **parâmetros da planta** que o gerou: ganho `K`, constante de tempo `τ`, atraso `θ`, frequência natural `ωn` e amortecimento `ζ`.

## Documentos

| | |
|---|---|
| [[Arquitetura do projeto]] | Visão macro: os quatro estágios, os algoritmos e por que cada um |
| [[Módulos]] | Um arquivo por módulo, com trechos do código, entradas e saídas |
| [[Decisões de projeto]] | O log de decisões — inclusive as duas refutadas |
| [[Critérios de qualidade]] | Como cada estágio é medido e de onde vem cada limiar |
| [[Anatomia da U-Net]] | O U desenhado: canais, resolução, parâmetros e custo por bloco |
| [[AIC e seleção de ordem]] | Como o Estágio D escolhe entre FOPDT e 2ª ordem |
| [[Proveniência]] | O que veio da literatura, o que é nosso, e as lacunas |

## Estado

| | |
|---|---|
| Estágios | 4, cada um com contrato próprio |
| Código de produção | 4 032 linhas |
| Suíte | 69 testes, 0 falhas |
| Cobertura da calibração física | 93,0 % |
| Erro geométrico da máscara | 0,799 px mediana · 1,703 px p95 |
| Modelo em produção | U-Net `base=32`, `in_ch=3`, 7 763 041 parâmetros |
| Parte 2 | fechada |
| Parte 3 | validação OOD, utilidade em controle, baseline CNN fim-a-fim |

## Como rodar

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

# identificar uma imagem
.venv/bin/python identificar.py grafico.png

# a suíte
.venv/bin/python -m pytest -q

# gerar corpus
.venv/bin/python -m dataset.generator data/train 6000 0
```

## Leitura recomendada, em ordem

1. [[Arquitetura do projeto]] — a cadeia de representações e os contratos de fronteira
2. [[Decisões de projeto#D1 · Saída em dois níveis]] — a decisão que organiza a saída
3. [[Decisões de projeto#D2 · RGB em vez de luminância]] — a mudança mais consequente
4. [[Critérios de qualidade#2.1 — o portão do Estágio A]] — por que IoU foi demitido
5. [[Módulos]] — o detalhe de implementação, módulo a módulo

## Dívidas abertas

- Guarda de parâmetro na borda da caixa (`K` no teto com `ok=true`)
- Sinalização de ordem ambígua
- Detecção/recusa de múltiplas curvas
- Latência com `base=32` (critério 3.11)
- Critério A.0 mede os defaults, não o checkpoint em produção
- As 13 imagens externas não estão versionadas como fixtures
- 5 das 13 serviram de diagnóstico antes de servir de validação

Detalhe em [[Decisões de projeto#Dívidas conhecidas]].
