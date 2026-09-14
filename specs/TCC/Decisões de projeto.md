---
tags: [tcc, decisoes, adr]
aliases: [ADR, Decisões]
---

# Decisões de projeto

> Cada entrada registra **o que foi decidido**, **contra o que**, **com que evidência** e **o que custaria se estiver errada**. Uma decisão sem contrafactual medido não é decisão — é preferência.

Duas decisões abaixo estão marcadas como **refutadas**. Elas ficam no documento de propósito: o processo que as derrubou é o resultado mais transferível do projeto.

Índice: [[Arquitetura do projeto]] · [[Módulos]] · [[Critérios de qualidade]]

---

## D1 · Saída em dois níveis

**Decisão.** A pipeline devolve `dimensionless` **sempre** e `physical` **só quando a calibração fecha**, em vez de abortar a amostra inteira quando os eixos não são lidos.

**Contra o que.** O comportamento anterior: `cal.ok == False` → amostra perdida.

**Evidência.** A calibração falhar custava **56 das ~68 amostras perdidas** do pipeline. Dessas, 53 foram recuperadas com ζ a 2,93 % de MAPE, contra 2,40 % do caminho físico — uma degradação de meio ponto percentual para recuperar 79 % das perdas.

**O argumento físico, que é o que sustenta a decisão.** *Amortecimento é adimensional.* ζ se lê da **forma** da curva, não da escala dos eixos. O que a calibração dá é a *unidade*: ωn em rad/s precisa dela, ζ não. Separar os dois níveis não é um paliativo — é reconhecer que as duas grandezas têm requisitos diferentes.

**Compatibilidade preservada, deliberadamente.** `params` e `order` no topo continuam sendo os do nível físico, e `ok` continua significando "há saída física", não "há resposta". Quem quer a estrutura sem calibração lê `order`; quem quer o nível adimensional lê `dimensionless`, que nunca é nulo.

**Custo se errada.** Consumidores que tratassem `dimensionless` como físico teriam números sem unidade. Mitigado por `physical` ser `None` — não `{}` — exatamente quando não existe.

Implementa: [[identify.pipeline]] · Assevera: critério 2.11 (300/300)

---

## D2 · RGB em vez de luminância

**Decisão.** A U-Net recebe **3 canais**. A projeção `0,299R + 0,587G + 0,114B` (ITU-R BT.601) foi removida do caminho da rede.

**Contra o que.** A versão original, que convertia para cinza antes da rede. Parecia inofensivo: a curva é escura sobre fundo claro, então a cor não deveria importar.

**Evidência.** É uma projeção de ℝ³ em ℝ¹, e portanto **destrutiva**. Medido numa imagem real: a curva verde `(44,160,44)` e a reta de referência vermelha `(230,61,61)` viram **o mesmo byte 112**. Onde se cruzavam, o contraste caía para 32 % do original e a máscara perdia **244 colunas consecutivas** — 39 % da largura.

**O que torna esta a decisão mais consequente do projeto.** Separar a curva da reta não era *difícil*: era **impossível**, porque a informação já tinha sido descartada antes da rede ver a imagem. Nenhuma quantidade de capacidade, dado ou treino conserta uma entrada da qual o sinal foi removido.

**Por que demorou a ser vista.** A suposição implícita de que luminância bastava nunca foi contradita pelas 900 amostras sintéticas — porque o gerador **não sorteava cores colidentes**. O corpus não continha o modo de falha. Ver [[#O padrão que aparece três vezes]].

**Custo.** +576 parâmetros (a primeira convolução) e um retreino completo. Os checkpoints de 1 canal continuam carregando, porque `load_model` infere `in_ch` do próprio arquivo.

Implementa: [[identify.extract]] · [[train_unet]] · Estrato de regressão: `reta_no_patamar`

---

## D3 · O blob de texto é a posição do tick

**Decisão.** A posição de um tick é o **centro do blob de texto** do rótulo, não a marca de tick desenhada.

**Contra o que.** A abordagem intuitiva: detectar as marcas e recortar o texto em volta de cada uma.

**Evidência.** A abordagem intuitiva falha sempre que há ticks menores **sem rótulo** entre os maiores — o recorte lê o rótulo do tick *vizinho*, e os pares saem com valor certo em posição errada. Medido: essa versão levava só **2 de 30** amostras a uma calibração aceita.

**Ganho estrutural.** Não depende de marca nenhuma existir — o que importa, porque o gerador sorteia figuras sem marcas visíveis (`has_major_ticks` é 90 %, não 100 %). A detecção de marcas continua implementada mas ficou **sem uso na cadeia principal**.

**Consequência posterior.** As marcas de tick, agora irrelevantes para a posição, passaram a ser **estorvo**: elas ficam dentro da faixa de rótulos e se fundem com o texto na dilatação. Daí `TICK_GAP = 8` e a redução de `BLOB_DILATE_X` de 8 para 3.

Implementa: [[identify.calibrate]] · Efeito medido: âncora do 1º tick a ≤3 px, **76,1 % → 99,9 %**

---

## D4 · RANSAC antes da consistência

**Decisão.** Ajustar a afim por RANSAC **primeiro**, e checar consistência (equiespaçamento) **depois**, sobre os *inliers*.

**Contra o que.** A ordem do esboço original: consistência antes do ajuste.

**Evidência.** O OCR erra cerca de **1 em cada 5 pares**. Checar consistência antes faz **um** valor errado reprovar a amostra inteira — exatamente o outlier que o RANSAC existe para descartar. Medido: essa era a causa de quase toda reprovação por `calibration_failed`.

**A consistência não foi removida.** Ela ainda pega o caso que motivou a existência dela: o RANSAC convergir para um subconjunto pequeno e espúrio. Só mudou de lugar na cadeia.

**Decisão irmã · o equiespaçamento tolera lacunas.** O OCR não lê 100 % dos rótulos, e um tick perdido no *meio* não é inconsistência — é lacuna. Cada diferença consecutiva precisa ser próxima de um **múltiplo inteiro** do menor espaçamento: cobre "sem lacuna" (razão 1) e "faltaram N" (razão N+1), e ainda reprova leitura errada (razão longe de qualquer inteiro).

Implementa: [[identify.calibrate]]

---

## D5 · União das componentes conexas

**Decisão.** A polilinha usa a **união de todas** as componentes acima de `MIN_COMPONENT_PX = 2`, não a maior.

**Contra o que.** A escolha padrão em visão computacional: manter a maior componente e descartar o resto como ruído.

**Evidência.** Uma curva tracejada ou pontilhada é, **por construção**, uma sequência de componentes desconectadas — cada travessão é a sua própria componente. Medido contra a máscara verdadeira: **40 de 300** amostras ficavam com menos de 10 pontos utilizáveis, e o RMSE do estrato `traco=:` estourava o alvo (2,43 px contra 2 px).

**Por que é seguro.** Na máscara verdadeira não há nada além da curva, então a união é sempre segura ali. Contra uma máscara *predita*, o limiar de 2 px ainda descarta ruído isolado de 1 px.

**Custo se errada.** Uma máscara predita com um distrator grande seria incorporada inteira. É por isso que a guarda `ajuste_inconsistente` existe a jusante.

Implementa: [[identify.polyline]]

---

## D6 · AIC com n efetivo

**Decisão.** A escolha entre FOPDT e 2ª ordem usa `n_eff = n·(1−ρ)/(1+ρ)` no lugar de `n`, onde ρ é a autocorrelação de defasagem 1 do resíduo.

**Contra o que.** O AIC clássico, com `n` cru.

**Evidência.** O AIC clássico pressupõe observações **independentes**. Aqui a série vem de uma polilinha extraída de imagem, onde pixels vizinhos carregam erro de extração **correlacionado**. Medido: ρ ≈ 0,71, e *n* mediano de **738** corresponde a *n* efetivo de **112** — o AIC tratava 738 pontos correlacionados como 738 evidências independentes.

**A consequência era estrutural, não de calibração.** A 2ª ordem vence quando `SSE₁/SSE₂ > exp(2/n)`; com n=806, bastava **0,234 %** de ganho de SSE. E ela consegue esse ganho ajustando **o próprio artefato de extração** com o polo extra — o NRMSE das duas estruturas empatava (0,00353 × 0,00351).

Resultado: **32 % das plantas de 1ª ordem** classificadas como 2ª ordem, contra **6 %** quando o mesmo estágio recebe a série verdadeira. A diferença entre 32 % e 6 % é exatamente o erro que a extração introduz.

**Por que BIC não resolveria.** Testado: corrige só **40 %** dos casos. O problema não é a constante da penalidade, é o *n* inflado. Trocar `2·k` por `ln(n)·k` mexe no lado errado da desigualdade.

**Detalhe que não é acidental.** ρ sai do resíduo da estrutura **mais flexível** (2ª ordem). Num modelo subespecificado, a correlação do resíduo mistura ruído de extração com erro de estrutura, e superestimaria a correção.

Implementa: [[identify.classical]]

---

## D7 · O estilo não pode ver o sistema

**Decisão.** `sample_style(rng)` tem assinatura `(rng)` e **nada mais**. Ela não recebe a `SystemSpec`.

**Contra o que.** A alternativa cômoda: passar o spec e ajustar o estilo a ele (escolher limites de eixo bonitos, por exemplo).

**Evidência.** É uma restrição *a priori*, não empírica — e essa é a questão. Se o estilo pudesse ver o rótulo, a rede poderia aprender atalhos ("figuras com grade tendem a ser de 1ª ordem") e obter acurácia alta no corpus que **desaparece no mundo real**. Vazamento de rótulo não aparece na métrica de validação, por definição.

**Como é garantido.** Um teste assevera que a assinatura continua sendo exatamente `["rng"]`. A restrição não depende da disciplina de quem escreve o código.

**Consequência de projeto.** Os três campos de estrato — `has_reference_line`, `has_annotation_arrow`, `has_settling_band` — são campos de *render*, marcados por `render_sample` via `dataclasses.replace`. Se fossem sorteados em `sample_style`, quebrariam a assinatura.

**Decisão irmã · três streams de RNG independentes.** `SeedSequence(seed).spawn(3)` dá streams separados para sistema, estilo e ruído. Acrescentar um sorteio no estilo **não desloca** o sorteio do sistema — sem isso, nenhuma comparação entre rodadas seria interpretável.

Implementa: [[dataset.randomize]] · [[dataset.generator]]

---

## D8 · Letterbox, não redimensionamento

**Decisão.** Escala **isotrópica** mais preenchimento para levar a figura a 512×512.

**Contra o que.** `cv2.resize` direto para o quadrado.

**Evidência.** Distorção anisotrópica muda a **inclinação local** da curva, e a inclinação é o que o estágio D lê como constante de tempo. Esticar a imagem envenenaria o parâmetro — e o efeito seria proporcional à razão de aspecto, que o gerador sorteia entre 240×180 e 1600×1200.

**Custo.** Área útil menor no quadrado quando a figura é muito alongada. Aceito: a alternativa corrompe o dado.

Implementa: [[identify.extract]] · Assevera: `test_letterbox_preserva_geometria_no_roundtrip`

---

## D9 · Aprovação por eixo, não conjunta

**Decisão.** `Calibration` declara `ok_x` e `ok_y` separados; `ok` passa a ser `ok_x and ok_y`.

**Evidência.** Medido em `data/test` (n=900):

| Exigência | Aprovadas |
|---|---|
| ambos os eixos | 81,3 % |
| eixo X sozinho | 89,3 % |
| eixo Y sozinho | 89,1 % |

**O ganho é assimétrico e material.** Só o eixo X já dá a **janela em segundos**, e com ela ωn, τ e θ saem em unidade física sem o eixo Y. Nas três imagens reais do bloco, **duas tinham X aprovado e Y reprovado** — ωn físico estava disponível e era descartado.

**Compatibilidade.** `ok` mantém a semântica antiga, então todo consumidor existente se comporta igual. O ganho entra por `physical_parcial`, que é **aditivo**.

Implementa: [[identify.calibrate]] · [[identify.pipeline]]

---

## D10 · O nível de fundo é a moda da borda

**Decisão.** `_fundo()` usa a **moda dos pixels da borda** da imagem, não a mediana global.

**Contra o que.** `np.median(gray)`, usado em três lugares.

**Evidência.** A mediana global assume **um fundo só**. Numa figura com `ax.set_facecolor()` diferente do fundo da figura, sendo a área de dados a maior região, a mediana cai no fundo *dos eixos* — e a moldura inteira vira tinta. Medido numa imagem externa de tema escuro (fundo da figura 43, dos eixos 30, mediana global 30): bbox `(0, 0, 799, 460)` em vez de `(100, 55, 720, 410)`, zero rótulos lidos nos dois eixos, calibração física perdida.

**Por que a borda.** A moldura de uma figura matplotlib nunca encosta na borda da imagem — a margem sempre existe. A borda é, por construção, fundo da *figura*.

**Verificação de não-regressão.** Idêntico nas 900 amostras de `data/test`, bbox igual em **900/900**.

Implementa: [[identify.calibrate]]

---

## D11 · O lote de OCR tem teto e recuo

**Decisão.** `_LOTE_MAX = 12` recortes por mosaico, com recuo para OCR individual quando o lote inteiro colapsa apesar de haver recortes úteis.

**Evidência.** Com `--psm 7` o Tesseract devolve **zero palavras** quando a análise de layout falha — **sem levantar exceção**. O lote inteiro vira nulo, silenciosamente. E não é previsível pelo tamanho:

| recortes | lidos |
|---:|---:|
| 14 | 11 |
| 16 | **0** |
| 20 | 17 |
| 21 | **0** |

Custava 16 amostras do corpus sem nenhum par.

**Por que as duas metades.** O teto reduz a probabilidade; o recuo garante que, quando acontecer mesmo assim, a amostra não é perdida. Nenhuma das duas sozinha resolve — não há tamanho seguro.

Implementa: [[identify.calibrate]]

---

## D12 · Guardas de plausibilidade

**Decisão.** Duas guardas, ambas calibradas contra o corpus **com falso positivo medido**.

| Guarda | Limiar | Precisão | Custo |
|---|---|---|---|
| `ajuste_inconsistente` | nrmse > 0,13 (p98 do corpus) | 88,2 % | 0,24 % (2/837) |
| `resposta_inversa` | undershoot > 0,08 | n=1 | 0,44 % (4/900) |

**O princípio.** Uma pipeline que **sempre responde** é pior que uma que às vezes recusa, porque o consumidor não tem como distinguir uma resposta boa de uma ruim.

**A guarda de resíduo não pega tudo, e não precisa.** Recall de 19 %. A função dela é recusar o absurdo, não auditar o aceitável.

**A definição de undershoot foi corrigida por física.** A primeira versão acusava falso positivo numa curva de ζ=0 legítima (0,147). Redefinida como *excursão contrária à do degrau **antes** de a resposta arrancar* — que é o que caracteriza fase não-mínima —, a mesma curva caiu para **0,0044**.

Implementa: [[identify.pipeline]]

---

## D13 · O alvo de treino é contínuo

**Decisão.** O alvo da U-Net usa o valor real que sai do `cv2.INTER_AREA` (0–255, normalizado), **sem nenhuma binarização**.

**Contra o que.** Três limiares tentados, todos medidos:

| Limiar | Resultado |
|---|---|
| 127 | perdia a curva quase toda em imagens grandes — cobertura de colunas a **0,4 %** |
| 0 | inflava a área do alvo até **3,03×**; IoU de teste piorou de 0,572 para 0,495 apesar do IoU de validação subir a 0,91 |
| 32 | melhor dos três (cobertura 85,1 %, inflação 2,14×), mas ζ ainda 0,64 p.p. acima do alvo |

**O raciocínio.** Em vez de caçar um **quarto** limiar mágico, remover a escolha. `dice_bce_loss` aceita alvo contínuo nativamente — é a definição matemática usual dos dois, sem mudança de código.

Uma caixa do downscale com pouca cobertura vira alvo **baixo mas não-zero**, preservando o gradiente de BCE (resolve o limiar 127); e uma caixa parcialmente coberta **não** empurra a rede a prever confiança alta (limita a inflação do limiar 0/32 pela raiz, não por um corte arbitrário).

**O sinal de alerta que motivou.** O limiar 0 subia o IoU de *validação* enquanto piorava o de *teste*. Uma métrica que melhora enquanto o resultado piora é evidência de que a métrica está errada — o que voltou a aparecer no [[Critérios de qualidade|Ruling 50]].

Implementa: [[train_unet]]

---

## D14 · Capacidade da rede: 32 canais base

**Decisão.** `base = 32` (7 763 041 parâmetros), elevado de 24 (4 368 073).

**Como foi testada, e não suposta.** Três modelos de `base=24` treinados com combinações diferentes de estratos **trocavam um caso real pelo outro**: o que acertava a imagem A errava a B, e vice-versa. Isso é assinatura de capacidade esgotada, não de dado insuficiente — mais dado teria movido os dois na mesma direção.

A hipótese foi testada elevando **só** os canais base, sem mudar mais nada de essencial. O modelo maior é o **primeiro que acerta as duas ao mesmo tempo**.

**O detalhe que mais informa.** O ganho em IoU de validação foi de **+0,005**. Se a decisão dependesse da métrica agregada, ela teria sido rejeitada. A diferença aparecia só nos casos reais — mais uma evidência de que a métrica agregada não é onde a diferença aparece.

**Custo.** +78 % de parâmetros, lote de treino de 8 para 6, e a latência do critério 3.11 **ainda não foi remedida**.

Implementa: [[identify.extract]] · [[train_unet]]

---

## Refutada · Guarda de descontinuidade da máscara

**Proposta.** Recusar quando o maior buraco entre colunas com tinta ultrapassasse um limiar.

**Por que parecia óbvia.** Numa imagem externa que errava, o buraco era **19,3 %** contra **5,5 %** da segunda pior. Separação convincente — em n=8.

**Por que morreu.** Contra o corpus de 895 amostras:

| Correlação de Spearman com o erro real | valor | p |
|---|---|---|
| maior buraco × erro | **+0,020** | 0,57 |
| densidade × erro | +0,003 | 0,93 |

E o maior buraco do próprio corpus (**20,9 %**) é *maior* que o da imagem que errava.

**O motivo estrutural, que vale mais que o resultado.** O corpus **não contém o modo de falha** que essa guarda existia para pegar. Medindo contra ele, só o **custo** aparecia — nunca o benefício. Uma guarda não pode ser validada num corpus que não tem a falha dela.

**Por que a guarda de resíduo escapa disso.** Ela não depende do modo de falha: depende de o ajuste ficar ruim quando a série é lixo, e disso o corpus tem **79 exemplos**.

**Consequência de método.** Isso inverteu a ordem de custo de três propostas: o estrato de gerador, que parecia o item mais caro, virou **pré-requisito** dos outros dois.

---

## Refutada · Primeira versão do estrato `reta_no_patamar`

**Proposta.** Renderizar uma reta de referência coincidente com o patamar, em cor de luminância colidente, para reproduzir o caso real.

**Por que morreu.** Não degradava nada. Cobertura mediana de 0,96, quando o critério pedia < 0,75.

**A causa, que virou uma decisão nova.** Com a janela em `0,5 a 6 × t_dom`, o corpus **praticamente nunca mostra o patamar** — a fração final já assentada tem mediana de **0,98 %** da janela, e zero de 60 amostras chegam a 30 %. Sem patamar visível, a reta toca a curva só na última coluna, e não há trecho colinear para ocluir.

Daí o estrato `janela_assentada`, com `_T_DOM_ESTRATO = 20.0` escolhido por varredura medida (n=40 por ponto). Acima de 20 o máximo vai a 0,83–0,89, **fora** da faixa real, e a mediana não sobe.

**O critério que ficou.** Um estrato precisa **degradar o modelo atual de forma mensurável**. Um que não move a métrica não reproduz o fenômeno, e treinar com ele seria desperdício.

---

## Um limiar calibrado com um exemplo é frágil a mudanças a montante

Não é uma decisão, é uma **lição registrada no código**.

O limiar de undershoot era 0,10, medido com o modelo anterior. Ao promover uma máscara melhor, a métrica **mudou de escala** — ela é normalizada pela faixa de *y* capturada, que é o **denominador**. A mesma imagem caiu de 0,143 para **0,0916**, furando o limiar por baixo e virando resposta confiante e errada.

Recalibrado para 0,08, no meio de um platô onde o custo é constante (0,075 a 0,090 → 4/900). O comentário no código diz o que fazer: **qualquer mudança no Estágio A exige remedir este número**.

---

## O padrão que aparece três vezes

Três achados independentes têm a mesma forma:

1. **RGB** — o gerador não sorteava cores colidentes, então 900 amostras nunca contradisseram "luminância basta".
2. **Guarda de descontinuidade** — o corpus não tem o modo de falha, então só o custo era mensurável.
3. **Estrato `reta_no_patamar` v1** — a janela curta escondia o fenômeno que o estrato queria exibir.

**A generalização:** um corpus sintético mede exatamente as hipóteses que o gerador dele encarna. Onde o gerador supõe algo, o corpus é cego — e a cegueira não aparece como erro, aparece como *ausência de sinal*.

É por isso que cada defeito que uma imagem real expôs virou um estrato opt-in: transforma um achado pontual em **regressão permanente**, e move a fronteira do que o corpus consegue medir.

---

## Dívidas conhecidas

| Item | Estado |
|---|---|
| Guarda de parâmetro na borda | proposta, não implementada — `K` no teto com `ok=true` é a pior saída possível |
| Sinalização de ordem ambígua | a pipeline escolhe a mais simples sem avisar |
| Detecção de múltiplas curvas | fora de escopo por decisão, mas sem recusa também |
| Latência com `base=32` | critério 3.11 não remedido depois de +78 % de parâmetros |
| Critério A.0 | mede `UNet()` com defaults, não o checkpoint em produção |
| Fixtures externas | as 13 imagens externas não estão versionadas |
| Validação externa limpa | 5 das 13 serviram de diagnóstico antes de servir de validação |

---

Ver também: [[Arquitetura do projeto]] · [[Critérios de qualidade]] · [[Módulos]]
