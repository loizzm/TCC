"""Gera as tres imagens de GANHO NEGATIVO usadas como fixture do Bloco 9.

Este arquivo e a FONTE DA VERDADE de `tests/fixtures/caso_real_neg_*.png`. A
verdade de cada figura esta declarada nas funcoes de transferencia abaixo, nao
inferida do grafico — foi assim que se descobriu que o titulo da Figura 2 dizia
zeta=0,3 enquanto o sistema tem zeta=0,2, e que o ajuste (0,201) estava certo.

  Figura 1 -> caso_real_neg_fopdt.png   K=-2, tau=0,5, theta do sistema=1 s,
                                        degrau em t=2 s (theta da janela = 3 s)
  Figura 2 -> caso_real_neg_sub.png     K=-1, wn=5, zeta=0,2, theta=2 s,
                                        degrau em t=0
  Figura 3 -> caso_real_neg_super.png   K=-3, wn=4, zeta=1,25,
                                        theta do sistema=0,5 s,
                                        degrau em t=3 s (theta da janela=3,5 s)
  Figura 4 -> caso_real_neg_super_legenda_movida.png
                                        IDENTICA a Figura 3, so com a legenda
                                        em 'upper right'. O par isola o efeito
                                        da OCLUSAO POR LEGENDA: uma variavel
                                        muda, e wn vai de 26 % de erro para
                                        2,97 %. Ver secao 43 do HANDOFF_P2_7.

O `rg.py` (sem sufixo) e o analogo do Bloco 8 e gera as fixtures de ganho
POSITIVO. Os dois convivem de proposito: cada geracao de fixtures mantem a sua
propria proveniencia versionada.

Tres tracos destas figuras que o corpus sintetico NAO produz, e que e por isso
que elas acham defeito:

  - ganho negativo (o gerador sorteia K > 0);
  - o DEGRAU DE ENTRADA plotado junto da resposta, como tracejada branca. E o
    "defeito 4": em `caso_real_neg_fopdt.png` a polilinha pula entre os dois
    objetos, o undershoot vai a 0,97 e a guarda recusa a imagem — mesmo com a
    fisica saindo certa por baixo (K=-1,997, tau=0,4998, theta=2,998);
  - `axvspan` cinza cobrindo a faixa do atraso, que em `caso_real_neg_sub.png`
    cai exatamente sobre o plato inicial da resposta.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ==========================================
# 1. Função para Aplicar o Atraso e o Degrau
# ==========================================
def step_response_custom(sistema, theta, amplitude_degrau, instante_degrau, vetor_tempo):
    """
    Calcula a resposta a um degrau com amplitude e instante de início arbitrários,
    além de considerar o atraso de transporte (theta) do próprio sistema.
    """
    y_out = np.zeros_like(vetor_tempo)
    
    # O sistema só começa a reagir no tempo do degrau + o atraso do sistema
    tempo_reacao = instante_degrau + theta
    mascara = vetor_tempo >= tempo_reacao
    
    t_ativo = vetor_tempo[mascara] - tempo_reacao
    
    if len(t_ativo) > 0:
        _, y_ativo = signal.step(sistema, T=t_ativo)
        y_out[mascara] = y_ativo * amplitude_degrau # Escala pelo tamanho/sinal do degrau
        
    return y_out

# ==========================================
# 2. Definição dos Sistemas
# ==========================================
# Sistema 1: FOPDT (Primeira Ordem com Atraso)
# G1(s) = 2 / (s + 2) -> K=1, tau=0.5
sys_1 = signal.TransferFunction([2], [1, 2])
theta_1 = 1.0

# Sistema 2: SOPDT Subamortecido
# G2(s) = 25 / (s^2 + 2s + 25) -> K=1, wn=5, zeta=0.2
sys_2 = signal.TransferFunction([25], [1, 2, 25])
theta_2 = 2.0

# Sistema 3: SOPDT Superamortecido
# G3(s) = 16 / (s^2 + 10s + 16) -> K=1, wn=4, zeta=1.25
sys_3 = signal.TransferFunction([16], [1, 10, 16])
theta_3 = 0.5

# ==========================================
# 3. Simulação
# ==========================================
t_vetor = np.linspace(0, 15, 1000)

# Resposta 1: Degrau Negativo de amplitude -2, aplicado em t=2s
y1 = step_response_custom(sys_1, theta_1, amplitude_degrau=-2, instante_degrau=2.0, vetor_tempo=t_vetor)

# Resposta 2: Degrau Negativo de amplitude -1, aplicado em t=0s
y2 = step_response_custom(sys_2, theta_2, amplitude_degrau=-1, instante_degrau=0.0, vetor_tempo=t_vetor)

# Resposta 3: Degrau Negativo de amplitude -3, aplicado em t=3s
y3 = step_response_custom(sys_3, theta_3, amplitude_degrau=-3, instante_degrau=3.0, vetor_tempo=t_vetor)


# ==========================================
# 4. Plotagem (Estilo Escuro Moderno)
# ==========================================
plt.style.use('dark_background')

# --- Figura 1: FOPDT com Degrau Atrasado ---
plt.figure(figsize=(8, 5))
plt.plot(t_vetor, y1, color='#00ffcc', linewidth=2.5, label='Saída do Sistema')
plt.step([0, 2.0, 15], [0, -2, -2], color='white', linestyle='--', where='post', label='Degrau de Entrada (t=2s)')
plt.axvspan(2.0, 2.0 + theta_1, color='#333333', alpha=0.8, label=f'Atraso do Sistema ($\\theta={theta_1}s$)')
plt.title('FOPDT: Degrau Negativo e Atraso de Processo', fontsize=12)
plt.xlabel('Tempo (s)'); plt.ylabel('Amplitude')
plt.legend(loc='upper right')
plt.grid(color='#444444', linestyle=':')
plt.xlim(0, 10)

# --- Figura 2: SOPDT Oscilatório (Degrau Negativo em t=0) ---
plt.figure(figsize=(8, 5))
plt.plot(t_vetor, y2, color='#ff66cc', linewidth=2.5, label='Saída do Sistema')
plt.step([0, 15], [-1, -1], color='white', linestyle='--', where='post', label='Degrau de Entrada (t=0s)')
plt.axvspan(0, theta_2, color='#333333', alpha=0.8, label=f'Atraso do Sistema ($\\theta={theta_2}s$)')
plt.title('SOPDT Subamortecido: Degrau Negativo', fontsize=12)
plt.xlabel('Tempo (s)'); plt.ylabel('Amplitude')
plt.legend(loc='upper right')
plt.grid(color='#444444', linestyle=':')
plt.xlim(0, 10)

# --- Figura 3: SOPDT Lento com Degrau Fortemente Atrasado ---
#
# DUAS VARIANTES, e o par e o ponto. Elas diferem em UMA coisa — a posicao da
# legenda — e por isso isolam o efeito da oclusao sem nenhuma outra variavel:
#
#   loc='lower left'  -> caso_real_neg_super.png
#       A caixa da legenda ocupa t 0,3..5,6 e y -2,4..-3,05, e a curva atravessa
#       essa faixa exatamente na ACOMODACAO, entre t=4,4 e t=5,6. A polilinha
#       segue a borda da caixa e cria um patamar falso em -2,93 onde a resposta
#       verdadeira ainda esta em -2,52. Resultado: wn=2,95 (erro 26 %) e
#       zeta=0,87 (erro 30 %).
#
#   loc='upper right' -> caso_real_neg_super_legenda_movida.png
#       Mesma planta, mesma janela, mesmo degrau. wn=3,88 (erro 2,97 %) e
#       zeta=1,22 (erro 2,23 %).
#
# Medido por faixa de t, o erro de extracao entre as duas: plato 0,0077 nas
# DUAS, arranque 0,0090 nas duas, transitorio rapido 0,0201 contra 0,0205,
# cauda assentada 0,0073 contra 0,0068 — e ACOMODACAO 0,1674 contra 0,0090.
# Uma faixa mudou, 19x. Ver HANDOFF_P2_7 secao 43.
for _loc, _nome in (('lower left', 'caso_real_neg_super.png'),
                    ('upper right', 'caso_real_neg_super_legenda_movida.png')):
    plt.figure(figsize=(8, 5))
    plt.plot(t_vetor, y3, color='#ffcc00', linewidth=2.5, label='Saída do Sistema')
    plt.step([0, 3.0, 15], [0, -3, -3], color='white', linestyle='--', where='post', label='Degrau de Entrada (t=3s)')
    plt.axvspan(3.0, 3.0 + theta_3, color='#333333', alpha=0.8, label=f'Atraso do Sistema ($\\theta={theta_3}s$)')
    plt.title('SOPDT Superamortecido: Degrau Aplicado em t=3s', fontsize=12)
    plt.xlabel('Tempo (s)'); plt.ylabel('Amplitude')
    plt.legend(loc=_loc)
    plt.grid(color='#444444', linestyle=':')
    plt.xlim(0, 15)

plt.show()