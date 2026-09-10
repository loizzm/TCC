import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ==========================================
# 1. Função de Superposição para Múltiplos Degraus
# ==========================================
def multi_step_response(sistema, theta, lista_degraus, vetor_tempo):
    """
    Calcula a resposta de um sistema a múltiplos degraus usando superposição.
    lista_degraus deve ser uma lista de tuplas: [(amplitude, instante_t), ...]
    """
    y_total = np.zeros_like(vetor_tempo)
    ref_entrada = np.zeros_like(vetor_tempo)
    
    # Soma cumulativa para a linha de referência (entrada)
    valor_acumulado = 0
    
    for amplitude, instante in lista_degraus:
        # 1. Adiciona o degrau na linha de referência da entrada
        mascara_entrada = vetor_tempo >= instante
        ref_entrada[mascara_entrada] = valor_acumulado + amplitude
        valor_acumulado += amplitude
        
        # 2. Calcula a resposta do sistema (considerando instante + atraso theta)
        tempo_reacao = instante + theta
        mascara_saida = vetor_tempo >= tempo_reacao
        t_ativo = vetor_tempo[mascara_saida] - tempo_reacao
        
        if len(t_ativo) > 0:
            _, y_ativo = signal.step(sistema, T=t_ativo)
            y_total[mascara_saida] += y_ativo * amplitude
            
    return y_total, ref_entrada

# ==========================================
# 2. Definição dos Sistemas e Entradas
# ==========================================
# Sistema 1: SOPDT Subamortecido (Sem atraso, theta=0)
sys_1 = signal.TransferFunction([10], [1, 2, 10])
degraus_sys_1 = [(-1.0, 1.0), (-2.0, 6.0)] # (Amplitude, Instante)

# Sistema 2: FOPDT (Com atraso, theta=0.5)
sys_2 = signal.TransferFunction([2], [1, 2])
theta_2 = 0.5
degraus_sys_2 = [(2.0, 1.0), (1.5, 4.0)]

# ==========================================
# 3. Simulação
# ==========================================
t_vetor = np.linspace(0, 12, 1000)

y1_out, u1_in = multi_step_response(sys_1, 0.0, degraus_sys_1, t_vetor)
y2_out, u2_in = multi_step_response(sys_2, theta_2, degraus_sys_2, t_vetor)

# ==========================================
# 4. Plotagem (Estilo Seaborn Darkgrid)
# ==========================================
plt.style.use('seaborn-v0_8-darkgrid')

# --- Figura 1: Dois Degraus Negativos ---
plt.figure(figsize=(9, 5))
plt.plot(t_vetor, y1_out, color='#e74c3c', linewidth=2.5, label='Saída do Sistema $c(t)$')
plt.plot(t_vetor, u1_in, color='black', linestyle='--', linewidth=1.5, drawstyle='steps-post', label='Sinal de Entrada Acumulado')

plt.title('Sistema Subamortecido: Dois Degraus Negativos Consecutivos', fontsize=13, fontweight='bold')
plt.xlabel('Tempo (s)'); plt.ylabel('Amplitude')
plt.legend(loc='upper right')
plt.xlim(0, 12); plt.ylim(-4, 0.5)

# --- Figura 2: Dois Degraus Positivos ---
plt.figure(figsize=(9, 5))
plt.plot(t_vetor, y2_out, color='#2ecc71', linewidth=2.5, label='Saída do Sistema $c(t)$')
plt.plot(t_vetor, u2_in, color='black', linestyle='--', linewidth=1.5, drawstyle='steps-post', label='Sinal de Entrada Acumulado')

# Destacando os períodos de tempo morto
plt.axvspan(1.0, 1.0 + theta_2, color='gray', alpha=0.2, label='Tempo Morto 1')
plt.axvspan(4.0, 4.0 + theta_2, color='gray', alpha=0.4, label='Tempo Morto 2')

plt.title('FOPDT: Dois Degraus Positivos Consecutivos', fontsize=13, fontweight='bold')
plt.xlabel('Tempo (s)'); plt.ylabel('Amplitude')
plt.legend(loc='lower right')
plt.xlim(0, 8); plt.ylim(0, 4)

plt.show()