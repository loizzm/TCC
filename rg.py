import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ==========================================
# 1. Função para Aplicar o Tempo Morto
# ==========================================
def step_com_atraso(sistema, theta, vetor_tempo):
    y_out = np.zeros_like(vetor_tempo)
    mascara = vetor_tempo >= theta
    t_ativo = vetor_tempo[mascara] - theta
    
    if len(t_ativo) > 0:
        _, y_ativo = signal.step(sistema, T=t_ativo)
        y_out[mascara] = y_ativo
        
    return y_out

# ==========================================
# 2. Definição dos Sistemas
# ==========================================
# 1. FOPDT: K=1.5, tau=0.5, theta=5
sys_1 = signal.TransferFunction([1.5], [0.5, 1])
theta_1 = 5.0

# 2. SOPDT (Criticamente Amortecido): K=1, wn=3, zeta=1.0, theta=1.2
sys_2 = signal.TransferFunction([9], [1, 6, 9])
theta_2 = 1.2

# 3. SOPDT (Amortecimento Ótimo): K=3, wn=4, zeta=0.7, theta=0.8
sys_3 = signal.TransferFunction([48], [1, 5.6, 16])
theta_3 = 0.8

# ==========================================
# 3. Simulação
# ==========================================
t_vetor = np.linspace(0, 10, 800)

y1 = step_com_atraso(sys_1, theta_1, t_vetor)
y2 = step_com_atraso(sys_2, theta_2, t_vetor)
y3 = step_com_atraso(sys_3, theta_3, t_vetor)

# ==========================================
# 4. Plotagem (Janelas Separadas)
# ==========================================
# Aplicando estilo de plotagem limpo
plt.style.use('seaborn-v0_8-whitegrid')

# --- Figura 1: FOPDT ---
plt.figure(figsize=(7, 4.5))
plt.plot(t_vetor, y1, color='#e74c3c', linewidth=2.5, label='Saída $c(t)$')
plt.axvspan(0, theta_1, color='#e74c3c', alpha=0.1, label='Latência (5s)')
plt.axhline(1.5, color='black', linestyle=':', label='Referência (K=1.5)')
plt.title('FOPDT: Dinâmica Rápida, Longo Atraso', fontsize=12, fontweight='bold')
plt.xlabel('Tempo (s)'); plt.ylabel('Amplitude')
plt.legend(loc='lower right')
plt.xlim(0, 10); plt.ylim(0, 1.8)

# --- Figura 2: SOPDT Criticamente Amortecido ---
plt.figure(figsize=(7, 4.5))
plt.plot(t_vetor, y2, color='#2980b9', linewidth=2.5, label=r'$\zeta = 1.0$')
plt.axvspan(0, theta_2, color='#2980b9', alpha=0.1, label='Tempo Morto')
plt.axhline(1.0, color='black', linestyle=':', label='Referência (K=1)')
plt.title('SOPDT: Criticamente Amortecido', fontsize=12, fontweight='bold')
plt.xlabel('Tempo (s)'); plt.ylabel('Amplitude')
plt.legend(loc='lower right')
plt.xlim(0, 10); plt.ylim(0, 1.2)

# --- Figura 3: SOPDT Amortecimento Ótimo ---
plt.figure(figsize=(7, 4.5))
plt.plot(t_vetor, y3, color='#27ae60', linewidth=2.5, label=r'$\zeta = 0.7$ (Ótimo)')
plt.axvspan(0, theta_3, color='#27ae60', alpha=0.1, label='Tempo Morto')
plt.axhline(3.0, color='black', linestyle=':', label='Referência (K=3)')
plt.title('SOPDT: Amortecimento Ótimo', fontsize=12, fontweight='bold')
plt.xlabel('Tempo (s)'); plt.ylabel('Amplitude')
plt.legend(loc='lower right')
plt.xlim(0, 10); plt.ylim(0, 3.5)

plt.show()