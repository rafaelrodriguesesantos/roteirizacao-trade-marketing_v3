# ==============================================================================
# TCC - ROTEIRIZAÇÃO INTELIGENTE EM TRADE MARKETING
# Script 2 (REVISADO): Matriz de Saaty 4x4, Teste de Consistência (CR) e Tiers
# ------------------------------------------------------------------------------
# Mudanças em relação à versão anterior:
#   - Corte dos Tiers: 40% / 20% / 40% (Tier 1 = top 40%, Tier 2 = 20% seguintes,
#     Tier 3 = 40% restantes), em vez de 20% / 40% / 40%.
#   - Mensagens de saída com os nomes de arquivo corretos e relatório dos Tiers.
#   - Os pesos são obtidos pelo método da média das colunas normalizadas
#     (aproximação do autovetor principal de Saaty).
# Uso: python Script2_AHP_Revisado.py   (CSV na mesma pasta, ou DATA_DIR=...)
# ==============================================================================

import os
import numpy as np
import pandas as pd

DATA_DIR = os.environ.get('DATA_DIR', '.')
OUT_DIR = os.environ.get('OUT_DIR', '.')
os.makedirs(OUT_DIR, exist_ok=True)

print("--------------------------------------------------")
print("--- ETAPA 2: MODELAGEM AHP ---")
print("--------------------------------------------------")

# 1. Base com o Score da PCA (saída do Script 1)
df_pdv = pd.read_csv(os.path.join(DATA_DIR, 'dpdvs_score_pca.csv'), sep=';')

# 2. Matriz de comparação par a par (escala de Saaty)
criterios = ['Potencial_PCA', 'Vendas_Hardware', 'Canal_Trade', 'Historico_Visitas']

matriz_saaty = np.array([
    [1.0, 2.0, 5.0, 7.0],   # Potencial_PCA
    [1/2, 1.0, 4.0, 6.0],   # Vendas_Hardware
    [1/5, 1/4, 1.0, 3.0],   # Canal_Trade
    [1/7, 1/6, 1/3, 1.0],   # Historico_Visitas
])

# 3. Pesos: normalização por coluna e média por linha
soma_colunas = matriz_saaty.sum(axis=0)
matriz_normalizada = matriz_saaty / soma_colunas
pesos_criterios = matriz_normalizada.mean(axis=1)

# 4. Consistência (CR)
vetor_ponderado = np.dot(matriz_saaty, pesos_criterios)
lambda_max = np.mean(vetor_ponderado / pesos_criterios)
n = len(criterios)
CI = (lambda_max - n) / (n - 1)
RI = 0.90   # índice aleatório de Saaty para n = 4
CR = CI / RI

print("Pesos dos critérios:")
for c, w in zip(criterios, pesos_criterios):
    print(f"   {c:<20s} {w*100:6.2f}%")
print(f"✅ Lambda máximo (λ_max): {lambda_max:.4f}")
print(f"✅ Índice de Consistência (CI): {CI:.4f}")
print(f"✅ Razão de Consistência (CR): {CR:.4f} ({CR*100:.2f}%)  -> aceitável se < 10%")

# 5. Pontuação ponderada por PDV (todos os critérios em escala 0-1)
df_ahp = df_pdv.copy()

smin, smax = df_ahp['Score_PCA_Potencial'].min(), df_ahp['Score_PCA_Potencial'].max()
df_ahp['C1_Potencial'] = (df_ahp['Score_PCA_Potencial'] - smin) / (smax - smin)

fat_hw_max = df_ahp['Fat_Hardware'].max()
df_ahp['C2_Hardware'] = df_ahp['Fat_Hardware'] / fat_hw_max if fat_hw_max > 0 else 0

df_ahp['C3_Canal'] = df_ahp['Canal Trade'].apply(lambda x: 1.0 if 'Marco' in str(x) else 0.2)

vmin, vmax = df_ahp['Total_Visitas'].min(), df_ahp['Total_Visitas'].max()
df_ahp['C4_Visitas'] = (df_ahp['Total_Visitas'] - vmin) / (vmax - vmin) if vmax > vmin else 0

df_ahp['Score_AHP_Final'] = (
    df_ahp['C1_Potencial'] * pesos_criterios[0] +
    df_ahp['C2_Hardware'] * pesos_criterios[1] +
    df_ahp['C3_Canal'] * pesos_criterios[2] +
    df_ahp['C4_Visitas'] * pesos_criterios[3]
)

# 6. Tiers: Tier 3 = 40% inferiores | Tier 2 = 20% intermediários | Tier 1 = 40% superiores
df_ahp['Prioridade_Rota'] = pd.qcut(
    df_ahp['Score_AHP_Final'],
    q=[0, 0.40, 0.60, 1.0],
    labels=['Tier 3 - Baixa', 'Tier 2 - Média', 'Tier 1 - Alta'],
)

print("\nPDVs por Tier:")
print(df_ahp['Prioridade_Rota'].value_counts().sort_index(ascending=False).to_string())
print("\nTier x Canal Trade:")
print(pd.crosstab(df_ahp['Prioridade_Rota'], df_ahp['Canal Trade']).to_string())

df_ranking = df_ahp.sort_values(by='Score_AHP_Final', ascending=False)
df_ranking.to_csv(os.path.join(OUT_DIR, 'dpdvs_priorizados_ahp_tcc.csv'), index=False, sep=';')
print("\n✅ Tabela 'dpdvs_priorizados_ahp_tcc.csv' exportada com sucesso!")