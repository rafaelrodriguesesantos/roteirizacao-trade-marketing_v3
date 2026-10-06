# ==============================================================================
# TCC - ROTEIRIZAÇÃO INTELIGENTE EM TRADE MARKETING
# Script 1: ETL, Premissas Multivariadas (KMO e Bartlett) e Aplicação da PCA
# ==============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("--------------------------------------------------")
print("--- INICIANDO ETAPA 1: ETL E APLICAÇÃO DE PCA ---")
print("--------------------------------------------------")

# 1. Carregando as bases de dados sintéticas/simuladas
df_base = pd.read_csv('Base_Ficticia_TCC_Completa.csv', sep=';')
df_lojas = pd.read_csv('dLojas.csv', sep=';')
df_produtos = pd.read_csv('dProdutos.csv', sep=';')
df_roteiro = pd.read_csv('fRoteiro.csv', sep=';')

# Limpeza de cabeçalhos
df_base.columns = [c.strip().replace('"','').replace('\n',' ') for c in df_base.columns]

# 2. Engenharia de Atributos e Consolidação por PDV
df_full = pd.merge(df_base, df_produtos, on=['ID Produto','Produto'], how='left')

# Faturamento por categoria (Hardware vs. Suprimento)
df_tipo_produto_fat = df_full.groupby(['ID','Tipo'])['Valor Total  (R$)'].sum().unstack(fill_value=0)
df_tipo_produto_fat.columns = ['Fat_Hardware','Fat_Suprimento']

# Volume físico em unidades
df_tipo_vol = df_full.groupby(['ID', 'Tipo'])['Qtde Total (Und)'].sum().unstack(fill_value=0)
df_tipo_vol.columns = ['Vol_Hardware', 'Vol_Suprimento']

# Sortimento (SKUs únicos comercializados)
df_sku = df_full.groupby('ID')['ID Produto'].nunique().rename('Skus_unicos')

# Histórico de atendimento presencial
df_rot_agd = df_roteiro.groupby('ID Loja')['Presença Promotor'].sum().rename('Total_Visitas')

# Unificação no Cadastro de PDVs
df_pdv = df_lojas.set_index('ID').join([df_tipo_produto_fat, df_tipo_vol, df_sku, df_rot_agd]).reset_index()
df_pdv = df_pdv.fillna({'Total_Visitas': 0})

# 3. Testes de Adequação da Matriz de Correlação (KMO e Bartlett)
var_columns = ['Fat_Hardware', 'Fat_Suprimento', 'Vol_Hardware', 'Vol_Suprimento', 'Skus_unicos', 'Total_Visitas']
X = df_pdv[var_columns]

# Teste de Esfericidade de Bartlett
R = np.corrcoef(X.values, rowvar=False)
n, p = X.shape
det_R = np.linalg.det(R)
statistic = - (n - 1 - (2*p + 5)/6) * np.log(det_R)
p_val_bartlett = stats.chi2.sf(statistic, df=p*(p-1)/2)

# Teste KMO
inv_R = np.linalg.inv(R)
partial_corr = np.zeros_like(R)
for i in range(p):
    for j in range(p):
        if i != j:
            partial_corr[i, j] = -inv_R[i, j] / np.sqrt(inv_R[i, i] * inv_R[j, j])

r2, p2 = R**2, partial_corr**2
np.fill_diagonal(r2, 0)
np.fill_diagonal(p2, 0)
kmo_total = np.sum(r2) / (np.sum(r2) + np.sum(p2))

print(f"✅ KMO Global: {kmo_total:.4f}")
print(f"✅ Teste de Bartlett (p-valor): {p_val_bartlett:.5e}")

# 4. Padronização Z-score e Extração de Componentes
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
pca.fit(X_scaled)

# 5. Exportação do Score de Potencial (PC1)
df_pdv['Score_PCA_Potencial'] = pca.transform(X_scaled)[:, 0]
df_pdv.to_csv('dpdvs_score_pca.csv', index=False, sep=';')
print("✅ Tabela 'pdvs_score_pca.csv' exportada com sucesso!")