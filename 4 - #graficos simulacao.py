# ==============================================================================
# TCC - ROTEIRIZAÇÃO INTELIGENTE EM TRADE MARKETING
# Figura 5 - Estudo de caso de um PDV do Tier 1 (março/2026)
# ------------------------------------------------------------------------------
# Lê as saídas do Script 3 (resultado_simulacao_mar_abr_2026.csv) e gera a mesma
# figura que consta no TCC. O que mudou em relação à versão anterior:
#   - Usa as visitas REAIS do baseline (Pres_Baseline) e as vendas OBSERVADAS;
#     não há mais simulação aleatória de visitas nem fator fixo de +12%.
#   - O cenário proposto vem da simulação like-for-like do Script 3
#     (Venda_Proposta), com o mesmo efeito de visita estimado nos dados.
#   - A loja é escolhida por critério definido a priori: a loja do Tier 1 cuja
#     variação de faturamento em março está mais próxima da MEDIANA do Tier 1
#     (evita escolher o melhor caso). Para forçar outra loja, defina ID_LOJA.
# Uso: python Script4_Figura_Estudo_Caso.py   (CSVs na mesma pasta, ou DATA_DIR=...)
# ==============================================================================

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DATA_DIR = os.environ.get('DATA_DIR', '.')
OUT_DIR = os.environ.get('OUT_DIR', '.')
os.makedirs(OUT_DIR, exist_ok=True)

ID_LOJA = None          # ex.: 111 para fixar uma loja; None = critério da mediana (Tier 1)
MES_INI, MES_FIM = '2026-03-01', '2026-03-31'

print("--------------------------------------------------")
print("--- GERANDO FIGURA DO ESTUDO DE CASO (TIER 1) ---")
print("--------------------------------------------------")

# 1. Saídas do Script 3 (decimal vírgula) e cadastro de lojas
res = pd.read_csv(os.path.join(DATA_DIR, 'resultado_simulacao_mar_abr_2026.csv'), sep=';', decimal=',')
res['Data'] = pd.to_datetime(res['Data'])
lojas = pd.read_csv(os.path.join(DATA_DIR, 'dLojas.csv'), sep=';', encoding='utf-8-sig').set_index('ID')

mar = res[(res['Data'] >= MES_INI) & (res['Data'] <= MES_FIM)]

# 2. Escolha da loja (critério a priori: mediana da variação de faturamento no Tier 1)
if ID_LOJA is None:
    dl = mar[mar['Tier'] == 'Tier 1'].groupby('ID Loja').apply(
        lambda d: d['Venda_Proposta'].sum() - d['Venda_Baseline'].sum())
    ID_LOJA = int((dl - dl.median()).abs().idxmin())
nome_loja = str(lojas.loc[ID_LOJA, 'Loja']).strip('-')
c = mar[mar['ID Loja'] == ID_LOJA].sort_values('Data')
print(f"✅ Loja do estudo de caso: {nome_loja} (ID {ID_LOJA}) | {c['Tier'].iloc[0]}")

fat_base, fat_prop = c['Venda_Baseline'].sum(), c['Venda_Proposta'].sum()
vis_base, vis_prop = int(c['Pres_Baseline'].sum()), int(c['Pres_Proposto'].sum())
print(f"   Faturamento mar/26: baseline R$ {fat_base:,.0f} | proposto R$ {fat_prop:,.0f} "
      f"({(fat_prop / fat_base - 1) * 100:+.1f}%) | visitas {vis_base} -> {vis_prop}")

# 3. Figura
fig, ax = plt.subplots(1, 2, figsize=(10, 4.0), gridspec_kw={'width_ratios': [2.3, 1]})

# Painel 1: vendas diárias e visitas
a1 = ax[0]
a1.plot(c['Data'], c['Venda_Baseline'], color='#8c8c8c', lw=1.4, label='Baseline (observado)')
a1.plot(c['Data'], c['Venda_Proposta'], color='#1f4e9c', lw=1.6, label='Proposto (PCA + AHP)')
vb_d = c[c['Pres_Baseline'] == 1]
vp_d = c[c['Pres_Proposto'] == 1]
a1.scatter(vb_d['Data'], vb_d['Venda_Baseline'], facecolors='none', edgecolors='#8c8c8c', s=34, label='Visita baseline')
a1.scatter(vp_d['Data'], vp_d['Venda_Proposta'], marker='^', color='#d9541e', s=34, label='Visita proposta')
a1.set_title(f'(1) Vendas diárias e visitas — {nome_loja} (Tier 1), mar/2026', fontsize=11)
a1.set_ylabel('R$ por dia', fontsize=10)
a1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
a1.tick_params(axis='both', labelsize=10)
a1.legend(fontsize=9, ncol=2)
a1.grid(alpha=.25)

# Painel 2: consolidado mensal
a2 = ax[1]
vals = [fat_base / 1000, fat_prop / 1000]
bars = a2.bar(['Baseline', 'Proposto'], vals, color=['#8c8c8c', '#1f4e9c'], width=.55)
for b, v, vis in zip(bars, vals, [vis_base, vis_prop]):
    a2.text(b.get_x() + b.get_width() / 2, v,
            f'R$ {v:,.1f}k\n{vis} visitas'.replace(',', 'X').replace('.', ',').replace('X', '.'),
            ha='center', va='bottom', fontsize=10)
a2.set_ylim(0, max(vals) * 1.25)
a2.set_title('(2) Faturamento e visitas no mês', fontsize=11)
a2.tick_params(axis='both', labelsize=10)
a2.set_ylabel('R$ mil', fontsize=10)
a2.grid(axis='y', alpha=.25)

plt.tight_layout()
destino = os.path.join(OUT_DIR, 'figura5_estudo_caso.png')
plt.savefig(destino, dpi=200)
plt.close()
print(f"✅ Figura '{destino}' gerada com sucesso!")