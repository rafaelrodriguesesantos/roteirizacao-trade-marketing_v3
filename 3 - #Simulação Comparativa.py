# ==============================================================================
# TCC - ROTEIRIZAÇÃO INTELIGENTE EM TRADE MARKETING
# Roteiro proposto, simulação comparativa, cenários, validação
# ------------------------------------------------------------------------------
# Lógica:
#   1) Baseline = vendas OBSERVADAS de mar/abr 2026 e roteiro REAL (fRoteiro.csv).
#   2) O efeito da visita é ESTIMADO nos dados (multiplicador por tipo de produto),
#      com intervalo de confiança por bootstrap por loja.
#   3) Comparação "like-for-like": remove-se o efeito das visitas do baseline e
#      aplica-se o efeito das visitas propostas, com a MESMA regra.
#   4) Roteiro proposto (cenário principal):
#        Tier 1 (top 40%)  -> segunda a sexta
#        Tier 2 (20% seg.) -> quarta-feira
#        Tier 3 (40% inf.) -> sem visita
#   5) Cenários alternativos e fronteira orçamento x faturamento: a regra é
#      escolhida com dados até dez/2025, validada em jan-fev/2026 e medida em mar-abr/2026.
#   6) MAE vem de backtest (treino até fev/2026, teste em mar/abr 2026).
# Uso: python Script3_Simulacao_Final.py   (CSVs na mesma pasta, ou DATA_DIR=...)
# ==============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DATA_DIR = os.environ.get('DATA_DIR', '.')
OUT_DIR = os.environ.get('OUT_DIR', '.')
os.makedirs(OUT_DIR, exist_ok=True)


def ler(nome):
    return pd.read_csv(os.path.join(DATA_DIR, nome), sep=';', encoding='utf-8-sig')


def salvar(df, nome):
    df.to_csv(os.path.join(OUT_DIR, nome), index=False, sep=';', decimal=',')


# ------------------------------- PARÂMETROS -----------------------------------
INI, FIM = pd.Timestamp('2026-03-01'), pd.Timestamp('2026-04-30')   # janela simulada
FIM_TREINO = pd.Timestamp('2026-02-28')                              # fim do treino (backtest)
CORTE_SELECAO = pd.Timestamp('2025-12-31')                           # dados usados p/ escolher regras
TIPOS = ['Hardware', 'Suprimento']
N_BOOT, SEED = 1000, 42
NOMES_DIA = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb']
# Regra do cenário principal (0 = segunda ... 5 = sábado)
FREQ_TIER = {'Tier 1': [0, 1, 2, 3, 4], 'Tier 2': [2], 'Tier 3': []}

print("--------------------------------------------------")
print("--- ETAPA 3: SIMULAÇÃO COMPARATIVA E VALIDAÇÃO ---")
print("--------------------------------------------------")

# ------------------------ 1. VENDAS DIÁRIAS POR LOJA --------------------------
vendas = ler('fVendas_Simuladas.csv')
vendas.columns = [c.strip().replace('"', '').replace('\n', ' ') for c in vendas.columns]
col_valor = [c for c in vendas.columns if c.startswith('Valor Total') and 'Quiosque' not in c][0]
prod = ler('dProdutos.csv')[['ID Produto', 'Tipo']]
vendas = vendas.merge(prod, on='ID Produto', how='left')
vendas['Data'] = pd.to_datetime(vendas['Data'])

diario = vendas.groupby(['ID', 'Data', 'Tipo'])[col_valor].sum().unstack('Tipo', fill_value=0)
for t in TIPOS:
    if t not in diario.columns:
        diario[t] = 0.0

ids = sorted(ler('dLojas.csv')['ID'].unique())
datas = pd.date_range(vendas['Data'].min(), vendas['Data'].max(), freq='D')
grade = pd.MultiIndex.from_product([ids, datas], names=['ID', 'Data'])
diario = diario[TIPOS].reindex(grade, fill_value=0.0).reset_index()   # dia sem venda = 0
diario['wd'] = diario['Data'].dt.weekday
print(f"✅ Vendas: {datas.min().date()} a {datas.max().date()} | {len(ids)} PDVs")

# ------------------------ 2. ROTEIRO BASELINE (REAL) --------------------------
rot = ler('fRoteiro.csv')
rot['Data'] = pd.to_datetime(rot['Data'])
rot = rot[rot['Presença Promotor'] == 1][['ID Loja', 'Data']].rename(columns={'ID Loja': 'ID'})
rot = rot.drop_duplicates()
rot['pb'] = 1
ids_atendidas = sorted(rot['ID'].unique())
diario = diario.merge(rot, on=['ID', 'Data'], how='left')
diario['pb'] = diario['pb'].fillna(0).astype(int)

# ------------------------ 3. TIERS (SAÍDA DO SCRIPT 2) ------------------------
tiers = ler('dpdvs_priorizados_ahp_tcc.csv')[['ID', 'Prioridade_Rota', 'Score_AHP_Final']].copy()
tiers['Tier'] = tiers['Prioridade_Rota'].str.extract(r'(Tier \d)')
ranking = tiers.sort_values('Score_AHP_Final', ascending=False)['ID'].tolist()
rpos = {i: p for p, i in enumerate(ranking)}                           # posição no ranking AHP
diario = diario.merge(tiers[['ID', 'Tier', 'Score_AHP_Final']], on='ID', how='left')
diario['rk'] = diario['ID'].map(rpos)

# ------------------------ 4. ROTEIRO PROPOSTO ---------------------------------
cal = pd.DataFrame({'Data': pd.date_range(INI, FIM, freq='D')})
cal['wd'] = cal['Data'].dt.weekday
cal = cal[cal['wd'] < 6]                                               # visitas de seg a sáb
prop = tiers[['ID', 'Tier']].merge(cal, how='cross')
prop['pp'] = [int(w in FREQ_TIER[t]) for t, w in zip(prop['Tier'], prop['wd'])]
rot_prop = prop[['Data', 'ID', 'pp']].rename(columns={'ID': 'ID Loja', 'pp': 'Presença Promotor'}).copy()
rot_prop['Data'] = rot_prop['Data'].dt.strftime('%Y-%m-%d')
salvar(rot_prop, 'fRoteiro_Proposto.csv')
diario = diario.merge(prop[['ID', 'Data', 'pp']], on=['ID', 'Data'], how='left')
diario['pp'] = diario['pp'].fillna(0).astype(int)
print("✅ fRoteiro_Proposto.csv gerado")

# ------------------------ 5. EFEITO DA VISITA (ESTIMADO) ----------------------
# Multiplicador = venda média em dias COM visita / dias SEM visita, na mesma loja e no
# mesmo mês, ajustando o dia da semana pelo índice das lojas nunca visitadas e usando
# só segunda a sábado. Agregado ponderado por valor.


def indice_semana(df, ate=None):
    na = df[~df['ID'].isin(ids_atendidas)]
    if ate is not None:
        na = na[na['Data'] <= ate]
    out = {}
    for t in TIPOS:
        s = na.groupby('wd')[t].mean()
        out[t] = s / s.mean()
    return out


def somas_celulas(df, ate=None):
    sw = indice_semana(df, ate)
    x = df[df['ID'].isin(ids_atendidas) & (df['wd'] < 6)]
    if ate is not None:
        x = x[x['Data'] <= ate]
    x = x.assign(ym=x['Data'].dt.to_period('M'))
    out = {}
    for t in TIPOS:
        x = x.assign(adj=x[t] / x['wd'].map(sw[t]))
        g = x.groupby(['ID', 'ym', 'pb'])['adj'].mean().unstack('pb').dropna()
        out[t] = g.groupby('ID').sum().reindex(ids_atendidas, fill_value=0.0)
    return out


def multiplicador(somas, pesos=None):
    w = np.ones(len(ids_atendidas)) if pesos is None else pesos
    return {t: float((somas[t][1].values * w).sum() / (somas[t][0].values * w).sum()) for t in TIPOS}


somas_full = somas_celulas(diario)
M = multiplicador(somas_full)
print(f"✅ Multiplicador estimado (visita / sem visita): Hardware {M['Hardware']:.3f}x | "
      f"Suprimento {M['Suprimento']:.3f}x")

rng = np.random.default_rng(SEED)
n_att = len(ids_atendidas)
boot_M = []
for _ in range(N_BOOT):
    cont = np.bincount(rng.integers(0, n_att, n_att), minlength=n_att).astype(float)
    boot_M.append(multiplicador(somas_full, cont))

# ------------------------ 6. SIMULAÇÃO LIKE-FOR-LIKE --------------------------
sim = diario[(diario['Data'] >= INI) & (diario['Data'] <= FIM)].copy().reset_index(drop=True)
Y = {t: sim[t].values for t in TIPOS}
PB = sim['pb'].values
Y_TOTAL = sim[TIPOS].sum(axis=1).values
BASE_OBS = Y_TOTAL.sum()
VIS_BASE = int(PB.sum())


def delta_pp(pp, Mt):
    """Variação de receita (proposto - baseline observado) para um vetor de presença pp."""
    d = 0.0
    for t in TIPOS:
        d += (Y[t] * (Mt[t] ** (pp - PB) - 1)).sum()
    return d


def ic_delta(pp):
    bs = np.array([delta_pp(pp, m) for m in boot_M])
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


pp_main = sim['pp'].values
res = sim[['Data', 'ID', 'Tier', 'pb', 'pp']].copy()
sv = np.zeros(len(sim)); pr = np.zeros(len(sim))
for t in TIPOS:
    y_sv = Y[t] / M[t] ** PB                     # remove o efeito das visitas do baseline
    sv += y_sv
    pr += y_sv * M[t] ** pp_main                 # aplica o efeito das visitas propostas
res['Venda_Baseline'] = Y_TOTAL
res['Venda_SemVisita'] = sv
res['Venda_Proposta'] = pr

base = res['Venda_Baseline'].sum()
prop_tot = res['Venda_Proposta'].sum()
vb, vp = int(res['pb'].sum()), int(res['pp'].sum())
attr_b = base - res['Venda_SemVisita'].sum()
attr_p = prop_tot - res['Venda_SemVisita'].sum()
ic_lo, ic_hi = ic_delta(pp_main)

tab1 = pd.DataFrame({
    'Métrica': ['Faturamento (bimestre, R$)', 'Total de visitas (bimestre)',
                'Receita atribuível às visitas (R$)', 'Receita atribuível por visita (R$)'],
    'Baseline (roteiro real)': [base, vb, attr_b, attr_b / vb],
    'Proposto (PCA + AHP)': [prop_tot, vp, attr_p, attr_p / vp],
    'Variação absoluta': [prop_tot - base, vp - vb, attr_p - attr_b, attr_p / vp - attr_b / vb],
    'Variação %': [(prop_tot / base - 1) * 100, (vp / vb - 1) * 100,
                   (attr_p / attr_b - 1) * 100, ((attr_p / vp) / (attr_b / vb) - 1) * 100],
})
salvar(tab1.round(2), 'tabela1_comparativo.csv')
marginal = (prop_tot - base) / (vp - vb)

g = res.groupby('Tier').agg(PDVs=('ID', 'nunique'), Fat_Baseline=('Venda_Baseline', 'sum'),
                            Fat_Proposto=('Venda_Proposta', 'sum'),
                            Visitas_Baseline=('pb', 'sum'), Visitas_Proposto=('pp', 'sum')).reset_index()
g['Incremental'] = g['Fat_Proposto'] - g['Fat_Baseline']
tot = g.drop(columns='Tier').sum(numeric_only=True)
g = pd.concat([g, pd.DataFrame([{'Tier': 'TOTAL', **tot.to_dict()}])], ignore_index=True)
salvar(g.round(2), 'tabela2_por_tier.csv')

# ------------------------ 7. SENSIBILIDADE AO EFEITO DA VISITA ----------------
b_main = np.array([delta_pp(pp_main, m) for m in boot_M])
sens = pd.DataFrame([
    {'Cenário do efeito': 'Efeito estimado (ponto)', 'Variação receita (R$)': delta_pp(pp_main, M)},
    {'Cenário do efeito': 'Limite inferior (IC95% bootstrap)', 'Variação receita (R$)': np.percentile(b_main, 2.5)},
    {'Cenário do efeito': 'Limite superior (IC95% bootstrap)', 'Variação receita (R$)': np.percentile(b_main, 97.5)},
    {'Cenário do efeito': 'Sem efeito negativo em Suprimento (multiplicador = 1)',
     'Variação receita (R$)': delta_pp(pp_main, {'Hardware': M['Hardware'], 'Suprimento': 1.0})},
    {'Cenário do efeito': 'Efeito de Hardware 20% menor que o estimado',
     'Variação receita (R$)': delta_pp(pp_main, {'Hardware': 1 + 0.8 * (M['Hardware'] - 1), 'Suprimento': M['Suprimento']})},
])
sens['Variação %'] = sens['Variação receita (R$)'] / base * 100
salvar(sens.round(2), 'tabela5_sensibilidade_efeito.csv')
mult_tab = pd.DataFrame({'Tipo': TIPOS, 'Multiplicador': [M[t] for t in TIPOS],
                         'IC95% inf': [np.percentile([b[t] for b in boot_M], 2.5) for t in TIPOS],
                         'IC95% sup': [np.percentile([b[t] for b in boot_M], 97.5) for t in TIPOS]})
salvar(mult_tab.round(3), 'multiplicadores_estimados.csv')

# ------------------------ 8. CENÁRIOS E FRONTEIRA (ESCOLHA SEM LOOK-AHEAD) ----
# Ranking de dias e multiplicadores usando SÓ dados até dez/2025; validação em jan-fev/2026.
M_sel = multiplicador(somas_celulas(diario, ate=CORTE_SELECAO))
t0 = diario[(diario['Data'] <= CORTE_SELECAO) & (diario['wd'] < 6)].copy()
for t in TIPOS:
    t0['nv_' + t] = t0[t] / M_sel[t] ** t0['pb']
t0['g'] = t0['nv_Hardware'] * (M_sel['Hardware'] - 1) + t0['nv_Suprimento'] * (M_sel['Suprimento'] - 1)
ganho_dia = t0[t0['Tier'] != 'Tier 3'].groupby('wd')['g'].mean().sort_values(ascending=False)
ordem_dias = ganho_dia.index.tolist()
wpos = {w: i for i, w in enumerate(ordem_dias)}
print("✅ Ranking de dias por ganho esperado (R$/visita, até dez/25):",
      [(NOMES_DIA[w], round(v)) for w, v in ganho_dia.items()])


def arrays(d):
    return {'rk': d['rk'].values, 'wd': d['wd'].values, 'wp': d['wd'].map(wpos).fillna(99).values,
            'pb': d['pb'].values, 'yH': d['Hardware'].values, 'yS': d['Suprimento'].values,
            'tot': (d['Hardware'] + d['Suprimento']).sum()}


A_val = arrays(diario[(diario['Data'] >= '2026-01-01') & (diario['Data'] <= '2026-02-28')])
A_tst = arrays(sim)


def delta_A(A, pp, Mt):
    return sum((A[k] * (Mt[t] ** (pp - A['pb']) - 1)).sum() for k, t in (('yH', 'Hardware'), ('yS', 'Suprimento')))


def pp_rank(A, N1, dias1, N2=0, dias2=()):
    """Top N1 lojas do ranking AHP nos dias1; as N2 seguintes nos dias2 (dias = 0..5 seg..sáb)."""
    m1 = (A['rk'] < N1) & np.isin(A['wd'], dias1)
    m2 = (A['rk'] >= N1) & (A['rk'] < N1 + N2) & np.isin(A['wd'], dias2)
    return (m1 | m2).astype(int)


# garante que o cenário principal equivale à regra por Tier
n_t1 = int((tiers['Tier'] == 'Tier 1').sum()); n_t2 = int((tiers['Tier'] == 'Tier 2').sum())
assert (pp_rank(A_tst, n_t1, FREQ_TIER['Tier 1'], n_t2, FREQ_TIER['Tier 2']) == pp_main).all()


def linha(nome, pp, extra=None):
    d0 = delta_A(A_tst, pp, M)
    lo, hi = ic_delta(pp)
    v = int(pp.sum())
    r = {'Cenário': nome, 'Visitas': v, 'Δ visitas %': (v / VIS_BASE - 1) * 100,
         'Δ faturamento (R$)': d0, 'Δ %': d0 / BASE_OBS * 100, 'IC95% inf (R$)': lo, 'IC95% sup (R$)': hi,
         'R$ por visita extra': d0 / (v - VIS_BASE) if v > VIS_BASE else np.nan}
    if extra:
        r.update(extra)
    return r


# Escolha por validação (jan-fev/2026), dentro de faixas de orçamento, entre top-N x k dias/semana
cand = []
for N in range(30, 135):
    for k in range(1, 7):
        dias = ordem_dias[:k]
        pv = pp_rank(A_val, N, dias); pt = pp_rank(A_tst, N, dias)
        cand.append((N, k, int(pt.sum()), delta_A(A_val, pv, M_sel) / A_val['tot'] * 100))
cand = pd.DataFrame(cand, columns=['N', 'k', 'Visitas_teste', 'Δ%_validação'])


def escolher(lo, hi):
    return cand[(cand['Visitas_teste'] >= lo) & (cand['Visitas_teste'] <= hi)].sort_values('Δ%_validação', ascending=False).iloc[0]


cen = []
for nome, (lo, hi) in {'Orçamento constante (≈100% das visitas)': (VIS_BASE * 0.95, VIS_BASE),
                       'Orçamento ≈125%': (VIS_BASE * 1.17, VIS_BASE * 1.27),
                       'Orçamento ≈150%': (VIS_BASE * 1.45, VIS_BASE * 1.55)}.items():
    x = escolher(lo, hi)
    dias = ordem_dias[:int(x.k)]
    desc = f"{nome} — {int(x.N)} lojas × {int(x.k)} dias ({'+'.join(NOMES_DIA[w] for w in dias)}), escolhida na validação"
    cen.append(linha(desc, pp_rank(A_tst, int(x.N), dias), {'Δ% na validação (jan-fev/26)': x['Δ%_validação']}))
cen.insert(0, linha(f'Top 40% ({n_t1} lojas), seg–sex; sem Tier 2', pp_rank(A_tst, n_t1, [0, 1, 2, 3, 4])))
cen.insert(1, linha(f'PRINCIPAL: Tier 1 ({n_t1}) seg–sex + Tier 2 ({n_t2}) quarta', pp_main))
cen.insert(2, linha(f'Tier 1 ({n_t1}) seg–sex + Tier 2 ({n_t2}) terça e quarta', pp_rank(A_tst, n_t1, [0, 1, 2, 3, 4], n_t2, [1, 2])))
tab3 = pd.DataFrame(cen)
salvar(tab3.round(2), 'tabela3_cenarios.csv')

# ------------------------ 9. BACKTEST (TREINO ATÉ FEV/2026) -------------------
M_tr = multiplicador(somas_celulas(diario, ate=FIM_TREINO))
sw_tr = indice_semana(diario, ate=FIM_TREINO)
janela = diario[(diario['Data'] > FIM_TREINO - pd.Timedelta(days=56)) & (diario['Data'] <= FIM_TREINO)]
teste = sim.copy()
teste['y'] = Y_TOTAL
pred = np.zeros(len(teste)); naive = np.zeros(len(teste))
for t in TIPOS:
    sj = janela['wd'].map(sw_tr[t])
    st = teste['wd'].map(sw_tr[t]).values
    b_vis = (janela[t] / (sj * M_tr[t] ** janela['pb'])).groupby(janela['ID']).mean()
    b_raw = (janela[t] / sj).groupby(janela['ID']).mean()
    pred += teste['ID'].map(b_vis).fillna(0).values * st * M_tr[t] ** teste['pb'].values
    naive += teste['ID'].map(b_raw).fillna(0).values * st
teste['pred'], teste['naive'] = pred, naive
teste['mes'] = teste['Data'].dt.to_period('M').astype(str)


def met(df, col):
    e = df[col] - df['y']
    return e.abs().mean(), e.abs().sum() / df['y'].sum() * 100, (df[col].sum() / df['y'].sum() - 1) * 100


linhas = []
for nivel, d in [('Loja-dia', teste),
                 ('Loja-mês', teste.groupby(['ID', 'mes'])[['y', 'pred', 'naive']].sum().reset_index())]:
    for nome, col in [('Modelo (com efeito de visita)', 'pred'), ('Benchmark (sem efeito de visita)', 'naive')]:
        mae, wape, vies = met(d, col)
        linhas.append({'Nível': nivel, 'Modelo': nome, 'MAE (R$)': mae, 'WAPE (%)': wape, 'Viés (%)': vies})
bt = pd.DataFrame(linhas)
salvar(bt.round(2), 'tabela4_backtest.csv')
vies_mes = teste.groupby('mes')[['y', 'pred']].sum()
vies_mes['Viés (%)'] = (vies_mes['pred'] / vies_mes['y'] - 1) * 100

# ------------------------ 10. ESTUDO DE CASO (LOJA REPRESENTATIVA DO TIER 1) --
# Critério definido a priori: loja do Tier 1 com variação de faturamento em março
# mais próxima da MEDIANA do Tier 1 (evita escolher a melhor loja).
mar = res[(res['Data'] >= '2026-03-01') & (res['Data'] <= '2026-03-31')]
dl = mar[mar['Tier'] == 'Tier 1'].groupby('ID').apply(lambda d: d['Venda_Proposta'].sum() - d['Venda_Baseline'].sum())
loja_caso = int((dl - dl.median()).abs().idxmin())
nome_loja = ler('dLojas.csv').set_index('ID').loc[loja_caso, 'Loja'].strip('-')
c = mar[mar['ID'] == loja_caso].sort_values('Data')
caso = pd.DataFrame({
    'Loja': [nome_loja], 'ID': [loja_caso],
    'Faturamento mar/26 baseline (R$)': [c['Venda_Baseline'].sum()],
    'Faturamento mar/26 proposto (R$)': [c['Venda_Proposta'].sum()],
    'Visitas baseline': [int(c['pb'].sum())], 'Visitas proposto': [int(c['pp'].sum())],
    'Venda média dia COM visita (R$)': [c.loc[c['pp'] == 1, 'Venda_Proposta'].mean()],
    'Venda média dia SEM visita (R$)': [c.loc[c['pp'] == 0, 'Venda_SemVisita'].mean()],
})
salvar(caso.round(2), 'estudo_de_caso.csv')

fig, ax = plt.subplots(1, 2, figsize=(10, 4.0), gridspec_kw={'width_ratios': [2.3, 1]})
a1 = ax[0]
a1.plot(c['Data'], c['Venda_Baseline'], color='#8c8c8c', lw=1.4, label='Baseline (observado)')
a1.plot(c['Data'], c['Venda_Proposta'], color='#1f4e9c', lw=1.6, label='Proposto (PCA + AHP)')
vb_d = c[c['pb'] == 1]; vp_d = c[c['pp'] == 1]
a1.scatter(vb_d['Data'], vb_d['Venda_Baseline'], facecolors='none', edgecolors='#8c8c8c', s=34, label='Visita baseline')
a1.scatter(vp_d['Data'], vp_d['Venda_Proposta'], marker='^', color='#d9541e', s=34, label='Visita proposta')
a1.set_title(f'(1) Vendas diárias e visitas — {nome_loja} (Tier 1), mar/2026', fontsize=11)
a1.set_ylabel('R$ por dia', fontsize=10); a1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m')); a1.tick_params(axis='both', labelsize=10); a1.legend(fontsize=9, ncol=2)
a1.grid(alpha=.25)
a2 = ax[1]
vals = [c['Venda_Baseline'].sum() / 1000, c['Venda_Proposta'].sum() / 1000]
bars = a2.bar(['Baseline', 'Proposto'], vals, color=['#8c8c8c', '#1f4e9c'], width=.55)
for b, v, vis in zip(bars, vals, [int(c['pb'].sum()), int(c['pp'].sum())]):
    a2.text(b.get_x() + b.get_width() / 2, v, f'R$ {v:,.1f}k\n{vis} visitas'.replace(',', 'X').replace('.', ',').replace('X', '.'),
            ha='center', va='bottom', fontsize=10)
a2.set_ylim(0, max(vals) * 1.25); a2.set_title('(2) Faturamento e visitas no mês', fontsize=11); a2.tick_params(axis='both', labelsize=10)
a2.set_ylabel('R$ mil', fontsize=10); a2.grid(axis='y', alpha=.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'figura5_estudo_caso.png'), dpi=200)
plt.close()

# ------------------------ 11. EXPORTAÇÃO + RESUMO -----------------------------
saida = res.rename(columns={'ID': 'ID Loja', 'pb': 'Pres_Baseline', 'pp': 'Pres_Proposto'}).copy()
saida['Data'] = saida['Data'].dt.strftime('%Y-%m-%d')
salvar(saida.round(2), 'resultado_simulacao_mar_abr_2026.csv')

pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')
print("\n=== TABELA 1 - Comparativo (cenário principal) ===");        print(tab1.to_string(index=False))
print(f"IC95% (bootstrap) da variação de faturamento: R$ {ic_lo:,.0f} a R$ {ic_hi:,.0f} "
      f"({ic_lo / base * 100:.2f}% a {ic_hi / base * 100:.2f}%) | receita marginal por visita extra: R$ {marginal:,.2f}")
print("\n=== TABELA 2 - Por Tier ===");                              print(g.to_string(index=False))
print("\n=== TABELA 3 - Cenários (regras escolhidas sem olhar o teste) ==="); print(tab3.to_string(index=False))
print("\n=== TABELA 4 - Backtest (treino até fev/26, teste mar-abr/26) ==="); print(bt.to_string(index=False))
print("\nViés do modelo por mês:");                                 print(vies_mes.to_string())
print("\n=== Multiplicadores estimados ===");                       print(mult_tab.to_string(index=False))
print("\n=== TABELA 5 - Sensibilidade ao efeito da visita ===");    print(sens.to_string(index=False))
print("\n=== Estudo de caso ===");                                  print(caso.T.to_string(header=False))
print("\n✅ Arquivos exportados em:", os.path.abspath(OUT_DIR))