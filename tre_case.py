"""risk mot tillväxt i tres telefonkrediter - all beräkningskod i en fil

kortversion av mitt case. bygger datan, skattar modellerna och räknar det som rapporten bygger på: tillväxtens pris,
motbudsmenyn, konjunkturkänsligheten, stresstestet och kreditstegen. den längre versionen med datagranskning (modul 1)
har jag kvar separat

data (ligger inte i repot pga kaggles villkor), lägg den under mappen TRE_DATA:
  hcdr/      Home Credit Default Risk: previous_application, installments_payments, bureau, bureau_balance,
             application_train, application_test
  lc/        Lending Club 2007-2018: accepted_2007_to_2018Q4.csv.gz
  macro/     Unemployment in America Per US State.csv (BLS via Kaggle)
  sverige/   aku_arbetsloshet_manad.csv (SCB) och prognos_ki_sep2026.csv (Konjunkturinstitutet)

kör:  TRE_DATA=/sökväg/till/data python tre_case.py   (ca 5 min, ger samma resultat varje gång)
krav: python 3.11+, polars, pandas, numpy, scipy, scikit-learn, statsmodels, lightgbm
ut:   outputs/resultat.json, outputs/tillvaxtens_pris.csv, outputs/kreditstege.csv

OBS om resultaten: risken är observerad i home credit och lending club, alltså inte hos tre. ekonomin, lgd, nekades risk,
acceptans för motbud, telekomfaktorn och köpkraftschocken är antaganden. bidrag, brytpunkter, menyns värde och
stresscenarier är simulerat = observerad risk + antaganden ihop
"""
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from scipy import sparse
from scipy.stats import chi2, norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DATA = Path(os.environ.get("TRE_DATA", "data"))
HC, OUT = DATA / "hcdr", Path("outputs")
SEED = 42

# population och utfall
LOOKBACK = 730            # samma fönster för alla lån, 730 dagar innan beslut
MIN_DAYS_DECISION = -2192 # så att hela fönstret ryms, tabellerna går bara 2922 dagar bakåt
OUTCOME_WINDOW = 365      # utfallsfönster från första förfallodagen
OBS_BUFFER = 35           # sista delbetalningen ska hinna bli >30 dagar sen
MATERIAL = 0.90           # väsentlighetsgräns. sen bara om <90% av delbetalningen är betald (annars småbelopp)
PLACEHOLDER, MONTH = 365243, 30.4375

# ekonomi i kr, antaganden. källor: ck hutchison + hi3g årsredovisningar 2025
ECON = dict(arpu=213.0, margin=157 / 213, churn=0.014, r_year=0.08, T=60, A=1700.0, C=420.0, H=8900.0,
            lgd=0.62, tau=6, term=36, r_fund=0.04)
SCENARIOS = {"låg": dict(margin=0.33), "bas": dict(), "hög": dict(margin=192 / 213, churn=0.013, A=1160.0)}
OFFERS = dict(cheap=dict(H=4000.0), sim=dict(H=0.0, arpu=180.0, churn=0.018, A=700.0, unpaid_months=2, tau=3))
K_REJECT = 2.0            # nekades verkliga risk = k ggr modellens PD, antagande (testat 1-10)


# =====================================================================================================
# 1. ekonomin - värdet av bra resp dålig kund + brytpunkten
# =====================================================================================================
def funding_cost(H, r_fund, term=36):
    """kostnad för räntefri delbetalning, H minus nuvärdet av alla månadsbetalningar"""
    v = (1 + r_fund) ** (-1 / 12)
    return H - H / term * v * (1 - v ** term) / (1 - v)


def values(w, **override):
    """G = m*S - A, L = A + LGD*EAD + C - m*tau, V_bad = -w*L + (1-w)*(G-C)
    w = andelen dåliga som faktiskt skrivs av"""
    e = {**ECON, **override}
    q = (1 - e["churn"]) / (1 + e["r_year"]) ** (1 / 12)
    S = q * (1 - q ** e["T"]) / (1 - q)                                   # diskonterad kundtid, blir 34,2 mån
    m = e["margin"] * e["arpu"]
    G = m * S - e["A"]
    L = e["A"] + e["lgd"] * e["H"] * (1 - e["tau"] / e["term"]) + e["C"] - m * e["tau"] + e.get("unpaid_months", 0) * e["arpu"]
    V_bad = -w * L + (1 - w) * (G - e["C"])
    F = funding_cost(e["H"], e["r_fund"], e["term"])
    return dict(S=S, G=G, L=L, V_bad=V_bad, F=F, p_star=(G - F) / (G - V_bad))


def expected_value(q, v):
    """förväntat värde om man godkänner en kund med sannolikhet q att bli dålig"""
    return (1 - q) * v["G"] + q * v["V_bad"] - v["F"]


# =====================================================================================================
# 2. data - population, utfall och variabler som fanns innan beslutet
# =====================================================================================================
def load_previous():
    prev = pl.read_csv(HC / "previous_application.csv", infer_schema_length=100_000)
    return prev.with_columns([pl.when(pl.col(c) == PLACEHOLDER).then(None).otherwise(pl.col(c)).alias(c)
                              for c in prev.columns if c.startswith("DAYS_")])


def load_installments():
    f32 = pl.Float32
    return pl.read_csv(HC / "installments_payments.csv", schema_overrides={
        "SK_ID_PREV": pl.Int32, "SK_ID_CURR": pl.Int32, "NUM_INSTALMENT_VERSION": f32, "NUM_INSTALMENT_NUMBER": pl.Int32,
        "DAYS_INSTALMENT": f32, "DAYS_ENTRY_PAYMENT": f32, "AMT_INSTALMENT": f32, "AMT_PAYMENT": f32})


def build_population(prev, inst):
    """beviljade telefonlån där hela 12 mån fönstret syns. y = någon delbetalning har <90% betalt 30 dagar efter
    förfallodagen (y60, y90 samma sak). obs 74% av raderna som är >30 dagar sena är bara små restbelopp!!"""
    phone = prev.filter((pl.col("NAME_GOODS_CATEGORY") == "Mobile") & (pl.col("NAME_CONTRACT_STATUS") == "Approved"))
    pi = inst.join(phone.select("SK_ID_PREV"), on="SK_ID_PREV", how="semi")
    phone = phone.join(pi.group_by("SK_ID_PREV").agg(pl.col("DAYS_INSTALMENT").min().alias("d1")), on="SK_ID_PREV")
    phone = phone.filter((pl.col("d1") + OUTCOME_WINDOW + OBS_BUFFER <= 0) & (pl.col("DAYS_DECISION") >= MIN_DAYS_DECISION))
    w = (pi.join(phone.select("SK_ID_PREV", "d1"), on="SK_ID_PREV")
         .filter(pl.col("DAYS_INSTALMENT") <= pl.col("d1") + OUTCOME_WINDOW)
         .with_columns([pl.when(pl.col("DAYS_ENTRY_PAYMENT") <= pl.col("DAYS_INSTALMENT") + lag).then(pl.col("AMT_PAYMENT"))
                        .otherwise(0.0).fill_null(0.0).alias(f"p{lag}") for lag in (30, 60, 90)]))
    per_inst = w.group_by(["SK_ID_PREV", "NUM_INSTALMENT_NUMBER", "NUM_INSTALMENT_VERSION"]).agg(
        pl.col("AMT_INSTALMENT").max().alias("due"), pl.col("p30").sum(), pl.col("p60").sum(), pl.col("p90").sum())
    out = per_inst.group_by("SK_ID_PREV").agg(
        *[(pl.col(f"p{lag}") < MATERIAL * pl.col("due")).any().cast(pl.Int8).alias(name)
          for lag, name in [(30, "y"), (60, "y60"), (90, "y90")]])
    return phone.join(out, on="SK_ID_PREV")


def features_A(rows):
    """A = ansökan. prisvariablerna skapas men används inte, de speglar home credits gamla riskpolicy"""
    cols = ["SK_ID_CURR", "CODE_GENDER", "NAME_EDUCATION_TYPE", "DAYS_BIRTH"]
    app = pl.concat([pl.read_csv(HC / "application_train.csv", columns=cols), pl.read_csv(HC / "application_test.csv", columns=cols)])
    a = rows.select(
        "SK_ID_PREV", "SK_ID_CURR", pl.col("DAYS_DECISION").alias("t0"),
        pl.col("AMT_APPLICATION").alias("A_amt_application"), pl.col("AMT_CREDIT").alias("A_amt_credit"),
        pl.col("AMT_GOODS_PRICE").alias("A_amt_goods"), pl.col("AMT_ANNUITY").alias("A_amt_annuity"),
        pl.col("AMT_DOWN_PAYMENT").alias("A_amt_down"), pl.col("RATE_DOWN_PAYMENT").alias("A_rate_down"),
        pl.col("CNT_PAYMENT").alias("A_cnt_payment"),
        (pl.col("AMT_CREDIT") / pl.col("AMT_APPLICATION")).alias("A_credit_to_app"),
        (pl.col("AMT_ANNUITY") / pl.col("AMT_CREDIT")).alias("A_annuity_to_credit"),
        pl.col("CHANNEL_TYPE").alias("A_channel"), pl.col("NAME_SELLER_INDUSTRY").alias("A_seller_industry"),
        pl.col("NAME_YIELD_GROUP").alias("A_yield_group"), pl.col("NFLAG_INSURED_ON_APPROVAL").alias("A_insured"),
        pl.col("NAME_TYPE_SUITE").fill_null("Missing").alias("A_type_suite"), pl.col("PRODUCT_COMBINATION").alias("A_product_comb"),
        pl.col("WEEKDAY_APPR_PROCESS_START").alias("A_weekday"), pl.col("HOUR_APPR_PROCESS_START").alias("A_hour"))
    return a.join(app, on="SK_ID_CURR", how="left").with_columns(
        ((pl.col("t0") - pl.col("DAYS_BIRTH")) / 365.25).alias("A_age_t0"), pl.col("CODE_GENDER").alias("A_gender"),
        pl.col("NAME_EDUCATION_TYPE").alias("A_education")).drop("DAYS_BIRTH", "CODE_GENDER", "NAME_EDUCATION_TYPE")


def features_B(keys):
    """B = kreditupplysning. bara krediter och månadsstatus från 730 dagar före t0"""
    b = pl.read_csv(HC / "bureau.csv", columns=["SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "DAYS_CREDIT",
                                                 "DAYS_CREDIT_ENDDATE", "DAYS_ENDDATE_FACT", "CREDIT_TYPE"])
    b = b.with_columns(pl.when(pl.col("CREDIT_ACTIVE") == "Active").then(0.0).when(pl.col("DAYS_ENDDATE_FACT").is_not_null())
                       .then(pl.col("DAYS_ENDDATE_FACT")).otherwise(pl.col("DAYS_CREDIT_ENDDATE")).alias("end_day"))
    j = keys.join(b, on="SK_ID_CURR").filter((pl.col("DAYS_CREDIT") < pl.col("t0")) & (pl.col("DAYS_CREDIT") >= pl.col("t0") - LOOKBACK))
    ct = pl.col("CREDIT_TYPE")
    agg = j.group_by("SK_ID_PREV").agg(
        pl.len().alias("B_n_credits"), (pl.col("DAYS_CREDIT") >= pl.col("t0") - 365).sum().alias("B_n_new_365"),
        (pl.col("end_day").is_null() | (pl.col("end_day") > pl.col("t0"))).sum().alias("B_n_active_t0"),
        (pl.col("t0").first() - pl.col("DAYS_CREDIT").min()).alias("B_hist_days"),
        (pl.col("t0").first() - pl.col("DAYS_CREDIT").max()).alias("B_days_since_last"),
        (ct == "Credit card").sum().alias("B_n_card"), (ct == "Consumer credit").sum().alias("B_n_consumer"),
        (ct == "Microloan").sum().alias("B_n_micro"), ct.is_in(["Car loan", "Mortgage"]).sum().alias("B_n_secured"))
    bb = pl.read_csv(HC / "bureau_balance.csv", schema_overrides={"SK_ID_BUREAU": pl.Int32, "MONTHS_BALANCE": pl.Int16})
    bb = bb.filter(pl.col("STATUS").is_in(list("012345"))).with_columns(pl.col("STATUS").cast(pl.Int8))
    link = j.select("SK_ID_PREV", "SK_ID_BUREAU", (pl.col("t0") / MONTH).floor().cast(pl.Int16).alias("t0_m"))
    jb = link.join(bb, on="SK_ID_BUREAU").filter((pl.col("MONTHS_BALANCE") < pl.col("t0_m")) & (pl.col("MONTHS_BALANCE") >= pl.col("t0_m") - LOOKBACK // 30))
    aggb = jb.group_by("SK_ID_PREV").agg(
        pl.len().alias("B_bb_months"), pl.col("STATUS").max().alias("B_bb_worst"), (pl.col("STATUS") >= 1).mean().alias("B_bb_share_late"),
        pl.col("STATUS").filter(pl.col("MONTHS_BALANCE") >= pl.col("t0_m") - 12).max().alias("B_bb_worst_12m"))
    out = keys.select("SK_ID_PREV").join(agg, on="SK_ID_PREV", how="left").join(aggb, on="SK_ID_PREV", how="left")
    return out.with_columns(pl.col("B_n_credits").is_not_null().cast(pl.Int8).alias("B_has_bureau"))


def features_C(keys, rows, prev, inst):
    """C = egen historik hos långivaren. förseningar som de såg ut vid t0 (betalt efter t0 räknas som fortfarnde sen)"""
    ct = rows.select("SK_ID_PREV", pl.col("NAME_CLIENT_TYPE").alias("C_client_type"))
    p = prev.select("SK_ID_CURR", pl.col("SK_ID_PREV").alias("other"), "DAYS_DECISION", "NAME_CONTRACT_STATUS",
                    "CODE_REJECT_REASON", "NAME_GOODS_CATEGORY")
    window = lambda c: (pl.col(c) < pl.col("t0")) & (pl.col(c) >= pl.col("t0") - LOOKBACK) & (pl.col("other") != pl.col("SK_ID_PREV"))
    st = pl.col("NAME_CONTRACT_STATUS")
    aggp = keys.join(p, on="SK_ID_CURR").filter(window("DAYS_DECISION")).group_by("SK_ID_PREV").agg(
        pl.len().alias("C_n_apps"), (st == "Approved").sum().alias("C_n_approved"), (st == "Refused").sum().alias("C_n_refused"),
        (pl.col("CODE_REJECT_REASON") == "LIMIT").sum().alias("C_n_limit_refused"),
        ((pl.col("NAME_GOODS_CATEGORY") == "Mobile") & (st == "Approved")).sum().alias("C_n_prev_phone"),
        (pl.col("t0").first() - pl.col("DAYS_DECISION").max()).alias("C_days_since_app"),
    ).with_columns((pl.col("C_n_refused") / pl.col("C_n_apps")).alias("C_share_refused"))
    i = inst.select("SK_ID_CURR", pl.col("SK_ID_PREV").alias("other"), "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT")
    ji = keys.join(i, on="SK_ID_CURR").filter(window("DAYS_INSTALMENT")).with_columns(
        pl.when(pl.col("DAYS_ENTRY_PAYMENT").is_null() | (pl.col("DAYS_ENTRY_PAYMENT") >= pl.col("t0")))
        .then(pl.col("t0") - pl.col("DAYS_INSTALMENT")).otherwise(pl.col("DAYS_ENTRY_PAYMENT") - pl.col("DAYS_INSTALMENT")).alias("late"),
        pl.when(pl.col("DAYS_ENTRY_PAYMENT") < pl.col("t0")).then(pl.col("DAYS_ENTRY_PAYMENT")).alias("pay"))
    aggi = ji.group_by("SK_ID_PREV").agg(
        pl.len().alias("C_n_inst"), pl.col("other").n_unique().alias("C_n_loans_paid"),
        *[(pl.col("late") > d).mean().alias(f"C_share_late{d}") for d in (0, 10, 30)],
        pl.col("late").max().alias("C_max_late"), pl.col("late").clip(-60, 120).mean().alias("C_mean_late"),
        (pl.col("t0").first() - pl.col("pay").max()).alias("C_days_since_pay"))
    out = keys.select("SK_ID_PREV").join(ct, on="SK_ID_PREV", how="left").join(aggp, on="SK_ID_PREV", how="left").join(aggi, on="SK_ID_PREV", how="left")
    return out.with_columns(pl.col("C_n_inst").is_not_null().cast(pl.Int8).alias("C_has_history"))


def features(rows, prev, inst):
    keys = rows.select("SK_ID_PREV", "SK_ID_CURR", pl.col("DAYS_DECISION").alias("t0"))
    return features_A(rows).join(features_B(keys), on="SK_ID_PREV").join(features_C(keys, rows, prev, inst), on="SK_ID_PREV")


def split_of(ids):
    """60/20/20 per kund med fast seed, samma kund hamnar alltid i samma del"""
    u = (pl.Series(ids).hash(seed=SEED) % 10_000) / 10_000
    return np.where(u < 0.6, "train", np.where(u < 0.8, "valid", "test"))


def build_dataset(prev, inst):
    phone = build_population(prev, inst)
    df = features(phone, prev, inst).join(phone.select("SK_ID_PREV", "y", "y60", "y90"), on="SK_ID_PREV").sort("SK_ID_PREV").to_pandas()
    df["split"] = split_of(df.SK_ID_CURR.values)
    return df


# =====================================================================================================
# 3. modeller - scorekort (beslutsmodell) och lightgbm (utmanare)
# =====================================================================================================
CAT = ["A_channel", "A_seller_industry", "A_yield_group", "A_type_suite", "A_product_comb", "A_weekday", "A_gender",
       "A_education", "C_client_type"]
PRICING = {"A_yield_group", "A_product_comb", "A_annuity_to_credit", "A_amt_annuity"}
MONOTONE_UP = {"C_share_late0", "C_share_late10", "C_share_late30", "C_max_late", "C_mean_late", "B_bb_worst",
               "B_bb_worst_12m", "B_bb_share_late", "C_share_refused"}


def feature_groups(df, groups, drop=()):
    cols = [c for c in df.columns if c[:2] in {f"{g}_" for g in groups} and c not in PRICING and c not in drop]
    return [c for c in cols if c not in CAT], [c for c in cols if c in CAT]


def woe_table(x_bin, y, smooth=0.5):
    t = pd.crosstab(x_bin, y).reindex(columns=[0, 1], fill_value=0)
    t.columns = ["good", "bad"]
    G, B = t.good.sum(), t.bad.sum()
    t["woe"] = np.log(((t.good + smooth) / G) / ((t.bad + smooth) / B))          # hög woe = låg risk
    t["iv"] = (t.good / G - t.bad / B) * t.woe
    t["n"], t["bad_rate"] = t.good + t.bad, t.bad / (t.good + t.bad)
    return t


def merge_bins(edges, x, y, min_share=0.05, min_bad=30):
    """slår ihop klasser bredvid varandra tills andelen dåliga är monoton och klasserna är tillräckligt stora"""
    while True:
        t = pd.crosstab(pd.cut(x, edges, include_lowest=True, duplicates="drop"), y).reindex(columns=[0, 1], fill_value=0)
        n, bad = t.sum(1).values, t[1].values
        rate, k = bad / np.maximum(n, 1), len(n)
        if k <= 2:
            return edges
        sign = np.sign(np.corrcoef(np.arange(k), rate)[0, 1]) or 1
        viol = [i for i in range(k - 1) if (rate[i + 1] - rate[i]) * sign < 0]
        small = [i for i in range(k) if n[i] < min_share * len(x) or bad[i] < min_bad]
        if not viol and not small:
            return edges
        i = viol[0] if viol else small[0]
        j = i - 1 if i == k - 1 else i + 1 if (i == 0 or viol) else (i - 1 if abs(rate[i - 1] - rate[i]) < abs(rate[i + 1] - rate[i]) else i + 1)
        edges = np.delete(edges, max(i, j))


class WoE:
    def fit(self, X, y, num, cat):
        self.spec, self.tab = {}, {}
        for c in num:
            x = X[c].astype(float)
            qs = np.unique(np.nanquantile(x[x.notna()], np.linspace(0, 1, 21)))
            if len(qs) < 2:
                continue
            self.spec[c] = ("num", merge_bins(np.r_[-np.inf, qs[1:-1], np.inf], x[x.notna()], y[x.notna()]))
            self.tab[c] = woe_table(self.bin(c, X[c]), y)
        for c in cat:
            share = X[c].fillna("Missing").value_counts(normalize=True)
            self.spec[c] = ("cat", set(share[share >= 0.02].index))
            self.tab[c] = woe_table(self.bin(c, X[c]), y)
        return self

    def bin(self, c, x):
        kind, s = self.spec[c]
        if kind == "num":
            return pd.cut(x.astype(float), s, include_lowest=True).astype(str).where(x.notna(), "Missing")
        x = x.fillna("Missing").astype(str)
        return x.where(x.isin(s), "Other")

    def transform(self, X):
        return pd.DataFrame({c: self.bin(c, X[c]).map(self.tab[c].woe).fillna(0.0).astype(float) for c in self.spec}, index=X.index)


class Scorecard:
    """woe + logistisk regression. urval: IV >= 0.02, korr < 0.7, rätt tecken och p < 0.05"""
    def fit(self, tr, num, cat):
        self.woe = WoE().fit(tr, tr.y, num, cat)
        iv = pd.Series({c: t.iv.sum() for c, t in self.woe.tab.items()}).sort_values(ascending=False)
        W = self.woe.transform(tr)
        keep = []
        for c in iv[iv >= 0.02].index:
            if all(abs(np.corrcoef(W[c], W[k])[0, 1]) < 0.7 for k in keep):
                keep.append(c)
        while True:
            res = sm.Logit(tr.y.values, sm.add_constant(W[keep])).fit(disp=0, maxiter=200)
            coef, p = res.params.drop("const"), res.pvalues.drop("const")
            if (coef > 0).any():
                keep.remove(coef.idxmax()); continue                      # woe = ln(bra/dålig) så koef måste vara negativ
            if p.max() > 0.05:
                keep.remove(p.idxmax()); continue
            break
        self.vars, self.res = keep, res
        return self

    def predict(self, df):
        return np.asarray(self.res.predict(sm.add_constant(self.woe.transform(df)[self.vars], has_constant="add")))


class GBM:
    """lightgbm, monotona villkor på förseningarna + early stopping på valideringsdelen"""
    P = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=200, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=SEED, deterministic=True,
             force_row_wise=True, metric="auc")

    def fit(self, tr, va, num, cat):
        self.cols, self.cat = num + cat, cat
        self.levels = {c: sorted(tr[c].fillna("Missing").astype(str).unique()) for c in cat}
        mono = [1 if c in MONOTONE_UP else 0 for c in self.cols]
        dtr = lgb.Dataset(self._X(tr), tr.y, categorical_feature=cat, free_raw_data=False)
        dva = lgb.Dataset(self._X(va), va.y, categorical_feature=cat, reference=dtr)
        self.model = lgb.train({**self.P, "monotone_constraints": mono}, dtr, 3000, valid_sets=[dva],
                               callbacks=[lgb.early_stopping(200, verbose=False)])
        return self

    def _X(self, df):
        X = df[self.cols].copy()
        for c in self.cat:
            X[c] = pd.Categorical(X[c].fillna("Missing").astype(str), categories=self.levels[c])
        return X

    def predict(self, df):
        return self.model.predict(self._X(df), num_iteration=self.model.best_iteration)


def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1


def hosmer_lemeshow(y, p, q=10):
    d = pd.DataFrame({"y": y, "p": p})
    d["g"] = pd.qcut(d.p.rank(method="first"), q, labels=False)
    t = d.groupby("g").agg(n=("y", "size"), obs=("y", "sum"), exp=("p", "sum"))
    hl = (((t.obs - t.exp) ** 2) / (t.exp * (1 - t.exp / t.n))).sum()
    return 1 - chi2.cdf(hl, q - 2)


def psi(a, b, q=10):
    """PSI. egentligen samma mått som IV (symmetrisk KL-divergens)"""
    e = np.unique(np.quantile(a, np.linspace(0, 1, q + 1)))
    e[0], e[-1] = -np.inf, np.inf
    pa, pb = (np.clip(np.histogram(x, e)[0] / len(x), 1e-6, None) for x in (a, b))
    return float(((pa - pb) * np.log(pa / pb)).sum())


def gini_diff_ci(y, p1, p2, groups, n_boot=1000):
    """skillnad i gini med 95% intervall. drar kunder och inte lån eftersom lån hos samma kund inte är oberoende"""
    rng = np.random.default_rng(SEED)
    codes, uniq = pd.factorize(groups)
    rows_by = pd.Series(np.arange(len(codes))).groupby(codes).apply(np.array).values
    d = []
    for _ in range(n_boot):
        idx = np.concatenate(rows_by[rng.integers(0, len(uniq), len(uniq))])
        d.append(gini(y[idx], p2[idx]) - gini(y[idx], p1[idx]))
    return np.quantile(d, [0.025, 0.975])


# =====================================================================================================
# 4. tillväxtens pris och motbudsmenyn, hela ansökningsflödet (beviljade + nekade)
# =====================================================================================================
def application_flow(df, prev, inst, w):
    """kalibrerad PD (lightgbm + platt) för beviljade och nekade ansökningar i testdelen.
    nekade har inget utfall så de viktas till sin andel av flödet och får antagen risk K_REJECT ggr PD"""
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ["train", "valid", "test"])
    num, cat = feature_groups(df, "ABC", drop={"A_cnt_payment"})            # saknas för 46% av nekade, därför bort
    gbm = GBM().fit(tr, va, num, cat)
    lg = lambda p: np.log(p / (1 - p)).reshape(-1, 1)
    platt = LogisticRegression(C=1e6).fit(lg(gbm.predict(va)), va.y.values)
    cal = lambda d: platt.predict_proba(lg(gbm.predict(d)))[:, 1]

    t_max = int(df.t0.max())
    window = (pl.col("NAME_GOODS_CATEGORY") == "Mobile") & (pl.col("DAYS_DECISION") >= MIN_DAYS_DECISION) & (pl.col("DAYS_DECISION") <= t_max)
    rows = prev.filter(window & (pl.col("NAME_CONTRACT_STATUS") == "Refused"))
    ref = features(rows, prev, inst).sort("SK_ID_PREV").to_pandas()
    for c in ["A_rate_down", "A_amt_down"]:
        ref[c] = ref[c].fillna(df[c].median())
    ref = ref[split_of(ref.SK_ID_CURR.values) == "test"].reset_index(drop=True)
    n_ref, n_app = rows.height, prev.filter(window & (pl.col("NAME_CONTRACT_STATUS") == "Approved")).height
    share = n_ref / (n_ref + n_app)
    wt = share / (1 - share) * len(te) / len(ref)
    return pd.DataFrame({"pd": np.r_[cal(te), cal(ref)], "vikt": np.r_[np.ones(len(te)), np.full(len(ref), wt)],
                         "nekad": np.r_[np.zeros(len(te), bool), np.ones(len(ref), bool)]})


def ead(H):
    return ECON["lgd"] * H * (1 - ECON["tau"] / ECON["term"])


def growth_ladder(f, w, rates=(0.70, 0.80, 0.85, 0.90, 0.95, 1.00), k=K_REJECT):
    """snitt mot marginal. andel sena bland alla godkända vs den senaste tranchen, kreditförlust och bidrag per
    ansökan, plus bidrag per extra kund (bas och fullt belastad marginal)"""
    P, W, R = f.pd.values, f.vikt.values, f.nekad.values
    q = np.where(R, np.minimum(1, k * P), P)
    order = np.argsort(P, kind="stable")
    cw = np.cumsum(W[order]) / W.sum()
    v = {s: values(w, **o) for s, o in SCENARIOS.items()}
    rows, prev = [], None
    for r in rates:
        a = np.zeros(len(P), bool)
        a[order[: np.searchsorted(cw, r, side="right")]] = True
        row = dict(godkanda=r, sena_alla=(q * W * a).sum() / (W * a).sum(),
                   kreditforlust_per_ansokan=(q * w * ead(ECON["H"]) * W * a).sum() / W.sum(),
                   bidrag_per_ansokan=(expected_value(q, v["bas"]) * W * a).sum() / W.sum())
        if prev is not None:
            t = a & ~prev
            row["sena_marginal"] = (q * W * t).sum() / (W * t).sum()
            for s in ["bas", "låg"]:
                row[f"bidrag_extra_kund_{s}"] = (expected_value(q, v[s]) * W * t).sum() / (W * t).sum()
        rows.append(row)
        prev = a
    return pd.DataFrame(rows)


def value_of_credit_decision(f, w, k=K_REJECT):
    """vad telefonkrediten är värd per ansökan jmf med att neka, om andelen s av nekade ändå tecknar abonnemang"""
    q = np.where(f.nekad.values, np.minimum(1, k * f.pd.values), f.pd.values)
    gain = lambda s: expected_value(q, values(w)) - s * expected_value(q, values(w, **OFFERS["sim"]))
    return {s: float((np.maximum(gain(s), 0) * f.vikt.values).sum() / f.vikt.sum()) for s in (0.0, 0.5, 0.9)}


def tornado(w):
    """brytpunkten p* för full telefon, ett antagande i taget"""
    tests = {"marginal": dict(margin=0.33), "marginal_hog": dict(margin=192 / 213), "pris_12000": dict(H=12000.0),
             "pris_6000": dict(H=6000.0), "lgd_80": dict(lgd=0.8), "lgd_30": dict(lgd=0.3), "cac_2250": dict(A=2250.0),
             "cac_1160": dict(A=1160.0), "churn_2": dict(churn=0.02), "churn_1": dict(churn=0.01),
             "ranta_7": dict(r_fund=0.07), "ranta_2": dict(r_fund=0.02)}
    return {"bas": values(w)["p_star"], **{n: values(w, **o)["p_star"] for n, o in tests.items()}}


def menu_vs_binary(f, w, cap, k=K_REJECT, m=1.0, a=0.7, s=0.0, shift=0.0):
    """ja/nej mot meny (full telefon, billigare telefon, abonnemang, nej) med samma tak för förväntad kreditförlust
    per kund. shift = förskjutning i log-odds för stresstestet, policyn ser den stressade PDn"""
    P = 1 / (1 + np.exp(-(np.log(f.pd.values / (1 - f.pd.values)) + shift)))
    W, R = f.vikt.values, f.nekad.values
    q = np.where(R, np.minimum(1, k * P), P)
    vf, vc, vs = values(w), values(w, **OFFERS["cheap"]), values(w, **OFFERS["sim"])
    el_sim = lambda p: p * w * OFFERS["sim"]["unpaid_months"] * OFFERS["sim"]["arpu"]
    full = P * w * ead(ECON["H"]) <= cap
    cheap = ~full & (m * P * w * ead(OFFERS["cheap"]["H"]) <= cap)
    sim = ~full & ~cheap & (el_sim(P) <= cap)
    sim_any = ~full & (el_sim(P) <= cap)
    ev_f, ev_c, ev_s = expected_value(q, vf), expected_value(np.minimum(1, m * q), vc), expected_value(q, vs)
    agg = lambda x: float((x * W).sum() / W.sum())
    as_ = max(a, s)
    return {"kunder_binar": agg(full + sim_any * s), "kunder_meny": agg(full + cheap * (a + (1 - a) * s) + sim * as_),
            "bidrag_binar": agg(full * ev_f + sim_any * s * ev_s),
            "bidrag_meny": agg(full * ev_f + cheap * (a * ev_c + (1 - a) * s * ev_s) + sim * as_ * ev_s),
            "andel_over_tak": agg(~full), "sena_bland_godkanda": float((full * q * W).sum() / (full * W).sum())}


def cap_at(f, w, rate=0.90):
    """taket i riskaptiten = förväntad kreditförlust för sista kunden in när ja/nej godkänner rate"""
    order = np.argsort(f.pd.values, kind="stable")
    cw = np.cumsum(f.vikt.values[order]) / f.vikt.sum()
    return float(f.pd.values[order][np.searchsorted(cw, rate) - 1] * w * ead(ECON["H"]))


# =====================================================================================================
# 5. konjunkturkänslighet - diskret hasardmodell på lending club mot arbetslöshet per delstat
# =====================================================================================================
STATES = dict(zip(
    ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "District of Columbia",
     "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine",
     "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
     "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
     "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia",
     "Washington", "West Virginia", "Wisconsin", "Wyoming"],
    "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI "
    "SC SD TN TX UT VT VA WA WV WI WY".split()))


def lending_club_cells():
    """lånemånader ihopslagna till celler (riskklass x delstat x månad x lånets ålder x årskull x löptid)
    tar lite tid första gången, sen cachas det"""
    cache = DATA / "lc" / "cells.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    mi = lambda s: s.dt.year * 12 + s.dt.month - 1
    d = pd.concat(pd.read_csv(DATA / "lc" / "accepted_2007_to_2018Q4.csv.gz", chunksize=200_000, dtype=str,
                              usecols=["issue_d", "loan_status", "addr_state", "grade", "term", "last_pymnt_d"]))
    d = d.dropna(subset=["issue_d", "addr_state", "grade"])
    d = d[~d.loan_status.str.startswith("Does not meet")]
    d["issue"] = mi(pd.to_datetime(d.issue_d, format="%b-%Y"))
    d["last"] = mi(pd.to_datetime(d.last_pymnt_d, format="%b-%Y", errors="coerce")).fillna(d.issue).astype(int)
    snap = int(d["last"].max())
    bad, paid = d.loan_status.isin(["Charged Off", "Default"]), d.loan_status.eq("Fully Paid")
    d["exit"] = np.clip(np.maximum(np.where(bad, d["last"] + 1, np.where(paid, d["last"], snap)), d.issue + 1), None, snap)
    d["event"], d["term_m"] = bad.astype(np.int8), d.term.str.extract(r"(\d+)").astype(int)
    parts = []
    for s in range(0, len(d), 150_000):
        c = d.iloc[s:s + 150_000]
        n = (c.exit - c.issue).values
        rep, mob = np.repeat(np.arange(len(c)), n), np.concatenate([np.arange(1, k + 1) for k in n])
        t = c.issue.values[rep] + mob
        x = pd.DataFrame({"grade": c.grade.values[rep], "state": c.addr_state.values[rep], "t": t,
                          "mob": pd.cut(mob, [0, 3, 6, 12, 24, 36, 60], labels=["01-03", "04-06", "07-12", "13-24", "25-36", "37-60"]).astype(str),
                          "vintage": c.issue.values[rep] // 12, "term": c.term_m.values[rep],
                          "ev": ((t == c.exit.values[rep]) & (c.event.values[rep] == 1)).astype(np.int32)})
        parts.append(x.groupby(["grade", "state", "t", "mob", "vintage", "term"], observed=True).ev.agg(["sum", "size"]).reset_index())
    cells = pd.concat(parts).groupby(["grade", "state", "t", "mob", "vintage", "term"], observed=True)[["sum", "size"]].sum().reset_index()
    return cells.rename(columns={"sum": "defaults", "size": "exposure"})


def hazard_fit(cells, cat, num="u", cluster="state", iters=25):
    """binomial logit med glesa dummys (IRLS), standardfel klustrade per delstat (sandwich).
    med fasta effekter per kalendermånad (t i cat) så identifieras beta bara av skillnader mellan delstater samma månad.
    utan det får beta fel tecken, lending club vidgade sin policy just när arbetslösheten föll"""
    cells = cells.groupby(list(dict.fromkeys(cat + [num, cluster])), observed=True)[["defaults", "exposure"]].sum().reset_index()
    blocks = [sparse.csr_matrix(np.ones((len(cells), 1)))]
    for c in cat:
        codes, uniq = pd.factorize(cells[c].astype(str), sort=True)
        blocks.append(sparse.csr_matrix((np.ones(len(cells)), (np.arange(len(cells)), codes)), shape=(len(cells), len(uniq)))[:, 1:])
    blocks.append(sparse.csr_matrix(cells[num].values.astype(float)[:, None]))
    X = sparse.hstack(blocks).tocsr()
    k, n = cells.defaults.values.astype(float), cells.exposure.values.astype(float)
    b = np.zeros(X.shape[1]); b[0] = np.log(k.sum() / (n.sum() - k.sum()))
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ b)))
        H = (X.T @ X.multiply((n * p * (1 - p))[:, None])).toarray()
        step = np.linalg.solve(H + 1e-9 * np.eye(len(b)), X.T @ (k - n * p))
        b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    p = 1 / (1 + np.exp(-(X @ b)))
    Hinv = np.linalg.inv((X.T @ X.multiply((n * p * (1 - p))[:, None])).toarray())
    cl, _ = pd.factorize(cells[cluster])
    G = (sparse.csr_matrix((np.ones(len(cl)), (cl, np.arange(len(cl))))) @ X.multiply((k - n * p)[:, None]).tocsc()).toarray()
    V = Hinv @ (G.T @ G * G.shape[0] / (G.shape[0] - 1)) @ Hinv
    return float(b[-1]), float(np.sqrt(V[-1, -1]))


def macro_sensitivity():
    u = pd.read_csv(DATA / "macro" / "Unemployment in America Per US State.csv")
    u = u[u["State/Area"].isin(STATES)].assign(state=lambda x: x["State/Area"].map(STATES),
                                                t=lambda x: x.Year.astype(int) * 12 + x.Month.astype(int) - 1,
                                                u=lambda x: pd.to_numeric(x["Percent (%) of Labor Force Unemployed in State/Area"].astype(str).str.strip(), errors="coerce"))
    cells = lending_club_cells().merge(u[["state", "t", "u"]].dropna(), on=["state", "t"])
    base = ["grade", "mob", "state", "vintage", "term"]
    out = {"utan_manadseffekter": hazard_fit(cells, base), "med_manadseffekter": hazard_fit(cells, base + ["t"])}
    for lab, mobs in [("tidiga_1_6", ["01-03", "04-06"]), ("senare_7_24", ["07-12", "13-24"])]:
        out[lab] = hazard_fit(cells[cells.mob.isin(mobs)], base + ["t"])
    return out


# =====================================================================================================
# 6. stresstest, omvänt stresstest och monte carlo
# =====================================================================================================
def stress(f, w, beta, beta_se, tau=1.0):
    """logit PD_stress = logit PD + tau * beta * du + ln(lambda). fast tak i kr, ja/nej mot meny"""
    cap = cap_at(f, w)
    ki = pd.read_csv(DATA / "sverige" / "prognos_ki_sep2026.csv").set_index("ar")
    du_base = ki.loc[2027, "arbetsloshet"] - 8.7
    scen = {"bas": (du_base, 1.0), "+1,5 pe": (1.5, 1.0), "+3 pe": (3.0, 1.0), "köpkraft x1,3": (0.0, 1.3),
            "köpkraft x1,5": (0.0, 1.5), "+3 pe och x1,3": (3.0, 1.3)}
    res = {n: menu_vs_binary(f, w, cap, shift=tau * beta * du + np.log(lam)) for n, (du, lam) in scen.items()}

    # omvänt stresstest: vid vilken oddsmultiplikator fördubblas kreditförlusten om alla godkänns
    q0 = lambda lam: np.where(f.nekad, np.minimum(1, K_REJECT * (lam * f.pd / (1 - f.pd + lam * f.pd))), lam * f.pd / (1 - f.pd + lam * f.pd))
    el = lambda lam: float((q0(lam) * w * ead(ECON["H"]) * f.vikt).sum() / f.vikt.sum())
    lam2 = next(l for l in np.round(np.arange(1, 15.01, 0.05), 2) if el(l) > 2 * el(1.0))

    # monte carlo: 12 mån förändringar i svensk arbetslöshet 2001-2026 runt KIs bana, osäker beta + telekomfaktor
    se = pd.read_csv(DATA / "sverige" / "aku_arbetsloshet_manad.csv", index_col=0)
    du_hist = (se.utjamnad - se.utjamnad.shift(12)).dropna().values
    rng = np.random.default_rng(SEED)
    du = rng.choice(du_hist, 20_000) + du_base
    b = np.clip(rng.normal(beta, beta_se, 20_000), 0, None)
    shift = rng.uniform(0.5, 1.0, 20_000) * b * du
    grid = np.linspace(shift.min() - 0.01, shift.max() + 0.01, 41)
    loss = [float((np.where(f.nekad, np.minimum(1, K_REJECT * p), p) * w * ead(ECON["H"]) * f.vikt).sum() / f.vikt.sum())
            for p in (1 / (1 + np.exp(-(np.log(f.pd / (1 - f.pd)) + g))) for g in grid)]
    mc = np.interp(shift, grid, loss)
    return res, {"tak_kr": cap, "odds_for_dubbel_forlust": lam2, "motsvarande_pe": np.log(lam2) / beta,
                 "mc_p99_over_medel": float(np.quantile(mc, 0.99) / mc.mean() - 1)}


# =====================================================================================================
# 7. kreditstegen - säger de första fakturorna mer än ansökan?
# =====================================================================================================
def credit_ladder(df, inst):
    """risk för väsentlig försening dom kommande 6 delbetalningarna givet hur de k första betalades.
    bara lån med minst 12 delbetalningar där alla 12 hunnit förfalla"""
    i = inst.join(pl.from_pandas(df[["SK_ID_PREV"]]), on="SK_ID_PREV", how="semi").to_pandas()
    for lag in (10, 30):
        i[f"p{lag}"] = i.AMT_PAYMENT.where(i.DAYS_ENTRY_PAYMENT <= i.DAYS_INSTALMENT + lag, 0).fillna(0)
    g = i.groupby(["SK_ID_PREV", "NUM_INSTALMENT_NUMBER", "NUM_INSTALMENT_VERSION"]).agg(due=("AMT_INSTALMENT", "max"), p10=("p10", "sum"), p30=("p30", "sum"))
    g = g.groupby(["SK_ID_PREV", "NUM_INSTALMENT_NUMBER"]).sum().reset_index()
    g["i_tid"], g["sen"] = g.p10 >= MATERIAL * g.due, g.p30 < MATERIAL * g.due
    g = g[g.SK_ID_PREV.isin(df.SK_ID_PREV[df.A_cnt_payment >= 12]) & (g.NUM_INSTALMENT_NUMBER <= 12)]
    tid, sen = (g.pivot(index="SK_ID_PREV", columns="NUM_INSTALMENT_NUMBER", values=c) for c in ("i_tid", "sen"))
    ok = tid.notna().all(axis=1) & sen.notna().all(axis=1)
    tid, sen = tid[ok].astype(bool), sen[ok].astype(bool)
    rows = []
    for k in (1, 3, 6):
        first_k_ontime = tid[list(range(1, k + 1))].all(axis=1)
        some_11_30 = ~first_k_ontime & ~sen[list(range(1, k + 1))].any(axis=1)        # sen men inte väsentligt sen än
        nxt = sen[list(range(k + 1, k + 7))].any(axis=1)
        rows.append((k, first_k_ontime.mean(), nxt[first_k_ontime].mean(), nxt[some_11_30].mean(), int(some_11_30.sum())))
    return pd.DataFrame(rows, columns=["fakturor", "andel_i_tid", "sena_om_i_tid", "sena_om_11_30_dagar", "antal_11_30"])


# =====================================================================================================
# körning
# =====================================================================================================
def main():
    OUT.mkdir(exist_ok=True)
    prev, inst = load_previous(), load_installments()
    df = build_dataset(prev, inst)
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ["train", "valid", "test"])
    w = float(tr.loc[tr.y == 1, "y90"].mean())                               # andel av de sena som når 90 dagar
    r = {"lan": len(df), "kunder": int(df.SK_ID_CURR.nunique()), "andel_daliga": df.y.mean(), "andel_90": df.y90.mean(), "w": w}

    num, cat = feature_groups(df, "ABC")
    sc, gb = Scorecard().fit(tr, num, cat), GBM().fit(tr, va, num, cat)
    p_sc, p_gb = sc.predict(te), gb.predict(te)
    ar = -te.t0 // 365
    r.update(gini_scorekort=gini(te.y, p_sc), gini_lightgbm=gini(te.y, p_gb), scorekort_variabler=sc.vars,
             gini_diff_ki=gini_diff_ci(te.y.values, p_sc, p_gb, te.SK_ID_CURR.values).tolist(),
             hl_p_scorekort=hosmer_lemeshow(te.y, p_sc), hl_p_lightgbm=hosmer_lemeshow(te.y, p_gb),
             psi_train_test=psi(sc.predict(tr), p_sc), psi_gamla_nya=psi(p_sc[(ar >= 4).values], p_sc[(ar <= 2).values]))
    for g in ["A", "AB"]:                                                    # vad varje datakälla tillför
        n_, c_ = feature_groups(df, g)
        r[f"gini_lightgbm_{g}"] = gini(te.y, GBM().fit(tr, va, n_, c_).predict(te))

    v = values(w)
    r.update(kundtid_S=v["S"], G=v["G"], V_bad=v["V_bad"], F=v["F"], brytpunkt=v["p_star"],
             brytpunkt_fullt_belastad=values(w, **SCENARIOS["låg"])["p_star"], tornado=tornado(w))

    f = application_flow(df, prev, inst, w)
    ladder = growth_ladder(f, w)
    cap = cap_at(f, w)
    r.update(tak_kr=cap, kreditbeslutets_varde=value_of_credit_decision(f, w),
             meny_vid_tak={rate: menu_vs_binary(f, w, cap_at(f, w, rate)) for rate in (0.85, 0.90, 0.95)})

    beta = macro_sensitivity()
    b, b_se = beta["med_manadseffekter"]
    r.update(beta=beta, oddskvot_3pe=np.exp(3 * b))
    scen, rev = stress(f, w, b, b_se)
    r.update(stress=scen, **rev)

    ladder_k = credit_ladder(df, inst)
    ladder.to_csv(OUT / "tillvaxtens_pris.csv", index=False)
    ladder_k.to_csv(OUT / "kreditstege.csv", index=False)
    (OUT / "resultat.json").write_text(json.dumps(r, indent=2, ensure_ascii=False, default=float))
    print(ladder.round(4).to_string(), "\n", ladder_k.round(4).to_string())
    print(json.dumps(r, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
