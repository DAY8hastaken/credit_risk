"""
Credit Risk Engine v12 (5 Cs) — Streamlit demo
==============================================
Raw applicant intake grouped by the 5 Cs -> the saved v12 model -> PD, decision,
and an explanation built around Character / Capacity / Capital / Conditions.

Run:
    streamlit run streamlit_app_v12.py

credit_risk_model_v12.pkl must be in this folder.
"""
import sys
import types
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "credit_risk_model_v12.pkl"


# ── Model class — must match the one pickled in File 2 (v12) ─────────────────
class CreditRiskModel:
    def __init__(self, pipeline, feature_order, latent_cols, leaf_features,
                 groups, minmax_ranges, home_map, global_default_rate,
                 purpose_risk_map=None, title_risk_map=None, gbn=None, version='v12'):
        self.pipeline = pipeline
        self.feature_order = feature_order
        self.latent_cols = latent_cols
        self.leaf_features = leaf_features
        self.groups = groups
        self.r = minmax_ranges
        self.home_map = home_map
        self.gmean = global_default_rate
        self.purpose_risk_map = purpose_risk_map or {}
        self.title_risk_map = title_risk_map or {}
        self.gbn = gbn
        self.version = version

    def _mm(self, col, v):
        lo, hi = self.r[col]['min'], self.r[col]['max']
        return (v - lo) / (hi - lo + 1e-9)

    def _inv(self, col, v):
        return 1.0 - self._mm(col, v)

    def engineer_from_raw(self, *, annual_inc, loan_amnt, int_rate, term_months,
                          emp_length_years, installment, dti,
                          fico_low, fico_high, credit_experience_years,
                          delinq_2yrs, pub_rec, total_acc, num_accts_ever_120_pd,
                          mths_since_last_delinq=None, home_ownership='RENT',
                          revol_bal=0.0, total_rev_hi_lim=0.0,
                          emp_title=None, purpose=None,
                          purpose_risk=None, title_risk=None):
        mi = annual_inc / 12.0
        FicoAvg = (fico_low + fico_high) / 2.0
        CreditExperienceYears = float(credit_experience_years)
        CreditDisciplineScore = (1 - (delinq_2yrs / (total_acc + 1))) - (num_accts_ever_120_pd / (total_acc + 1))
        HasCleanHistory = 1.0 if (delinq_2yrs == 0 and num_accts_ever_120_pd == 0) else 0.0
        MthsSinceDelinq = 200.0 if mths_since_last_delinq is None else min(float(mths_since_last_delinq), 200.0)
        Delinq2yrs = float(delinq_2yrs); PubRec = float(pub_rec)
        EmploymentYears = float(emp_length_years)
        MonthlyIncome = float(np.log1p(mi))
        DTI = float(dti)
        FreeCashFlow = float(mi - installment)
        InstallmentToIncome = float(min(installment / (mi + 1.0), 1.0))
        HomeOwnerScore = float(self.home_map.get(str(home_ownership).upper(), 0.3))
        RevolUtilityBurden = 0.0 if not total_rev_hi_lim else float(min(revol_bal / total_rev_hi_lim, 2.0))
        InterestRate = float(int_rate); TermMonths = float(term_months)
        LoanAmount = float(np.log1p(loan_amnt))
        _t = '' if emp_title is None else str(emp_title).strip().lower()
        has_title = 1.0 if _t not in ('', 'nan', 'none') else 0.0
        if purpose_risk is None:
            purpose_risk = self.purpose_risk_map.get(str(purpose).strip().lower(), self.gmean)
        if title_risk is None:
            title_risk = self.title_risk_map.get(_t, self.gmean)
        PurposeRisk = float(purpose_risk); TitleRisk = float(title_risk)

        f = {'FicoAvg': FicoAvg, 'CreditExperienceYears': CreditExperienceYears,
             'CreditDisciplineScore': CreditDisciplineScore, 'HasCleanHistory': HasCleanHistory,
             'Delinq2yrs': Delinq2yrs, 'PubRec': PubRec, 'MthsSinceDelinq': MthsSinceDelinq,
             'EmploymentYears': EmploymentYears, 'TitleRisk': TitleRisk, 'has_title': has_title,
             'MonthlyIncome': MonthlyIncome, 'DTI': DTI, 'FreeCashFlow': FreeCashFlow,
             'InstallmentToIncome': InstallmentToIncome, 'HomeOwnerScore': HomeOwnerScore,
             'RevolUtilityBurden': RevolUtilityBurden, 'PurposeRisk': PurposeRisk,
             'InterestRate': InterestRate, 'TermMonths': TermMonths, 'LoanAmount': LoanAmount}

        mm, inv = self._mm, self._inv
        f['CH'] = (0.22 * mm('FicoAvg', FicoAvg) + 0.15 * mm('CreditDisciplineScore', CreditDisciplineScore) +
                   0.12 * mm('CreditExperienceYears', CreditExperienceYears) + 0.10 * HasCleanHistory +
                   0.10 * mm('MthsSinceDelinq', MthsSinceDelinq) + 0.10 * mm('EmploymentYears', EmploymentYears) +
                   0.08 * inv('TitleRisk', TitleRisk) + 0.05 * has_title +
                   0.04 * inv('Delinq2yrs', Delinq2yrs) + 0.04 * inv('PubRec', PubRec))
        f['CA'] = (0.40 * inv('DTI', DTI) + 0.30 * mm('FreeCashFlow', FreeCashFlow) +
                   0.20 * inv('InstallmentToIncome', InstallmentToIncome) + 0.10 * mm('MonthlyIncome', MonthlyIncome))
        f['CP'] = (0.60 * mm('HomeOwnerScore', HomeOwnerScore) + 0.40 * inv('RevolUtilityBurden', RevolUtilityBurden))
        f['CO'] = (0.40 * inv('InterestRate', InterestRate) + 0.25 * inv('PurposeRisk', PurposeRisk) +
                   0.20 * inv('TermMonths', TermMonths) + 0.15 * inv('LoanAmount', LoanAmount))
        return f

    def predict_proba_from_raw(self, **raw):
        feats = self.engineer_from_raw(**raw)
        X = pd.DataFrame([feats])[self.feature_order]
        return float(self.pipeline.predict_proba(X)[:, 1][0])

    def predict_proba(self, X_df):
        return self.pipeline.predict_proba(X_df[self.feature_order])[:, 1]


# make the class resolvable however the pickle recorded its module
setattr(sys.modules["__main__"], "CreditRiskModel", CreditRiskModel)
_shim = types.ModuleType("app"); _shim.CreditRiskModel = CreditRiskModel
sys.modules.setdefault("app", _shim)


# ── Encodings & options ──────────────────────────────────────────────────────
PURPOSE_RISK = {
    "debt_consolidation": 0.21, "credit_card": 0.18, "home_improvement": 0.17,
    "major_purchase": 0.17, "small_business": 0.31, "car": 0.15, "medical": 0.21,
    "moving": 0.22, "vacation": 0.21, "house": 0.18, "wedding": 0.16,
    "renewable_energy": 0.23, "educational": 0.23, "other": 0.22,
}
GLOBAL_TITLE_RISK = 0.21
TITLE_RISK = {
    "Not provided": (GLOBAL_TITLE_RISK, 0.0), "Manager / Executive": (0.15, 1.0),
    "Professional (eng/IT/finance)": (0.15, 1.0), "Healthcare (nurse/doctor)": (0.16, 1.0),
    "Teacher / Education": (0.17, 1.0), "Government / Military": (0.16, 1.0),
    "Skilled trade": (0.20, 1.0), "Office / Administrative": (0.20, 1.0),
    "Sales / Retail": (0.23, 1.0), "Driver / Transport": (0.24, 1.0),
    "Service / Hospitality": (0.25, 1.0), "Self-employed": (0.27, 1.0), "Other": (0.21, 1.0),
}
HOME_OPTIONS = ["RENT", "MORTGAGE", "OWN", "OTHER"]

# the four Cs: (name, plain description, weight on RiskScore)
C_INFO = {
    "CH": ("Character", "credit history, discipline and stability", 2.2),
    "CA": ("Capacity", "ability to afford the repayments", 1.8),
    "CP": ("Capital", "assets and credit cushion", 0.9),
    "CO": ("Conditions", "the loan terms and purpose", 1.1),
}
C_WEAK = {
    "CH": "Weak Character — credit history or track record is a concern",
    "CA": "Weak Capacity — the repayments may strain their budget",
    "CP": "Weak Capital — little ownership or credit cushion",
    "CO": "Unfavourable Conditions — the loan terms add risk",
}
C_STRONG = {
    "CH": "Strong Character — solid credit history and stability",
    "CA": "Strong Capacity — comfortably affords the repayments",
    "CP": "Strong Capital — good ownership / credit cushion",
    "CO": "Favourable Conditions — reasonable loan terms",
}
C_TIP = {
    "CH": "a longer track record and fewer delinquencies would strengthen this",
    "CA": "a smaller loan or higher income would ease the repayment burden",
    "CP": "more savings/ownership or a lower card balance would help",
    "CO": "a lower rate, shorter term, or smaller amount would improve the terms",
}


def band_decision(pd_val):
    if pd_val < 0.15:
        return "LOW", "APPROVE", "#1f9d55"
    if pd_val < 0.35:
        return "MEDIUM", "REVIEW", "#d97706"
    return "HIGH", "DECLINE", "#dc2626"


def monthly_payment(principal, annual_rate_pct, n_months):
    r = annual_rate_pct / 100 / 12
    if r <= 0:
        return principal / n_months
    return principal * r / (1 - (1 + r) ** (-n_months))


def _blend(target, a):
    tr, tg, tb = int(target[1:3], 16), int(target[3:5], 16), int(target[5:7], 16)
    r = int(255 + (tr - 255) * a); g = int(255 + (tg - 255) * a); b = int(255 + (tb - 255) * a)
    return f"#{r:02x}{g:02x}{b:02x}"


def c_fill(score):
    if score >= 0.5:
        return _blend("#1f9d55", min((score - 0.5) / 0.5, 1) * 0.85 + 0.15)
    return _blend("#dc2626", min((0.5 - score) / 0.5, 1) * 0.85 + 0.15)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def build_raw(b):
    """Form inputs -> the model's raw kwargs (computes annual income + installment)."""
    annual = b["monthly_inc"] * 12
    inst = monthly_payment(b["loan_amnt"], b["int_rate"], b["term_months"])
    return dict(
        annual_inc=annual, loan_amnt=b["loan_amnt"], int_rate=b["int_rate"],
        term_months=b["term_months"], emp_length_years=b["emp_length_years"],
        installment=inst, dti=b["dti"], fico_low=b["fico"] - 2, fico_high=b["fico"] + 2,
        credit_experience_years=b["credit_experience_years"], delinq_2yrs=b["delinq_2yrs"],
        pub_rec=b["pub_rec"], total_acc=b["total_acc"],
        num_accts_ever_120_pd=b["num_accts_ever_120_pd"],
        mths_since_last_delinq=b["mths_since_last_delinq"], home_ownership=b["home_ownership"],
        revol_bal=b["revol_bal"], total_rev_hi_lim=b["total_rev_hi_lim"],
        emp_title=b["emp_title_str"], purpose_risk=b["purpose_risk"], title_risk=b["title_risk"])


def c_breakdown(feats):
    """Score, level and risk-pressure for each of the four Cs."""
    rows = []
    for c, (name, desc, w) in C_INFO.items():
        s = float(feats[c])
        pressure = w * (1 - s)               # how much this C pushes risk up
        rows.append({"c": c, "name": name, "desc": desc, "score": s,
                     "pressure": pressure, "weight": w})
    return rows


def what_if(model, b, base_pd):
    out = []

    def run(label, **ch):
        bb = dict(b); bb.update(ch)
        p = model.predict_proba_from_raw(**build_raw(bb))
        band, _, _ = band_decision(p)
        out.append({"label": label, "pd": p, "delta": p - base_pd, "band": band})

    run("If income were 20% higher", monthly_inc=b["monthly_inc"] * 1.2)
    run("If the loan were 20% smaller", loan_amnt=b["loan_amnt"] * 0.8)
    if b["int_rate"] > 8:
        run("If the rate were 3 points lower", int_rate=b["int_rate"] - 3)
    if b["fico"] < 820:
        run("If their credit score were 30 higher", fico=min(b["fico"] + 30, 850))
    if b["dti"] > 5:
        run("If their DTI were 5 points lower", dti=b["dti"] - 5)
    return out


# ── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Credit Risk Engine v12", page_icon="🏦", layout="centered")
st.title("🏦 Inclusive Credit Risk Engine")
st.caption("5 Cs of Credit · Character · Capacity · Capital · Conditions")

try:
    model = load_model()
except FileNotFoundError:
    st.error(f"Model file `{MODEL_PATH}` not found — put it in this folder.")
    st.stop()

with st.form("application"):
    name = st.text_input("Applicant name", "Maria Kovacs")
    tabs = st.tabs(["👤 Character", "💵 Capacity", "🏠 Capital", "📄 Conditions"])

    with tabs[0]:
        fico = st.slider("Credit score (FICO)", 300, 850, 690)
        credit_experience_years = st.slider("Years of credit history", 0.0, 40.0, 10.0, 0.5)
        emp_title = st.selectbox("Job title", list(TITLE_RISK.keys()))
        emp_length_years = st.slider("Years employed", 0.0, 40.0, 5.0, 0.5)
        c1, c2 = st.columns(2)
        delinq_2yrs = c1.number_input("Delinquencies (2 yrs)", 0, 30, 0)
        pub_rec = c2.number_input("Public records", 0, 30, 0)
        total_acc = c1.number_input("Total credit accounts", 1, 100, 15)
        num_accts_ever_120_pd = c2.number_input("Accounts ever 120+ past due", 0, 30, 0)
        no_delinq = st.checkbox("No past delinquency", value=True)
        mths_since_last_delinq = None if no_delinq else st.slider(
            "Months since last delinquency", 0, 120, 24)

    with tabs[1]:
        monthly_inc = st.number_input("Monthly income (USD)", 100, 100_000, 3000, step=100)
        dti = st.slider("Debt-to-income (DTI %)", 0.0, 50.0, 18.0, 0.5)

    with tabs[2]:
        home_ownership = st.selectbox("Home ownership", HOME_OPTIONS)
        revol_bal = st.number_input("Credit-card balance (USD)", 0, 500_000, 4000, step=500)
        total_rev_hi_lim = st.number_input("Total credit limit (USD)", 0, 1_000_000, 15000, step=500)

    with tabs[3]:
        loan_amnt = st.number_input("Loan amount (USD)", 500, 100_000, 9000, step=500)
        int_rate = st.slider("Interest rate (%)", 5.0, 30.0, 12.5, 0.1)
        term_months = st.selectbox("Term (months)", [36, 60])
        purpose = st.selectbox("Loan purpose", list(PURPOSE_RISK.keys()),
                               format_func=lambda s: s.replace("_", " ").title())

    submitted = st.form_submit_button("Score application", use_container_width=True)

if submitted:
    title_risk, has_title_flag = TITLE_RISK[emp_title]
    b = dict(monthly_inc=monthly_inc, loan_amnt=loan_amnt, int_rate=int_rate,
             term_months=term_months, emp_length_years=emp_length_years, dti=dti,
             fico=fico, credit_experience_years=credit_experience_years,
             delinq_2yrs=delinq_2yrs, pub_rec=pub_rec, total_acc=total_acc,
             num_accts_ever_120_pd=num_accts_ever_120_pd,
             mths_since_last_delinq=mths_since_last_delinq, home_ownership=home_ownership,
             revol_bal=revol_bal, total_rev_hi_lim=total_rev_hi_lim,
             emp_title_str=("" if emp_title == "Not provided" else emp_title),
             purpose_risk=PURPOSE_RISK[purpose], title_risk=title_risk)

    raw = build_raw(b)
    pd_val = model.predict_proba_from_raw(**raw)
    feats = model.engineer_from_raw(**raw)
    band, decision, color = band_decision(pd_val)
    cs = c_breakdown(feats)
    installment = monthly_payment(loan_amnt, int_rate, term_months)

    # ── headline ─────────────────────────────────────────────────────────────
    st.markdown("---")
    in10 = round((1 - pd_val) * 10)
    st.markdown(
        f"<div style='border-left:4px solid {color};padding:.6rem 1rem;"
        f"background:rgba(0,0,0,.03);border-radius:4px'>"
        f"<b>{name or 'The applicant'}</b> is assessed as <b style='color:{color}'>{band} risk</b> "
        f"— about <b>{pd_val:.0%}</b> chance of default. Roughly <b>{in10} in 10</b> similar "
        f"borrowers repay in full.</div>", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Probability of default", f"{pd_val:.1%}")
    m2.markdown(f"**Risk band**<br><span style='color:{color};font-size:1.5rem;"
                f"font-weight:700'>{band}</span>", unsafe_allow_html=True)
    m3.markdown(f"**Decision**<br><span style='color:{color};font-size:1.5rem;"
                f"font-weight:700'>{decision}</span>", unsafe_allow_html=True)
    st.progress(min(pd_val, 1.0))

    # ── affordability ─────────────────────────────────────────────────────────
    st.subheader("💵 Can they afford it?")
    pti = installment / (monthly_inc + 1e-9)
    afford = ("comfortable" if pti < 0.20 else "manageable" if pti < 0.35
              else "tight" if pti < 0.50 else "a stretch")
    a1, a2, a3 = st.columns(3)
    a1.metric("Est. monthly payment", f"${installment:,.0f}")
    a2.metric("Share of income", f"{pti:.0%}")
    a3.metric("Total interest", f"${installment*term_months - loan_amnt:,.0f}")
    st.markdown(f"The **${installment:,.0f}/month** repayment takes **{pti:.0%}** of income "
                f"— that looks **{afford}**.")

    # ── the 5 Cs explanation ──────────────────────────────────────────────────
    st.subheader("What's behind this decision — the 5 Cs")
    concerns = sorted([c for c in cs if c["score"] < 0.5], key=lambda x: -x["pressure"])
    favours = sorted([c for c in cs if c["score"] >= 0.55], key=lambda x: -x["score"])
    st.caption(f"Found **{len(concerns)}** of the 4 Cs raising risk and "
               f"**{len(favours)}** working in the borrower's favour.")

    cols = st.columns(4)
    for col, c in zip(cols, cs):
        col.markdown(
            f"<div style='text-align:center'>"
            f"<div style='font-size:.8rem;opacity:.7'>{c['name']}</div>"
            f"<div style='font-size:1.6rem;font-weight:700;color:{c_fill(c['score'])}'>"
            f"{c['score']*100:.0f}</div>"
            f"<div style='font-size:.72rem;opacity:.55'>/ 100</div></div>",
            unsafe_allow_html=True)

    e1, e2 = st.columns(2)
    with e1:
        st.markdown(f"##### 🔴 Concerns ({len(concerns)})")
        for c in concerns:
            st.markdown(f"<div style='color:#dc2626;margin:.3rem 0'>● {C_WEAK[c['c']]}</div>",
                        unsafe_allow_html=True)
        if not concerns:
            st.caption("No weak Cs.")
    with e2:
        st.markdown(f"##### 🟢 In their favour ({len(favours)})")
        for c in favours:
            st.markdown(f"<div style='color:#1f9d55;margin:.3rem 0'>● {C_STRONG[c['c']]}</div>",
                        unsafe_allow_html=True)
        if not favours:
            st.caption("Nothing strongly in their favour.")

    if concerns and band != "LOW":
        worst = concerns[0]
        st.success(f"💡 To improve the outcome, focus on **{worst['name']}**: {C_TIP[worst['c']]}.")

    # ── the C network ─────────────────────────────────────────────────────────
    st.subheader("🔗 How the Cs feed the score")
    dot = ['digraph G {', 'rankdir=LR;', 'bgcolor="transparent";',
           'node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11 color="#cccccc"];']
    for c in cs:
        dot.append(f'{c["c"]} [label="{c["name"]}\\n{c["score"]*100:.0f}/100" '
                   f'fillcolor="{c_fill(c["score"])}"];')
    dot.append(f'RISK [label="RISK SCORE" shape=ellipse fillcolor="{color}" '
               f'fontcolor="white" penwidth=0];')
    for c in cs:
        dot.append(f'{c["c"]} -> RISK [color="#9aa0a6"];')
    dot.append('}')
    st.graphviz_chart("\n".join(dot), use_container_width=True)
    st.caption("Each C is scored 0–100 from the borrower's data, then the four "
               "combine into the risk score. Green = helping, red = raising risk.")

    # ── what-if ────────────────────────────────────────────────────────────────
    st.subheader("🔍 What-if questions")
    for w in what_if(model, b, pd_val):
        better = w["delta"] < 0
        arrow = "↓" if better else "↑"
        col = "#1f9d55" if better else "#dc2626"
        mark = "✅" if w["delta"] <= -0.02 else "•"
        st.markdown(f"<div style='margin:.25rem 0'>{mark} {w['label']} → <b>{w['pd']:.0%}</b> "
                    f"<span style='color:{col}'>({arrow}{abs(w['delta']):.0%})</span> "
                    f"<span style='opacity:.5;font-size:.85rem'>· {w['band']}</span></div>",
                    unsafe_allow_html=True)

    with st.expander("📊 Technical details (for analysts)"):
        st.write({k: round(v, 3) for k, v in feats.items()})
