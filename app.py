"""
Riwaaya Analytics Dashboard, single-file build.

Deliberately written as ONE file with no custom subfolders or local package
imports, so there is nothing for a hosting platform's file upload step to
drop or misplace. Only standard library + installed packages are imported.
The only other file this app needs sitting next to it is
riwaaya_survey_cleaned.csv.
"""
import os
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from scipy import stats

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              RandomForestRegressor, GradientBoostingRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve, confusion_matrix,
                              r2_score, mean_squared_error, mean_absolute_error)

# ===================================================================
# STYLING: one accent color, two neutrals, used everywhere below
# ===================================================================
ACCENT = "#A6432F"
ACCENT_LIGHT = "#D98C7A"
DARK = "#2E2E2E"
GRAY = "#B8B8B8"
GRAY_LIGHT = "#EDEAE6"
BG = "#FFFFFF"
PALETTE = [ACCENT, DARK, GRAY, ACCENT_LIGHT, "#6B6B6B", "#E0C9A6"]

_template = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Helvetica, Arial, sans-serif", color=DARK, size=13),
        paper_bgcolor=BG, plot_bgcolor=BG, colorway=PALETTE,
        title=dict(font=dict(size=18, color=DARK), x=0.02, xanchor="left"),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=GRAY, ticks="outside"),
        yaxis=dict(showgrid=True, gridcolor=GRAY_LIGHT, zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=60, b=40),
    )
)
pio.templates["riwaaya"] = _template
pio.templates.default = "riwaaya"


def inject_css():
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {BG}; }}
        h1, h2, h3 {{ color: {DARK}; font-family: Helvetica, Arial, sans-serif; }}
        [data-testid="stMetricValue"] {{ color: {ACCENT}; }}
        [data-testid="stSidebar"] {{ background-color: {GRAY_LIGHT}; }}
        .insight-box {{
            background-color: {GRAY_LIGHT};
            border-left: 4px solid {ACCENT};
            padding: 14px 18px;
            border-radius: 4px;
            margin: 10px 0px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def pretty(name):
    return (name.replace("interest_", "").replace("channel_", "").replace("trust_", "")
                .replace("_", " "))


# ===================================================================
# DATA LOADING AND FEATURE ENGINEERING
# ===================================================================
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "riwaaya_survey_cleaned.csv")

REGION_MAP = {
    "UAE": "GCC", "India": "South Asia",
    "UK": "Europe", "Germany": "Europe", "Switzerland": "Europe", "Italy": "Europe",
    "France": "Europe", "Netherlands": "Europe", "Sweden": "Europe",
    "USA": "North America", "Canada": "North America",
    "Singapore": "APAC", "Australia": "APAC", "Hong Kong": "APAC",
}
INCOME_ORDER = ["Under $25K", "$25-50K", "$50-100K", "$100-200K", "$200K+", "Prefer not to say"]
AGE_ORDER = ["18-24", "25-34", "35-44", "45-54", "55+"]


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["region"] = df["country"].map(REGION_MAP)
    df["target_membership_interest"] = (df["membership_interest"] == "Yes, definitely").astype(int)
    df["wants_launch_notification_int"] = df["wants_launch_notification"].astype(int)
    return df


def split_multiselect(value):
    if pd.isna(value):
        return []
    return [v.strip() for v in str(value).split(";") if v.strip()]


def safe_name(text):
    text = text.replace("'", "")
    return re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_")


def build_feature_matrix(df: pd.DataFrame, drop_cols=None):
    drop_cols = set(drop_cols or [])
    numeric_features = [
        "age_range_ordinal", "income_bracket_ordinal", "items_purchased_12m_ordinal",
        "items_purchased_12m_raw", "appeal_score", "trust_count", "nps_score",
        "max_price_small_piece_usd", "max_price_large_piece_usd", "authenticity_premium_pct",
        "future_purchases_per_year", "annual_spending_potential_usd",
        "high_intent_customer", "premium_buyer", "high_trust_customer",
        "wants_launch_notification_int",
    ]
    numeric_features = [c for c in numeric_features if c not in drop_cols]

    raw_or_numeric_trust_cols = {"buying_channels", "product_interests", "trust_factors", "trust_count"}
    existing_onehot = [
        c for c in df.columns
        if c.startswith(("channel_", "interest_", "trust_")) and c not in raw_or_numeric_trust_cols
    ]
    nominal_to_encode = ["gender", "region", "respondent_role", "heritage_connection",
                         "top_frustration", "main_purchase_blocker", "top_wishlist_feature"]
    df_encoded = pd.get_dummies(df[nominal_to_encode], prefix=nominal_to_encode, drop_first=False)
    X = pd.concat([df[numeric_features], df[existing_onehot], df_encoded], axis=1)
    return X


# ===================================================================
# MODEL TRAINING (cached so grid search only runs once per session)
# ===================================================================
CLASSIFIER_GRIDS = {
    "KNN": (KNeighborsClassifier(), {
        "n_neighbors": [5, 7, 9, 11, 15, 21], "weights": ["uniform", "distance"], "p": [1, 2],
    }),
    "Decision Tree": (DecisionTreeClassifier(random_state=42), {
        "max_depth": [3, 5, 7, 10, None], "min_samples_split": [2, 5, 10], "criterion": ["gini", "entropy"],
    }),
    "Random Forest": (RandomForestClassifier(random_state=42), {
        "n_estimators": [100, 200], "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5], "max_features": ["sqrt", "log2"],
    }),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=42), {
        "n_estimators": [100, 200], "learning_rate": [0.01, 0.05, 0.1, 0.2], "max_depth": [2, 3, 4],
    }),
}


@st.cache_resource(show_spinner="Training and tuning classification models (runs once per session)...")
def train_classification_models(df: pd.DataFrame):
    X = build_feature_matrix(df)
    y = df["target_membership_interest"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    for name, (estimator, grid) in CLASSIFIER_GRIDS.items():
        gs = GridSearchCV(estimator, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        gs.fit(X_train_scaled, y_train)
        best_model = gs.best_estimator_

        train_pred = best_model.predict(X_train_scaled)
        test_pred = best_model.predict(X_test_scaled)
        test_proba = best_model.predict_proba(X_test_scaled)[:, 1]

        cm = confusion_matrix(y_test, test_pred)
        tn, fp, fn, tp = cm.ravel()
        total_errors = fp + fn
        fpr, tpr, _ = roc_curve(y_test, test_proba)

        results[name] = {
            "model": best_model, "best_params": gs.best_params_, "cv_best_score": gs.best_score_,
            "train_accuracy": accuracy_score(y_train, train_pred),
            "test_accuracy": accuracy_score(y_test, test_pred),
            "precision": precision_score(y_test, test_pred),
            "recall": recall_score(y_test, test_pred),
            "f1": f1_score(y_test, test_pred),
            "roc_auc": roc_auc_score(y_test, test_proba),
            "confusion_matrix": cm, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "fp_pct_of_errors": (fp / total_errors * 100) if total_errors else 0,
            "fn_pct_of_errors": (fn / total_errors * 100) if total_errors else 0,
            "fp_pct_of_test": fp / len(y_test) * 100, "fn_pct_of_test": fn / len(y_test) * 100,
            "fpr": fpr, "tpr": tpr, "test_proba": test_proba,
        }

    comparison = pd.DataFrame({k: {
        "Train Accuracy": v["train_accuracy"], "Test Accuracy": v["test_accuracy"],
        "Precision": v["precision"], "Recall": v["recall"], "F1 Score": v["f1"], "ROC-AUC": v["roc_auc"],
    } for k, v in results.items()}).T.sort_values("ROC-AUC", ascending=False)
    best_name = comparison.index[0]

    importance_source = best_name if hasattr(results[best_name]["model"], "feature_importances_") else "Random Forest"
    importances = pd.Series(
        results[importance_source]["model"].feature_importances_, index=feature_names
    ).sort_values(ascending=False)

    return {
        "results": results, "comparison": comparison, "best_model_name": best_name,
        "feature_importances": importances, "X_test": X_test, "y_test": y_test,
        "feature_names": feature_names, "scaler": scaler,
    }


@st.cache_resource(show_spinner="Training regression models...")
def train_regression_models(df: pd.DataFrame, target_col: str, extra_drop_cols=None):
    extra_drop_cols = extra_drop_cols or []
    drop_cols = set(extra_drop_cols) | {target_col}
    X = build_feature_matrix(df, drop_cols=drop_cols)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        results[name] = {
            "model": model, "r2": r2_score(y_test, pred),
            "rmse": mean_squared_error(y_test, pred) ** 0.5, "mae": mean_absolute_error(y_test, pred),
            "y_test": y_test, "pred": pred,
        }
    comparison = pd.DataFrame({k: {"R2": v["r2"], "RMSE": v["rmse"], "MAE": v["mae"]}
                               for k, v in results.items()}).T.sort_values("R2", ascending=False)
    return {"results": results, "comparison": comparison, "best_model_name": comparison.index[0]}


def chi_square_test(df, col, target="target_membership_interest"):
    ct = pd.crosstab(df[col], df[target])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    return chi2, p


@st.cache_resource
def quick_importance(df):
    X = build_feature_matrix(df)
    y = df["target_membership_interest"]
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)


# ===================================================================
# PAGE 0: OVERVIEW
# ===================================================================
def page_overview(df):
    st.title("Riwaaya: Consumer Analytics Dashboard")
    st.markdown(
        "##### Authenticated Indian handicrafts marketplace: data analytics, "
        "diagnostic insight, and machine learning for go-to-market decisions"
    )
    st.divider()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Survey Respondents", f"{len(df):,}")
    col2.metric("Strong Membership Intent", f"{df['target_membership_interest'].mean()*100:.1f}%")
    col3.metric("High Intent Customers", f"{df['high_intent_customer'].mean()*100:.1f}%")
    col4.metric("Premium Buyers", f"{df['premium_buyer'].mean()*100:.1f}%")
    col5.metric("Avg Annual Spend Potential", f"${df['annual_spending_potential_usd'].mean():,.0f}")

    st.divider()
    st.markdown("### How to use this dashboard")
    left, right = st.columns([2, 1])
    with left:
        st.markdown(
            """
            Use the sidebar to move through the analysis. Each page builds on the last:

            1. **Descriptive Analytics**: who the respondents are, how they buy, what they'll pay,
               what they want, and what they trust.
            2. **Diagnostic Analytics**: what actually drives membership interest and spending,
               broken down by age, income, customer type, heritage connection, and geography,
               with significance testing so the patterns aren't just noise.
            3. **Predictive Modeling**: four classification algorithms (KNN, Decision Tree,
               Random Forest, Gradient Boosting), tuned and cross-validated, predicting who
               will commit to a Riwaaya membership.
            4. **Predictive Analytics**: regression models forecasting annual spending
               potential and future purchase frequency, plus a conversion-likelihood scoring tool.
            5. **Prescriptive Strategy**: what Riwaaya should actually do about all of this:
               segmentation, pricing, marketing, membership design, and expansion priorities,
               closing with a sustainability verdict.
            """
        )
    with right:
        st.markdown(
            f"""
            <div class="insight-box">
            <b>About this data</b><br><br>
            Synthetic consumer survey built to validate the Riwaaya concept: a premium,
            authenticated handicrafts marketplace. 1,000 respondents across five hypothesized
            segments, covering demographics, buying behavior, willingness to pay, trust
            requirements, and stated purchase intent.
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.divider()
    st.caption("Riwaaya Analytics Dashboard - Individual & Group PBL, Data Analytics for Insights and Decision Making")


# ===================================================================
# PAGE 1: DESCRIPTIVE ANALYTICS
# ===================================================================
def page_descriptive(df):
    st.title("Descriptive Analytics")
    st.markdown("Who the respondents are, how they buy, what they'll pay, what they want, and what they trust.")
    st.divider()

    tabs = st.tabs(["Demographics", "Buying Behavior", "Willingness to Pay",
                    "Product Preferences", "Trust Factors", "Purchase Intent"])

    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            age_counts = df["age_range"].value_counts().reindex(AGE_ORDER)
            fig = px.bar(x=age_counts.index, y=age_counts.values, labels={"x": "Age range", "y": "Respondents"},
                        title="Respondents by Age Range", color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, width='stretch')
        with c2:
            gender_counts = df["gender"].value_counts()
            fig = px.bar(x=gender_counts.values, y=gender_counts.index, orientation="h",
                        labels={"x": "Respondents", "y": ""}, title="Respondents by Gender",
                        color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, width='stretch')

        c3, c4 = st.columns(2)
        with c3:
            region_counts = df["region"].value_counts()
            fig = px.bar(x=region_counts.index, y=region_counts.values,
                        labels={"x": "Region", "y": "Respondents"}, title="Respondents by Region",
                        color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, width='stretch')
        with c4:
            income_counts = df["income_bracket"].value_counts().reindex(INCOME_ORDER)
            fig = px.bar(x=income_counts.index, y=income_counts.values,
                        labels={"x": "Income bracket", "y": "Respondents"}, title="Respondents by Income Bracket",
                        color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, width='stretch')

        c5, c6 = st.columns(2)
        with c5:
            role_counts = df["respondent_role"].value_counts()
            fig = px.bar(x=role_counts.values, y=role_counts.index, orientation="h",
                        labels={"x": "Respondents", "y": ""}, title="Respondents by Customer Type",
                        color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, width='stretch')
        with c6:
            her_counts = df["heritage_connection"].value_counts()
            fig = px.bar(x=her_counts.values, y=her_counts.index, orientation="h",
                        labels={"x": "Respondents", "y": ""}, title="Respondents by Heritage Connection",
                        color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, width='stretch')

        st.markdown(
            """<div class="insight-box">North America and Europe together make up the majority of respondents,
            with the GCC (mostly UAE) the third largest pool. Income is fairly spread across brackets, which
            matters later: income turns out to be one of the strongest predictors of both spend and membership
            interest.</div>""", unsafe_allow_html=True)

    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            items_counts = df["items_purchased_12m"].value_counts().reindex(["0", "1-2", "3-5", "6+"])
            fig = px.bar(x=items_counts.index, y=items_counts.values,
                        labels={"x": "Items purchased, last 12 months", "y": "Respondents"},
                        title="Items Purchased in the Last 12 Months", color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, width='stretch')
        with c2:
            future_counts = df["future_purchases_per_year"].round(0).value_counts().sort_index()
            fig = px.bar(x=future_counts.index, y=future_counts.values,
                        labels={"x": "Expected future purchases / year", "y": "Respondents"},
                        title="Expected Future Purchase Frequency", color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, width='stretch')

        channel_cols = [c for c in df.columns if c.startswith("channel_") and c != "channel_no_response"]
        channel_totals = df[channel_cols].sum().sort_values(ascending=False)
        channel_totals.index = [pretty(c.replace("channel_", "")) for c in channel_totals.index]
        fig = px.bar(x=channel_totals.values, y=channel_totals.index, orientation="h",
                    labels={"x": "Respondents selecting this channel", "y": ""},
                    title="Preferred Buying Channels (multi-select)", color_discrete_sequence=[ACCENT])
        st.plotly_chart(fig, width='stretch')

        blocker_counts = df["main_purchase_blocker"].value_counts()
        fig = px.bar(x=blocker_counts.values, y=blocker_counts.index, orientation="h",
                    labels={"x": "Respondents", "y": ""}, title="Main Purchase Blocker",
                    color_discrete_sequence=[ACCENT])
        st.plotly_chart(fig, width='stretch')

        st.markdown(
            """<div class="insight-box">Online marketplaces dominate as a buying channel, but a meaningful share
            still buy direct from artisans or in galleries, channels Riwaaya can't fully replace, only complement.
            Trust and authenticity doubts are the single biggest blocker to purchase, ahead of price.</div>""",
            unsafe_allow_html=True)

    with tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x="max_price_small_piece_usd", nbins=30,
                               labels={"max_price_small_piece_usd": "Max price, small piece (USD)"},
                               title="Willingness to Pay: Small Piece", color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, width='stretch')
        with c2:
            fig = px.histogram(df, x="max_price_large_piece_usd", nbins=30,
                               labels={"max_price_large_piece_usd": "Max price, large piece (USD)"},
                               title="Willingness to Pay: Large Piece", color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, width='stretch')

        c3, c4 = st.columns(2)
        with c3:
            fig = px.histogram(df, x="authenticity_premium_pct", nbins=25,
                               labels={"authenticity_premium_pct": "Premium willing to pay for authenticity (%)"},
                               title="Authenticity Premium Respondents Will Pay", color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, width='stretch')
        with c4:
            price_by_income = df.groupby("income_bracket")["max_price_small_piece_usd"].median().reindex(INCOME_ORDER)
            fig = px.bar(x=price_by_income.index, y=price_by_income.values,
                        labels={"x": "Income bracket", "y": "Median max price, small piece (USD)"},
                        title="Price Willingness by Income", color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, width='stretch')

        ratio = (df[df['income_bracket'] == '$200K+']['max_price_small_piece_usd'].median() /
                df[df['income_bracket'] == 'Under $25K']['max_price_small_piece_usd'].median())
        st.markdown(
            f"""<div class="insight-box">Median willingness to pay for a small piece sits around
            ${df['max_price_small_piece_usd'].median():.0f}, and rises sharply with income, the $200K+ bracket
            pays roughly {ratio:.1f}x what the under-$25K bracket pays. The average authenticity premium
            respondents will pay is {df['authenticity_premium_pct'].mean():.0f}%, a real, monetizable signal
            that authentication is a feature people will pay for, not just a nice-to-have.</div>""",
            unsafe_allow_html=True)

    with tabs[3]:
        interest_cols = [c for c in df.columns if c.startswith("interest_") and c != "interest_no_response"]
        interest_totals = df[interest_cols].sum().sort_values(ascending=False)
        interest_totals.index = [pretty(c.replace("interest_", "")) for c in interest_totals.index]
        fig = px.bar(x=interest_totals.values, y=interest_totals.index, orientation="h",
                    labels={"x": "Respondents interested", "y": ""},
                    title="Product Category Interest (multi-select)", color_discrete_sequence=[ACCENT])
        st.plotly_chart(fig, width='stretch')

        wishlist_counts = df["top_wishlist_feature"].value_counts()
        fig = px.bar(x=wishlist_counts.values, y=wishlist_counts.index, orientation="h",
                    labels={"x": "Respondents", "y": ""}, title="Top Wishlist Feature",
                    color_discrete_sequence=[ACCENT])
        st.plotly_chart(fig, width='stretch')

        st.markdown(
            f"""<div class="insight-box">{interest_totals.index[0]} and {interest_totals.index[1]} lead product
            interest, useful for prioritizing the initial catalog rather than launching with everything at
            once.</div>""", unsafe_allow_html=True)

    with tabs[4]:
        trust_cols = [c for c in df.columns
                     if c.startswith("trust_") and c not in ("trust_count", "trust_no_response", "trust_factors")]
        trust_totals = df[trust_cols].sum().sort_values(ascending=False)
        trust_totals.index = [pretty(c.replace("trust_", "")) for c in trust_totals.index]
        fig = px.bar(x=trust_totals.values, y=trust_totals.index, orientation="h",
                    labels={"x": "Respondents selecting this factor", "y": ""},
                    title="What Builds Trust to Buy (multi-select)", color_discrete_sequence=[ACCENT])
        st.plotly_chart(fig, width='stretch')

        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x="trust_count", nbins=6,
                               labels={"trust_count": "Number of trust factors selected"},
                               title="How Many Trust Signals Respondents Need", color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, width='stretch')
        with c2:
            fig = px.histogram(df, x="nps_score", nbins=11, labels={"nps_score": "NPS score (0-10)"},
                               title="Net Promoter Score Distribution", color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, width='stretch')

        promoters = (df["nps_score"] >= 9).mean() * 100
        detractors = (df["nps_score"] <= 6).mean() * 100
        st.markdown(
            f"""<div class="insight-box">{trust_totals.index[0]} is the single most-cited trust factor, this is
            the feature Riwaaya's authentication layer needs to lead with in messaging. NPS skews positive:
            {promoters:.0f}% are promoters (score 9-10) versus {detractors:.0f}% detractors (score 0-6),
            a healthy starting signal for a pre-launch concept test.</div>""", unsafe_allow_html=True)

    with tabs[5]:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x="appeal_score", nbins=5,
                               labels={"appeal_score": "Concept appeal score (1-5)"},
                               title="Concept Appeal Score", color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, width='stretch')
        with c2:
            mem_counts = df["membership_interest"].value_counts().reindex(["No", "Possibly", "Yes, definitely"])
            fig = px.bar(x=mem_counts.index, y=mem_counts.values,
                        labels={"x": "Membership interest", "y": "Respondents"},
                        title="Membership Interest", color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, width='stretch')

        c3, c4 = st.columns(2)
        with c3:
            flag_summary = pd.DataFrame({
                "Flag": ["High Intent Customer", "Premium Buyer", "High Trust Customer", "Strong Membership Intent"],
                "Share (%)": [
                    df["high_intent_customer"].mean() * 100, df["premium_buyer"].mean() * 100,
                    df["high_trust_customer"].mean() * 100, df["target_membership_interest"].mean() * 100,
                ]
            })
            fig = px.bar(flag_summary, x="Flag", y="Share (%)", title="Customer Quality Flags, Overall",
                        color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, width='stretch')
        with c4:
            fig = px.histogram(df, x="annual_spending_potential_usd", nbins=30,
                               labels={"annual_spending_potential_usd": "Annual spending potential (USD)"},
                               title="Annual Spending Potential Distribution", color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, width='stretch')

        st.markdown(
            f"""<div class="insight-box">{df['target_membership_interest'].mean()*100:.0f}% of respondents say
            they'd "definitely" join a membership program, a strong, almost evenly-split signal that makes this
            a clean target for the predictive models on the next pages. Spending potential is right-skewed: a
            small group of high-value respondents pulls the average well above the median
            (${df['annual_spending_potential_usd'].median():.0f}), exactly the kind of concentration worth
            segmenting around rather than averaging away.</div>""", unsafe_allow_html=True)


# ===================================================================
# PAGE 2: DIAGNOSTIC ANALYTICS
# ===================================================================
def page_diagnostic(df):
    st.title("Diagnostic Analytics")
    st.markdown("What actually drives membership interest and spending, with significance testing so the "
                "patterns aren't just noise.")
    st.divider()

    tabs = st.tabs(["Key Drivers", "Age & Income", "Customer Type & Heritage", "Geography"])

    with tabs[0]:
        st.markdown("#### What predicts strong membership interest?")
        importances = quick_importance(df)
        top15 = importances.head(15).sort_values()
        top15.index = [pretty(i) for i in top15.index]
        fig = px.bar(x=top15.values, y=top15.index, orientation="h",
                    labels={"x": "Relative importance (Random Forest)", "y": ""},
                    title="Top 15 Drivers of Strong Membership Interest", color_discrete_sequence=[ACCENT])
        fig.update_layout(height=500)
        st.plotly_chart(fig, width='stretch')

        st.markdown(
            """<div class="insight-box">The strongest predictors are behavioral and economic, not demographic:
            annual spending potential, future purchase frequency, NPS score, and price willingness dominate the
            list. Age, gender, and heritage connection barely register. In plain terms: membership interest is
            an extension of how much someone already loves and spends on the category, not who they are on
            paper.</div>""", unsafe_allow_html=True)

        st.markdown("#### Correlation matrix, key numeric variables")
        corr_cols = ["appeal_score", "trust_count", "nps_score", "income_bracket_ordinal",
                    "max_price_small_piece_usd", "max_price_large_piece_usd", "authenticity_premium_pct",
                    "future_purchases_per_year", "annual_spending_potential_usd", "target_membership_interest"]
        corr = df[corr_cols].corr()
        labels = [pretty(c).title() for c in corr_cols]
        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=labels, y=labels, colorscale="RdGy_r", zmin=-1, zmax=1,
            text=np.round(corr.values, 2), texttemplate="%{text}", textfont=dict(size=10),
        ))
        fig.update_layout(title="Correlation Matrix", height=550)
        st.plotly_chart(fig, width='stretch')

        st.markdown(
            f"""<div class="insight-box">Annual spending potential correlates most with income
            ({df[['income_bracket_ordinal','annual_spending_potential_usd']].corr().iloc[0,1]:.2f}) and concept
            appeal ({df[['appeal_score','annual_spending_potential_usd']].corr().iloc[0,1]:.2f}), confirming that
            willingness to pay is an economic story first, an enthusiasm story second.</div>""",
            unsafe_allow_html=True)

    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            by_age = (df.groupby("age_range")["target_membership_interest"].mean() * 100).reindex(AGE_ORDER)
            fig = px.bar(x=by_age.index, y=by_age.values,
                        labels={"x": "Age range", "y": "% strong membership interest"},
                        title="Membership Interest by Age", color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, width='stretch')
            chi2, p = chi_square_test(df, "age_range")
            st.caption(f"Chi-square = {chi2:.2f}, p = {p:.4f}, "
                      f"{'statistically significant' if p < 0.05 else 'not statistically significant'} at the 5% level")
        with c2:
            by_income = (df.groupby("income_bracket")["target_membership_interest"].mean() * 100).reindex(INCOME_ORDER)
            fig = px.bar(x=by_income.index, y=by_income.values,
                        labels={"x": "Income bracket", "y": "% strong membership interest"},
                        title="Membership Interest by Income", color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, width='stretch')
            chi2, p = chi_square_test(df, "income_bracket")
            st.caption(f"Chi-square = {chi2:.2f}, p = {p:.4f}, "
                      f"{'statistically significant' if p < 0.05 else 'not statistically significant'} at the 5% level")

        spend_by_income = df.groupby("income_bracket")["annual_spending_potential_usd"].mean().reindex(INCOME_ORDER)
        fig = px.bar(x=spend_by_income.index, y=spend_by_income.values,
                    labels={"x": "Income bracket", "y": "Avg annual spending potential (USD)"},
                    title="Annual Spending Potential by Income", color_discrete_sequence=[ACCENT])
        st.plotly_chart(fig, width='stretch')

        gap = by_income.get("$200K+", 0) - by_income.get("Under $25K", 0)
        st.markdown(
            f"""<div class="insight-box">Income is the single sharpest demographic lever: membership interest
            rises from {by_income.get('Under $25K', 0):.0f}% in the under-$25K bracket to
            {by_income.get('$200K+', 0):.0f}% in the $200K+ bracket, a {gap:.0f} point swing. Age moves interest
            too (older respondents lean in more), but the effect is smaller and less actionable than income or
            the behavioral signals on the Key Drivers tab.</div>""", unsafe_allow_html=True)

    with tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            by_role = (df.groupby("respondent_role")["target_membership_interest"].mean() * 100).sort_values(ascending=False)
            fig = px.bar(x=by_role.values, y=by_role.index, orientation="h",
                        labels={"x": "% strong membership interest", "y": ""},
                        title="Membership Interest by Customer Type", color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, width='stretch')
            chi2, p = chi_square_test(df, "respondent_role")
            st.caption(f"Chi-square = {chi2:.2f}, p = {p:.5f}, "
                      f"{'statistically significant' if p < 0.05 else 'not statistically significant'} at the 5% level")
        with c2:
            by_her = (df.groupby("heritage_connection")["target_membership_interest"].mean() * 100).sort_values(ascending=False)
            fig = px.bar(x=by_her.values, y=by_her.index, orientation="h",
                        labels={"x": "% strong membership interest", "y": ""},
                        title="Membership Interest by Heritage Connection", color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, width='stretch')
            chi2, p = chi_square_test(df, "heritage_connection")
            st.caption(f"Chi-square = {chi2:.2f}, p = {p:.4f}, "
                      f"{'statistically significant' if p < 0.05 else 'not statistically significant'} at the 5% level")

        st.markdown(
            f"""<div class="insight-box">Customer type is the strongest categorical driver in the entire dataset
            (chi-square p &lt; 0.0001). Trade buyers, retailers/boutique owners ({by_role.iloc[0]:.0f}%) and
            interior designers ({by_role.get('Interior designer or architect', 0):.0f}%), show dramatically
            higher membership interest than one-off gift buyers
            ({by_role.get('Gift buyer (personal)', 0):.0f}%) or corporate gifting buyers
            ({by_role.get('Corporate gifting buyer', 0):.0f}%). Heritage connection tells a more nuanced story:
            "professional interest" beats "Indian-origin" as a predictor of interest, meaning cultural identity
            alone is a weaker membership driver than how someone actually uses the product.</div>""",
            unsafe_allow_html=True)

    with tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            by_region = (df.groupby("region")["target_membership_interest"].mean() * 100).sort_values(ascending=False)
            fig = px.bar(x=by_region.index, y=by_region.values,
                        labels={"x": "Region", "y": "% strong membership interest"},
                        title="Membership Interest by Region", color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, width='stretch')
            chi2, p = chi_square_test(df, "region")
            st.caption(f"Chi-square = {chi2:.2f}, p = {p:.4f}, "
                      f"{'statistically significant' if p < 0.05 else 'not statistically significant'} at the 5% level")
        with c2:
            spend_by_region = df.groupby("region")["annual_spending_potential_usd"].mean().sort_values(ascending=False)
            fig = px.bar(x=spend_by_region.index, y=spend_by_region.values,
                        labels={"x": "Region", "y": "Avg annual spending potential (USD)"},
                        title="Spending Potential by Region", color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, width='stretch')

        seg_spend = df.groupby("segment_true")["annual_spending_potential_usd"].mean().sort_values(ascending=False)
        seg_count = df["segment_true"].value_counts().reindex(seg_spend.index)
        fig = go.Figure()
        fig.add_bar(x=seg_spend.index, y=seg_spend.values, name="Avg spending potential (USD)",
                   marker_color=ACCENT, yaxis="y1")
        fig.add_scatter(x=seg_count.index, y=seg_count.values, name="Respondent count", mode="lines+markers",
                        marker_color=DARK, yaxis="y2")
        fig.update_layout(
            title="Segment Value vs Segment Size",
            yaxis=dict(title="Avg annual spending potential (USD)"),
            yaxis2=dict(title="Respondent count", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(fig, width='stretch')

        st.markdown(
            f"""<div class="insight-box">Europe leads on membership interest ({by_region.iloc[0]:.0f}%), South
            Asia trails ({by_region.iloc[-1]:.0f}%), but the more important pattern is the segment chart above:
            Collectors Circle Prospect is Riwaaya's smallest segment by headcount but by far its highest-value
            one, while Diaspora Heritage Buyer is the largest by volume but mid-tier on spend.</div>""",
            unsafe_allow_html=True)


# ===================================================================
# PAGE 3: PREDICTIVE MODELING (CLASSIFICATION)
# ===================================================================
def page_predictive_modeling(df):
    st.title("Predictive Modeling: Who Will Commit to Membership")
    st.markdown(
        """
        **Target variable:** `target_membership_interest`: 1 if a respondent said **"Yes, definitely"**
        to joining a Riwaaya membership program, 0 otherwise (Possibly / No). This split is close to
        50/50 ({:.1f}% positive), which is what makes it a clean classification target rather than an
        imbalanced one that would need resampling.
        """.format(df["target_membership_interest"].mean() * 100)
    )

    with st.expander("Methodology: features, split, tuning and cross-validation", expanded=False):
        st.markdown(
            """
            - **Features:** 75 engineered features covering demographics, ordinal-encoded scales (age,
              income, items purchased, membership interest), one-hot encoded multi-select fields (buying
              channels, product interests, trust factors), one-hot encoded nominal fields (gender, region,
              customer type, heritage connection, frustrations, blockers, wishlist), and the four engineered
              business flags (high intent, premium buyer, high trust, annual spending potential). The raw
              ground-truth segment label was deliberately excluded to avoid leaking the answer.
            - **Split:** 80/20 train/test, stratified on the target so both sets keep the same class balance.
            - **Scaling:** StandardScaler fit on the training set only, applied to both sets.
            - **Tuning:** GridSearchCV, 5-fold stratified cross-validation, optimizing ROC-AUC.
            - **Models:** K-Nearest Neighbors, Decision Tree, Random Forest, Gradient Boosting.
            """
        )

    with st.spinner("Training and tuning all four models..."):
        output = train_classification_models(df)

    results = output["results"]
    comparison = output["comparison"]
    best_name = output["best_model_name"]

    st.divider()
    st.markdown("### Model Comparison")
    st.dataframe(comparison.style.format("{:.3f}").highlight_max(axis=0, color="#F0DCD5"), width='stretch')

    fig = go.Figure()
    metrics_to_plot = ["Test Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    for i, model_name in enumerate(comparison.index):
        fig.add_trace(go.Bar(name=model_name, x=metrics_to_plot,
                             y=[comparison.loc[model_name, m] for m in metrics_to_plot],
                             marker_color=PALETTE[i % len(PALETTE)]))
    fig.update_layout(title="Model Performance Comparison", barmode="group", yaxis=dict(range=[0, 1.05]))
    st.plotly_chart(fig, width='stretch')

    st.markdown(
        f"""<div class="insight-box"><b>{best_name}</b> is the best-performing model by ROC-AUC
        ({comparison.loc[best_name, 'ROC-AUC']:.3f}) and test accuracy
        ({comparison.loc[best_name, 'Test Accuracy']*100:.1f}%). Note the gap between training and test
        accuracy across the tree-based models, classic overfitting on a training set of only 800 rows
        with 75 features; this is exactly why cross-validated tuning and a held-out test set both matter
        here, a model that only looked good on training data would be misleading.</div>""",
        unsafe_allow_html=True)

    st.divider()
    st.markdown("### Per-Model Detail")
    model_tabs = st.tabs(list(results.keys()))
    for tab, name in zip(model_tabs, results.keys()):
        with tab:
            r = results[name]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Train Accuracy", f"{r['train_accuracy']*100:.1f}%")
            c2.metric("Test Accuracy", f"{r['test_accuracy']*100:.1f}%")
            c3.metric("Precision", f"{r['precision']*100:.1f}%")
            c4.metric("Recall", f"{r['recall']*100:.1f}%")
            c5.metric("ROC-AUC", f"{r['roc_auc']:.3f}")
            st.caption(f"Best hyperparameters (5-fold CV, optimizing ROC-AUC): {r['best_params']}")

            cc1, cc2 = st.columns(2)
            with cc1:
                cm = r["confusion_matrix"]
                fig = go.Figure(data=go.Heatmap(
                    z=cm, x=["Predicted: Not Definite", "Predicted: Definite"],
                    y=["Actual: Not Definite", "Actual: Definite"],
                    colorscale=[[0, "#FFFFFF"], [1, ACCENT]],
                    text=cm, texttemplate="%{text}", textfont=dict(size=16), showscale=False,
                ))
                fig.update_layout(title=f"{name}: Confusion Matrix", height=380)
                st.plotly_chart(fig, width='stretch')
            with cc2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines",
                                         line=dict(color=ACCENT, width=3),
                                         name=f"ROC curve (AUC = {r['roc_auc']:.3f})"))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                         line=dict(color=GRAY, width=1, dash="dash"), name="Random guess"))
                fig.update_layout(title=f"{name}: ROC Curve", xaxis_title="False Positive Rate",
                                  yaxis_title="True Positive Rate", height=380)
                st.plotly_chart(fig, width='stretch')

            fp_fn_df = pd.DataFrame({
                "Error type": ["False Positives", "False Negatives"],
                "% of total errors": [r["fp_pct_of_errors"], r["fn_pct_of_errors"]],
                "% of test set": [r["fp_pct_of_test"], r["fn_pct_of_test"]],
            })
            cc3, cc4 = st.columns([1, 1])
            with cc3:
                fig = px.bar(fp_fn_df, x="Error type", y="% of total errors",
                            title=f"{name}: Error Composition", color="Error type",
                            color_discrete_sequence=[ACCENT, DARK])
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width='stretch')
            with cc4:
                st.markdown(
                    f"""
                    <div class="insight-box">
                    <b>{name}</b> made {r['fp']+r['fn']} errors on {len(output['y_test'])} test cases.<br><br>
                    <b>False Positives:</b> {r['fp']} cases ({r['fp_pct_of_test']:.1f}% of test set):
                    predicted as "definitely joining" but actually weren't. Business cost: wasted marketing
                    spend targeting them.<br><br>
                    <b>False Negatives:</b> {r['fn']} cases ({r['fn_pct_of_test']:.1f}% of test set):
                    predicted as not interested but actually were. Business cost: missed membership
                    revenue, a real customer who wanted in but wasn't targeted.
                    </div>
                    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### All Models, One ROC Chart")
    fig = go.Figure()
    for i, (name, r) in enumerate(results.items()):
        fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines",
                                 line=dict(color=PALETTE[i % len(PALETTE)], width=2.5),
                                 name=f"{name} (AUC={r['roc_auc']:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color=GRAY, width=1, dash="dash"), name="Random guess"))
    fig.update_layout(title="ROC Curve Comparison, All Models", xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate", height=500)
    st.plotly_chart(fig, width='stretch')

    st.markdown("### What Drives the Prediction")
    importances = output["feature_importances"].head(15).sort_values()
    labels = [pretty(i) for i in importances.index]
    fig = px.bar(x=importances.values, y=labels, orientation="h",
                labels={"x": "Relative importance", "y": ""},
                title="Top 15 Features, Best Tree-Based Model", color_discrete_sequence=[ACCENT])
    fig.update_layout(height=500)
    st.plotly_chart(fig, width='stretch')

    st.markdown(
        f"""<div class="insight-box"><b>Bottom line:</b> {best_name} is the model to put into production for
        membership targeting. It correctly separates likely members from unlikely ones using mostly behavioral
        signals (spending potential, purchase frequency, satisfaction, price willingness) rather than
        demographics, which means a lean signup form focused on buying behavior will predict membership
        interest better than one focused on who the person is.</div>""", unsafe_allow_html=True)


# ===================================================================
# PAGE 4: PREDICTIVE ANALYTICS (REGRESSION + CONVERSION SCORING)
# ===================================================================
def page_predictive_analytics(df):
    st.title("Predictive Analytics: Forecasting Spend, Purchases, and Conversion")
    st.markdown(
        "Beyond classifying who joins, how much will customers spend, how often will they buy, "
        "and who's most likely to convert right now."
    )
    st.divider()

    tab1, tab2, tab3 = st.tabs(["Annual Spending Potential", "Future Purchase Frequency", "Conversion Likelihood"])

    with tab1:
        st.markdown("#### Forecasting annual spending potential (USD)")
        st.caption(
            "Predictors deliberately exclude future_purchases_per_year, max_price_small_piece_usd, "
            "high_intent_customer and premium_buyer, since those are the direct components or near-duplicates "
            "of the target itself."
        )
        with st.spinner("Training regression models..."):
            reg_out = train_regression_models(
                df, "annual_spending_potential_usd",
                extra_drop_cols=["future_purchases_per_year", "max_price_small_piece_usd",
                                 "high_intent_customer", "premium_buyer"]
            )
        comp = reg_out["comparison"]
        st.dataframe(comp.style.format("{:.3f}"), width='stretch')

        best_reg_name = reg_out["best_model_name"]
        best_reg = reg_out["results"][best_reg_name]

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=best_reg["y_test"], y=best_reg["pred"], mode="markers",
                                     marker=dict(color=ACCENT, opacity=0.6), name="Predictions"))
            max_val = max(best_reg["y_test"].max(), best_reg["pred"].max())
            fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines",
                                     line=dict(color=GRAY, dash="dash"), name="Perfect prediction"))
            fig.update_layout(title=f"{best_reg_name}: Actual vs Predicted",
                              xaxis_title="Actual annual spending potential (USD)",
                              yaxis_title="Predicted annual spending potential (USD)")
            st.plotly_chart(fig, width='stretch')
        with c2:
            fig = px.bar(x=list(comp.index), y=comp["R2"], labels={"x": "Model", "y": "R-squared"},
                        title="R-squared by Model", color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, width='stretch')

        st.markdown(
            f"""<div class="insight-box">The best model ({best_reg_name}) explains about
            {comp.loc[best_reg_name, 'R2']*100:.0f}% of the variation in spending potential (R-squared =
            {comp.loc[best_reg_name, 'R2']:.2f}), with a typical error of about
            ${comp.loc[best_reg_name, 'MAE']:.0f} against an average spend potential of
            ${df['annual_spending_potential_usd'].mean():.0f}. Good enough to rank and prioritize customers,
            not precise enough to bill them off of.</div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("#### Forecasting future purchase frequency (purchases / year)")
        with st.spinner("Training regression models..."):
            reg_out2 = train_regression_models(
                df, "future_purchases_per_year",
                extra_drop_cols=["annual_spending_potential_usd", "high_intent_customer"]
            )
        comp2 = reg_out2["comparison"]
        st.dataframe(comp2.style.format("{:.3f}"), width='stretch')

        best_reg2_name = reg_out2["best_model_name"]
        best_reg2 = reg_out2["results"][best_reg2_name]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=best_reg2["y_test"], y=best_reg2["pred"], mode="markers",
                                 marker=dict(color=ACCENT, opacity=0.6), name="Predictions"))
        max_val2 = max(best_reg2["y_test"].max(), best_reg2["pred"].max())
        fig.add_trace(go.Scatter(x=[0, max_val2], y=[0, max_val2], mode="lines",
                                 line=dict(color=GRAY, dash="dash"), name="Perfect prediction"))
        fig.update_layout(title=f"{best_reg2_name}: Actual vs Predicted",
                          xaxis_title="Actual future purchases / year",
                          yaxis_title="Predicted future purchases / year")
        st.plotly_chart(fig, width='stretch')

        st.markdown(
            f"""<div class="insight-box"><b>Honest finding, not a polished one:</b> purchase frequency is much
            harder to predict from static profile data, R-squared of only {comp2.loc[best_reg2_name,'R2']:.2f}.
            That's a real result worth reporting as-is. The business implication: don't try to forecast
            purchase cadence from a signup form, track actual purchase behavior over time instead.</div>""",
            unsafe_allow_html=True)

    with tab3:
        st.markdown("#### Conversion likelihood: scoring every respondent")
        st.caption(
            "Uses the tuned classification model from the Predictive Modeling page to score every "
            "respondent's probability of strong membership interest, then buckets them into targeting tiers."
        )
        with st.spinner("Scoring respondents..."):
            clf_out = train_classification_models(df)

        best_name = clf_out["best_model_name"]
        best_model = clf_out["results"][best_name]["model"]
        X_full = build_feature_matrix(df)
        scaler = clf_out["scaler"]
        X_full_scaled = scaler.transform(X_full)
        df_scored = df.copy()
        df_scored["conversion_probability"] = best_model.predict_proba(X_full_scaled)[:, 1]

        def tier(p):
            if p >= 0.7:
                return "High (70%+)"
            elif p >= 0.4:
                return "Medium (40-70%)"
            return "Low (<40%)"

        df_scored["conversion_tier"] = df_scored["conversion_probability"].apply(tier)
        tier_order = ["High (70%+)", "Medium (40-70%)", "Low (<40%)"]

        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df_scored, x="conversion_probability", nbins=30,
                               labels={"conversion_probability": "Predicted conversion probability"},
                               title=f"Conversion Probability Distribution ({best_name})",
                               color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, width='stretch')
        with c2:
            tier_counts = df_scored["conversion_tier"].value_counts().reindex(tier_order)
            fig = px.bar(x=tier_counts.index, y=tier_counts.values,
                        labels={"x": "Conversion tier", "y": "Respondents"},
                        title="Respondents by Conversion Tier", color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, width='stretch')

        tier_profile = df_scored.groupby("conversion_tier")[
            ["annual_spending_potential_usd", "income_bracket_ordinal", "appeal_score", "nps_score"]
        ].mean().reindex(tier_order)
        st.markdown("##### What each tier looks like")
        st.dataframe(tier_profile.style.format("{:.1f}"), width='stretch')

        high_share = (df_scored["conversion_tier"] == "High (70%+)").mean() * 100
        high_spend = tier_profile.loc["High (70%+)", "annual_spending_potential_usd"]
        low_spend = tier_profile.loc["Low (<40%)", "annual_spending_potential_usd"]
        st.markdown(
            f"""<div class="insight-box">{high_share:.0f}% of respondents score in the high-conversion tier, and
            they carry roughly {high_spend/low_spend:.1f}x the spending potential of the low tier
            (${high_spend:.0f} vs ${low_spend:.0f}). This is the segment to prioritize in a launch campaign:
            smaller in number, far higher expected return per acquisition dollar.</div>""",
            unsafe_allow_html=True)


# ===================================================================
# PAGE 5: PRESCRIPTIVE STRATEGY AND SUSTAINABILITY VERDICT
# ===================================================================
def page_prescriptive(df):
    st.title("Prescriptive Strategy & Sustainability Verdict")
    st.markdown("Turning the analysis into decisions: who to target, what to charge, how to message, "
                "what to build, and where to expand.")
    st.divider()

    seg = df.groupby("segment_true").agg(
        count=("respondent_id", "count"),
        avg_spend=("annual_spending_potential_usd", "mean"),
        pct_member=("target_membership_interest", "mean"),
        pct_premium=("premium_buyer", "mean"),
    ).reset_index()
    seg["pct_member"] *= 100

    st.markdown("## 1. Customer Segmentation Strategy")
    fig = px.scatter(seg, x="pct_member", y="avg_spend", size="count", color="segment_true",
                     size_max=60, color_discrete_sequence=PALETTE,
                     labels={"pct_member": "% strong membership interest",
                            "avg_spend": "Avg annual spending potential (USD)", "segment_true": "Segment"},
                     title="Segment Priority Map: Value vs Intent (bubble size = respondent count)")
    fig.update_layout(height=500)
    st.plotly_chart(fig, width='stretch')

    st.markdown(
        """
        **Tier 1, Anchor segments (lead with these):** Collectors Circle Prospect and Global Design & Decor
        Collector. Smaller combined headcount, but the highest spend potential and membership conversion in
        the data. These are the founding members and the people whose retention defines whether the
        membership program pays for itself.

        **Tier 2, Volume segment (nurture, don't over-invest yet):** Diaspora Heritage Buyer. The single
        largest segment by headcount, moderate spend, and middling membership interest. Treat this as the
        top-of-funnel audience: good for brand reach and community building, not the primary membership
        revenue engine on day one.

        **Tier 3, Opportunistic segments (serve, don't chase):** Gifting & Corporate Buyer and Conscious
        Luxury Consumer. Lowest spend potential and membership interest of the five. Worth serving well
        since they still buy, but not worth a dedicated acquisition budget until the anchor segments are
        proven out.
        """
    )

    st.markdown("## 2. Pricing Strategy")
    median_small = df["max_price_small_piece_usd"].median()
    median_large = df["max_price_large_piece_usd"].median()
    avg_premium = df["authenticity_premium_pct"].mean()
    st.markdown(
        f"""
        - **Anchor catalog pricing to actual willingness to pay, not assumption.** Median willingness to pay
          sits at \\${median_small:.0f} for a small piece and \\${median_large:.0f} for a large one. Price the
          entry catalog around these medians rather than guessing high and discounting later.
        - **Monetize the authentication layer directly.** Respondents will pay an average
          {avg_premium:.0f}% premium for verified authenticity. That's a number Riwaaya can build into
          margin, an authenticated piece should be priced {avg_premium:.0f}% above an unverified equivalent.
        - **Tier the membership fee to the segment, not a single flat price.** A single membership price
          will either be too cheap for Collectors Circle Prospect or too expensive for Diaspora Heritage
          Buyer. Recommend two tiers: a premium "Collector" tier and a lighter "Member" tier.
        """
    )

    st.markdown("## 3. Marketing Campaigns")
    top_role = df.groupby("respondent_role")["target_membership_interest"].mean().idxmax()
    st.markdown(
        f"""
        - **Lead every campaign with authentication, not heritage.** Authentication/certification is the
          single most-cited trust factor, and trust/authenticity doubts are the top purchase blocker.
        - **Build a dedicated B2B trade channel.** {top_role}s show the highest membership interest of any
          customer type by a wide margin. A trade program (bulk pricing, dedicated account support, early
          catalog access) will convert far more efficiently than a generic consumer campaign.
        - **Use NPS promoters as a referral engine.** A structured referral incentive for that group is
          cheaper than paid acquisition and targets people who are already convinced.
        """
    )

    st.markdown("## 4. Membership Offerings")
    st.markdown(
        """
        | Tier | Target segment | Core offer |
        |---|---|---|
        | **Trade Pro** | Retailers, boutique owners, interior designers | Bulk pricing, early catalog access, dedicated account support |
        | **Collector** | Collectors Circle Prospect, Global Design & Decor Collector | Authentication priority, exclusive/limited pieces, concierge sourcing |
        | **Member** | Diaspora Heritage Buyer, Conscious Luxury Consumer | Standard authentication, community content, modest repeat-purchase discount |
        """
    )

    st.markdown("## 5. Personalization Opportunities")
    interest_cols = [c for c in df.columns if c.startswith("interest_") and c != "interest_no_response"]
    top_interest_by_role = {}
    for role in df["respondent_role"].unique():
        sub = df[df["respondent_role"] == role]
        top = pretty(sub[interest_cols].mean().idxmax().replace("interest_", ""))
        top_interest_by_role[role] = top
    top_interest_df = pd.DataFrame(list(top_interest_by_role.items()), columns=["Customer type", "Top product interest"])
    st.dataframe(top_interest_df, width='stretch', hide_index=True)
    st.markdown(
        """
        Product interest clearly differs by customer type, which means the catalog landing page shouldn't be
        one-size-fits-all. Personalize the homepage and email campaigns by customer type at minimum, and
        layer in the predicted conversion tier from the Predictive Analytics page to decide who gets a
        premium catalog view versus a standard one.
        """
    )

    st.markdown("## 6. Expansion Priorities")
    region_summary = df.groupby("region").agg(
        count=("respondent_id", "count"), pct_member=("target_membership_interest", "mean"),
        avg_spend=("annual_spending_potential_usd", "mean"),
    ).round(1).sort_values("avg_spend", ascending=False)
    region_summary["pct_member"] *= 100
    st.dataframe(region_summary.style.format({"pct_member": "{:.1f}%", "avg_spend": "${:.0f}"}), width='stretch')
    st.markdown(
        """
        Europe and the GCC show the strongest combination of membership interest and spend, despite North
        America carrying the largest respondent base. Recommended sequencing: launch and prove the model in
        the GCC and Europe, use North America for volume growth once the membership mechanics are validated,
        and treat South Asia as a longer-term market requiring a different price point.
        """
    )

    st.divider()
    st.markdown("## 7. Key Findings, Managerial Insights, and Sustainability Verdict")
    c1, c2, c3 = st.columns(3)
    c1.metric("Strong Membership Intent", f"{df['target_membership_interest'].mean()*100:.0f}%")
    c2.metric("Premium Buyers", f"{df['premium_buyer'].mean()*100:.0f}%")
    c3.metric("Avg Authenticity Premium", f"{df['authenticity_premium_pct'].mean():.0f}%")

    st.markdown(
        """
        ### What the data actually says

        1. **Demand for the core concept is real and reasonably strong**, just over half of respondents
           say they'd definitely join a membership, and willingness to pay an authenticity premium is
           widespread.
        2. **The business is not one customer base, it's at least two with very different economics.**
           Trade buyers and serious collectors drive most of the value; gift buyers and casual browsers
           drive volume but little margin.
        3. **Behavior predicts loyalty far better than identity does.** Purchase frequency, satisfaction,
           and price willingness outpredict age, gender, and even heritage connection.
        4. **Spending potential is forecastable from profile data with real, if moderate, accuracy.**
           Purchase frequency is not, current data can rank likely high-spenders but not reliably predict
           how often any individual will come back.

        ### Is Riwaaya commercially sustainable?

        **Conditionally, yes.** The data supports the model the original business case argued for, an
        authentication-led premium marketplace, provided three things hold:

        - **Execution stays focused on the anchor segments.** If acquisition spend drifts toward the
          broader, lower-intent diaspora and gifting audiences too early, blended unit economics will look
          much weaker than this analysis suggests.
        - **The authentication layer has to be real, not cosmetic.** It's the single biggest reason people
          say they'd pay more and the single biggest blocker when it's in doubt.
        - **Membership design matches the tiered reality of the customer base**, not a single flat price.

        The risk isn't that the concept is wrong. It's that a generic, undifferentiated rollout would spend
        against the wrong segments and undersell the one feature, authentication, that's actually driving
        willingness to pay. Sustainability depends on discipline in where Riwaaya points its budget, not on
        whether the underlying demand exists.
        """
    )


# ===================================================================
# MAIN: PAGE CONFIG, NAVIGATION, DISPATCH
# ===================================================================
st.set_page_config(
    page_title="Riwaaya Analytics Dashboard",
    page_icon="🏺",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

df = load_data()

st.sidebar.title("Riwaaya")
st.sidebar.caption("Consumer Analytics Dashboard")
nav_choice = st.sidebar.radio(
    "Go to",
    ["Overview", "Descriptive Analytics", "Diagnostic Analytics",
     "Predictive Modeling", "Predictive Analytics", "Prescriptive Strategy"],
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.caption(
    "Individual & Group PBL, Data Analytics for Insights and Decision Making"
)

PAGES = {
    "Overview": page_overview,
    "Descriptive Analytics": page_descriptive,
    "Diagnostic Analytics": page_diagnostic,
    "Predictive Modeling": page_predictive_modeling,
    "Predictive Analytics": page_predictive_analytics,
    "Prescriptive Strategy": page_prescriptive,
}
PAGES[nav_choice](df)
