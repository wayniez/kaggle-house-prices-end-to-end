import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from scipy.special import boxcox1p
import os

# ── Page settings ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction")
st.caption("Models: XGBoost + LightGBM + Ridge blend | Dataset: Kaggle House Prices")

# ── Models load ──────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    models_dir = os.path.join(BASE_DIR, "models")
    data_dir = os.path.join(BASE_DIR, "data", "processed")

    ridge = joblib.load(os.path.join(models_dir, "ridge.pkl"))
    xgb_model = joblib.load(os.path.join(models_dir, "xgb_model.pkl"))
    lgb_model = joblib.load(os.path.join(models_dir, "lgb_model.pkl"))
    blend_config = joblib.load(os.path.join(models_dir, "blend_config.pkl"))
    prep_config = joblib.load(os.path.join(models_dir, "preprocessing_config.pkl"))

    feature_names = pd.read_csv(
        os.path.join(data_dir, "feature_names.csv")
    )["feature"].tolist()

    return ridge, xgb_model, lgb_model, blend_config, prep_config, feature_names


try:
    ridge, xgb_model, lgb_model, blend_config, prep_config, feature_names = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"Model load error: {e}")
    models_loaded = False
    st.stop()

w_xgb   = blend_config["w_xgb"]
w_lgb   = blend_config["w_lgb"]
w_ridge = blend_config["w_ridge"]

# ── Mappings for categorical fields ────────────────────────────────────────
# Label Encoding must match the one used in 02_feature_engineering
# We use sorted() — this is exactly how the sklearn LabelEncoder works

NEIGHBORHOODS = sorted([
    'Blmngtn','Blueste','BrDale','BrkSide','ClearCr','CollgCr',
    'Crawfor','Edwards','Gilbert','IDOTRR','MeadowV','Mitchel',
    'NAmes','NoRidge','NPkVill','NridgHt','NWAmes','OldTown',
    'Sawyer','SawyerW','Somerst','StoneBr','SWISU','Timber','Veenker'
])
OVERALL_QUAL  = list(range(1, 11))
EXTER_QUAL    = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
KITCHEN_QUAL  = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
BSMT_QUAL     = ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
GARAGE_FINISH = ['None', 'Unf', 'RFn', 'Fin']
FIREPLACE_QU  = ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
MS_ZONING     = sorted(['A','C (all)','FV','I','RH','RL','RP','RM'])
HOUSE_STYLE   = sorted(['1Story','1.5Fin','1.5Unf','2Story','2.5Fin','2.5Unf','SFoyer','SLvl'])
SALE_TYPE     = sorted(['COD','Con','ConLD','ConLI','ConLw','CWD','New','Oth','WD'])

# Ordered mappings (correspond to 02_feature_engineering)
ORDINAL = {
    'ExterQual':   {'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5},
    'KitchenQual': {'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5},
    'BsmtQual':    {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5},
    'GarageFinish':{'None':0,'Unf':1,'RFn':2,'Fin':3},
    'FireplaceQu': {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5},
}

def encode_label(series_val, all_values):
    """Behaves like the sklearn LabelEncoder: sorted unique values → index."""
    mapping = {v: i for i, v in enumerate(sorted(set(all_values)))}
    return mapping.get(series_val, 0)

# ── Input form ───────────────────────────────────────────────────────────────
st.subheader("Object params")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Area and layout**")
    gr_liv_area   = st.number_input("Living area (sq. m.)", 400, 6000, 1500, step=50)
    total_bsmt_sf = st.number_input("Basement area (sq. m.)", 0, 3000, 800, step=50)
    first_flr_sf  = st.number_input("First-floor area (sq. m.)", 300, 4000, 1000, step=50)
    second_flr_sf = st.number_input("Area of the second floor (sq. m.)", 0, 2500, 0, step=50)
    lot_area      = st.number_input("Lot area (sq. m.)", 1000, 100000, 9000, step=500)
    lot_frontage  = st.number_input("Frontage (m)", 20, 200, 70, step=5)

with col2:
    st.markdown("**Quality and condition**")
    overall_qual  = st.selectbox("Overall quality (1–10)", OVERALL_QUAL, index=5)
    overall_cond  = st.selectbox("Overall condition (1–10)", list(range(1, 11)), index=4)
    exter_qual    = st.selectbox("Exterior quality", EXTER_QUAL, index=2)
    kitchen_qual  = st.selectbox("Quality of the kitchen", KITCHEN_QUAL, index=2)
    bsmt_qual     = st.selectbox("Basement quality", BSMT_QUAL, index=3)
    fireplace_qu  = st.selectbox("Fireplace quality", FIREPLACE_QU, index=0)

with col3:
    st.markdown("**Garage, year, other**")
    garage_area   = st.number_input("Garage area (sq. m.)", 0, 1500, 480, step=20)
    garage_cars   = st.selectbox("Garage spaces", [0, 1, 2, 3, 4], index=2)
    garage_finish = st.selectbox("Garage Finishing", GARAGE_FINISH, index=2)
    year_built    = st.number_input("Year built", 1872, 2010, 1980, step=1)
    year_remod    = st.number_input("The Year of Renovation", 1950, 2010, 1995, step=1)
    full_bath     = st.selectbox("Full bathrooms", [0, 1, 2, 3, 4], index=2)
    fireplaces    = st.selectbox("Fireplaces", [0, 1, 2, 3], index=0)
    neighborhood  = st.selectbox("Neighborhood", NEIGHBORHOODS, index=12)

# ── Предсказание ──────────────────────────────────────────────────────────────
predict_btn = st.button("🔮 Predict price", type="primary", use_container_width=True)

if predict_btn:
    yr_sold = 2010  

    total_sf           = total_bsmt_sf + first_flr_sf + second_flr_sf
    total_bath         = full_bath
    house_age          = yr_sold - year_built
    remod_age          = yr_sold - year_remod
    garage_age         = max(0, yr_sold - year_built)
    is_remodeled       = int(year_built != year_remod)
    is_new             = int(year_built == yr_sold)
    has_garage         = int(garage_area > 0)
    has_bsmt           = int(total_bsmt_sf > 0)
    has_fireplace      = int(fireplaces > 0)
    has_2nd_floor      = int(second_flr_sf > 0)
    has_pool           = 0
    total_porch_sf     = 0
    oq_total_sf        = overall_qual * total_sf
    oq_gr_liv          = overall_qual * gr_liv_area
    oq_total_bath      = overall_qual * total_bath

    eq_enc  = ORDINAL['ExterQual'][exter_qual]
    kq_enc  = ORDINAL['KitchenQual'][kitchen_qual]
    bq_enc  = ORDINAL['BsmtQual'][bsmt_qual]
    gf_enc  = ORDINAL['GarageFinish'][garage_finish]
    fq_enc  = ORDINAL['FireplaceQu'][fireplace_qu]

    neigh_enc = encode_label(neighborhood, NEIGHBORHOODS)
    oc_enc    = int(overall_cond)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(BASE_DIR, "models")
    medians = joblib.load(os.path.join(models_dir, "medians.pkl"))
    row = medians.copy()


    known = {
        'LotFrontage':          lot_frontage,
        'LotArea':              lot_area,
        'OverallQual':          overall_qual,
        'OverallCond':          oc_enc,
        'YearBuilt':            year_built,
        'YearRemodAdd':         year_remod,
        'MasVnrArea':           0,
        'ExterQual':            eq_enc,
        'BsmtQual':             bq_enc,
        'TotalBsmtSF':          total_bsmt_sf,
        'BsmtUnfSF':            max(0, total_bsmt_sf - 400),
        '1stFlrSF':             first_flr_sf,
        '2ndFlrSF':             second_flr_sf,
        'GrLivArea':            gr_liv_area,
        'FullBath':             full_bath,
        'KitchenQual':          kq_enc,
        'Fireplaces':           fireplaces,
        'FireplaceQu':          fq_enc,
        'GarageFinish':         gf_enc,
        'GarageCars':           garage_cars,
        'GarageArea':           garage_area,
        'Neighborhood':         neigh_enc,
        'TotalSF':              total_sf,
        'TotalBath':            total_bath,
        'TotalPorchSF':         total_porch_sf,
        'HouseAge':             house_age,
        'RemodAge':             remod_age,
        'GarageAge':            garage_age,
        'IsRemodeled':          is_remodeled,
        'IsNew':                is_new,
        'HasPool':              has_pool,
        'HasGarage':            has_garage,
        'HasBsmt':              has_bsmt,
        'HasFireplace':         has_fireplace,
        'Has2ndFloor':          has_2nd_floor,
        'OverallQual_TotalSF':  oq_total_sf,
        'OverallQual_GrLivArea':oq_gr_liv,
        'OverallQual_TotalBath':oq_total_bath,
    }
    row.update(known)

    input_df = pd.DataFrame([row])[feature_names]

    skewed_feats = prep_config['skewed_feats']
    lam          = prep_config['box_cox_lambda']
    for feat in skewed_feats:
        if feat in input_df.columns:
            input_df[feat] = boxcox1p(input_df[feat], lam)

    # Prediction
    pred_ridge = ridge.predict(input_df)[0]
    pred_xgb   = xgb_model.predict(input_df)[0]
    pred_lgb   = lgb_model.predict(input_df)[0]
    pred_blend = w_xgb * pred_xgb + w_lgb * pred_lgb + w_ridge * pred_ridge
    price      = np.expm1(pred_blend)

    cv_rmsle   = blend_config.get("cv_rmsle", 0.115)
    price_low  = np.expm1(pred_blend - 2 * cv_rmsle)
    price_high = np.expm1(pred_blend + 2 * cv_rmsle)

    # ── Result ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Result")

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted price",  f"${price:,.0f}")
    m2.metric("📉 Lower limit (95%)", f"${price_low:,.0f}")
    m3.metric("📈 Upper limit (95%)", f"${price_high:,.0f}")

    # ── SHAP ─────────────────────────────────────────────────────────────────
    st.subheader("Top influence factors (SHAP)")

    with st.spinner("Calculating SHAP values..."):
        explainer   = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(input_df)

        shap_series = pd.Series(
            shap_values[0],
            index=feature_names
        ).sort_values(key=abs, ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#e74c3c' if v > 0 else '#3498db' for v in shap_series.values[::-1]]
        bars = ax.barh(shap_series.index[::-1], shap_series.values[::-1], color=colors)

        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_xlabel("SHAP value (impact on log(price))")
        ax.set_title("Top 15 Factors Affecting Price")

        for bar, val in zip(bars, shap_series.values[::-1]):
            sign = "+" if val > 0 else ""
            ax.text(
                val + (0.001 if val >= 0 else -0.001),
                bar.get_y() + bar.get_height() / 2,
                f"{sign}{val:.3f}",
                va='center',
                ha='left' if val >= 0 else 'right',
                fontsize=9
            )

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Legend ──────────────────────────────────────────────────────────────
    st.caption(
        "🔴 Red — indicates a **price increase**  |  "
        "🔵 Blue — indicates a **price decrease**  |  "
        "Length = strength of influence"
    )

    # ── pred details ───────────────────────────────────────────────────
    with st.expander("Prediction details for each model"):
        dc1, dc2, dc3, dc4 = st.columns(4)
        dc1.metric("Ridge",    f"${np.expm1(pred_ridge):,.0f}")
        dc2.metric("XGBoost",  f"${np.expm1(pred_xgb):,.0f}")
        dc3.metric("LightGBM", f"${np.expm1(pred_lgb):,.0f}")
        dc4.metric("Blend",    f"${price:,.0f}")
        st.caption(f"Веса: XGBoost={w_xgb}, LightGBM={w_lgb}, Ridge={w_ridge}")

# ── Annotation ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("The model was trained on the Ames Housing dataset (Kaggle). The predictions are for illustrative purposes only.")
