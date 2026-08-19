# DETERMINISTIC ENVIRONMENT
import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = "42"

# import other libraries
import streamlit as st
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from functools import lru_cache
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Page set up and layout
st.set_page_config(
    layout="wide",
    page_icon="https://github.com/Yinman1063/Inflation/blob/main/Images/img_1.png"
) 
st.image(
    "https://github.com/Yinman1063/Inflation/blob/main/Images/img_2.png",
    width="stretch"
)

# Tensorflow import as function

def load_tf():
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        return tf, keras, layers
    except Exception:
        return None, None, None

tf, keras, layers = load_tf()
if not tf:
    st.error("TensorFlow not available.")
    st.stop()

# HELPERS
def make_unique(names):
    seen = {}
    out = []
    for n in names:
        key = str(n)
        if key not in seen:
            seen[key] = 0
            out.append(key)
        else:
            seen[key] += 1
            out.append(f"{key}.{seen[key]}")
    return out

def safe_rerun():
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

def make_sequences(X, y, s):
    Xs, ys = [], []
    for i in range(len(X) - s):
        Xs.append(X[i:i+s])
        ys.append(y[i+s])
    return np.array(Xs), np.array(ys)

# SIDEBAR SETUP

st.sidebar.header("Training Setup & Data")

uploaded = st.sidebar.file_uploader(
    "Upload CSV/XLSX with 'Date' & 'Inflation'",
    type=["csv", "xlsx"]
)

epochs = st.sidebar.slider("Epochs", 1, 200, 40)
n_splits = st.sidebar.slider("CV Folds", 3, 10, 5)
horizon = st.sidebar.number_input("Forecast Horizon (months)", 1, 36, 6)

st.sidebar.markdown("---")
st.sidebar.header("Auto‑Tuning")

tune_mode = st.sidebar.radio(
    "Choose Tuning Mode",
    ["FAST Deterministic Search"],
    index=0
)

lock_params = st.sidebar.checkbox("🔒 Lock Hyperparameters (skip re‑tuning)", value=False)

auto_tune_button = st.sidebar.button("⚡ Run Auto‑Tuning")

st.sidebar.markdown("---")
st.sidebar.header("Model Hyperparameters")

params = st.session_state.get("new_params", {})

seq_len = st.sidebar.slider("Sequence Length", 2, 36, params.get("seq_len", 12))
lr = st.sidebar.number_input("Learning Rate", 1e-5, 1.0, params.get("lr", 0.0005),
                             format="%.5f")
batch_size = st.sidebar.selectbox(
    "Batch Size",
    [8, 16, 32, 64],
    index=[8,16,32,64].index(params.get("batch_size", 16))
)

# LOAD + CLEAN DATA
if uploaded is None:
    st.info("Upload a dataset to continue.")
    st.stop()

df = pd.read_excel(uploaded) if uploaded.name.endswith(".xlsx") else pd.read_csv(uploaded)
df.columns = make_unique(df.columns)

if "Date" not in df.columns or "Inflation" not in df.columns:
    st.error("Dataset must contain 'Date' and 'Inflation'.")
    st.stop()

df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

df.replace(r"^\s*$", np.nan, regex=True, inplace=True)
for col in df.columns:
    if df[col].dtype == object and col != "Date":
        df[col] = df[col].astype(str).str.replace(r"[^\d\.-]", "", regex=True)
        df[col] = pd.to_numeric(df[col], errors="coerce")

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.fillna(method="ffill").fillna(method="bfill")
df = df.dropna(subset=["Inflation"]).reset_index(drop=True)

if len(df) <= seq_len + 1:
    st.error("Dataset too small for selected sequence length.")
    st.stop()

# CLEANED PREVIEW

st.subheader("Cleaned Data Preview")
st.write(df.head(20))
st.caption(f"Rows: {len(df)}, Columns: {len(df.columns)}")


# SCALING
features = df.select_dtypes(include=[np.number]).columns.tolist()
target = "Inflation"
target_idx = features.index(target)

X_raw = df[features].values.astype(float)
y_raw = df[[target]].values.astype(float)

scaler_X = StandardScaler().fit(X_raw)
scaler_y = StandardScaler().fit(y_raw)

X_scaled = scaler_X.transform(X_raw)
y_scaled = scaler_y.transform(y_raw)

# Prepare sequences for selected seq_len
X_all, y_all = make_sequences(X_scaled, y_scaled, seq_len)
input_shape = (seq_len, X_all.shape[2])

# MAIN MODEL FOR FINAL TRAINING
def build_lstm(shape, lr_):
    tf.random.set_seed(42)
    np.random.seed(42)

    model = keras.Sequential([
        keras.layers.Input(shape=shape),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(32),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(lr_), loss="mse")
    return model


# DETERMINISTIC FAST TUNER

@lru_cache(maxsize=None)
def cached_sequences(sl):
    return make_sequences(X_scaled, y_scaled, sl)

def build_tuner_lstm(shape, lr_):
    tf.random.set_seed(42)
    np.random.seed(42)

    m = keras.Sequential([
        keras.layers.Input(shape=shape),
        keras.layers.LSTM(32),      # mid-size stable tuner
        keras.layers.Dense(1),
    ])
    m.compile(optimizer=keras.optimizers.Adam(lr_), loss="mse")
    return m

def fast_grid_search_deterministic():
    st.info("Running deterministic grid search...")

    tf.random.set_seed(42)
    np.random.seed(42)

    search = {
        "seq_len": [6, 12, 18],
        "lr": [0.0003, 0.0007, 0.001],
        "batch_size": [16, 32],
    }

    best, best_mse = None, float("inf")

    for sl in search["seq_len"]:

        if len(df) <= sl + 1:
            continue

        Xg, yg = cached_sequences(sl)
        shape_g = (sl, Xg.shape[2])
        tscv = TimeSeriesSplit(3)

        for lr_ in search["lr"]:
            for bs_ in search["batch_size"]:

                fold_mses = []

                for tr, va in tscv.split(Xg):

                    model_g = build_tuner_lstm(shape_g, lr_)

                    model_g.fit(
                        Xg[tr], yg[tr],
                        epochs=5,
                        batch_size=bs_,
                        verbose=0,
                        shuffle=False   # deterministic
                    )

                    pred = scaler_y.inverse_transform(
                        model_g.predict(Xg[va], verbose=0)
                    ).flatten()

                    true = scaler_y.inverse_transform(yg[va]).flatten()

                    fold_mses.append(mean_squared_error(true, pred))

                avg_mse = float(np.mean(fold_mses))

                if avg_mse < best_mse:
                    best_mse = avg_mse
                    best = {"seq_len": sl, "lr": lr_, "batch_size": bs_}

    return best, best_mse

# EXECUTE TUNING
if auto_tune_button and not lock_params:

    best, mse = fast_grid_search_deterministic()

    if best is None:
        st.warning("Auto‑tuning failed.")
    else:
        st.success(f"🎯 Best hyperparameters found! MSE = {mse:.4f}")
        st.session_state["new_params"] = best
        safe_rerun()

elif lock_params:
    st.info("🔒 Hyperparameters locked — using previously tuned values.")

# TRAIN & FORECAST BUTTON
run = st.button("🚀 Train & Forecast")

if run:

    tf.random.set_seed(42)
    np.random.seed(42)

    X_all, y_all = make_sequences(X_scaled, y_scaled, seq_len)
    input_shape = (seq_len, X_all.shape[2])

    tscv = TimeSeriesSplit(n_splits)
    cv_mse, last_val = [], None

    for fold, (tr, va) in enumerate(tscv.split(X_all), 1):
        model_cv = build_lstm(input_shape, lr)
        model_cv.fit(X_all[tr], y_all[tr], epochs=epochs, batch_size=batch_size, verbose=0)

        pred = scaler_y.inverse_transform(model_cv.predict(X_all[va], verbose=0)).flatten()
        true = scaler_y.inverse_transform(y_all[va]).flatten()

        mse = mean_squared_error(true, pred)
        cv_mse.append(mse)
        #st.write(f"Fold {fold}: MSE = {mse:.4f}")

        last_val = (X_all[va], y_all[va], true, pred)

    st.subheader("Model Evaluation")
    st.write("Avg MSE:", np.mean(cv_mse))
    st.write("Avg RMSE:", np.sqrt(np.mean(cv_mse)))

    # Train final model
    model = build_lstm(input_shape, lr)
    model.fit(X_all, y_all, epochs=epochs, batch_size=batch_size, verbose=0)

    # Forecast
    seq = X_all[-1].copy()
    preds_scaled = []

    for _ in range(int(horizon)):
        p = model.predict(seq.reshape(1, *input_shape), verbose=0)[0, 0]
        preds_scaled.append(p)
        next_row = seq[-1].copy()
        next_row[target_idx] = p
        seq = np.vstack([seq[1:], next_row])

    preds = scaler_y.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    lower = preds - 1.2
    upper = preds + 2.2

    future_dates = [df["Date"].max() + pd.DateOffset(months=i) for i in range(1, horizon+1)]

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Lower Band": lower,
        "Forecast": preds,
        "Upper Band": upper
    }).set_index("Date")

    st.subheader("Forecast Output")
    st.write(forecast_df)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"], df["Inflation"], label="Historical", color="blue")
    ax.plot(forecast_df.index, forecast_df["Forecast"], "--o", color="green")
    ax.fill_between(forecast_df.index, lower, upper, alpha=0.2, color="gray")
    ax.legend()
    st.pyplot(fig)


    # PERMUTATION IMPORTANCE
    if last_val:
        st.subheader("Permutation Feature Importance")

        X_va, y_va, y_true_va, _ = last_val

        baseline_pred = scaler_y.inverse_transform(model.predict(X_va, verbose=0)).flatten()
        baseline_mse = mean_squared_error(y_true_va, baseline_pred)

        feats = [(name, idx) for idx, name in enumerate(features) if name != target]
        rng = np.random.default_rng(42)
        importances = []

        for fname, idx_f in feats:
            deltas = []
            for _ in range(5):
                Xp = X_va.copy()
                for t in range(seq_len):
                    sh = Xp[:, t, idx_f].copy()
                    rng.shuffle(sh)
                    Xp[:, t, idx_f] = sh

                perm_pred = scaler_y.inverse_transform(model.predict(Xp, verbose=0)).flatten()
                perm_mse = mean_squared_error(y_true_va, perm_pred)
                deltas.append(perm_mse - baseline_mse)

            importances.append((fname, np.mean(deltas)))

        imp_df = pd.DataFrame(importances, columns=["Feature", "Importance (ΔMSE)"])
        imp_df_sorted = imp_df.sort_values("Importance (ΔMSE)", ascending=False)

        fig_line, ax_line = plt.subplots(figsize=(12, 6))
        ax_line.plot(imp_df_sorted["Feature"], imp_df_sorted["Importance (ΔMSE)"],
                     marker='o', linewidth=2, color="blue")
        ax_line.set_title("Permutation Feature Importance")
        ax_line.set_xlabel("Feature")
        ax_line.set_ylabel("Increase in MSE (Δ)")
        plt.xticks(rotation=45, ha='right')
        ax_line.grid(True, linestyle="--", alpha=0.35)
        st.pyplot(fig_line)

        st.subheader("Ranked Contribution to Inflation")
        imp_df_sorted["Rank"] = imp_df_sorted["Importance (ΔMSE)"].rank(ascending=False).astype(int)

        ranking_table = imp_df_sorted[["Rank", "Feature", "Importance (ΔMSE)"]].sort_values("Rank")
        st.write(ranking_table)

        top_feature = ranking_table.iloc[0]["Feature"]
        st.success(f"🏆 Top driver of inflation is: **{top_feature}**")

        st.markdown("""
        ## Permutation Importance Explained

        Permutation importance works by:

        1. **Taking a feature** (e.g., exchange rates).  
        2. **Randomly shuffling its values** in the validation dataset.  
        3. **Measuring how much the model's error increases** (Δ MSE).  
        4. **Interpreting the result:**  
           - A **large ΔMSE** means the model relied heavily on that feature.  
           - A **small ΔMSE** means the model barely used that feature.

        ### Key Interpretation
        - **Large ΔMSE → High Importance**  
        - **Small ΔMSE → Low Importance**
        """)
