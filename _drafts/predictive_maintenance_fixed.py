import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
np.random.seed(42)

plt.rcParams.update({'font.family': 'serif','axes.spines.top': False,'axes.spines.right': False,'axes.linewidth': 0.8})

def save_fig(path: str):
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()

# NASA CMAPSS FD001 required:
# - train_FD001.txt (space-separated, no header)
# - RUL_FD001.txt   (single column)

def load_fd001():
    try:
        df = pd.read_csv("train_FD001.txt", sep="\s+", header=None)
        df.dropna(axis=1, inplace=True)
        df.columns = ['unit', 'time', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
        rul = pd.read_csv("RUL_FD001.txt", header=None)
    except FileNotFoundError as e:
        raise FileNotFoundError("NASA CMAPSS FD001 files not found. See DATA_NASA_CMAPSS.md for instructions.") from e
    rul.columns = ['max_RUL']
    rul['unit'] = rul.index + 1
    last_cycle = df.groupby('unit')['time'].max().reset_index().rename(columns={'time': 'max_time'})
    df = df.merge(last_cycle, on='unit')
    df['RUL'] = df['max_time'] - df['time']
    df['distress'] = df['RUL'] < 20
    return df

def make_rf_sequences(df, sensor, window=30):
    X, y, units = [], [], []
    for uid, g in df.groupby('unit'):
        vals = g[sensor].values; target = g['distress'].values
        for i in range(len(vals) - window):
            X.append(vals[i:i+window]); y.append(target[i+window]); units.append(uid)
    return np.array(X), np.array(y), np.array(units)

def make_lstm_sequences(df, sensor_cols, window=30):
    X, y, units = [], [], []
    for uid, g in df.groupby('unit'):
        vals = g[sensor_cols].values; target = g['distress'].values
        for i in range(len(vals) - window):
            X.append(vals[i:i+window]); y.append(target[i+window]); units.append(uid)
    return np.array(X), np.array(y), np.array(units)

def unit_time_series_splits(units: np.ndarray, n_splits: int = 5):
    unique_units = np.array(sorted(pd.unique(units)))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for tr_u, te_u in tscv.split(unique_units):
        tr_set, te_set = set(unique_units[tr_u]), set(unique_units[te_u])
        train_idx = np.where(np.isin(units, list(tr_set)))[0]
        test_idx = np.where(np.isin(units, list(te_set)))[0]
        yield train_idx, test_idx

def compute_metrics(y_true, y_prob):
    y_hat = (y_prob >= 0.5).astype(int)
    return {
        'accuracy': float(accuracy_score(y_true, y_hat)),
        'roc_auc': float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        'pr_auc': float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        'f1': float(f1_score(y_true, y_hat)) if len(np.unique(y_hat)) > 1 else np.nan,
    }

def main(plot: bool = False):
    df = load_fd001()
    selected_sensors = ['sensor_9', 'sensor_14', 'sensor_4', 'sensor_3', 'sensor_17', 'sensor_2']

    # RF sequence classifier
    X_rf, y_rf, u_rf = make_rf_sequences(df, 'sensor_9', window=30)
    rf_results = []
    for tr, te in unit_time_series_splits(u_rf, n_splits=5):
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_rf[tr], y_rf[tr])
        yprob = clf.predict_proba(X_rf[te])[:, 1]
        rf_results.append(compute_metrics(y_rf[te], yprob))
    logger.info('RF mean metrics:', {k: np.nanmean([m[k] for m in rf_results]) for k in rf_results[0]})

    # LSTM leakage-free (fit scaler+PCA on train per fold)
    X, y, u = make_lstm_sequences(df, selected_sensors, window=30)
    lstm_results = []
    callbacks = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True), ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=0)]
    last_fold = None
    for tr, te in unit_time_series_splits(u, n_splits=5):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        n_features = Xtr.shape[2]
        Xtrf, Xtef = Xtr.reshape(-1, n_features), Xte.reshape(-1, n_features)
        scaler = StandardScaler().fit(Xtrf)
        Xtr_s = scaler.transform(Xtrf).reshape(Xtr.shape)
        Xte_s = scaler.transform(Xtef).reshape(Xte.shape)
        pca = PCA(n_components=3).fit(Xtr_s.reshape(-1, n_features))
        Xtr_seq = pca.transform(Xtr_s.reshape(-1, n_features)).reshape(Xtr.shape[0], Xtr.shape[1], 3)
        Xte_seq = pca.transform(Xte_s.reshape(-1, n_features)).reshape(Xte.shape[0], Xte.shape[1], 3)
        model = Sequential([LSTM(64, input_shape=(Xtr_seq.shape[1], Xtr_seq.shape[2])), Dense(1, activation='sigmoid')])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(Xtr_seq, ytr, epochs=20, batch_size=32, shuffle=False, validation_data=(Xte_seq, yte), callbacks=callbacks, verbose=0)
        yprob = model.predict(Xte_seq, verbose=0).flatten()
        lstm_results.append(compute_metrics(yte, yprob))
        last_fold = (scaler, pca, model)
    logger.info('LSTM mean metrics:', {k: np.nanmean([m[k] for m in lstm_results]) for k in lstm_results[0]})

    # Engine plot from last fold transforms
    if last_fold is not None:
        scaler, pca, model = last_fold
        unit_id = int(df['unit'].iloc[0]); window = 30
        eng = df[df['unit'] == unit_id].sort_values('time')
        raw = np.array([eng[selected_sensors].values[i:i+window] for i in range(len(eng) - window)])
        if len(raw):
            n_features = raw.shape[2]
            raw_s = scaler.transform(raw.reshape(-1, n_features)).reshape(raw.shape)
            seq = pca.transform(raw_s.reshape(-1, n_features)).reshape(raw.shape[0], raw.shape[1], 3)
            cycles = eng['time'].values[window:]
            pred = model.predict(seq, verbose=0).flatten()
    if plot:
                plt.figure(figsize=(8, 4))
                plt.plot(cycles, pred, label='LSTM'); plt.axhline(0.5, color='red', linestyle=':', linewidth=0.8)
                plt.title(f'Engine {unit_id} – LSTM Distress Probability'); plt.xlabel('Cycle'); plt.ylabel('Probability'); plt.legend(); save_fig('pm_lstm_engine.png')

if __name__ == "__main__":
    main()
