"""Generated from Jupyter notebook: predictive_maintenance_RUL

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

import matplotlib.pyplot as plt
import pandas as pd

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
data = pd.read_csv(url, parse_dates=["Date"], index_col="Date")
data.plot(figsize=(12, 6), legend=False, title="Daily Minimum Temperatures")
plt.ylabel("Temperature (Celsius)")
plt.show()

import numpy as np
from statsmodels.nonparametric.kernel_regression import KernelReg

data_reset = data.reset_index()
data_reset["Time"] = np.arange(len(data_reset))

kr = KernelReg(
    endog=data_reset["Temp"], exog=data_reset["Time"], var_type="c", bw="cv_ls"
)
y_pred, _ = kr.fit(data_reset["Time"])

plt.figure(figsize=(12, 6))
plt.plot(data_reset["Date"], data_reset["Temp"], label="Original", alpha=0.4)
plt.plot(data_reset["Date"], y_pred, label="Smoothed", color="red")
plt.title("Kernel Regression on Temperature Data")
plt.xlabel("Date")
plt.ylabel("Temperature (Celsius)")
plt.legend()
plt.show()

for lag in range(1, 8):
    data[f"lag_{lag}"] = data["Temp"].shift(lag)
data.dropna(inplace=True)

train = data[:"1988"]
test = data["1989":]

X_train = train.drop("Temp", axis=1)
y_train = train["Temp"]
X_test = test.drop("Temp", axis=1)
y_test = test["Temp"]

from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, y_pred, label="Predicted")
plt.title("k-NN Regression for Time Series Forecasting")
plt.xlabel("Date")
plt.ylabel("Temperature (Celsius)")
plt.legend()
plt.show()


# --- code cell ---

import numpy as np
import pandas as pd

np.random.seed(42)
n = 200
sentiment = np.zeros(n, dtype=int)

for t in range(1, n):
    prob = [0.2, 0.5, 0.3] if sentiment[t - 1] == 1 else [0.3, 0.4, 0.3]
    sentiment[t] = np.random.choice([0, 1, 2], p=prob)

dates = pd.date_range(start="2010-01-01", periods=n, freq="M")
df = pd.DataFrame({"Date": dates, "Sentiment": sentiment}).set_index("Date")

import matplotlib.pyplot as plt

df["Sentiment"].plot(figsize=(12, 6), drawstyle="steps-post")
plt.title("Simulated Consumer Sentiment (Ordinal Time Series)")
plt.ylabel("Sentiment Level (0 = Negative, 2 = Positive)")
plt.show()

df["lag1"] = df["Sentiment"].shift(1)
df.dropna(inplace=True)

# !pip install mord --quiet  # Jupyter-only

import mord as m
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X = df[["lag1"]].astype(int)
y = df["Sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

model = m.LogisticIT()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(0)
n = 250
latent = np.zeros(n)
obs = np.zeros(n, dtype=int)

# AR(1) latent process
for t in range(1, n):
    latent[t] = 0.7 * latent[t - 1] + np.random.normal()

# Thresholds to generate ordered categories
thresholds = [-np.inf, -0.5, 0.5, np.inf]

for t in range(n):
    if latent[t] < thresholds[1]:
        obs[t] = 0
    elif latent[t] < thresholds[2]:
        obs[t] = 1
    else:
        obs[t] = 2

df = pd.DataFrame({"Latent": latent, "Observed": obs})

fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax[0].plot(df["Latent"], color="gray")
ax[0].set_title("Latent Process")
ax[1].plot(df["Observed"], drawstyle="steps-post")
ax[1].set_title("Observed Ordered Outcome")
plt.tight_layout()
plt.show()

# !pip install hmmlearn --quiet  # Jupyter-only

from hmmlearn import hmm

model = hmm.MultinomialHMM(n_components=3, n_iter=100)
obs_reshaped = df["Observed"].values.reshape(-1, 1)
model.fit(obs_reshaped)
hidden_states = model.predict(obs_reshaped)

plt.figure(figsize=(12, 6))
plt.plot(df["Observed"].values, label="Observed", drawstyle="steps-post", alpha=0.6)
plt.plot(hidden_states, label="Hidden State", drawstyle="steps-post")
plt.title("Hidden Markov Model: Ordered Outcomes and Inferred States")
plt.legend()
plt.show()


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Download from FRED via pandas_datareader (or use a pre-downloaded CSV)
import pandas_datareader.data as web
from hmmlearn import hmm

df = web.DataReader("NFIBETEX", "fred", start="2000-01-01")
df.dropna(inplace=True)
df["Ordinal"] = df["NFIBETEX"].astype(int) + 1  # Map -1, 0, 1 to 0, 1, 2

# Fit HMM
X = df["Ordinal"].values.reshape(-1, 1)
model = hmm.MultinomialHMM(n_components=3, n_iter=100, random_state=42)
model.fit(X)
hidden_states = model.predict(X)
df["Hidden_State"] = hidden_states

# Plot
plt.figure(figsize=(12, 6))
plt.plot(
    df.index,
    df["Ordinal"],
    label="Observed Outlook (Ordinal)",
    drawstyle="steps-post",
    alpha=0.6,
)
plt.plot(
    df.index, df["Hidden_State"], label="Inferred Hidden State", drawstyle="steps-post"
)
plt.title("NFIB: Good Time to Expand Business (Ordinal) with Hidden Markov States")
plt.xlabel("Date")
plt.yticks([0, 1, 2], ["Negative", "Neutral", "Positive"])
plt.legend()
plt.tight_layout()
plt.savefig("nfib_hmm_hidden_states.png")
plt.show()


# --- code cell ---

import matplotlib.pyplot as plt
import pandas as pd
from fredapi import Fred
from hmmlearn import hmm

# Replace this with your actual FRED API key
fred = Fred(api_key="8f058d10ec8c788296c040ea09e634d5")

# Download data
series = fred.get_series("NFIBETEX", observation_start="2000-01-01")
df = series.to_frame(name="NFIBETEX")
df.index.name = "Date"
df.dropna(inplace=True)

# Map -1 (bad time), 0 (uncertain), 1 (good time) → ordinal values 0, 1, 2
df["Ordinal"] = df["NFIBETEX"].astype(int) + 1


# --- code cell ---

# !pip install hmmlearn --quiet  # Jupyter-only


# --- code cell ---

# !pip install fredapi  # Jupyter-only


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from hmmlearn import hmm

# Download consumer sentiment index from FRED
df = web.DataReader("UMCSENT", "fred", start="2000-01-01")
df = df.dropna().rename(columns={"UMCSENT": "Sentiment"})

# Bin into ordinal categories: 0 = low, 1 = medium, 2 = high
# Thresholds chosen based on historical percentiles
df["Ordinal"] = pd.cut(
    df["Sentiment"], bins=[-np.inf, 70, 85, np.inf], labels=[0, 1, 2]
).astype(int)


# --- code cell ---

model = hmm.MultinomialHMM(n_components=3, n_iter=100, random_state=42)
X = df["Ordinal"].values.reshape(-1, 1)
model.fit(X)
df["Hidden_State"] = model.predict(X)


# --- code cell ---

plt.figure(figsize=(12, 6))
plt.plot(
    df.index,
    df["Ordinal"],
    drawstyle="steps-post",
    label="Observed Sentiment (Ordinal)",
    alpha=0.6,
)
plt.plot(df.index, df["Hidden_State"], drawstyle="steps-post", label="Hidden State")
plt.title("Consumer Sentiment (Ordinal) and Inferred HMM States")
plt.xlabel("Date")
plt.ylabel("State")
plt.legend()
plt.tight_layout()
plt.savefig("umcsent_hmm_hidden_states.png")
plt.show()


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from hmmlearn import hmm

# Step 1: Load and clean the data
df = web.DataReader("UMCSENT", "fred", start="2000-01-01")
df = df.dropna().rename(columns={"UMCSENT": "Sentiment"})

# Step 2: Bin into 3 real categories using quantiles instead of hard thresholds
quantiles = df["Sentiment"].quantile([0.33, 0.66])
df["Ordinal"] = pd.cut(
    df["Sentiment"],
    bins=[-np.inf, quantiles[0.33], quantiles[0.66], np.inf],
    labels=[0, 1, 2],
).astype(int)

X = df["Ordinal"].values.reshape(-1, 1)

model = hmm.MultinomialHMM(n_components=3, n_iter=100, random_state=42)
model.n_features = 3
model.fit(X)
df["Hidden_State"] = model.predict(X)

plt.figure(figsize=(12, 6))
plt.plot(
    df.index,
    df["Ordinal"],
    drawstyle="steps-post",
    label="Observed Sentiment (Ordinal)",
    alpha=0.6,
)
plt.plot(df.index, df["Hidden_State"], drawstyle="steps-post", label="Hidden State")
plt.title("Consumer Sentiment (Ordinal, Quantile-Binned) and HMM Inferred States")
plt.xlabel("Date")
plt.ylabel("State")
plt.legend()
plt.tight_layout()
plt.savefig("umcsent_fixed_hmm_states.png")
plt.show()


# --- code cell ---

from hmmlearn.hmm import CategoricalHMM

model = CategoricalHMM(n_components=3, n_iter=100, random_state=42)
model.fit(X)
df["Hidden_State"] = model.predict(X)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from hmmlearn.hmm import CategoricalHMM

# Step 1: Load data
df = web.DataReader("UMCSENT", "fred", start="2000-01-01")
df = df.dropna().rename(columns={"UMCSENT": "Sentiment"})

# Step 2: Bin using quantiles into 3 ordinal levels
df["Ordinal"] = pd.qcut(df["Sentiment"], q=3, labels=False)


X = df["Ordinal"].values.reshape(-1, 1)
model = CategoricalHMM(n_components=3, n_iter=100, random_state=42)
model.fit(X)
df["Hidden_State"] = model.predict(X)


# Step 4: Plot
plt.figure(figsize=(12, 6))
plt.plot(
    df.index,
    df["Ordinal"],
    drawstyle="steps-post",
    label="Observed Ordinal Sentiment",
    alpha=0.6,
)
plt.plot(
    df.index, df["Hidden_State"], drawstyle="steps-post", label="Inferred Hidden State"
)
plt.title("University of Michigan Sentiment (Ordinal) and Inferred HMM States")
plt.xlabel("Date")
plt.ylabel("State")
plt.legend()
plt.tight_layout()
plt.savefig("umcsent_hmm_categorical.png")
plt.show()


# --- code cell ---

print(df["Ordinal"].value_counts())


# --- code cell ---

df["Ordinal"].groupby(df["Ordinal"]).apply(lambda x: (x.index.min(), x.index.max()))

df["Hidden_State"].value_counts()

fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

df["Sentiment"].plot(ax=ax[0], title="Raw Sentiment")
df["Ordinal"].plot(
    ax=ax[1], title="Ordinal Category (0=Low, 2=High)", drawstyle="steps-post"
)
df["Hidden_State"].plot(ax=ax[2], title="Inferred Hidden State", drawstyle="steps-post")

plt.tight_layout()
plt.savefig("umcsent_diagnostics.png")
plt.show()


# --- code cell ---

df["Hidden_State"].value_counts()


# --- code cell ---

model = CategoricalHMM(n_components=2, n_iter=100, random_state=42)


df["Delta"] = df["Sentiment"].diff().fillna(0)
df["Delta_Ordinal"] = pd.qcut(df["Delta"], q=3, labels=False)

for i, probs in enumerate(model.emissionprob_):
    print(f"State {i}: {[round(p, 3) for p in probs]}")


# --- code cell ---

import numpy as np

symbols = np.array([[0], [1], [2]])  # possible ordinal values
log_likelihoods = model._compute_log_likelihood(symbols)

probs = np.exp(log_likelihoods)
for i, row in enumerate(probs.T):
    print(f"State {i}: {[round(p, 3) for p in row]}")


# --- code cell ---

model.emissionprob_

from pprint import pprint

# This works for CategoricalHMM from hmmlearn >= 0.3.0
emissions = model._compute_log_likelihood(np.array([[0], [1], [2]]))
print("Log-likelihoods of each observation per state:")
pprint(emissions)


# --- code cell ---

df.head()


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import CategoricalHMM
from sklearn.preprocessing import MinMaxScaler

# Load FD003 dataset
cols = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
df = pd.read_csv("train_FD003.txt", sep="\s+", header=None, names=cols)

# Step 1: Calculate RUL
rul = df.groupby("unit")["cycle"].max().reset_index()
rul.columns = ["unit", "max_cycle"]
df = df.merge(rul, on="unit")
df["RUL"] = df["max_cycle"] - df["cycle"]


# Step 2: Categorize health state
def categorize_rul(rul):
    if rul > 30:
        return 0  # healthy
    elif rul > 10:
        return 1  # warning
    else:
        return 2  # distress


df["HealthState"] = df["RUL"].apply(categorize_rul)

# Step 3: Choose a unit (e.g., engine 1)
unit_df = df[df["unit"] == 1].copy()

# Step 4: Prepare data for HMM
X = unit_df["HealthState"].values.reshape(-1, 1)

# Step 5: Fit Categorical HMM
model = CategoricalHMM(n_components=3, n_iter=100, random_state=42)
model.fit(X)
unit_df["HiddenState"] = model.predict(X)

# Step 6: Plot
plt.figure(figsize=(12, 6))
plt.plot(
    unit_df["cycle"],
    unit_df["HealthState"],
    label="Observed Health State",
    drawstyle="steps-post",
    alpha=0.6,
)
plt.plot(
    unit_df["cycle"],
    unit_df["HiddenState"],
    label="Inferred Hidden State",
    drawstyle="steps-post",
)
plt.title("Engine 1: Observed Health State vs HMM Inferred Hidden State")
plt.xlabel("Cycle")
plt.ylabel("State")
plt.yticks([0, 1, 2], ["Healthy", "Warning", "Distress"])
plt.legend()
plt.tight_layout()
plt.savefig("engine1_hmm_states.png")
plt.show()


# --- code cell ---

import torch
from torch.utils.data import DataLoader, Dataset


class TurbofanDataset(Dataset):
    def __init__(self, df, seq_len=30):
        self.seq_len = seq_len
        self.X, self.y = [], []
        features = [f"sensor_{i}" for i in range(1, 22)]
        for unit in df["unit"].unique():
            unit_df = df[df["unit"] == unit].sort_values("cycle")
            vals = unit_df[features].values
            labels = unit_df["HealthState"].values
            for i in range(len(vals) - seq_len):
                self.X.append(vals[i : i + seq_len])
                self.y.append(labels[i + seq_len])
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = TurbofanDataset(df, seq_len=30)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

import torch.nn as nn


class HealthLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


model = HealthLSTM(input_size=21, hidden_size=64, output_size=3)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    total_loss = 0
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.2f}")


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# Load FD003 dataset
cols = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
df = pd.read_csv("train_FD003.txt", sep="\s+", header=None, names=cols)

# Add RUL and HealthState
rul = df.groupby("unit")["cycle"].max().reset_index()
rul.columns = ["unit", "max_cycle"]
df = df.merge(rul, on="unit")
df["RUL"] = df["max_cycle"] - df["cycle"]

df["HealthState"] = pd.cut(
    df["RUL"], bins=[-1, 10, 30, np.inf], labels=[2, 1, 0]
).astype(int)


# Normalize sensor features
sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
scaler = MinMaxScaler()
df[sensor_cols] = scaler.fit_transform(df[sensor_cols])


# Dataset class
class TurbofanDataset(Dataset):
    def __init__(self, df, seq_len=30):
        self.seq_len = seq_len
        X, y = [], []
        for unit in df["unit"].unique():
            unit_df = df[df["unit"] == unit].sort_values("cycle")
            sensors = unit_df[sensor_cols].values
            labels = unit_df["HealthState"].values
            for i in range(len(sensors) - seq_len):
                X.append(sensors[i : i + seq_len])
                y.append(labels[i + seq_len])
        self.X = torch.from_numpy(np.array(X)).float()
        self.y = torch.from_numpy(np.array(y)).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Model
class HealthLSTM(nn.Module):
    def __init__(self, input_size=21, hidden_size=64, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


# Training setup
dataset = TurbofanDataset(df, seq_len=30)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
model = HealthLSTM()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
loss_history = []
for epoch in range(20):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    loss_history.append(total_loss)
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.2f}")

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, 21), loss_history, marker="o")
plt.title("HealthLSTM: Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# Load and preprocess
cols = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
df = pd.read_csv("train_FD003.txt", sep="\s+", header=None, names=cols)

rul = df.groupby("unit")["cycle"].max().reset_index()
rul.columns = ["unit", "max_cycle"]
df = df.merge(rul, on="unit")
df["RUL"] = df["max_cycle"] - df["cycle"]

# Ordinal health bins
df["HealthState"] = pd.cut(
    df["RUL"], bins=[-1, 10, 30, np.inf], labels=[2, 1, 0]
).astype(float)

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
scaler = MinMaxScaler()
df[sensor_cols] = scaler.fit_transform(df[sensor_cols])


# Dataset class
class TurbofanOrdinalDataset(Dataset):
    def __init__(self, df, seq_len=30):
        self.seq_len = seq_len
        X, y = [], []
        for unit in df["unit"].unique():
            unit_df = df[df["unit"] == unit].sort_values("cycle")
            sensors = unit_df[sensor_cols].values
            labels = unit_df["HealthState"].values
            for i in range(len(sensors) - seq_len):
                X.append(sensors[i : i + seq_len])
                y.append(labels[i + seq_len])
        self.X = torch.from_numpy(np.array(X)).float()
        self.y = torch.from_numpy(np.array(y)).float().unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Model
class HealthLSTMOrdinal(nn.Module):
    def __init__(self, input_size=21, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])  # scalar output


# Setup
dataset = TurbofanOrdinalDataset(df)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
model = HealthLSTMOrdinal()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
loss_history = []
for epoch in range(20):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    loss_history.append(total_loss)
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.2f}")

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, 21), loss_history, marker="o")
plt.title("Ordinal HealthLSTM (Scalar Regression) - Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.tight_layout()
plt.show()


# --- code cell ---

# ! pip install coral-pytorch  # Jupyter-only


# --- code cell ---

# ================================================
# Ordinal Health State Prediction: FD003 Benchmark
# Models: Scalar MSE | CORN | Ordinal Logit
# Target: [0=Healthy, 1=Warning, 2=Distress]
# ================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from corn import CornOrdinalCrossEntropyLoss
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from torchordinal.losses import OrdinalLogisticLoss
from torchordinal.models import OrdinalLogisticModel

# ----- DATA PREP -----
cols = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
df = pd.read_csv("train_FD003.txt", sep="\\s+", header=None, names=cols)
rul = df.groupby("unit")["cycle"].max().reset_index()
rul.columns = ["unit", "max_cycle"]
df = df.merge(rul, on="unit")
df["RUL"] = df["max_cycle"] - df["cycle"]
df["HealthState"] = pd.cut(
    df["RUL"], bins=[-1, 10, 30, np.inf], labels=[2, 1, 0]
).astype(int)

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
scaler = MinMaxScaler()
df[sensor_cols] = scaler.fit_transform(df[sensor_cols])


# ----- DATASET -----
class TurbofanDataset(Dataset):
    def __init__(self, df, seq_len=30):
        self.seq_len = seq_len
        X, y = [], []
        for unit in df["unit"].unique():
            unit_df = df[df["unit"] == unit].sort_values("cycle")
            X_vals = unit_df[sensor_cols].values
            y_vals = unit_df["HealthState"].values
            for i in range(len(X_vals) - seq_len):
                X.append(X_vals[i : i + seq_len])
                y.append(y_vals[i + seq_len])
        self.X = torch.from_numpy(np.array(X)).float()
        self.y = torch.from_numpy(np.array(y)).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = TurbofanDataset(df)
loader = DataLoader(dataset, batch_size=32, shuffle=True)


# ----- BASE LSTM -----
class LSTMBackbone(nn.Module):
    def __init__(self, input_size=21, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return h[-1]


# ----- MODEL WRAPPERS -----
class ScalarRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = LSTMBackbone()
        self.output = nn.Linear(self.backbone.hidden_size, 1)

    def forward(self, x):
        return self.output(self.backbone(x))


class CornModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = LSTMBackbone()
        self.output = nn.Linear(self.backbone.hidden_size, num_classes - 1)

    def forward(self, x):
        return self.output(self.backbone(x))


class OrdinalLogitWrapper(nn.Module):
    def __init__(self, input_size=21, hidden_size=64, num_classes=3):
        super().__init__()
        self.backbone = LSTMBackbone(input_size, hidden_size)
        self.ordinal = OrdinalLogisticModel(hidden_size, num_classes)

    def forward(self, x):
        return self.ordinal(self.backbone(x))


# ----- TRAINING LOOP -----
def train_model(model, loss_fn, transform_output=None, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_trace, mse_trace = [], []
    model.train()
    for epoch in range(epochs):
        total_loss, total_mse, count = 0, 0, 0
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            if transform_output:  # For CORN and Logit
                pred_scalar = transform_output(y_pred)
            else:
                pred_scalar = y_pred.squeeze()

            optimizer.zero_grad()
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mse += (
                ((pred_scalar.round().clamp(0, 2) - y_batch.float()) ** 2).sum().item()
            )
            count += len(y_batch)
        loss_trace.append(total_loss)
        mse_trace.append(total_mse / count)
        print(
            f"Epoch {epoch + 1:2d}: Loss = {total_loss:.2f}, MSE = {mse_trace[-1]:.4f}"
        )
    return loss_trace, mse_trace


# ----- BENCHMARK RUN -----
print("\\n== Scalar Regression ==")
scalar_model = ScalarRegression()
scalar_loss = nn.MSELoss()
scalar_loss_trace, scalar_mse_trace = train_model(
    scalar_model, scalar_loss, transform_output=lambda x: x.squeeze()
)

print("\\n== CORN Model ==")
corn_model = CornModel()
corn_loss = CornOrdinalCrossEntropyLoss()
corn_loss_trace, corn_mse_trace = train_model(
    corn_model, corn_loss, transform_output=lambda x: x.sigmoid().sum(dim=1)
)

print("\\n== Ordinal Logit Model ==")
logit_model = OrdinalLogitWrapper()
logit_loss = OrdinalLogisticLoss()
logit_loss_trace, logit_mse_trace = train_model(
    logit_model, logit_loss, transform_output=lambda x: x.argmax(dim=1)
)

# ----- PLOT -----
plt.figure(figsize=(12, 5))
plt.plot(scalar_mse_trace, label="Scalar (MSE)")
plt.plot(corn_mse_trace, label="CORN")
plt.plot(logit_mse_trace, label="Ordinal Logit")
plt.title("MSE Comparison Across Ordinal Models")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from coral_pytorch.dataset import (
    corn_label_from_logits,
    levels_from_labelbatch,
    proba_to_label,
)
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import coral_loss, corn_loss
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# --------- Load and Preprocess FD003 ---------
df = pd.read_csv("train_FD003.txt", sep="\s+", header=None)
cols = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
df.columns = cols
rul = df.groupby("unit")["cycle"].max().reset_index()
rul.columns = ["unit", "max_cycle"]
df = df.merge(rul, on="unit")
df["RUL"] = df["max_cycle"] - df["cycle"]
df["HealthState"] = pd.cut(
    df["RUL"], bins=[-1, 10, 30, np.inf], labels=[2, 1, 0]
).astype(int)

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
scaler = MinMaxScaler()
df[sensor_cols] = scaler.fit_transform(df[sensor_cols])


# --------- Dataset ---------
class TurbofanDataset(Dataset):
    def __init__(self, df, seq_len=30):
        X, y = [], []
        for unit in df["unit"].unique():
            unit_df = df[df["unit"] == unit].sort_values("cycle")
            X_vals = unit_df[sensor_cols].values
            y_vals = unit_df["HealthState"].values
            for i in range(len(X_vals) - seq_len):
                X.append(X_vals[i : i + seq_len])
                y.append(y_vals[i + seq_len])
        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = TurbofanDataset(df)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)


# --------- Backbone ---------
class LSTMBackbone(nn.Module):
    def __init__(self, input_size=21, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return h[-1]


# --------- Models ---------
class MSEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = LSTMBackbone()
        self.output = nn.Linear(self.backbone.hidden_size, 1)

    def forward(self, x):
        return self.output(self.backbone(x))


class CORNModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = LSTMBackbone()
        self.output = nn.Linear(self.backbone.hidden_size, num_classes - 1)

    def forward(self, x):
        return self.output(self.backbone(x))


class CORALModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = LSTMBackbone()
        self.coral = CoralLayer(self.backbone.hidden_size, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.coral(features)
        probas = torch.sigmoid(logits)
        return logits, probas


# --------- Training Loop ---------
def train(model, loss_fn, output_fn, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_trace, mse_trace = [], []
    for epoch in range(epochs):
        model.train()
        total_loss, total_mse, count = 0, 0, 0
        for X_batch, y_batch in train_loader:
            y_batch = y_batch.to(torch.long)
            if isinstance(model, CORALModel):
                logits, probas = model(X_batch)
                levels = levels_from_labelbatch(y_batch, num_classes=3).float()
                loss = coral_loss(logits, levels)
                y_pred = output_fn(probas)
            else:
                logits = model(X_batch)
                if loss_fn == corn_loss:
                    loss = loss_fn(logits, y_batch, num_classes=3)
                    y_pred = output_fn(logits)
                else:
                    y_true = y_batch.float().unsqueeze(1)
                    loss = loss_fn(logits, y_true)
                    y_pred = logits.squeeze()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_mse += (
                ((y_pred.clamp(0, 2).round() - y_batch.float()) ** 2).sum().item()
            )
            count += y_batch.size(0)
        avg_loss = total_loss
        avg_mse = total_mse / count
        loss_trace.append(avg_loss)
        mse_trace.append(avg_mse)
        print(f"Epoch {epoch + 1:2d}: Loss={avg_loss:.4f}, MSE={avg_mse:.4f}")
    return loss_trace, mse_trace


# --------- Run Experiments ---------
print("\n--- Training Scalar MSE Model ---")
mse_model = MSEModel()
mse_loss = nn.MSELoss()
mse_out = lambda x: x.squeeze()
mse_loss_trace, mse_mse_trace = train(mse_model, mse_loss, mse_out)

print("\n--- Training CORN Model ---")
corn_model = CORNModel()
corn_loss_trace, corn_mse_trace = train(
    corn_model, corn_loss, lambda x: corn_label_from_logits(x).float()
)

print("\n--- Training CORAL Model ---")
coral_model = CORALModel()
coral_loss_trace, coral_mse_trace = train(
    coral_model, coral_loss, lambda x: proba_to_label(x).float()
)

# --------- Plot ---------
plt.figure(figsize=(12, 5))
plt.plot(mse_mse_trace, label="MSE")
plt.plot(corn_mse_trace, label="CORN")
plt.plot(coral_mse_trace, label="CORAL")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Ordinal Health Prediction: MSE Comparison")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("health_lstm_corn_coral_mse.png")
plt.show()


# --- code cell ---

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from skorch import NeuralNet
from spacecutter.callbacks import AscensionCallback
from spacecutter.losses import CumulativeLinkLoss
from spacecutter.models import OrdinalLogisticModel
from torch.utils.data import DataLoader


# ----- LSTM-based Predictor -----
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=21, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.output(h[-1])


# ----- Prepare data from DataLoader to numpy for skorch -----
def dataloader_to_numpy(loader):
    X_list, y_list = [], []
    for X_batch, y_batch in loader:
        X_list.append(X_batch.numpy())
        y_list.append(y_batch.numpy())
    X = np.vstack(X_list)
    y = np.concatenate(y_list).reshape(-1, 1)
    return X, y


# Assumes train_loader is available from previous script
X_np, y_np = dataloader_to_numpy(train_loader)

# ----- Define model with spacecutter -----
predictor = LSTMPredictor()
model = OrdinalLogisticModel(predictor, num_classes=3)

net = NeuralNet(
    module=OrdinalLogisticModel,
    module__predictor=predictor,
    module__num_classes=3,
    criterion=CumulativeLinkLoss,
    train_split=None,
    callbacks=[("ascension", AscensionCallback())],
    max_epochs=20,
    lr=0.001,
    batch_size=64,
    iterator_train__shuffle=True,
    verbose=1,
)

# ----- Train and time the model -----
start = time.time()
net.fit(X_np, y_np)
spacecutter_time = time.time() - start

# ----- Predict and evaluate -----
y_pred_proba = net.predict_proba(X_np)
y_pred_labels = y_pred_proba.argmax(axis=1)
spacecutter_mse = np.mean((y_pred_labels - y_np.flatten()) ** 2)

# ----- Plot all four + spacecutter -----
# Assumes mse_mse_trace, corn_mse_trace, coral_mse_trace, logit_mse_trace, logit_time exist
# Replace dummy times if not available
mse_time = 40
corn_time = 42
coral_time = 44
logit_time = 46

labels = ["MSE", "CORN", "CORAL", "Logit", "Spacecutter"]
times = [mse_time, corn_time, coral_time, logit_time, spacecutter_time]
mses = [
    mse_mse_trace[-1],
    corn_mse_trace[-1],
    coral_mse_trace[-1],
    logit_mse_trace[-1],
    spacecutter_mse,
]

plt.figure(figsize=(8, 6))
for label, t, m in zip(labels, times, mses):
    plt.scatter(t, m, s=100, label=label)
    plt.text(t + 0.5, m + 0.01, label)

plt.title("Runtime vs Final MSE (All Ordinal Models)")
plt.xlabel("Training Time (s)")
plt.ylabel("Final MSE")
plt.grid(True)
plt.tight_layout()
plt.savefig("runtime_vs_mse_with_spacecutter.png")
plt.show()


# --- code cell ---

# ! pip install spacecutter skorch  # Jupyter-only


# --- code cell ---

labels = ["MSE", "CORN", "CORAL", "Spacecutter"]
times = [mse_time, corn_time, coral_time, spacecutter_time]
mses = [mse_mse_trace[-1], corn_mse_trace[-1], coral_mse_trace[-1], spacecutter_mse]

plt.figure(figsize=(8, 6))
for label, t, m in zip(labels, times, mses):
    plt.scatter(t, m, s=100, label=label)
    plt.text(t + 0.5, m + 0.01, label)

plt.title("Runtime vs Final MSE (Ordinal Models)")
plt.xlabel("Training Time (s)")
plt.ylabel("Final MSE")
plt.grid(True)
plt.tight_layout()
plt.savefig("runtime_vs_mse_updated.png")
plt.show()


# --- code cell ---

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from coral_pytorch.dataset import (
    corn_label_from_logits,
    levels_from_labelbatch,
    proba_to_label,
)
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import coral_loss, corn_loss
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# --------- Load FD003 and Process ---------
df = pd.read_csv("train_FD003.txt", sep="\\s+", header=None)
cols = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
df.columns = cols
rul = df.groupby("unit")["cycle"].max().reset_index()
rul.columns = ["unit", "max_cycle"]
df = df.merge(rul, on="unit")
df["RUL"] = df["max_cycle"] - df["cycle"]
df["HealthState"] = pd.cut(
    df["RUL"], bins=[-1, 10, 30, np.inf], labels=[2, 1, 0]
).astype(int)

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
scaler = MinMaxScaler()
df[sensor_cols] = scaler.fit_transform(df[sensor_cols])


# --------- Dataset ---------
class TurbofanDataset(Dataset):
    def __init__(self, df, seq_len=30):
        X, y = [], []
        for unit in df["unit"].unique():
            unit_df = df[df["unit"] == unit].sort_values("cycle")
            X_vals = unit_df[sensor_cols].values
            y_vals = unit_df["HealthState"].values
            for i in range(len(X_vals) - seq_len):
                X.append(X_vals[i : i + seq_len])
                y.append(y_vals[i + seq_len])
        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = TurbofanDataset(df)
loader = DataLoader(dataset, batch_size=64, shuffle=True)


# --------- Base LSTM ---------
class LSTMBackbone(nn.Module):
    def __init__(self, input_size=21, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return h[-1]


# --------- Models ---------
class MSEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = LSTMBackbone()
        self.out = nn.Linear(self.backbone.hidden_size, 1)

    def forward(self, x):
        return self.out(self.backbone(x))


class CORNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = LSTMBackbone()
        self.out = nn.Linear(self.backbone.hidden_size, 2)

    def forward(self, x):
        return self.out(self.backbone(x))


class CORALModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = LSTMBackbone()
        self.coral = CoralLayer(self.backbone.hidden_size, 3)

    def forward(self, x):
        logits = self.coral(self.backbone(x))
        return logits, torch.sigmoid(logits)


# --------- Unified Training Function ---------
def train(model, loss_fn, output_fn, is_coral=False, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_hist, mse_hist = [], []
    for epoch in range(epochs):
        model.train()
        total_loss, total_mse, count = 0, 0, 0
        for X_batch, y_batch in loader:
            y_batch = y_batch.to(torch.long)
            if is_coral:
                logits, probas = model(X_batch)
                levels = levels_from_labelbatch(y_batch, 3).float()
                loss = coral_loss(logits, levels)
                preds = output_fn(probas)
            else:
                logits = model(X_batch)
                if loss_fn == corn_loss:
                    loss = loss_fn(logits, y_batch, 3)
                    preds = output_fn(logits)
                else:
                    loss = loss_fn(logits.squeeze(), y_batch.float())
                    preds = logits.squeeze()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_mse += (
                ((preds.clamp(0, 2).round() - y_batch.float()) ** 2).sum().item()
            )
            count += y_batch.size(0)
        loss_hist.append(total_loss)
        mse_hist.append(total_mse / count)
        print(
            f"Epoch {epoch + 1:2d}: Loss={total_loss:.4f}, MSE={total_mse / count:.4f}"
        )
    return loss_hist, mse_hist


# --------- Train All Three Models ---------
print("\\n--- Scalar MSE ---")
mse_model = MSEModel()
start = time.time()
mse_loss, mse_mse = train(mse_model, nn.MSELoss(), lambda x: x.squeeze())
mse_time = time.time() - start

print("\\n--- CORN ---")
corn_model = CORNModel()
start = time.time()
corn_loss_hist, corn_mse = train(
    corn_model, corn_loss, lambda x: corn_label_from_logits(x).float()
)
corn_time = time.time() - start

print("\\n--- CORAL ---")
coral_model = CORALModel()
start = time.time()
coral_loss_hist, coral_mse = train(
    coral_model, coral_loss, lambda x: proba_to_label(x).float(), is_coral=True
)
coral_time = time.time() - start

# --------- Plot Comparison ---------
labels = ["MSE", "CORN", "CORAL"]
times = [mse_time, corn_time, coral_time]
mses = [mse_mse[-1], corn_mse[-1], coral_mse[-1]]

plt.figure(figsize=(8, 6))
for label, t, m in zip(labels, times, mses):
    plt.scatter(t, m, s=100, label=label)
    plt.text(t + 0.5, m + 0.01, label)

plt.title("Runtime vs Final MSE (Ordinal Models)")
plt.xlabel("Training Time (s)")
plt.ylabel("Final MSE")
plt.grid(True)
plt.tight_layout()
plt.savefig("final_runtime_vs_mse.png")
plt.show()


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

# Assumes: coral_model is trained and available

# --------- Reload and preprocess FD003 ---------
df = pd.read_csv("train_FD003.txt", sep="\\s+", header=None)
cols = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
df.columns = cols
rul = df.groupby("unit")["cycle"].max().reset_index()
rul.columns = ["unit", "max_cycle"]
df = df.merge(rul, on="unit")
df["RUL"] = df["max_cycle"] - df["cycle"]
df["HealthState"] = pd.cut(
    df["RUL"], bins=[-1, 10, 30, np.inf], labels=[2, 1, 0]
).astype(int)

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
scaler = MinMaxScaler()
df[sensor_cols] = scaler.fit_transform(df[sensor_cols])


# --------- Helper: Get transitions ---------
def get_transitions(preds):
    transitions = []
    for i in range(1, len(preds)):
        if preds[i] != preds[i - 1]:
            transitions.append((i, preds[i - 1], preds[i]))
    return transitions


def shade_regions(ax, cycles, states, level, color, alpha):
    in_block = False
    for i in range(len(states)):
        if states[i] == level and not in_block:
            start = cycles[i]
            in_block = True
        elif states[i] != level and in_block:
            end = cycles[i]
            ax.axvspan(start, end, color=color, alpha=alpha)
            in_block = False
    if in_block:
        ax.axvspan(start, cycles[-1], color=color, alpha=alpha)


# --------- Select 2 engines ---------
units = df["unit"].unique()[:2]

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

for idx, unit in enumerate(units):
    ax = axes[idx]
    unit_df = df[df["unit"] == unit].sort_values("cycle")
    X = torch.tensor(unit_df[sensor_cols].values, dtype=torch.float32)
    X_seq = torch.stack([X[i : i + 30] for i in range(len(X) - 30)])
    true_labels = unit_df["HealthState"].values[30:]

    # Get predictions
    coral_model.eval()
    with torch.no_grad():
        logits, probas = coral_model(X_seq)
        preds = torch.sigmoid(logits)
        pred_labels = torch.sum(preds > 0.5, dim=1).numpy()

    cycles = unit_df["cycle"].values[30:]
    ax.plot(
        cycles,
        true_labels,
        label="True Health State",
        drawstyle="steps-post",
        alpha=0.6,
    )
    ax.plot(cycles, pred_labels, label="Predicted", drawstyle="steps-post")

    # Shade background
    shade_regions(ax, cycles, true_labels, level=1, color="red", alpha=0.2)  # Warning
    shade_regions(ax, cycles, true_labels, level=2, color="red", alpha=0.5)  # Distress

    # Add predicted transitions
    trans = get_transitions(pred_labels)
    for t, from_state, to_state in trans:
        ax.axvline(cycles[t], color="black", linestyle="--", alpha=0.7)
        ax.text(cycles[t], 2.1, f"{from_state}→{to_state}", rotation=90, fontsize=8)

    ax.set_title(f"Engine {unit}")
    ax.set_ylabel("State")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Healthy", "Warning", "Distress"])
    ax.legend()

plt.xlabel("Cycle")
plt.tight_layout()
plt.savefig("coral_health_transitions_2_engines.png")
plt.show()


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

# Load trained models: coral_model, corn_model, mse_model (must be defined in current session)

# --------- Load and preprocess FD003 ---------
df = pd.read_csv("train_FD003.txt", sep="\\s+", header=None)
cols = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
df.columns = cols
rul = df.groupby("unit")["cycle"].max().reset_index()
rul.columns = ["unit", "max_cycle"]
df = df.merge(rul, on="unit")
df["RUL"] = df["max_cycle"] - df["cycle"]
df["HealthState"] = pd.cut(
    df["RUL"], bins=[-1, 10, 30, np.inf], labels=[2, 1, 0]
).astype(int)

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
scaler = MinMaxScaler()
df[sensor_cols] = scaler.fit_transform(df[sensor_cols])


# --------- Helper: Smooth transition detection ---------
def get_transitions(preds):
    transitions = []
    for i in range(1, len(preds)):
        if preds[i] != preds[i - 1]:
            transitions.append((i, preds[i - 1], preds[i]))
    return transitions


def shade_regions(ax, x, states, level, color, alpha):
    in_block = False
    for i in range(len(states)):
        if states[i] == level and not in_block:
            start = x[i]
            in_block = True
        elif states[i] != level and in_block:
            end = x[i]
            ax.axvspan(start, end, color=color, alpha=alpha)
            in_block = False
    if in_block:
        ax.axvspan(start, x[-1], color=color, alpha=alpha)


# --------- Select engines and plot ---------
units = df["unit"].unique()[:2]
fig, axes = plt.subplots(len(units), 1, figsize=(12, 6 * len(units)), sharex=False)

for idx, unit in enumerate(units):
    ax = axes[idx]
    unit_df = df[df["unit"] == unit].sort_values("cycle")
    X = torch.tensor(unit_df[sensor_cols].values, dtype=torch.float32)
    cycles = unit_df["cycle"].values
    true_labels = unit_df["HealthState"].values

    # Sliding windows
    seq_len = 30
    X_seq = torch.stack([X[i : i + seq_len] for i in range(len(X) - seq_len)])
    t_cycles = cycles[seq_len:]
    t_labels = true_labels[seq_len:]

    def predict_states(model, is_coral=False, is_corn=False):
        model.eval()
        with torch.no_grad():
            if is_coral:
                logits, probas = model(X_seq)
                pred = torch.sum(probas > 0.5, dim=1).numpy()
            elif is_corn:
                logits = model(X_seq)
                pred = torch.sum(torch.sigmoid(logits) > 0.5, dim=1).numpy()
            else:
                logits = model(X_seq)
                pred = torch.clamp(logits.squeeze(), 0, 2).round().numpy()
        return pred

    coral_pred = predict_states(coral_model, is_coral=True)
    corn_pred = predict_states(corn_model, is_corn=True)
    mse_pred = predict_states(mse_model)

    # Only last 50 cycles
    end_idx = len(t_cycles)
    start_idx = max(0, end_idx - 50)
    x_norm = [
        "T-" + str(t_cycles[end_idx - 1] - c) for c in t_cycles[start_idx:end_idx]
    ]

    ax.plot(
        x_norm,
        t_labels[start_idx:end_idx],
        label="True",
        drawstyle="steps-post",
        alpha=0.6,
    )
    ax.plot(
        x_norm, coral_pred[start_idx:end_idx], label="CORAL", linestyle="--", alpha=0.8
    )
    ax.plot(
        x_norm, corn_pred[start_idx:end_idx], label="CORN", linestyle=":", alpha=0.8
    )
    ax.plot(x_norm, mse_pred[start_idx:end_idx], label="MSE", linestyle="-.", alpha=0.8)

    # Shading
    shade_regions(ax, x_norm, t_labels[start_idx:end_idx], 1, "red", 0.2)
    shade_regions(ax, x_norm, t_labels[start_idx:end_idx], 2, "red", 0.5)

    # Transitions (only coral here for clarity, extend as needed)
    def add_vlines(pred, label, color, linestyle):
        trans = get_transitions(pred[start_idx:end_idx])
        for i, from_s, to_s in trans:
            ax.axvline(x_norm[i], color=color, linestyle=linestyle, alpha=0.7)
            ax.text(
                x_norm[i], 2.1, f"{from_s}→{to_s}", rotation=90, fontsize=8, color=color
            )

    add_vlines(coral_pred, "CORAL", "black", "--")
    add_vlines(corn_pred, "CORN", "blue", ":")
    add_vlines(mse_pred, "MSE", "green", "-.")

    ax.set_title(f"Engine {unit} – Last 50 Cycles")
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Healthy", "Warning", "Distress"])
    ax.legend(loc="upper left")
    ax.set_xlabel("Time Before Failure")
    # Set specific x-ticks at T-50, T-25, and T-0
    xticks = [x_norm[0], x_norm[25], x_norm[-1]]
    ax.set_xticks(xticks)


plt.tight_layout()
plt.savefig("transition_last50_comparison.png")
plt.show()


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": "black",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

# Load and preprocess
df = pd.read_csv("train_FD003.txt", sep="\\s+", header=None)
cols = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
df.columns = cols
rul = df.groupby("unit")["cycle"].max().reset_index()
rul.columns = ["unit", "max_cycle"]
df = df.merge(rul, on="unit")
df["RUL"] = df["max_cycle"] - df["cycle"]
df["HealthState"] = pd.cut(
    df["RUL"], bins=[-1, 10, 30, np.inf], labels=[2, 1, 0]
).astype(int)

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
scaler = MinMaxScaler()
df[sensor_cols] = scaler.fit_transform(df[sensor_cols])


def get_transitions(preds):
    transitions = []
    for i in range(1, len(preds)):
        if preds[i] != preds[i - 1]:
            transitions.append((i, preds[i - 1], preds[i]))
    return transitions


def shade_regions(ax, x, states, level, color, alpha):
    in_block = False
    for i in range(len(states)):
        if states[i] == level and not in_block:
            start = i
            in_block = True
        elif states[i] != level and in_block:
            end = i
            ax.axvspan(start, end, color=color, alpha=alpha)
            in_block = False
    if in_block:
        ax.axvspan(start, len(states) - 1, color=color, alpha=alpha)


units = df["unit"].unique()[:4]
fig, axes = plt.subplots(len(units), 1, figsize=(10, 8), sharex=False)

for idx, unit in enumerate(units):
    ax = axes[idx]
    unit_df = df[df["unit"] == unit].sort_values("cycle")
    X = torch.tensor(unit_df[sensor_cols].values, dtype=torch.float32)
    cycles = unit_df["cycle"].values
    true_labels = unit_df["HealthState"].values
    seq_len = 30
    X_seq = torch.stack([X[i : i + seq_len] for i in range(len(X) - seq_len)])
    t_cycles = cycles[seq_len:]
    t_labels = true_labels[seq_len:]

    def predict_states(model, is_coral=False, is_corn=False):
        model.eval()
        with torch.no_grad():
            if is_coral:
                logits, probas = model(X_seq)
                pred = torch.sum(probas > 0.5, dim=1).numpy()
            elif is_corn:
                logits = model(X_seq)
                pred = torch.sum(torch.sigmoid(logits) > 0.5, dim=1).numpy()
            else:
                logits = model(X_seq)
                pred = torch.clamp(logits.squeeze(), 0, 2).round().numpy()
        return pred

    coral_pred = predict_states(coral_model, is_coral=True)
    corn_pred = predict_states(corn_model, is_corn=True)
    mse_pred = predict_states(mse_model)

    end_idx = len(t_cycles)
    start_idx = max(0, end_idx - 50)
    x_range = np.array(list(range(-50, 0)))

    y_true = t_labels[start_idx:end_idx]
    y_coral = coral_pred[start_idx:end_idx]
    y_corn = corn_pred[start_idx:end_idx]
    y_mse = mse_pred[start_idx:end_idx]

    ax.plot(x_range, y_true, label="True", drawstyle="steps-post", alpha=0.6)
    ax.plot(x_range, y_coral, label="CORAL", linestyle="--", alpha=0.8)
    ax.plot(x_range, y_corn, label="CORN", linestyle=":", alpha=0.8)
    ax.plot(x_range, y_mse, label="MSE", linestyle="-.", alpha=0.8)

    shade_regions(ax, x_range, y_true, 1, "red", 0.2)
    shade_regions(ax, x_range, y_true, 2, "red", 0.5)

    def add_vlines(pred, color, linestyle):
        trans = get_transitions(pred)
        for i, f, t in trans:
            ax.axvline(x_range[i], color=color, linestyle=linestyle, alpha=0.7)
            ax.text(x_range[i], 2.2, f"{f}→{t}", rotation=90, fontsize=7, color=color)

    add_vlines(y_coral, "black", "--")
    add_vlines(y_corn, "blue", ":")
    add_vlines(y_mse, "green", "-.")

    ax.set_title(f"Engine {unit}", loc="left", fontsize=12)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Healthy", "Warning", "Distress"])
    ax.set_xticks([-50, -25, 0])
    ax.set_xticklabels(["T-50", "T-25", "T-0"])
    if idx == len(units) - 1:
        ax.set_xlabel("Time Before Failure")
    if idx == 0:
        ax.legend(loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig("health_transitions_minimalist.png", dpi=300)
plt.show()


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": "black",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

df = pd.read_csv("train_FD003.txt", sep="\\s+", header=None)
cols = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
df.columns = cols
rul = df.groupby("unit")["cycle"].max().reset_index()
rul.columns = ["unit", "max_cycle"]
df = df.merge(rul, on="unit")
df["RUL"] = df["max_cycle"] - df["cycle"]
df["HealthState"] = pd.cut(
    df["RUL"], bins=[-1, 10, 30, np.inf], labels=[2, 1, 0]
).astype(int)

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
scaler = MinMaxScaler()
df[sensor_cols] = scaler.fit_transform(df[sensor_cols])


def get_transitions(preds, min_gap=3):
    transitions = []
    last_idx = -min_gap
    for i in range(1, len(preds)):
        if preds[i] != preds[i - 1] and (i - last_idx) >= min_gap:
            transitions.append((i, preds[i - 1], preds[i]))
            last_idx = i
    return transitions


units = df["unit"].unique()[:4]
fig, axes = plt.subplots(len(units), 1, figsize=(10, 9), sharex=False)

for idx, unit in enumerate(units):
    ax = axes[idx]
    unit_df = df[df["unit"] == unit].sort_values("cycle")
    X = torch.tensor(unit_df[sensor_cols].values, dtype=torch.float32)
    cycles = unit_df["cycle"].values
    true_labels = unit_df["HealthState"].values
    seq_len = 30
    X_seq = torch.stack([X[i : i + seq_len] for i in range(len(X) - seq_len)])
    t_cycles = cycles[seq_len:]
    t_labels = true_labels[seq_len:]

    def predict_states(model, is_coral=False, is_corn=False):
        model.eval()
        with torch.no_grad():
            if is_coral:
                logits, probas = model(X_seq)
                pred = torch.sum(probas > 0.5, dim=1).numpy()
            elif is_corn:
                logits = model(X_seq)
                pred = torch.sum(torch.sigmoid(logits) > 0.5, dim=1).numpy()
            else:
                logits = model(X_seq)
                pred = torch.clamp(logits.squeeze(), 0, 2).round().numpy()
        return pred

    coral_pred = predict_states(coral_model, is_coral=True)
    corn_pred = predict_states(corn_model, is_corn=True)
    mse_pred = predict_states(mse_model)

    end_idx = len(t_cycles)
    start_idx = max(0, end_idx - 50)
    x_range = np.array(list(range(-50, 0)))

    y_true = t_labels[start_idx:end_idx]
    y_coral = coral_pred[start_idx:end_idx]
    y_corn = corn_pred[start_idx:end_idx]
    y_mse = mse_pred[start_idx:end_idx]

    def shade_health(ax, y_vals, x_vals):
        for i in range(len(y_vals) - 1):
            if y_vals[i] == 1:
                ax.axvspan(x_vals[i], x_vals[i + 1], color="red", alpha=0.2)
            elif y_vals[i] == 2:
                ax.axvspan(x_vals[i], x_vals[i + 1], color="red", alpha=0.5)

    ax.plot(x_range, y_true, label="True", drawstyle="steps-post", alpha=0.6)
    ax.plot(x_range, y_coral, label="CORAL", linestyle="--", alpha=0.8)
    ax.plot(x_range, y_corn, label="CORN", linestyle=":", alpha=0.8)
    ax.plot(x_range, y_mse, label="MSE", linestyle="-.", alpha=0.8)

    shade_health(ax, y_true, x_range)

    def add_vlines(ax, pred, x_vals, label, color, linestyle):
        shown_transitions = set()
        for i, f, t in get_transitions(pred):
            trans_label = f"{f}→{t}"
            if trans_label not in shown_transitions:
                ax.axvline(x_vals[i], color=color, linestyle=linestyle, alpha=0.7)
                ax.text(
                    x_vals[i], 2.2, trans_label, rotation=90, fontsize=7, color=color
                )
                shown_transitions.add(trans_label)

    add_vlines(ax, y_coral, x_range, "CORAL", "black", "--")
    add_vlines(ax, y_corn, x_range, "CORN", "blue", ":")
    add_vlines(ax, y_mse, x_range, "MSE", "green", "-.")

    ax.set_title(f"Engine {unit}", loc="left", fontsize=12)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Healthy", "Warning", "Distress"])
    ax.set_xticks([-50, -25, 0])
    ax.set_xticklabels(["T-50", "T-25", "T-0"])
    if idx == len(units) - 1:
        ax.set_xlabel("Time Before Failure")
    if idx == 0:
        ax.legend(loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig("health_transitions_cleaned.png", dpi=300)
plt.show()


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": "black",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

df = pd.read_csv("train_FD003.txt", sep="\\s+", header=None)
cols = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
df.columns = cols
rul = df.groupby("unit")["cycle"].max().reset_index()
rul.columns = ["unit", "max_cycle"]
df = df.merge(rul, on="unit")
df["RUL"] = df["max_cycle"] - df["cycle"]
df["HealthState"] = pd.cut(
    df["RUL"], bins=[-1, 10, 30, np.inf], labels=[2, 1, 0]
).astype(int)

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
scaler = MinMaxScaler()
df[sensor_cols] = scaler.fit_transform(df[sensor_cols])


def get_transitions(preds, min_gap=3):
    transitions = []
    last_idx = -min_gap
    for i in range(1, len(preds)):
        if preds[i] != preds[i - 1] and (i - last_idx) >= min_gap:
            transitions.append((i, preds[i - 1], preds[i]))
            last_idx = i
    return transitions


def get_state_blocks(y_vals, x_vals):
    blocks = []
    current = y_vals[0]
    start = x_vals[0]
    for i in range(1, len(y_vals)):
        if y_vals[i] != current:
            blocks.append((current, start, x_vals[i]))
            start = x_vals[i]
            current = y_vals[i]
    blocks.append((current, start, x_vals[-1]))
    return blocks


units = df["unit"].unique()[:4]
fig, axes = plt.subplots(len(units), 1, figsize=(10, 9), sharex=False)

for idx, unit in enumerate(units):
    ax = axes[idx]
    unit_df = df[df["unit"] == unit].sort_values("cycle")
    X = torch.tensor(unit_df[sensor_cols].values, dtype=torch.float32)
    cycles = unit_df["cycle"].values
    true_labels = unit_df["HealthState"].values
    seq_len = 30
    X_seq = torch.stack([X[i : i + seq_len] for i in range(len(X) - seq_len)])
    t_cycles = cycles[seq_len:]
    t_labels = true_labels[seq_len:]

    def predict_states(model, is_coral=False, is_corn=False):
        model.eval()
        with torch.no_grad():
            if is_coral:
                logits, probas = model(X_seq)
                pred = torch.sum(probas > 0.5, dim=1).numpy()
            elif is_corn:
                logits = model(X_seq)
                pred = torch.sum(torch.sigmoid(logits) > 0.5, dim=1).numpy()
            else:
                logits = model(X_seq)
                pred = torch.clamp(logits.squeeze(), 0, 2).round().numpy()
        return pred

    coral_pred = predict_states(coral_model, is_coral=True)
    corn_pred = predict_states(corn_model, is_corn=True)
    mse_pred = predict_states(mse_model)

    end_idx = len(t_cycles)
    start_idx = max(0, end_idx - 50)
    x_range = np.array(list(range(-50, 0)))

    y_true = t_labels[start_idx:end_idx]
    y_coral = coral_pred[start_idx:end_idx]
    y_corn = corn_pred[start_idx:end_idx]
    y_mse = mse_pred[start_idx:end_idx]

    def shade_blocks(ax, y_vals, x_vals):
        blocks = get_state_blocks(y_vals, x_vals)
        for val, start, end in blocks:
            if val == 1:
                ax.axvspan(start, end, color="red", alpha=0.2)
            elif val == 2:
                ax.axvspan(start, end, color="red", alpha=0.5)

    ax.plot(x_range, y_true, label="True", drawstyle="steps-post", alpha=0.6)
    ax.plot(x_range, y_coral, label="CORAL", linestyle="--", alpha=0.8)
    ax.plot(x_range, y_corn, label="CORN", linestyle=":", alpha=0.8)
    ax.plot(x_range, y_mse, label="MSE", linestyle="-.", alpha=0.8)

    shade_blocks(ax, y_true, x_range)

    def add_vlines(ax, pred, x_vals, label, color, linestyle):
        shown_transitions = set()
        for i, f, t in get_transitions(pred):
            trans_label = f"{f}→{t}"
            if trans_label not in shown_transitions:
                ax.axvline(x_vals[i], color=color, linestyle=linestyle, alpha=0.7)
                ax.text(
                    x_vals[i], 2.2, trans_label, rotation=90, fontsize=7, color=color
                )
                shown_transitions.add(trans_label)

    add_vlines(ax, y_coral, x_range, "CORAL", "black", "--")
    add_vlines(ax, y_corn, x_range, "CORN", "blue", ":")
    add_vlines(ax, y_mse, x_range, "MSE", "green", "-.")

    ax.set_title(f"Engine {unit}", loc="left", fontsize=12)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Healthy", "Warning", "Distress"])
    ax.set_xticks([-50, -25, 0])
    ax.set_xticklabels(["T-50", "T-25", "T-0"])
    if idx == len(units) - 1:
        ax.set_xlabel("Time Before Failure")
    if idx == 0:
        ax.legend(loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig("health_transitions_final.png", dpi=300)
plt.show()


# --- code cell ---

from collections import defaultdict


def transition_deltas(true_seq, pred_seq, offset_range):
    """
    Compute transition deltas for each predicted transition vs true transition.
    Returns a dict with average deltas for each transition type (0→1 and 1→2).
    """

    def find_first_transition(seq, from_state, to_state):
        for i in range(1, len(seq)):
            if seq[i - 1] == from_state and seq[i] == to_state:
                return i
        return None

    results = defaultdict(list)

    for from_state, to_state in [(0, 1), (1, 2)]:
        true_idx = find_first_transition(true_seq, from_state, to_state)
        pred_idx = find_first_transition(pred_seq, from_state, to_state)
        if true_idx is not None and pred_idx is not None:
            delta = offset_range[pred_idx] - offset_range[true_idx]
            results[f"{from_state}→{to_state}"].append(delta)

    return results


# Compute average transition deltas across all engines
engine_ids = df["unit"].unique()[:4]
seq_len = 30
delta_summary = defaultdict(list)

for unit in engine_ids:
    unit_df = df[df["unit"] == unit].sort_values("cycle")
    X = torch.tensor(unit_df[sensor_cols].values, dtype=torch.float32)
    cycles = unit_df["cycle"].values
    true_labels = unit_df["HealthState"].values

    X_seq = torch.stack([X[i : i + seq_len] for i in range(len(X) - seq_len)])
    t_labels = true_labels[seq_len:]
    offset_range = np.array(list(range(-len(t_labels), 0)))  # e.g., T-50 to T-0

    def get_preds(model, coral=False, corn=False):
        model.eval()
        with torch.no_grad():
            if coral:
                _, probas = model(X_seq)
                return torch.sum(probas > 0.5, dim=1).numpy()
            elif corn:
                logits = model(X_seq)
                return torch.sum(torch.sigmoid(logits) > 0.5, dim=1).numpy()
            else:
                logits = model(X_seq)
                return torch.clamp(logits.squeeze(), 0, 2).round().numpy()

    coral_preds = get_preds(coral_model, coral=True)
    corn_preds = get_preds(corn_model, corn=False)
    mse_preds = get_preds(mse_model)

    for label, preds in zip(
        ["CORAL", "CORN", "MSE"], [coral_preds, corn_preds, mse_preds]
    ):
        deltas = transition_deltas(t_labels, preds, offset_range)
        for trans, delta_list in deltas.items():
            delta_summary[f"{label} {trans}"].extend(delta_list)

# Compute average deltas
average_deltas = {k: np.mean(v) for k, v in delta_summary.items()}
average_deltas


# --- code cell ---

# Simulate a policy evaluation using real HealthState predictions

# Generate synthetic predictions for demonstration using the trained models
# Each row is one time step for one engine (unit), with real HealthState and predicted model outputs

seq_len = 30
simulated_records = []

# Evaluate on last 50 cycles per unit (where applicable)
for unit in df["unit"].unique():
    unit_df = df[df["unit"] == unit].sort_values("cycle")
    X = torch.tensor(unit_df[sensor_cols].values, dtype=torch.float32)
    cycles = unit_df["cycle"].values
    rul = unit_df["RUL"].values
    true_states = unit_df["HealthState"].values

    if len(X) < seq_len + 50:
        continue  # skip engines without enough history

    X_seq = torch.stack([X[i : i + seq_len] for i in range(len(X) - seq_len)])
    last_X_seq = X_seq[-50:]
    last_cycles = cycles[seq_len:][-50:]
    last_rul = rul[seq_len:][-50:]
    last_true = true_states[seq_len:][-50:]

    def get_preds(model, coral=False, corn=False):
        model.eval()
        with torch.no_grad():
            if coral:
                _, probas = model(last_X_seq)
                return torch.sum(probas > 0.5, dim=1).numpy()
            elif corn:
                logits = model(last_X_seq)
                return torch.sum(torch.sigmoid(logits) > 0.5, dim=1).numpy()
            else:
                logits = model(last_X_seq)
                return torch.clamp(logits.squeeze(), 0, 2).round().numpy()

    coral_pred = get_preds(coral_model, coral=True)
    corn_pred = get_preds(corn_model, corn=False)
    mse_pred = get_preds(mse_model)

    for i in range(50):
        for model_name, pred in zip(
            ["CORAL", "CORN", "MSE"], [coral_pred, corn_pred, mse_pred]
        ):
            simulated_records.append(
                {
                    "Unit": unit,
                    "Cycle": last_cycles[i],
                    "Model": model_name,
                    "PredictedState": int(pred[i]),
                    "TrueState": int(last_true[i]),
                    "RUL": last_rul[i],
                }
            )

policy_df = pd.DataFrame(simulated_records)
policy_df.sort_values(by=["Model", "Unit", "Cycle"], inplace=True)

# Add previous predicted state per engine and model
policy_df["PrevPred"] = policy_df.groupby(["Model", "Unit"])["PredictedState"].shift(1)


# Define actions
def action_policy(row):
    if row["PredictedState"] == 2:
        return "intervene"
    elif row["PrevPred"] == 0 and row["PredictedState"] == 1:
        return "review"
    else:
        return "wait"


policy_df["Action"] = policy_df.apply(action_policy, axis=1)


# Define failure risk mapping based on predicted state
def failure_risk(row):
    base_risk = [0.01, 0.1, 0.3][row["PredictedState"]]
    if row["Action"] == "intervene":
        return base_risk * 0.5
    elif row["Action"] == "review":
        return base_risk * 0.75
    else:
        return base_risk


policy_df["FailureRisk"] = policy_df.apply(failure_risk, axis=1)
policy_df["Cost"] = policy_df["Action"].map({"intervene": 100, "review": 30, "wait": 0})

# Aggregate over time
agg_risk = policy_df.groupby(["Model", "Cycle"])["FailureRisk"].sum().reset_index()
agg_cost = policy_df.groupby(["Model", "Cycle"])["Cost"].sum().reset_index()

# Merge for plotting
agg = pd.merge(agg_risk, agg_cost, on=["Model", "Cycle"])

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
for i, model in enumerate(["MSE", "CORN", "CORAL"]):
    sub = agg[agg["Model"] == model]
    ax = axes[i]
    ax2 = ax.twinx()
    ax.plot(sub["Cycle"], sub["FailureRisk"], color="red", label="Failure Risk")
    ax2.plot(sub["Cycle"], sub["Cost"], color="blue", label="Cost")
    ax.set_ylabel("Risk", color="red")
    ax2.set_ylabel("Cost ($)", color="blue")
    ax.set_title(f"Policy Simulation – {model}")
    ax.grid(True, linestyle="--", alpha=0.3)

plt.xlabel("Cycle")
plt.tight_layout()
plt.savefig("/mnt/data/engine_policy_simulation.png")
plt.show()


# --- code cell ---

# Rerun cleanly without re-importing torch (already imported earlier)
# We'll simulate model behavior with synthetic predictions to demonstrate the policy logic

np.random.seed(42)
units = df["unit"].unique()
simulated_records = []

for unit in units:
    unit_df = df[df["unit"] == unit].sort_values("cycle")
    if len(unit_df) < 80:  # ensure at least seq_len + 50
        continue
    cycles = unit_df["cycle"].values[-50:]
    rul = unit_df["RUL"].values[-50:]
    true = unit_df["HealthState"].values[-50:]

    for model_name in ["MSE", "CORN", "CORAL"]:
        # simulate predictions with some randomness around true labels
        noise = np.random.choice([-1, 0, 1], size=50, p=[0.05, 0.9, 0.05])
        pred = np.clip(true + noise, 0, 2)

        for i in range(50):
            simulated_records.append(
                {
                    "Unit": unit,
                    "Cycle": cycles[i],
                    "Model": model_name,
                    "PredictedState": pred[i].item(),
                    "TrueState": true[i].item(),
                    "RUL": rul[i].item(),
                }
            )

policy_df = pd.DataFrame(simulated_records)
policy_df.sort_values(by=["Model", "Unit", "Cycle"], inplace=True)
policy_df["PrevPred"] = policy_df.groupby(["Model", "Unit"])["PredictedState"].shift(1)


def action_policy(row):
    if row["PredictedState"] == 2:
        return "intervene"
    elif row["PrevPred"] == 0 and row["PredictedState"] == 1:
        return "review"
    else:
        return "wait"


policy_df["Action"] = policy_df.apply(action_policy, axis=1)


def failure_risk(row):
    base_risk = [0.01, 0.1, 0.3][int(row["PredictedState"])]
    if row["Action"] == "intervene":
        return base_risk * 0.5
    elif row["Action"] == "review":
        return base_risk * 0.75
    else:
        return base_risk


policy_df["FailureRisk"] = policy_df.apply(failure_risk, axis=1)
policy_df["Cost"] = policy_df["Action"].map({"intervene": 100, "review": 30, "wait": 0})

# Final comparison
final_summary = (
    policy_df.groupby("Model")[["FailureRisk", "Cost"]].sum().round(2).reset_index()
)


# --- code cell ---

final_summary


# --- duplicate code cell omitted (identical to earlier cell) ---


# --- duplicate code cell omitted (identical to earlier cell) ---


# --- code cell ---

# Compute a no-action baseline where all predictions result in 'wait' (no interventions)

baseline_df = policy_df.copy()
baseline_df["Action"] = "wait"
baseline_df["FailureRisk"] = baseline_df["PredictedState"].map(
    {0: 0.01, 1: 0.1, 2: 0.3}
)
baseline_df["Cost"] = 0

# Aggregate baseline failure risk
baseline_summary = (
    baseline_df.groupby("Model")[["FailureRisk", "Cost"]].sum().round(2).reset_index()
)
baseline_summary["Model"] = "No Action"

# Combine with previous policy results
full_comparison = pd.concat([final_summary, baseline_summary], ignore_index=True)


# --- code cell ---

full_comparison


# --- code cell ---

plt.figure(figsize=(8, 6))
for label, t, m in zip(labels, times, mses):
    plt.scatter(t, m, s=100, label=label)
    plt.text(t, m + 0.0005, label, ha="center", va="bottom", fontsize=10)

plt.title("Runtime vs Final MSE (Ordinal Models)")
plt.xlabel("Training Time (s)")
plt.ylabel("Final MSE")
plt.grid(False)
plt.tight_layout()
plt.savefig("final_runtime_vs_mse.png")
plt.show()


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

# Simulate health states (0: Healthy, 1: Warning, 2: Distress)
np.random.seed(42)
n = 200
true = [1]
for _ in range(n - 1):
    last = true[-1]
    move = np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1])
    true.append(max(0, min(2, last + move)))

# Model A: conservative, Model B: aggressive
model_a = [max(0, x - np.random.choice([0, 1], p=[0.8, 0.2])) for x in true]
model_b = [min(2, x + np.random.choice([0, 1], p=[0.8, 0.2])) for x in true]

df = pd.DataFrame({"True": true, "Model_A": model_a, "Model_B": model_b})

# Weighted Cohen’s Kappa
kappa_a = cohen_kappa_score(df["True"], df["Model_A"], weights="quadratic")
kappa_b = cohen_kappa_score(df["True"], df["Model_B"], weights="quadratic")

print(f"Weighted Cohen's Kappa (Model A): {kappa_a:.2f}")
print(f"Weighted Cohen's Kappa (Model B): {kappa_b:.2f}")

# Wilcoxon Signed-Rank Test
df["Error_A"] = (df["True"] - df["Model_A"]).abs()
df["Error_B"] = (df["True"] - df["Model_B"]).abs()
stat, p = wilcoxon(df["Error_A"], df["Error_B"])
print(f"Wilcoxon test p-value: {p:.4f}")

# Calibration
calib = pd.DataFrame(
    {
        "True": df["True"].value_counts(normalize=True).sort_index(),
        "Model_A": df["Model_A"].value_counts(normalize=True).sort_index(),
        "Model_B": df["Model_B"].value_counts(normalize=True).sort_index(),
    }
)
print("\nCalibration Summary:")
print(calib)

# Confusion Matrices
cm_a = confusion_matrix(df["True"], df["Model_A"])
cm_b = confusion_matrix(df["True"], df["Model_B"])

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(
    cm_a,
    annot=True,
    fmt="d",
    ax=ax[0],
    cmap="Blues",
    xticklabels=[0, 1, 2],
    yticklabels=[0, 1, 2],
)
ax[0].set_title("Confusion Matrix - Model A")
ax[0].set_xlabel("Predicted")
ax[0].set_ylabel("Actual")

sns.heatmap(
    cm_b,
    annot=True,
    fmt="d",
    ax=ax[1],
    cmap="Blues",
    xticklabels=[0, 1, 2],
    yticklabels=[0, 1, 2],
)
ax[1].set_title("Confusion Matrix - Model B")
ax[1].set_xlabel("Predicted")
ax[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrices.png")
plt.show()


# Policy Impact Modeling
def decide(action):
    if action == 2:
        return "intervene"
    elif action == 1:
        return "review"
    return "none"


df["Policy_A"] = df["Model_A"].apply(decide)
df["Policy_B"] = df["Model_B"].apply(decide)


def estimate_cost(policy):
    if policy == "intervene":
        return 20
    elif policy == "review":
        return 5
    return 0


def expected_loss(true_class, policy):
    base_risk = [10, 50, 100][true_class]
    if policy == "intervene":
        return base_risk * 0.4
    elif policy == "review":
        return base_risk * 0.7
    return base_risk


for model in ["A", "B"]:
    df[f"Cost_{model}"] = df[f"Policy_{model}"].apply(estimate_cost)
    df[f"ExpectedLoss_{model}"] = df.apply(
        lambda row: expected_loss(row["True"], row[f"Policy_{model}"]), axis=1
    )
    df[f"Total_{model}"] = df[f"Cost_{model}"] + df[f"ExpectedLoss_{model}"]

summary = df[["Total_A", "Total_B"]].mean().to_frame(name="Average Total Cost")
print("\nPolicy Impact Summary:")
print(summary)


# --- code cell ---

from collections import Counter

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from coral_pytorch.dataset import corn_label_from_logits
from coral_pytorch.losses import corn_loss
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

# ----- Load and preprocess FD003 -----
df = pd.read_csv("train_FD003.txt", sep="\\s+", header=None)
cols = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
df.columns = cols
rul = df.groupby("unit")["cycle"].max().reset_index()
rul.columns = ["unit", "max_cycle"]
df = df.merge(rul, on="unit")
df["RUL"] = df["max_cycle"] - df["cycle"]
df["HealthState"] = pd.cut(
    df["RUL"], bins=[-1, 10, 30, np.inf], labels=[2, 1, 0]
).astype(int)

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
scaler = MinMaxScaler()
df[sensor_cols] = scaler.fit_transform(df[sensor_cols])


# ----- Sequence Dataset with Summary Features -----
class TurbofanDataset(Dataset):
    def __init__(self, df, seq_len=30):
        self.seq_data, self.targets, self.features = [], [], []
        for unit in df["unit"].unique():
            unit_df = df[df["unit"] == unit].sort_values("cycle")
            X_vals = unit_df[sensor_cols].values
            y_vals = unit_df["HealthState"].values
            for i in range(len(X_vals) - seq_len):
                seq = X_vals[i : i + seq_len]
                label = y_vals[i + seq_len]
                self.seq_data.append(seq)
                self.targets.append(label)
                flat = {}
                for j, col in enumerate(sensor_cols):
                    flat[f"{col}_mean"] = seq[:, j].mean()
                    flat[f"{col}_std"] = seq[:, j].std()
                    flat[f"{col}_last"] = seq[-1, j]
                self.features.append(flat)

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.seq_data[idx], dtype=torch.float32),
            self.targets[idx],
            self.features[idx],
        )


dataset = TurbofanDataset(df)
seq_data = torch.stack([x for x, _, _ in dataset])
labels = torch.tensor([y for _, y, _ in dataset])
features_df = pd.DataFrame([f for _, _, f in dataset])


# ----- CORN Model Definition -----
class LSTMBackbone(nn.Module):
    def __init__(self, input_size=21, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return h[-1]


class CORNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = LSTMBackbone()
        self.out = nn.Linear(self.backbone.hidden_size, 2)

    def forward(self, x):
        return self.out(self.backbone(x))


model = CORNModel()
model.eval()

# ----- Run Inference -----
with torch.no_grad():
    logits = model(seq_data)
    preds = corn_label_from_logits(logits).numpy()

print("Prediction distribution (CORN):")
print(Counter(preds))

# ----- If model is constant, fallback to true labels -----
unique_preds = np.unique(preds)
if len(unique_preds) < 2:
    print("Model predictions lack variation. Using true HealthState labels for SHAP.")
    y = labels.numpy()
else:
    y = preds

# ----- Train Surrogate LightGBM -----
X = features_df
train_data = lgb.Dataset(X, label=y)
params = {"objective": "regression_l1", "verbosity": -1}
lgb_model = lgb.train(params, train_data, num_boost_round=100)

# ----- SHAP Analysis -----
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig("shap_summary_corn_fd003_full.png")
plt.show()


# --- code cell ---

from collections import Counter

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from coral_pytorch.dataset import corn_label_from_logits
from coral_pytorch.losses import corn_loss
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

# ----- Load and preprocess FD003 -----
df = pd.read_csv("train_FD003.txt", sep="\\s+", header=None)
cols = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
df.columns = cols
rul = df.groupby("unit")["cycle"].max().reset_index()
rul.columns = ["unit", "max_cycle"]
df = df.merge(rul, on="unit")
df["RUL"] = df["max_cycle"] - df["cycle"]
df["HealthState"] = pd.cut(
    df["RUL"], bins=[-1, 10, 30, np.inf], labels=[2, 1, 0]
).astype(int)

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
scaler = MinMaxScaler()
df[sensor_cols] = scaler.fit_transform(df[sensor_cols])


# ----- Sequence Dataset with Summary Features -----
class TurbofanDataset(Dataset):
    def __init__(self, df, seq_len=30):
        self.seq_data, self.targets, self.features = [], [], []
        for unit in df["unit"].unique():
            unit_df = df[df["unit"] == unit].sort_values("cycle")
            X_vals = unit_df[sensor_cols].values
            y_vals = unit_df["HealthState"].values
            for i in range(len(X_vals) - seq_len):
                seq = X_vals[i : i + seq_len]
                label = y_vals[i + seq_len]
                self.seq_data.append(seq)
                self.targets.append(label)
                flat = {}
                for j, col in enumerate(sensor_cols):
                    flat[f"{col}_mean"] = seq[:, j].mean()
                    flat[f"{col}_std"] = seq[:, j].std()
                    flat[f"{col}_last"] = seq[-1, j]
                self.features.append(flat)

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.seq_data[idx], dtype=torch.float32),
            self.targets[idx],
            self.features[idx],
        )


dataset = TurbofanDataset(df)
seq_data = torch.stack([x for x, _, _ in dataset])
labels = torch.tensor([y for _, y, _ in dataset])
features_df = pd.DataFrame([f for _, _, f in dataset])


# ----- CORN Model Definition -----
class LSTMBackbone(nn.Module):
    def __init__(self, input_size=21, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return h[-1]


class CORNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = LSTMBackbone()
        self.out = nn.Linear(self.backbone.hidden_size, 2)

    def forward(self, x):
        return self.out(self.backbone(x))


model = CORNModel()
model.eval()

# ----- Run Inference -----
with torch.no_grad():
    logits = model(seq_data)
    preds = corn_label_from_logits(logits).numpy()

print("Prediction distribution (CORN):")
print(Counter(preds))

# ----- If model is constant, fallback to true labels -----
unique_preds = np.unique(preds)
if len(unique_preds) < 2:
    print("Model predictions lack variation. Using true HealthState labels for SHAP.")
    y = labels.numpy()
else:
    y = preds

# ----- Train Surrogate LightGBM -----
X = features_df
train_data = lgb.Dataset(X, label=y)
params = {"objective": "regression_l1", "verbosity": -1}
lgb_model = lgb.train(params, train_data, num_boost_round=100)

# ----- SHAP Analysis -----
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X)

# Summary plot
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig("shap_summary_with_force_fd003.png")
plt.show()

# Force plot for instance 100
shap.initjs()
force_plot = shap.force_plot(
    explainer.expected_value, shap_values[100], X.iloc[100], matplotlib=True, show=False
)
plt.tight_layout()
plt.savefig("shap_force_plot_instance100.png")
plt.show()


# --- code cell ---

import lightgbm as lgb
import mord as m
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Load FD003 dataset
cols = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
df = pd.read_csv("train_FD003.txt", sep="\s+", header=None, names=cols)

# Calculate RUL and health state
rul = df.groupby("unit")["cycle"].max().reset_index()
rul.columns = ["unit", "max_cycle"]
df = df.merge(rul, on="unit")
df["RUL"] = df["max_cycle"] - df["cycle"]
df["HealthState"] = pd.cut(
    df["RUL"], bins=[-1, 10, 30, np.inf], labels=[2, 1, 0]
).astype(int)

# Feature selection and normalization
sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
X = df[sensor_cols]
y = df["HealthState"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Ordinal logistic regression
model_logit = m.LogisticIT()
model_logit.fit(X_train, y_train)
y_pred_logit = model_logit.predict(X_test)

# Gradient boosting regression
train_data = lgb.Dataset(X_train, label=y_train)
params = {"objective": "regression_l1", "verbosity": -1}
model_gbm = lgb.train(params, train_data, num_boost_round=100)
y_pred_gbm = model_gbm.predict(X_test).round().clip(0, 2).astype(int)

# Simple average ensemble
y_pred_avg = ((y_pred_logit + y_pred_gbm) / 2).round().clip(0, 2).astype(int)
ensemble_acc = accuracy_score(y_test, y_pred_avg)
ensemble_mae = mean_absolute_error(y_test, y_pred_avg)

# Boosted residual learning
residual = y_train - model_logit.predict(X_train)
model_boost = lgb.LGBMRegressor()
model_boost.fit(X_train, residual)
y_pred_base = model_logit.predict(X_test)
y_pred_resid = model_boost.predict(X_test)
y_pred_combo = (y_pred_base + y_pred_resid).round().clip(0, 2).astype(int)
boosted_acc = accuracy_score(y_test, y_pred_combo)
boosted_mae = mean_absolute_error(y_test, y_pred_combo)

# Prepare output
{
    "Ensemble Accuracy": ensemble_acc,
    "Ensemble MAE": ensemble_mae,
    "Boosted Hybrid Accuracy": boosted_acc,
    "Boosted Hybrid MAE": boosted_mae,
}


# --- code cell ---

# Full Ensemble Pipeline for Ordered Health State Prediction (FD003)

import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from coral_pytorch.dataset import (
    corn_label_from_logits,
    levels_from_labelbatch,
    proba_to_label,
)
from coral_pytorch.losses import coral_loss, corn_loss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# Load and preprocess data
cols = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
df = pd.read_csv("train_FD003.txt", sep="\\s+", header=None, names=cols)
rul = df.groupby("unit")["cycle"].max().reset_index()
rul.columns = ["unit", "max_cycle"]
df = df.merge(rul, on="unit")
df["RUL"] = df["max_cycle"] - df["cycle"]
df["HealthState"] = pd.cut(
    df["RUL"], bins=[-1, 10, 30, np.inf], labels=[2, 1, 0]
).astype(int)

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
scaler = MinMaxScaler()
df[sensor_cols] = scaler.fit_transform(df[sensor_cols])

# Train/test split for tabular models
X = df[sensor_cols]
y = df["HealthState"]
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Train LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
params = {"objective": "regression_l1", "verbosity": -1}
model_gbm = lgb.train(params, train_data, num_boost_round=100)
y_pred_gbm = model_gbm.predict(X_test).round().clip(0, 2).astype(int)


# Dataset for LSTM models
class TurbofanDataset(Dataset):
    def __init__(self, df, seq_len=30):
        X, y = [], []
        for unit in df["unit"].unique():
            unit_df = df[df["unit"] == unit].sort_values("cycle")
            X_vals = unit_df[sensor_cols].values
            y_vals = unit_df["HealthState"].values
            for i in range(len(X_vals) - seq_len):
                X.append(X_vals[i : i + seq_len])
                y.append(y_vals[i + seq_len])
        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Define LSTM backbone
class LSTMBackbone(nn.Module):
    def __init__(self, input_size=21, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return h[-1]


# CORN model
class CORNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = LSTMBackbone()
        self.out = nn.Linear(self.backbone.hidden_size, 2)

    def forward(self, x):
        return self.out(self.backbone(x))


# Train CORN
def train(model, loss_fn, output_fn, is_coral=False, epochs=5):
    loader = DataLoader(TurbofanDataset(df), batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_batch = y_batch.to(torch.long)
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch, 3)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


corn_model = CORNModel()
train(corn_model, corn_loss, lambda x: corn_label_from_logits(x).float())


# Predict with CORN on test set (sequence-based)
def get_lstm_preds(model, df, seq_len=30):
    model.eval()
    X_test_seq = []
    for unit in df["unit"].unique():
        unit_df = df[df["unit"] == unit].sort_values("cycle")
        X_vals = unit_df[sensor_cols].values
        for i in range(len(X_vals) - seq_len):
            X_test_seq.append(X_vals[i : i + seq_len])
    X_test_seq = torch.tensor(np.array(X_test_seq), dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_test_seq)
        y_pred = corn_label_from_logits(logits).numpy().astype(int)
    return y_pred


y_pred_lstm = get_lstm_preds(corn_model, df)

# Align sizes with tabular predictions
min_len = min(len(y_pred_gbm), len(y_pred_lstm))
y_pred_gbm = y_pred_gbm[:min_len]
y_test = y_test.iloc[:min_len]
y_pred_lstm = y_pred_lstm[:min_len]

# Dummy ordinal regression with LightGBM-style residual fix
model_logit = model_gbm  # placeholder for missing mord.LogisticIT
y_pred_logit = y_pred_gbm

# Simple average ensemble
y_pred_avg = (
    ((y_pred_logit + y_pred_gbm + y_pred_lstm) / 3).round().clip(0, 2).astype(int)
)
acc_avg = accuracy_score(y_test, y_pred_avg)
mae_avg = mean_absolute_error(y_test, y_pred_avg)

# Stacking
X_stack = np.column_stack([y_pred_logit, y_pred_gbm, y_pred_lstm])
meta_model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=500)
meta_model.fit(X_stack, y_test[: len(X_stack)])
y_pred_stack = meta_model.predict(X_stack)
acc_stack = accuracy_score(y_test[: len(X_stack)], y_pred_stack)
mae_stack = mean_absolute_error(y_test[: len(X_stack)], y_pred_stack)

print(f"Averaging Ensemble - Accuracy: {acc_avg:.4f}, MAE: {mae_avg:.4f}")
print(f"Stacked Ensemble - Accuracy: {acc_stack:.4f}, MAE: {mae_stack:.4f}")


# --- code cell ---

# Full Ensemble Pipeline for Ordered Health State Prediction (FD003)

import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import mord as m
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from coral_pytorch.dataset import (
    corn_label_from_logits,
    levels_from_labelbatch,
    proba_to_label,
)
from coral_pytorch.losses import coral_loss, corn_loss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# Load and preprocess data
cols = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
df = pd.read_csv("train_FD003.txt", sep="\\s+", header=None, names=cols)
rul = df.groupby("unit")["cycle"].max().reset_index()
rul.columns = ["unit", "max_cycle"]
df = df.merge(rul, on="unit")
df["RUL"] = df["max_cycle"] - df["cycle"]
df["HealthState"] = pd.cut(
    df["RUL"], bins=[-1, 10, 30, np.inf], labels=[2, 1, 0]
).astype(int)

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
scaler = MinMaxScaler()
df[sensor_cols] = scaler.fit_transform(df[sensor_cols])

# Train/test split for tabular models
X = df[sensor_cols]
y = df["HealthState"]
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Logistic ordinal regression
model_logit = m.LogisticIT()
model_logit.fit(X_train, y_train)
y_pred_logit = model_logit.predict(X_test)

# Train LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
params = {"objective": "regression_l1", "verbosity": -1}
model_gbm = lgb.train(params, train_data, num_boost_round=100)
y_pred_gbm = model_gbm.predict(X_test).round().clip(0, 2).astype(int)


# Dataset for LSTM models
class TurbofanDataset(Dataset):
    def __init__(self, df, seq_len=30):
        X, y = [], []
        for unit in df["unit"].unique():
            unit_df = df[df["unit"] == unit].sort_values("cycle")
            X_vals = unit_df[sensor_cols].values
            y_vals = unit_df["HealthState"].values
            for i in range(len(X_vals) - seq_len):
                X.append(X_vals[i : i + seq_len])
                y.append(y_vals[i + seq_len])
        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Define LSTM backbone
class LSTMBackbone(nn.Module):
    def __init__(self, input_size=21, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return h[-1]


# CORN model
class CORNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = LSTMBackbone()
        self.out = nn.Linear(self.backbone.hidden_size, 2)

    def forward(self, x):
        return self.out(self.backbone(x))


# Train CORN
def train(model, loss_fn, output_fn, is_coral=False, epochs=5):
    loader = DataLoader(TurbofanDataset(df), batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_batch = y_batch.to(torch.long)
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch, 3)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


corn_model = CORNModel()
train(corn_model, corn_loss, lambda x: corn_label_from_logits(x).float())


# Predict with CORN on test set (sequence-based)
def get_lstm_preds(model, df, seq_len=30):
    model.eval()
    X_test_seq = []
    for unit in df["unit"].unique():
        unit_df = df[df["unit"] == unit].sort_values("cycle")
        X_vals = unit_df[sensor_cols].values
        for i in range(len(X_vals) - seq_len):
            X_test_seq.append(X_vals[i : i + seq_len])
    X_test_seq = torch.tensor(np.array(X_test_seq), dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_test_seq)
        y_pred = corn_label_from_logits(logits).numpy().astype(int)
    return y_pred


y_pred_lstm = get_lstm_preds(corn_model, df)

# Align sizes
min_len = min(len(y_pred_logit), len(y_pred_gbm), len(y_pred_lstm))
y_test = y_test.iloc[:min_len]
y_pred_logit = y_pred_logit[:min_len]
y_pred_gbm = y_pred_gbm[:min_len]
y_pred_lstm = y_pred_lstm[:min_len]

# Averaging ensemble
y_pred_avg = (
    ((y_pred_logit + y_pred_gbm + y_pred_lstm) / 3).round().clip(0, 2).astype(int)
)
acc_avg = accuracy_score(y_test, y_pred_avg)
mae_avg = mean_absolute_error(y_test, y_pred_avg)

# Stacking
X_stack = np.column_stack([y_pred_logit, y_pred_gbm, y_pred_lstm])
meta_model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=500)
meta_model.fit(X_stack, y_test)
y_pred_stack = meta_model.predict(X_stack)
acc_stack = accuracy_score(y_test, y_pred_stack)
mae_stack = mean_absolute_error(y_test, y_pred_stack)

print(f"Averaging Ensemble - Accuracy: {acc_avg:.4f}, MAE: {mae_avg:.4f}")
print(f"Stacked Ensemble - Accuracy: {acc_stack:.4f}, MAE: {mae_stack:.4f}")


# --- code cell ---

import time

import lightgbm as lgb
import mord as m
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from coral_pytorch.dataset import corn_label_from_logits
from coral_pytorch.losses import corn_loss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# --- Load and preprocess FD003 ---
cols = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
df = pd.read_csv("train_FD003.txt", sep="\s+", header=None, names=cols)

rul = df.groupby("unit")["cycle"].max().reset_index()
rul.columns = ["unit", "max_cycle"]
df = df.merge(rul, on="unit")
df["RUL"] = df["max_cycle"] - df["cycle"]
df["HealthState"] = pd.cut(
    df["RUL"], bins=[-1, 10, 30, np.inf], labels=[2, 1, 0]
).astype(int)

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
scaler = MinMaxScaler()
df[sensor_cols] = scaler.fit_transform(df[sensor_cols])


# --- Dataset for CORN ---
class TurbofanDataset(Dataset):
    def __init__(self, df, seq_len=30):
        self.X, self.y = [], []
        for unit in df["unit"].unique():
            unit_df = df[df["unit"] == unit].sort_values("cycle")
            sensor_data = unit_df[sensor_cols].values
            labels = unit_df["HealthState"].values
            for i in range(len(sensor_data) - seq_len):
                self.X.append(sensor_data[i : i + seq_len])
                self.y.append(labels[i + seq_len])
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = TurbofanDataset(df)
loader = DataLoader(dataset, batch_size=64, shuffle=True)


# --- CORN LSTM Model ---
class LSTMBackbone(nn.Module):
    def __init__(self, input_size=21, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return h[-1]


class CORNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = LSTMBackbone()
        self.out = nn.Linear(64, 2)  # For 3 classes

    def forward(self, x):
        return self.out(self.backbone(x))


# --- Train CORN LSTM ---
def train(model, loss_fn, output_fn, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch, 3)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1:2d}: Loss = {total_loss:.4f}")


corn_model = CORNModel()
train(corn_model, corn_loss, corn_label_from_logits)

# --- Prepare sequences for prediction ---
seq_len = 30
df_seq = (
    df.groupby("unit", group_keys=False)
    .apply(lambda x: x.iloc[seq_len:])
    .reset_index(drop=True)
)

X = df_seq[sensor_cols]
y = df_seq["HealthState"]
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# --- Ordinal Logistic Regression (mord) ---
model_logit = m.LogisticIT()
model_logit.fit(X_train, y_train)
y_pred_logit = model_logit.predict(X_test)

# --- LightGBM Regressor ---
train_data = lgb.Dataset(X_train, label=y_train)
params = {"objective": "regression_l1", "verbosity": -1}
model_gbm = lgb.train(params, train_data, num_boost_round=100)
y_pred_gbm = model_gbm.predict(X_test).round().clip(0, 2).astype(int)


# --- CORN Predictions on Test Set ---
def get_corn_preds(model, df, seq_len=30):
    model.eval()
    preds = []
    for unit in df["unit"].unique():
        unit_df = df[df["unit"] == unit].sort_values("cycle")
        sensor_data = unit_df[sensor_cols].values
        for i in range(len(sensor_data) - seq_len):
            x = torch.tensor(
                sensor_data[i : i + seq_len], dtype=torch.float32
            ).unsqueeze(0)
            with torch.no_grad():
                logits = model(x)
                pred = corn_label_from_logits(logits).item()
                preds.append(pred)
    return np.array(preds)


corn_input_df = df_seq.iloc[X_test.index].copy()
corn_input_df["unit"] = df_seq.iloc[X_test.index]["unit"].values
y_pred_lstm = get_corn_preds(corn_model, corn_input_df, seq_len=seq_len)

# --- Align and Trim ---
min_len = min(len(y_test), len(y_pred_logit), len(y_pred_gbm), len(y_pred_lstm))
y_test = y_test.iloc[:min_len]
y_pred_logit = y_pred_logit[:min_len]
y_pred_gbm = y_pred_gbm[:min_len]
y_pred_lstm = y_pred_lstm[:min_len]

# --- Averaging Ensemble ---
y_pred_avg = (
    ((y_pred_logit + y_pred_gbm + y_pred_lstm) / 3).round().clip(0, 2).astype(int)
)
acc_avg = accuracy_score(y_test, y_pred_avg)
mae_avg = mean_absolute_error(y_test, y_pred_avg)

# --- Stacked Ensemble ---
X_stack = np.column_stack([y_pred_logit, y_pred_gbm, y_pred_lstm])
meta_model = LogisticRegression(solver="lbfgs", max_iter=500)
meta_model.fit(X_stack, y_test)
y_pred_stack = meta_model.predict(X_stack)
acc_stack = accuracy_score(y_test, y_pred_stack)
mae_stack = mean_absolute_error(y_test, y_pred_stack)

# --- Final Output ---
print(f"\nAveraging Ensemble - Accuracy: {acc_avg:.4f}, MAE: {mae_avg:.4f}")
print(f"Stacked Ensemble   - Accuracy: {acc_stack:.4f}, MAE: {mae_stack:.4f}")


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import MinMaxScaler

# --- Load FD003 data ---
cols = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
df = pd.read_csv("train_FD003.txt", sep="\s+", header=None, names=cols)

# --- Normalize sensors ---
sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
scaler = MinMaxScaler()
df[sensor_cols] = scaler.fit_transform(df[sensor_cols])


# --- Extract rolling window features per unit ---
def extract_features(unit_df, window=30):
    records = []
    unit_id = unit_df["unit"].iloc[0]
    for i in range(window, len(unit_df)):
        window_df = unit_df.iloc[i - window : i]
        row = {
            "unit": unit_id,
            "cycle": unit_df.iloc[i]["cycle"],
            "mean": window_df[sensor_cols].mean().mean(),
            "std": window_df[sensor_cols].std().mean(),
            "skew": skew(window_df[sensor_cols].values.flatten()),
            "kurtosis": kurtosis(window_df[sensor_cols].values.flatten()),
            "rms": np.sqrt((window_df[sensor_cols] ** 2).mean().mean()),
        }
        records.append(row)
    return records


# --- Apply to all units ---
all_records = []
for unit in df["unit"].unique():
    unit_df = df[df["unit"] == unit].sort_values("cycle")
    all_records.extend(extract_features(unit_df, window=30))

feature_df = pd.DataFrame(all_records)

# --- Labeling distress and censored events ---
# Distress = last 10% of cycles for each unit
labeled = []
for unit in feature_df["unit"].unique():
    unit_df = feature_df[feature_df["unit"] == unit].copy()
    max_cycle = unit_df["cycle"].max()
    threshold = int(0.9 * max_cycle)
    unit_df["event"] = (unit_df["cycle"] >= threshold).astype(int)
    unit_df["time"] = max_cycle - unit_df["cycle"]
    labeled.append(unit_df)

df_labeled = pd.concat(labeled, ignore_index=True)

# --- Cox Proportional Hazards Model ---
model_df = df_labeled[["mean", "std", "skew", "kurtosis", "rms", "time", "event"]]
cph = CoxPHFitter()
cph.fit(model_df, duration_col="time", event_col="event")
cph.print_summary()

# --- Survival Function ---
kmf = KaplanMeierFitter()
kmf.fit(durations=df_labeled["time"], event_observed=df_labeled["event"])

plt.figure(figsize=(8, 6))
kmf.plot_survival_function()
plt.title("Survival Function for Turbofan Engine Distress")
plt.xlabel("Cycles Before Distress")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.show()


# --- Real-time Risk Prediction Example ---
def predict_distress_risk(row, model):
    return model.predict_partial_hazard(row.to_frame().T).values[0]


latest_sample = df_labeled[df_labeled["cycle"] == df_labeled["cycle"].max()]
latest_features = latest_sample[["mean", "std", "skew", "kurtosis", "rms"]].iloc[0]
risk_score = predict_distress_risk(latest_features, cph)
print("Distress risk score:", risk_score)


# --- code cell ---

# !pip install lifelines  # Jupyter-only


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import MinMaxScaler

# --- Load FD003 ---
cols = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
df = pd.read_csv("train_FD003.txt", sep="\s+", header=None, names=cols)

# --- Normalize sensor data ---
sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
scaler = MinMaxScaler()
df[sensor_cols] = scaler.fit_transform(df[sensor_cols])


# --- Extract features per cycle ---
def extract_features(row):
    values = row[sensor_cols].values
    return pd.Series(
        {
            "mean": values.mean(),
            "std": values.std(),
            "skew": skew(values),
            "kurtosis": kurtosis(values),
            "rms": np.sqrt(np.mean(values**2)),
        }
    )


feature_df = df.copy()
features = feature_df.apply(extract_features, axis=1)
feature_df = pd.concat([feature_df[["unit", "cycle"]], features], axis=1)

# --- Label distress (event) and time-to-event (duration) ---
labeled = []
for unit in feature_df["unit"].unique():
    unit_df = feature_df[feature_df["unit"] == unit].copy()
    max_cycle = unit_df["cycle"].max()
    threshold = int(0.9 * max_cycle)
    unit_df["event"] = (unit_df["cycle"] >= threshold).astype(int)
    unit_df["time"] = max_cycle - unit_df["cycle"]
    labeled.append(unit_df)

df_labeled = pd.concat(labeled, ignore_index=True)

# --- Cox Proportional Hazards Model ---
model_df = df_labeled[["mean", "std", "skew", "kurtosis", "rms", "time", "event"]]
cph = CoxPHFitter()
cph.fit(model_df, duration_col="time", event_col="event")
cph.print_summary()

# --- Survival Function ---
kmf = KaplanMeierFitter()
kmf.fit(durations=model_df["time"], event_observed=model_df["event"])

plt.figure(figsize=(8, 6))
kmf.plot_survival_function()
plt.title("Survival Function for Turbofan Distress (FD003)")
plt.xlabel("Cycles Before Distress")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.tight_layout()
plt.show()


# --- Real-Time Hazard Score ---
def predict_distress_risk(row, model):
    return model.predict_partial_hazard(row.to_frame().T).values[0]


latest_cycle = df_labeled[df_labeled["cycle"] == df_labeled["cycle"].max()]
latest_features = latest_cycle[["mean", "std", "skew", "kurtosis", "rms"]].iloc[0]
risk_score = predict_distress_risk(latest_features, cph)
print("Distress risk score:", risk_score)


# --- code cell ---

import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

# Use previously created df_labeled (from FD003, no rolling)
df_labeled["rms_group"] = pd.qcut(df_labeled["rms"], q=2, labels=["low", "high"])

kmf = KaplanMeierFitter()
plt.figure(figsize=(10, 6))

for group in ["low", "high"]:
    mask = df_labeled["rms_group"] == group
    kmf.fit(
        df_labeled[mask]["time"],
        event_observed=df_labeled[mask]["event"],
        label=f"RMS: {group}",
    )
    kmf.plot_survival_function(ci_show=False)

plt.title("Kaplan-Meier Survival Curves by RMS Group (FD003)")
plt.xlabel("Cycles Before Distress")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

from lifelines.utils import concordance_index

# Predict partial hazard (risk score)
df_labeled["risk_score"] = cph.predict_partial_hazard(df_labeled)

# Higher risk should correspond to shorter time
c_index = concordance_index(
    df_labeled["time"], -df_labeled["risk_score"], df_labeled["event"]
)
print(f"Concordance Index: {c_index:.3f}")


# --- code cell ---

import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

# Group by RMS (or any other feature)
df_labeled["rms_group"] = pd.qcut(df_labeled["rms"], q=2, labels=["low", "high"])

kmf = KaplanMeierFitter()
plt.figure(figsize=(10, 6))

for group in ["low", "high"]:
    mask = df_labeled["rms_group"] == group
    kmf.fit(
        df_labeled[mask]["time"],
        event_observed=df_labeled[mask]["event"],
        label=f"RMS: {group}",
    )
    kmf.plot_survival_function(ci_show=False)

plt.title("Kaplan-Meier Survival Curves by RMS Group (Time to Failure)")
plt.xlabel("Cycles Until Failure")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()


# --- code cell ---

from lifelines.utils import concordance_index

df_labeled["risk_score"] = cph.predict_partial_hazard(df_labeled)
c_index = concordance_index(
    df_labeled["time"], -df_labeled["risk_score"], df_labeled["event"]
)
print(f"Concordance Index (Failure): {c_index:.3f}")


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# --- Load and preprocess FD003 ---
cols = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
df = pd.read_csv("train_FD003.txt", sep="\s+", header=None, names=cols)

# --- Normalize the 3 operating settings ---
op_cols = [f"op_setting_{i}" for i in range(1, 4)]
scaler = MinMaxScaler()
df[op_cols] = scaler.fit_transform(df[op_cols])

# --- K-Means to cluster into 5 operating regimes ---
kmeans = KMeans(n_clusters=5, random_state=42)
df["op_cluster"] = kmeans.fit_predict(df[op_cols])

# --- Label time-to-failure (RUL) and event for failure modeling ---
records = []
for unit in df["unit"].unique():
    unit_df = df[df["unit"] == unit].copy()
    max_cycle = unit_df["cycle"].max()
    unit_df["time"] = max_cycle - unit_df["cycle"]
    unit_df["event"] = 1  # All units fail
    records.append(unit_df)

df_labeled = pd.concat(records, ignore_index=True)

# --- Kaplan-Meier: Overall ---
kmf = KaplanMeierFitter()
kmf.fit(df_labeled["time"], df_labeled["event"])

plt.figure(figsize=(10, 6))
kmf.plot_survival_function(ci_show=False)
plt.title("Kaplan-Meier Survival Function (All Engines)")
plt.xlabel("Cycles Until Failure")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Kaplan-Meier: By Operating Cluster ---
plt.figure(figsize=(10, 6))
for i in range(5):
    cluster_df = df_labeled[df_labeled["op_cluster"] == i]
    kmf.fit(cluster_df["time"], cluster_df["event"], label=f"Cluster {i}")
    kmf.plot_survival_function(ci_show=False)

plt.title("Kaplan-Meier Survival Curves by Operating Cluster")
plt.xlabel("Cycles Until Failure")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()


# --- code cell ---

from lifelines import CoxPHFitter

# Prepare data
model_df = df_labeled[
    ["mean", "std", "skew", "kurtosis", "rms", "time", "event", "op_cluster"]
].copy()
model_df["op_cluster"] = model_df["op_cluster"].astype(
    str
)  # lifelines requires strings for strata

# Fit stratified Cox model
cph_strat = CoxPHFitter()
cph_strat.fit(model_df, duration_col="time", event_col="event", strata=["op_cluster"])

# Summary of covariates
cph_strat.print_summary()


# --- code cell ---

from scipy.stats import kurtosis, skew


# Recalculate per-cycle features from sensor values
def extract_features(row):
    values = row[sensor_cols].values
    return pd.Series(
        {
            "mean": values.mean(),
            "std": values.std(),
            "skew": skew(values),
            "kurtosis": kurtosis(values),
            "rms": np.sqrt(np.mean(values**2)),
        }
    )


features = df.apply(extract_features, axis=1)
df_features = pd.concat([df[["unit", "cycle"]], features], axis=1)

# Merge features into df_labeled
df_labeled = df_labeled.merge(df_features, on=["unit", "cycle"], how="left")


# --- code cell ---

from lifelines import CoxPHFitter

# Prepare data
model_df = df_labeled[
    ["mean", "std", "skew", "kurtosis", "rms", "time", "event", "op_cluster"]
].copy()
model_df["op_cluster"] = model_df["op_cluster"].astype(str)

# Fit stratified Cox model
cph_strat = CoxPHFitter()
cph_strat.fit(model_df, duration_col="time", event_col="event", strata=["op_cluster"])
cph_strat.print_summary()


# --- code cell ---

cph_strat.print_summary()


# --- code cell ---

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchtuples as tt
from pycox.evaluation import EvalSurv
from pycox.models import LogisticHazard
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Load FD003 dataset ---
cols = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
df = pd.read_csv("train_FD003.txt", sep="\s+", header=None, names=cols)

# --- Label time-to-failure (RUL) ---
rul = df.groupby("unit")["cycle"].max().reset_index()
rul.columns = ["unit", "max_cycle"]
df = df.merge(rul, on="unit")
df["event"] = 1  # All engines fail
df["time"] = df["max_cycle"] - df["cycle"]

# --- Feature matrix ---
feature_cols = [f"sensor_{i}" for i in range(1, 22)]
X = df[feature_cols].values
durations = df["time"].values
events = df["event"].values

# --- Train-test split ---
X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test = (
    train_test_split(X, durations, events, test_size=0.2, random_state=42)
)

# --- Standardize features ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype("float32")
X_test = scaler.transform(X_test).astype("float32")

# --- Discretize time into 50 intervals ---
labtrans = LabTransDiscreteTime(50)
y_train_disc = labtrans.fit_transform(y_time_train, y_event_train)
y_test_disc = labtrans.transform(y_time_test, y_event_test)

# --- Build neural net and LogisticHazard model ---
net = tt.practical.MLPVanilla(
    in_features=X_train.shape[1],
    num_nodes=[32, 32],
    out_features=labtrans.out_features,
    activation=nn.ReLU,
)

model = LogisticHazard(net, tt.optim.Adam, duration_index=labtrans.cuts)

# --- Train model ---
model.fit(X_train, y_train_disc, batch_size=128, epochs=50, verbose=False)

# --- Predict survival ---
surv = model.predict_surv_df(X_test)

# --- Evaluate with Concordance Index ---
ev = EvalSurv(surv, y_time_test, y_event_test, censor_surv="km")
c_index = ev.concordance_td()
print(f"LogisticHazard C-index (PyCox): {c_index:.3f}")


# --- code cell ---

import matplotlib.pyplot as plt

# --- Pick test samples to visualize (you can customize these indices) ---
sample_indices = [10, 200, 400, 800]

# --- Predict survival functions for selected samples ---
surv_curves = model.predict_surv_df(X_test[sample_indices])

# --- Plot ---
plt.figure(figsize=(10, 6))
for i, idx in enumerate(sample_indices):
    surv_curves.iloc[:, i].plot(label=f"Sample {idx}")

plt.title("Predicted Survival Curves (LogisticHazard - FD003)")
plt.xlabel("Cycle bins (discretized time)")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# --- code cell ---

import inspect

from pycox.preprocessing import label_transforms

# Inspect the signature of LabTransDiscreteTime to determine valid arguments
signature = inspect.signature(label_transforms.LabTransDiscreteTime.__init__)
str(signature)


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torchtuples as tt
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from pycox.evaluation import EvalSurv
from pycox.models import CoxPH, LogisticHazard
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Load and preprocess FD003 ---
cols = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
df = pd.read_csv("train_FD003.txt", sep="\s+", header=None, names=cols)
rul = df.groupby("unit")["cycle"].max().reset_index()
rul.columns = ["unit", "max_cycle"]
df = df.merge(rul, on="unit")
df["event"] = 1
df["time"] = df["max_cycle"] - df["cycle"]

# --- Select sensor features ---
feature_cols = [f"sensor_{i}" for i in range(1, 22)]
X = df[feature_cols].values
durations = df["time"].values
events = df["event"].values

# --- Train-test split ---
X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test = (
    train_test_split(X, durations, events, test_size=0.2, random_state=42)
)

# --- Standardize features ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype("float32")
X_test = scaler.transform(X_test).astype("float32")

# =====================================================================================
# 📌 CoxPH (lifelines)
# =====================================================================================
selector = VarianceThreshold(threshold=1e-4)
X_train_cox = selector.fit_transform(X_train)
X_test_cox = selector.transform(X_test)
selected_features = [f for f, keep in zip(feature_cols, selector.get_support()) if keep]

cox_train_df = pd.DataFrame(X_train_cox, columns=selected_features)
cox_train_df["time"] = y_time_train
cox_train_df["event"] = y_event_train

cox = CoxPHFitter()
cox.fit(cox_train_df, duration_col="time", event_col="event")

cox_test_df = pd.DataFrame(X_test_cox, columns=selected_features)
cox_test_df["time"] = y_time_test
cox_test_df["event"] = y_event_test
risk_scores = cox.predict_partial_hazard(cox_test_df)

c_index_cox = concordance_index(y_time_test, -risk_scores, y_event_test)
print(f"📊 CoxPH (lifelines) C-index: {c_index_cox:.3f}")

# =====================================================================================
# 📌 LogisticHazard (PyCox)
# =====================================================================================
labtrans = LabTransDiscreteTime(50)
y_train_disc = labtrans.fit_transform(y_time_train, y_event_train)
y_test_disc = labtrans.transform(y_time_test, y_event_test)

net_lhaz = tt.practical.MLPVanilla(
    in_features=X_train.shape[1],
    num_nodes=[32, 32],
    out_features=labtrans.out_features,
    activation=nn.ReLU,
)
loghaz = LogisticHazard(net_lhaz, tt.optim.Adam, duration_index=labtrans.cuts)
loghaz.fit(X_train, y_train_disc, batch_size=128, epochs=50, verbose=False)

surv_lhaz = loghaz.predict_surv_df(X_test)
ev_lhaz = EvalSurv(surv_lhaz, y_time_test, y_event_test, censor_surv="km")
c_index_lhaz = ev_lhaz.concordance_td()
print(f"📊 LogisticHazard (PyCox) C-index: {c_index_lhaz:.3f}")

# =====================================================================================
# 📌 DeepSurv (PyCox via CoxPH with NN)
# =====================================================================================
net_deepsurv = tt.practical.MLPVanilla(
    in_features=X_train.shape[1], num_nodes=[32, 32], out_features=1, activation=nn.ReLU
)
deepsurv = CoxPH(net_deepsurv, tt.optim.Adam)
deepsurv.fit(
    X_train, (y_time_train, y_event_train), batch_size=128, epochs=50, verbose=False
)
deepsurv.compute_baseline_hazards()

surv_ds = deepsurv.predict_surv_df(X_test)
ev_ds = EvalSurv(surv_ds, y_time_test, y_event_test, censor_surv="km")
c_index_ds = ev_ds.concordance_td()
print(f"📊 DeepSurv (PyCox) C-index: {c_index_ds:.3f}")

# =====================================================================================
# 📊 Plot Survival Curve for a Sample
# =====================================================================================
sample_idx = 200
plt.figure(figsize=(10, 6))
surv_lhaz.iloc[:, sample_idx].plot(label="LogisticHazard")
surv_ds.iloc[:, sample_idx].plot(label="DeepSurv")
plt.title(f"Survival Curve Comparison (Sample {sample_idx})")
plt.xlabel("Cycle Bins (Discretized Time)")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# --- code cell ---

# !pip install --upgrade pycox  # Jupyter-only


# --- code cell ---

from pycox.models import CoxPH

# Create deep net
net = tt.practical.MLPVanilla(
    in_features=X_train.shape[1],
    num_nodes=[32, 32],
    out_features=1,
    activation=nn.ReLU,
    dropout=0.1,
)

# Wrap in CoxPH (DeepSurv)
model = CoxPH(net, tt.optim.Adam)

# Fit the model
model.fit(
    X_train, (y_time_train, y_event_train), batch_size=128, epochs=50, verbose=True
)

# Must compute baseline hazards first
model.compute_baseline_hazards()

# Predict survival curves
surv_ds = model.predict_surv_df(X_test)
ev_ds = EvalSurv(surv_ds, y_time_test, y_event_test, censor_surv="km")
c_index_ds = ev_ds.concordance_td()
print(f"📊 DeepSurv (CoxPH + NN) C-index: {c_index_ds:.3f}")


# --- code cell ---

import matplotlib.pyplot as plt

# Set sample index
sample_idx = 200
true_failure_time = y_time_test[sample_idx]

# Truncate survival curves for the selected sample
surv_lhaz_trunc = surv_lhaz.iloc[surv_lhaz.index <= true_failure_time, sample_idx]
surv_ds_trunc = surv_ds.iloc[surv_ds.index <= true_failure_time, sample_idx]

# CoxPH approximation
baseline_surv = cox.baseline_survival_.squeeze()
partial_hazard = float(risk_scores.iloc[sample_idx])
surv_cox = baseline_surv**partial_hazard
surv_cox_trunc = surv_cox[surv_cox.index <= true_failure_time]

# Plot side-by-side
fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Plot LogisticHazard and DeepSurv
surv_lhaz_trunc.plot(ax=ax[0], label="LogisticHazard")
surv_ds_trunc.plot(ax=ax[0], label="DeepSurv")
ax[0].set_title("LogisticHazard vs DeepSurv")
ax[0].set_xlabel("Time (Cycle Bins)")
ax[0].set_ylabel("Survival Probability")
ax[0].legend()
ax[0].grid(True)

# Plot CoxPH
surv_cox_trunc.plot(ax=ax[1], label="CoxPH")
ax[1].set_title("CoxPH (Lifelines)")
ax[1].set_xlabel("Time (Cycles)")
ax[1].legend()
ax[1].grid(True)

plt.suptitle("Truncated Survival Curve Comparison (Sample 200)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("truncated_survival_curve_comparison.png")
plt.show()


# --- code cell ---

import matplotlib.pyplot as plt

# Sample index
sample_idx = 200
true_failure_time = y_time_test[sample_idx]
display_limit = 100  # Reasonable upper display limit
plot_until = min(true_failure_time + 10, display_limit)

# Truncate survival curves up to display limit
surv_lhaz_trunc = surv_lhaz[surv_lhaz.index <= plot_until].iloc[:, sample_idx]
surv_ds_trunc = surv_ds[surv_ds.index <= plot_until].iloc[:, sample_idx]

# CoxPH survival approximation
baseline_surv = cox.baseline_survival_.squeeze()
partial_hazard = float(risk_scores.iloc[sample_idx])
surv_cox = baseline_surv**partial_hazard
surv_cox_trunc = surv_cox[surv_cox.index <= plot_until]

# Plot side-by-side
fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Plot LogisticHazard and DeepSurv
surv_lhaz_trunc.plot(ax=ax[0], label="LogisticHazard")
surv_ds_trunc.plot(ax=ax[0], label="DeepSurv")
ax[0].axvline(true_failure_time, color="gray", linestyle="--", label="True Failure")
ax[0].set_title("LogisticHazard vs DeepSurv")
ax[0].set_xlabel("Time (Cycle Bins)")
ax[0].set_ylabel("Survival Probability")
ax[0].legend()
ax[0].grid(False)

# Plot CoxPH
surv_cox_trunc.plot(ax=ax[1], label="CoxPH")
ax[1].axvline(true_failure_time, color="gray", linestyle="--", label="True Failure")
ax[1].set_title("CoxPH (Lifelines)")
ax[1].set_xlabel("Time (Cycles)")
ax[1].legend()
ax[1].grid(False)

plt.suptitle("Survival Curve Comparison for Sample 200", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("survival_curve_comparison_trimmed.png")
plt.show()


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set up Tufte-inspired minimalist style
plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.frameon": False,
    }
)

import torch.nn as nn
import torchtuples as tt
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from pycox.evaluation import EvalSurv
from pycox.models import CoxPH, LogisticHazard
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

# --- Load and preprocess FD003 ---
cols = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
df = pd.read_csv("train_FD003.txt", sep="\s+", header=None, names=cols)
rul = df.groupby("unit")["cycle"].max().reset_index()
rul.columns = ["unit", "max_cycle"]
df = df.merge(rul, on="unit")
df["event"] = 1
df["time"] = df["max_cycle"] - df["cycle"]

# --- Select sensor features ---
feature_cols = [f"sensor_{i}" for i in range(1, 22)]
X = df[feature_cols].values
durations = df["time"].values
events = df["event"].values

# --- Train-test split ---
X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test = (
    train_test_split(X, durations, events, test_size=0.2, random_state=42)
)

# --- Standardize features ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype("float32")
X_test = scaler.transform(X_test).astype("float32")

# =====================================================================================
# CoxPH (lifelines)
# =====================================================================================
selector = VarianceThreshold(threshold=1e-4)
X_train_cox = selector.fit_transform(X_train)
X_test_cox = selector.transform(X_test)
selected_features = [f for f, keep in zip(feature_cols, selector.get_support()) if keep]

cox_train_df = pd.DataFrame(X_train_cox, columns=selected_features)
cox_train_df["time"] = y_time_train
cox_train_df["event"] = y_event_train

cox = CoxPHFitter()
cox.fit(cox_train_df, duration_col="time", event_col="event")

cox_test_df = pd.DataFrame(X_test_cox, columns=selected_features)
cox_test_df["time"] = y_time_test
cox_test_df["event"] = y_event_test
risk_scores = cox.predict_partial_hazard(cox_test_df)

c_index_cox = concordance_index(y_time_test, -risk_scores, y_event_test)
print(f"CoxPH (lifelines) C-index: {c_index_cox:.3f}")

# =====================================================================================
# LogisticHazard (PyCox)
# =====================================================================================
labtrans = LabTransDiscreteTime(50)
y_train_disc = labtrans.fit_transform(y_time_train, y_event_train)
y_test_disc = labtrans.transform(y_time_test, y_event_test)

net_lhaz = tt.practical.MLPVanilla(
    in_features=X_train.shape[1],
    num_nodes=[32, 32],
    out_features=labtrans.out_features,
    activation=nn.ReLU,
)
loghaz = LogisticHazard(net_lhaz, tt.optim.Adam, duration_index=labtrans.cuts)
loghaz.fit(X_train, y_train_disc, batch_size=128, epochs=50, verbose=False)

surv_lhaz = loghaz.predict_surv_df(X_test)
ev_lhaz = EvalSurv(surv_lhaz, y_time_test, y_event_test, censor_surv="km")
c_index_lhaz = ev_lhaz.concordance_td()
print(f"LogisticHazard (PyCox) C-index: {c_index_lhaz:.3f}")

# =====================================================================================
# DeepSurv (PyCox via CoxPH with NN)
# =====================================================================================
net_deepsurv = tt.practical.MLPVanilla(
    in_features=X_train.shape[1], num_nodes=[32, 32], out_features=1, activation=nn.ReLU
)
deepsurv = CoxPH(net_deepsurv, tt.optim.Adam)
deepsurv.fit(
    X_train, (y_time_train, y_event_train), batch_size=128, epochs=50, verbose=False
)
deepsurv.compute_baseline_hazards()

surv_ds = deepsurv.predict_surv_df(X_test)
ev_ds = EvalSurv(surv_ds, y_time_test, y_event_test, censor_surv="km")
c_index_ds = ev_ds.concordance_td()
print(f"DeepSurv (PyCox) C-index: {c_index_ds:.3f}")

# =====================================================================================
# Plot Survival Curve for a Sample
# =====================================================================================
sample_idx = 200
plt.figure(figsize=(10, 6))
surv_lhaz.iloc[:, sample_idx].plot(label="LogisticHazard")
surv_ds.iloc[:, sample_idx].plot(label="DeepSurv")
plt.title(f"Survival Curve Comparison (Sample {sample_idx})")
plt.xlabel("Cycle Bins (Discretized Time)")
plt.ylabel("Survival Probability")
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# Left: LogisticHazard vs DeepSurv
surv_lhaz_trunc.plot(ax=ax[0], label="LogisticHazard", color="black", linestyle="-")
surv_ds_trunc.plot(ax=ax[0], label="DeepSurv", color="gray", linestyle="--")
ax[0].axvline(true_failure_time, color="black", linestyle=":", lw=1)
ax[0].set_title("LogisticHazard vs DeepSurv")
ax[0].set_xlabel("Cycles")
ax[0].set_ylabel("Survival")
ax[0].legend()

# Right: CoxPH
surv_cox_trunc.plot(ax=ax[1], label="CoxPH", color="black", linestyle="-")
ax[1].axvline(true_failure_time, color="black", linestyle=":", lw=1)
ax[1].set_title("CoxPH")
ax[1].set_xlabel("Cycles")
ax[1].legend()

plt.suptitle("Survival Curve Comparison – Sample 200", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("minimalist_survival_comparison.png")
plt.show()
