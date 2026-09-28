import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Ustawienie ziarna losowości
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Używane urządzenie: {device}")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Wczytanie danych
df_train = pd.read_parquet(PROCESSED_DIR / "train_2022.parquet")
df_test = pd.read_parquet(PROCESSED_DIR / "test_2022.parquet")

features = [
    'temp_era5',
    'altitude',
    'urban_fraction_500m',
    'sin_hour',
    'cos_hour',
    'sin_doy',
    'cos_doy'
]
target = 'temp_ground'

X_train_raw = df_train[features].values
y_train_raw = df_train[target].values
era5_train = df_train['temp_era5'].values

X_test_raw = df_test[features].values
y_test_raw = df_test[target].values
era5_test = df_test['temp_era5'].values

# Cel sieci: modelujemy rezyduum (różnicę): delta = temp_ground - temp_era5
delta_train = y_train_raw - era5_train
delta_test = y_test_raw - era5_test

# 2. Skalowanie cech (fit TYLKO na treningowym!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)
joblib.dump(scaler, MODELS_DIR / "deep_learning_scaler.joblib")

# 3. Dataset i DataLoader
class ClimateDataset(Dataset):
    def __init__(self, X, delta):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.delta = torch.tensor(delta, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.delta[idx]

train_dataset = ClimateDataset(X_train_scaled, delta_train)
test_dataset = ClimateDataset(X_test_scaled, delta_test)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# 4. Architektura Residual MLP
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.act = nn.GELU()
        
    def forward(self, x):
        return self.act(x + self.block(x))

class DownscalingResMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_blocks=3, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        self.out_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        h = self.in_proj(x)
        for b in self.res_blocks:
            h = b(h)
        return self.out_head(h)

model = DownscalingResMLP(input_dim=len(features), hidden_dim=128, num_blocks=3, dropout=0.1).to(device)

# 5. Trening
criterion = nn.HuberLoss(delta=1.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

epochs = 35
print("\nRozpoczynanie treningu Residual MLP...")
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    for batch_x, batch_delta in train_loader:
        batch_x, batch_delta = batch_x.to(device), batch_delta.to(device)
        
        optimizer.zero_grad()
        pred_delta = model(batch_x)
        loss = criterion(pred_delta, batch_delta)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_x)
        
    train_loss = total_loss / len(train_dataset)
    scheduler.step(train_loss)
    
    if epoch % 5 == 0 or epoch == epochs:
        print(f"Epoch [{epoch:02d}/{epochs}] - Loss (Huber): {train_loss:.5f} - LR: {optimizer.param_groups[0]['lr']:.6f}")

# 6. Ewaluacja na zbiorze testowym
model.eval()
preds_delta = []
with torch.no_grad():
    for batch_x, _ in test_loader:
        batch_x = batch_x.to(device)
        preds_delta.append(model(batch_x).cpu().numpy())

pred_delta_all = np.vstack(preds_delta).flatten()

# Rekonstrukcja końcowej temperatury: T_pred = T_era5 + delta
y_pred_mlp = era5_test + pred_delta_all

# Metryki
mae_mlp = mean_absolute_error(y_test_raw, y_pred_mlp)
rmse_mlp = np.sqrt(mean_squared_error(y_test_raw, y_pred_mlp))
r2_mlp = r2_score(y_test_raw, y_pred_mlp)
bias_mlp = (y_pred_mlp - y_test_raw).mean()

# Pobranie wyników poprzednich modeli
mae_era5 = mean_absolute_error(y_test_raw, era5_test)
rmse_era5 = np.sqrt(mean_squared_error(y_test_raw, era5_test))
r2_era5 = r2_score(y_test_raw, era5_test)
bias_era5 = (era5_test - y_test_raw).mean()

print("\n" + "="*70)
print("📊 ZESTAWIENIE MODELI NA ZBIORZE TESTOWYM (42 DNI)")
print("="*70)
comp = pd.DataFrame([
    {'Model': 'Baseline 0 (Surowe ERA5)', 'MAE [°C]': round(mae_era5, 3), 'RMSE [°C]': round(rmse_era5, 3), 'R²': round(r2_era5, 4), 'BIAS [°C]': round(bias_era5, 3)},
    {'Model': 'Random Forest (Baseline ML)', 'MAE [°C]': 0.470, 'RMSE [°C]': 0.811, 'R²': 0.9845, 'BIAS [°C]': 0.022},
    {'Model': 'Deep Learning (Residual MLP)', 'MAE [°C]': round(mae_mlp, 3), 'RMSE [°C]': round(rmse_mlp, 3), 'R²': round(r2_mlp, 4), 'BIAS [°C]': round(bias_mlp, 3)}
])
print(comp.to_string(index=False))
print("="*70)

# Zapis wag modelu
torch.save(model.state_dict(), MODELS_DIR / "residual_mlp_downscaling.pth")
print(f"✅ Zapisano model do: {MODELS_DIR / 'residual_mlp_downscaling.pth'}")