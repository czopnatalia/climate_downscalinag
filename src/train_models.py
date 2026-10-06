import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# =========================================================
# 1. WCZYTANIE DANYCH (TRENING I TEST)
# =========================================================
print("1. Wczytywanie danych...")
df_train = pd.read_parquet("data/processed/train_2022.parquet")
df_test = pd.read_parquet("data/processed/test_2022.parquet")

features = [
    'temp_era5',
    'altitude',
    'urban_fraction_500m',
    'sin_hour',
    'cos_hour',
    'sin_doy',
    'cos_doy'
]

X_train = df_train[features].values
y_train = df_train['temp_ground'].values

X_test = df_test[features].values
y_test = df_test['temp_ground'].values

# Rezydua: sieć uczy się mikroklimatycznej poprawki (delta)
delta_train = y_train - df_train['temp_era5'].values
delta_test = y_test - df_test['temp_era5'].values

# =========================================================
# 2. STANDARYZACJA CECH WEJŚCIOWYCH
# =========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# 3. BASELINE 0: SUROWE ERA5-LAND
# =========================================================
print("\n--- BASELINE 0: SUROWE ERA5-LAND ---")
y_pred_era5 = df_test['temp_era5'].values

mae_era5 = mean_absolute_error(y_test, y_pred_era5)
rmse_era5 = np.sqrt(mean_squared_error(y_test, y_pred_era5))
r2_era5 = r2_score(y_test, y_pred_era5)
bias_era5 = np.mean(y_pred_era5 - y_test)

print(f"MAE:  {mae_era5:.3f} °C")
print(f"RMSE: {rmse_era5:.3f} °C")
print(f"R²:   {r2_era5:.4f}")
print(f"BIAS: {bias_era5:.3f} °C")

# =========================================================
# 4. MODEL ML: RANDOM FOREST REGRESSOR
# =========================================================
print("\n2. Trening Random Forest...")
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=16,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)

y_pred_rf = rf.predict(X_test_scaled)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)
bias_rf = np.mean(y_pred_rf - y_test)

print(f"MAE:  {mae_rf:.3f} °C")
print(f"RMSE: {rmse_rf:.3f} °C")
print(f"R²:   {r2_rf:.4f}")
print(f"BIAS: {bias_rf:.3f} °C")

# =========================================================
# 5. MODEL DL: KERAS SEQUENTIAL (MLP)
# =========================================================
print("\n3. Trening sieci neuronowej Keras...")

model = Sequential([
    Dense(128, activation='relu', input_shape=(len(features),)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(
    optimizer='adam',
    loss='mae',
    metrics=['mae', 'mse']
)

model.summary()

# Early stopping zabezpiecza przed przeuczeniem
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

history = model.fit(
    X_train_scaled,
    delta_train,
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# Rekonstrukcja: T_pred = T_era5 + delta
predicted_delta = model.predict(X_test_scaled).flatten()
y_pred_dl = df_test['temp_era5'].values + predicted_delta

mae_dl = mean_absolute_error(y_test, y_pred_dl)
rmse_dl = np.sqrt(mean_squared_error(y_test, y_pred_dl))
r2_dl = r2_score(y_test, y_pred_dl)
bias_dl = np.mean(y_pred_dl - y_test)

# =========================================================
# 6. PODSUMOWANIE WSZYSTKICH MODELI
# =========================================================
results = pd.DataFrame([
    {
        "Model": "Baseline 0 (Surowe ERA5)",
        "MAE [°C]": mae_era5,
        "RMSE [°C]": rmse_era5,
        "R²": r2_era5,
        "BIAS [°C]": bias_era5
    },
    {
        "Model": "Random Forest",
        "MAE [°C]": mae_rf,
        "RMSE [°C]": rmse_rf,
        "R²": r2_rf,
        "BIAS [°C]": bias_rf
    },
    {
        "Model": "Keras MLP (Deep Learning)",
        "MAE [°C]": mae_dl,
        "RMSE [°C]": rmse_dl,
        "R²": r2_dl,
        "BIAS [°C]": bias_dl
    }
])

print("\n" + "=" * 65)
print("📊 PODSUMOWANIE WYNIKÓW (ZBIÓR TESTOWY - 42 DNI)")
print("=" * 65)
print(results.round(3).to_string(index=False))

# Zapisanie wag modelu
model.save("models/keras_mlp_model.keras")
print("\n✅ Zapisano model Keras do pliku models/keras_mlp_model.keras")