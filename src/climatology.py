import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def annual_harmonic(t, t0, a, phi):
    """
    Klimatologiczny model rocznej fali harmonicznej (I rząd Fouriera):
    T(t) = T0 + A * sin(2*pi*t / 365.25 + phi)
    
    Parametry:
      - t: dzień roku (day of year, 1-365/366)
      - t0: średnia roczna temperatura
      - a: amplituda fali rocznej
      - phi: przesunięcie fazowe
    """
    return t0 + a * np.sin(2.0 * np.pi * t / 365.25 + phi)


def compute_yearly_guminski_seasons(df_year):
    """
    Wyznacza ciągłe daty graniczne termicznych pór roku Gumińskiego (1948)
    dla pojedynczego roku kalendarzowego.

    Klasyczne progi Gumińskiego:
      - Zima:         T < 0°C
      - Przedwiośnie: 0°C <= T < 5°C   (faza ocieplenia / wiosenna)
      - Wiosna:       5°C <= T < 15°C  (faza ocieplenia / wiosenna)
      - Lato:         T >= 15°C
      - Jesień:       15°C > T >= 5°C  (faza ochłodzenia / jesienna)
      - Przedzimie:   5°C > T >= 0°C   (faza ochłodzenia / jesienna)
    """
    df_y = df_year.sort_values('date').copy()
    doy = df_y['doy'].values
    temp = df_y['temp_era5_daily'].values

    # 1. Dopasowanie fali rocznej
    try:
        popt, _ = curve_fit(
            annual_harmonic,
            doy,
            temp,
            p0=[10.0, 10.0, -np.pi / 2.0]
        )
        t0, a, phi = popt
        df_y['temp_fit'] = annual_harmonic(doy, t0, a, phi)
    except Exception:
        # Fallback na wygładzenie 30-dniowe w razie problemu z estymacją nieliniową
        df_y['temp_fit'] = df_y['temp_era5_daily'].rolling(30, center=True, min_periods=1).mean()

    # 2. Wyznaczenie punktów zwrotnych fali rocznej (minimum i maksimum)
    min_doy = df_y.loc[df_y['temp_fit'].idxmin(), 'doy']
    max_doy = df_y.loc[df_y['temp_fit'].idxmax(), 'doy']

    warmup = df_y[(df_y['doy'] >= min_doy) & (df_y['doy'] <= max_doy)]
    cooldown = df_y[df_y['doy'] >= max_doy]

    # 3. Daty przecięcia progów termicznych w fazie ocieplenia (0°C, 5°C, 15°C)
    d_przedwiosnie = warmup[warmup['temp_fit'] >= 0.0]['date'].min()
    d_wiosna = warmup[warmup['temp_fit'] >= 5.0]['date'].min()
    d_lato = warmup[warmup['temp_fit'] >= 15.0]['date'].min()

    # 4. Daty przecięcia progów termicznych w fazie ochłodzenia (15°C, 5°C, 0°C)
    d_jesien = cooldown[cooldown['temp_fit'] < 15.0]['date'].min()
    d_przedzimie = cooldown[cooldown['temp_fit'] < 5.0]['date'].min()
    d_zima_end = cooldown[cooldown['temp_fit'] < 0.0]['date'].min()

    # 5. Przypisanie jednoznacznej etykiety dla każdego dnia w roku
    def assign_single_date(d):
        if pd.notna(d_przedwiosnie) and d < d_przedwiosnie:
            return 'zima'
        elif pd.notna(d_wiosna) and d < d_wiosna:
            return 'przedwiosnie'
        elif pd.notna(d_lato) and d < d_lato:
            return 'wiosna'
        elif pd.notna(d_jesien) and d < d_jesien:
            return 'lato'
        elif pd.notna(d_przedzimie) and d < d_przedzimie:
            return 'jesien'
        elif pd.notna(d_zima_end) and d < d_zima_end:
            return 'przedzimie'
        else:
            return 'zima'

    df_y['guminski_season'] = df_y['date'].apply(assign_single_date)
    return df_y[['date', 'guminski_season']]


def add_guminski_seasons_harmonic(df, timestamp_col='timestamp', temp_col='temp_era5'):
    """
    Główna funkcja modułu:
    1. Agreguje cogodzinne dane do średnich dobowych ERA5 (tło makroklimatyczne).
    2. Grupuje dane rocznikami i dla każdego roku dopasowuje falę roczną.
    3. Mapuje wyznaczoną porę roku z powrotem do każdego wiersza godzinowego.
    """
    df_out = df.copy()
    ts = pd.to_datetime(df_out[timestamp_col])

    # 1. Średnia dobowa tła makroskalowego ERA5
    daily = df_out.groupby(ts.dt.date)[temp_col].mean().reset_index()
    daily.columns = ['date', 'temp_era5_daily']
    daily['date'] = pd.to_datetime(daily['date'])
    daily['year'] = daily['date'].dt.year
    daily['doy'] = daily['date'].dt.dayofyear

    # 2. Wyznaczenie sezonów dla każdego roku w zbiorze
    yearly_results = []
    for year, group in daily.groupby('year'):
        yearly_results.append(compute_yearly_guminski_seasons(group))

    seasons_calendar = pd.concat(yearly_results, ignore_index=True)
    calendar_map = dict(zip(seasons_calendar['date'].dt.date, seasons_calendar['guminski_season']))

    # 3. Zmapowanie do danych godzinowych
    df_out['guminski_season'] = ts.dt.date.map(calendar_map)
    return df_out