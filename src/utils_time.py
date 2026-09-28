import numpy as np
import pandas as pd


def localize_to_utc(series_or_df):
    """
    Przyjmuje serię datetime lub kolumnę i bezpiecznie konwertuje czas polski do UTC,
    obsługując zmiany czasu letni/zimowy (ambiguous='NaT', nonexistent='shift_forward').
    """
    ts = pd.to_datetime(series_or_df)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize('Europe/Warsaw', ambiguous='NaT', nonexistent='shift_forward')
    return ts.dt.tz_convert('UTC').dt.tz_localize(None)


def add_cyclic_temporal_features(df, timestamp_col='timestamp'):
    """Dodaje harmoniczne cechy dobowe i roczne (sin/cos)."""
    df_out = df.copy()
    ts = pd.to_datetime(df_out[timestamp_col])
    
    df_out['sin_hour'] = np.sin(2.0 * np.pi * ts.dt.hour / 24.0)
    df_out['cos_hour'] = np.cos(2.0 * np.pi * ts.dt.hour / 24.0)
    df_out['sin_doy'] = np.sin(2.0 * np.pi * ts.dt.dayofyear / 365.25)
    df_out['cos_doy'] = np.cos(2.0 * np.pi * ts.dt.dayofyear / 365.25)
    return df_out