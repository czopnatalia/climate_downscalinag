import os
import zipfile
import requests
import pandas as pd
from pathlib import Path
from config import RAW_DATA_DIR


# ==========================================
# 1. KRAKÓW - BALICE (STACJA SYNOPTYCZNA)
# ==========================================
def fetch_and_merge_balice(start_year: int = 2000, end_year: int = 2025):
    station_id = "566"
    station_name = "KRAKOW_BALICE"
    base_url = "https://danepubliczne.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/terminowe/synop/"
    
    download_dir = RAW_DATA_DIR / "imgw" / "balice"
    download_dir.mkdir(parents=True, exist_ok=True)
    out_merged = RAW_DATA_DIR / "krakow_balice_imgw_2000_2025.csv"

    print("--- 1. POBIERANIE DANYCH DLA STACJI KRAKÓW-BALICE ---")
    for year in range(start_year, end_year + 1):
        year_file = download_dir / f"{station_name}_{year}.csv"
        if year_file.exists():
            continue

        year_folder = "1996_2000" if 1996 <= year <= 2000 else str(year)
        zip_filename = f"1996_2000_{station_id}_s.zip" if year == 2000 else f"{year}_{station_id}_s.zip"
        url = f"{base_url}{year_folder}/{zip_filename}"
        
        try:
            r = requests.get(url, timeout=25)
            if r.status_code == 200:
                temp_zip = download_dir / f"temp_{zip_filename}"
                with open(temp_zip, "wb") as f:
                    f.write(r.content)

                with zipfile.ZipFile(temp_zip, "r") as z:
                    csvs = [f for f in z.namelist() if station_id in f and f.endswith(".csv")]
                    if csvs:
                        with z.open(csvs[0]) as f_csv:
                            df = pd.read_csv(f_csv, header=None, encoding='iso-8859-2', low_memory=False)
                            # Indeksy IMGW Synop: 2=rok, 3=mc, 4=dz, 5=gg, 29=temp
                            df_sub = df[[2, 3, 4, 5, 29]].copy()
                            df_sub.columns = ['year', 'month', 'day', 'hour', 'temp']
                            df_sub['temp'] = pd.to_numeric(df_sub['temp'], errors='coerce')
                            df_year = df_sub[df_sub['year'] == year].copy()
                            
                            df_year.to_csv(year_file, index=False)
                            print(f"   ✅ Balice {year}: zapisano {year_file.name} ({len(df_year)} wierszy)")
                temp_zip.unlink(missing_ok=True)
            else:
                print(f"   ⚠️ Balice {year}: brak na serwerze (kod {r.status_code})")
        except Exception as e:
            print(f"   ❌ Balice {year}: błąd ({e})")

    # Scalanie plików rocznych Balic do jednego CSV
    yearly_files = sorted(download_dir.glob(f"{station_name}_*.csv"))
    if yearly_files:
        df_all = pd.concat([pd.read_csv(f) for f in yearly_files], ignore_index=True)
        df_all.sort_values(by=['year', 'month', 'day', 'hour'], inplace=True)
        df_all.to_csv(out_merged, index=False)
        print(f"✅ Utworzono plik zbiorczy Balic: {out_merged} ({len(df_all):,} wierszy)\n")


# ==========================================
# 2. KRAKÓW - OBSERWATORIUM (STACJA KLIMATYCZNA)
# ==========================================
def fetch_and_merge_obserwatorium(start_year: int = 2000, end_year: int = 2025):
    station_id = 250190390
    station_name = "KRAKOW_OBSERWATORIUM"
    base_url = "https://danepubliczne.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/terminowe/klimat/"
    
    download_dir = RAW_DATA_DIR / "imgw" / "obserwatorium"
    download_dir.mkdir(parents=True, exist_ok=True)
    out_merged = RAW_DATA_DIR / "krakow_obserwatorium_imgw_2000_2025.csv"

    print("--- 2. POBIERANIE DANYCH DLA STACJI KRAKÓW-OBSERWATORIUM ---")
    for year in range(start_year, end_year + 1):
        year_file = download_dir / f"{station_name}_{year}.csv"
        if year_file.exists():
            continue

        year_records = []
        folders = [str(year), "1996_2000"] if year == 2000 else [str(year)]

        # 1. Próba pobrania paczek miesięcznych (_k.zip)
        found_monthly = False
        for month in range(1, 13):
            for folder in folders:
                zip_filename = f"{year}_{month:02d}_k.zip"
                url = f"{base_url}{folder}/{zip_filename}"
                try:
                    r = requests.get(url, timeout=15)
                    if r.status_code == 200:
                        temp_zip = download_dir / f"temp_{zip_filename}"
                        with open(temp_zip, "wb") as f:
                            f.write(r.content)
                        with zipfile.ZipFile(temp_zip, "r") as z:
                            csvs = [f for f in z.namelist() if f.endswith(".csv")]
                            if csvs:
                                with z.open(csvs[0]) as f_csv:
                                    df = pd.read_csv(f_csv, header=None, encoding='iso-8859-2', low_memory=False)
                                    df_st = df[df[0] == station_id][[2, 3, 4, 5, 6]].copy()
                                    df_st.columns = ['year', 'month', 'day', 'hour', 'temp']
                                    df_st['temp'] = pd.to_numeric(df_st['temp'], errors='coerce')
                                    df_st = df_st[df_st['year'] == year]
                                    if not df_st.empty:
                                        year_records.append(df_st)
                                        found_monthly = True
                        temp_zip.unlink(missing_ok=True)
                        break
                except Exception:
                    pass

        # 2. Próba paczki rocznej (jeśli w danym roku nie było paczek miesięcznych)
        if not found_monthly:
            for folder in folders:
                zip_filename = f"{year}_k.zip"
                url = f"{base_url}{folder}/{zip_filename}"
                try:
                    r = requests.get(url, timeout=20)
                    if r.status_code == 200:
                        temp_zip = download_dir / f"temp_{zip_filename}"
                        with open(temp_zip, "wb") as f:
                            f.write(r.content)
                        with zipfile.ZipFile(temp_zip, "r") as z:
                            csvs = [f for f in z.namelist() if f.endswith(".csv")]
                            if csvs:
                                with z.open(csvs[0]) as f_csv:
                                    df = pd.read_csv(f_csv, header=None, encoding='iso-8859-2', low_memory=False)
                                    df_st = df[df[0] == station_id][[2, 3, 4, 5, 6]].copy()
                                    df_st.columns = ['year', 'month', 'day', 'hour', 'temp']
                                    df_st['temp'] = pd.to_numeric(df_st['temp'], errors='coerce')
                                    df_st = df_st[df_st['year'] == year]
                                    if not df_st.empty:
                                        year_records.append(df_st)
                        temp_zip.unlink(missing_ok=True)
                        break
                except Exception:
                    pass

        # Zapis pliku rocznego dla Obserwatorium
        if year_records:
            df_year = pd.concat(year_records, ignore_index=True)
            df_year.sort_values(by=['month', 'day', 'hour'], inplace=True)
            df_year.to_csv(year_file, index=False)
            print(f"   ✅ Obserwatorium {year}: zapisano {year_file.name} ({len(df_year)} wierszy)")
        else:
            print(f"   ⚠️ Obserwatorium {year}: nie odnaleziono danych")

    # Scalanie plików rocznych Obserwatorium do jednego CSV
    yearly_files = sorted(download_dir.glob(f"{station_name}_*.csv"))
    if yearly_files:
        df_all = pd.concat([pd.read_csv(f) for f in yearly_files], ignore_index=True)
        df_all.sort_values(by=['year', 'month', 'day', 'hour'], inplace=True)
        df_all.to_csv(out_merged, index=False)
        print(f"✅ Utworzono plik zbiorczy Obserwatorium: {out_merged} ({len(df_all):,} wierszy)\n")


if __name__ == '__main__':
    fetch_and_merge_balice()
    fetch_and_merge_obserwatorium()