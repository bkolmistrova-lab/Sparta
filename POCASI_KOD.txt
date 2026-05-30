#Čištění OpenMeteo
# 1. načtení vstupní tabulky 
df = pd.read_csv("/data/in/tables/open-meteo-20260412_1.csv")

# 2. převod času
df["time"] = pd.to_datetime(df["time"], errors="coerce")

# 3. lokalizace času
df["time"] = df["time"].dt.tz_localize(
    "Europe/Berlin",
    nonexistent="NaT",
    ambiguous="NaT"
)

# 4. záloha a problematické řádky
df_zaloha = df.copy()
problematic_rows = df[df["time"].isna()].copy()

# 5. odstranění řádků s neplatným časem
df = df.dropna(subset=["time"]).copy()

# 6. duplicity
dup_rows = df[df.duplicated(subset=["location_id", "time"], keep=False)].copy()
df = df.drop_duplicates(subset=["location_id", "time"]).copy()

# 7. kontrola podezřelých hodnot
df = df[
    (df["temperature_2m (°C)"] >= -80) &
    (df["temperature_2m (°C)"] <= 60) &
    (df["rain (mm)"] >= 0) &
    (df["snowfall (cm)"] >= 0) &
    (df["precipitation (mm)"] >= 0)
].copy()

=======================

#Merge
import pandas as pd

# 1. načtení souborů
meteo = pd.read_csv("open-meteo-vycisteno_od_13.csv")
sparta = pd.read_csv("Sparta_ocistene_.csv")

# 2. převod ID sloupců na celá čísla
meteo["location_id"] = meteo["location_id"].astype("Int64")
sparta["ID"] = sparta["ID"].astype("Int64")

# 3. převod času počasí na datetime a pražský čas
meteo["time"] = pd.to_datetime(meteo["time"], errors="coerce", utc=True)
meteo["time"] = meteo["time"].dt.tz_convert("Europe/Prague")

# 4. vytvoření datetime začátku zápasu
sparta["zacatek_zapasu"] = pd.to_datetime(
    sparta["Date"] + " " + sparta["Time"],
    errors="coerce"
).dt.tz_localize("Europe/Prague")

# 5. sem budeme ukládat dílčí výsledky
vysledky = []

# 6. projdeme zápasy jeden po druhém
for _, zapas in sparta.iterrows():
    od = zapas["zacatek_zapasu"] - pd.Timedelta(hours=1)
    do = zapas["zacatek_zapasu"] + pd.Timedelta(hours=2)

    # 7. vybereme jen správnou lokaci a správné časové okno
    vyber = meteo[
        (meteo["location_id"] == zapas["ID"]) &
        (meteo["time"] >= od) &
        (meteo["time"] <= do)
    ].copy()

    # 8. ke každému vybranému počasí doplníme údaje o zápasu
    for sloupec in sparta.columns:
        vyber[sloupec] = zapas[sloupec]

    vysledky.append(vyber)

# 9. spojení všech částí dohromady
vysledek_filtr = pd.concat(vysledky, ignore_index=True)


====================


#Meteorit (shoda času)

#1. Instalace 
py -m pip install lxml
python -m pip install lxml


#2. Merge sparta_final.csv s meteority.csv
import pandas as pd
from pathlib import Path

# 3. cesta ke složce, kde je tento skript
slozka = Path(__file__).parent

# 4. načtení souborů
sparta = pd.read_csv(slozka / "SPARTA_FINAL.csv")
meteority = pd.read_csv(slozka / "meteority.csv")

# 5. kontrola názvů sloupců
print("Sloupce Sparta:")
print(sparta.columns)

print("\nSloupce Meteority:")
print(meteority.columns)

# 6. převod datumových sloupců na datum
# uprav názvy sloupců podle skutečnosti, pokud je máš jiné
sparta["DATE"] = pd.to_datetime(sparta["DATE"], errors="coerce")
meteority["Date (UTC)"] = pd.to_datetime(meteority["Date (UTC)"], errors="coerce")

# 7. merge podle data
spolecne_dny = pd.merge(
    sparta,
    meteority,
    left_on="DATE",
    right_on="Date (UTC)",
    how="inner"
)
