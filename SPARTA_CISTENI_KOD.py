import pandas as pd
import re
import numpy as np

# Načtení datového souboru s korektním kódováním a oddělovačem
Sparta = pd.read_csv("Sparta_puvodni.csv", sep=";", encoding="utf-8")

# Převod sloupce s datem na skutečný datový typ datetime
Sparta["Date"] = pd.to_datetime(Sparta["Date"], dayfirst=True)

# Nahrazení české zkratky "odp." za standardní "PM" pro odpolední časy
time_text_clean = Sparta["Time"].str.replace("odp.", "PM", regex=False)
time_clean = time_text_clean.str.strip()

# Převod textu na časový formát (zpracování 12hodinového formátu s AM/PM)
time_final = pd.to_datetime(
    time_clean, 
    format="%I:%M %p", 
    errors='coerce'
).dt.time

Sparta["Time"] = time_final

# 1. Základní očištění: Odstranění textu v závorkách a oříznutí mezer
Sparta["Home team"] = Sparta["Home team"].astype(str).str.split("(").str[0].str.strip()
Sparta["Away team"] = Sparta["Away team"].astype(str).str.split("(").str[0].str.strip()

# 2. Odstranění úvodních číslic a teček na začátku názvu (např. "1. Sparta" -> "Sparta")
Sparta["Home team"] = Sparta["Home team"].replace(r'^[0-9.\s]+', '', regex=True)
Sparta["Away team"] = Sparta["Away team"].replace(r'^[0-9.\s]+', '', regex=True)

# 3. Sjednocení sponzorských názvů Zlína (Fastav, Trinity, Zlín) pod jeden jednotný název
zlin_regex = r".*(Zl[ií]n|Trinity|Fastav).*"
Sparta["Home team"] = Sparta["Home team"].replace(zlin_regex, "Zlin", regex=True)
Sparta["Away team"] = Sparta["Away team"].replace(zlin_regex, "Zlin", regex=True)


# 1. Přídání sezon
sezony = []

for datum in Sparta['Date']:
    rok = datum.year
    mesic = datum.month
    
    if mesic >= 7:
        sezona = f"{rok}/{rok + 1}"
    else:
     
        sezona = f"{rok - 1}/{rok}"
    
    sezony.append(sezona)

Sparta['Season'] = sezony

# Odstranění teček, které reprezentovaly oddělovače tisíců (např. 17.202 -> 17202)
Sparta["Attendance"] = Sparta["Attendance"].astype(str).str.replace(".", "", regex=False)

nova_cisla = []

# Ošetření neplatných hodnot, chybějících údajů a převod na čísla
for hodnota in Sparta["Attendance"]:
    text_hodnoty = str(hodnota)
    if text_hodnoty == "x" or text_hodnoty == "nan":
        nova_cisla.append(0)
    else:
        nova_cisla.append(int(text_hodnoty))
Sparta["Attendance"] = nova_cisla

prazdne_casy = Sparta[Sparta["Time"].isna()]
Sparta["Time"] = Sparta["Time"].fillna("15:00:00")

# vymazání nepotřebných sloupců
Sparta = Sparta.drop(["Result", "System of play"], axis=1)


# 1. Nový zápas v dubnu
sloupce_novy_zapas = [
    'temperature_2m (°C)', 
    'rain (mm)', 
    'snowfall (cm)', 
    'precipitation (mm)',
    'Coach', 
    'Attendance', 
    'location_id', 
    'time',
    'Date',
    'Home team',
    'Away team'
]

# 2. Připravíme si hodnoty pro nový zápas
hodnoty_novy_zapas = [
    20.9,                  
    0,                    
    0,                     
    0,                     
    'Brian Priske',        
    17514,                 
    25,                    
    '18.04.2026 18:00:00', 
    '2026-04-18',          
    'Sparta Prague',       
    'Jablonec'             
]

# 3. Vytvoříme z těchto hodnot nový jednořádkový DataFrame
novy_radek = pd.DataFrame([hodnoty_novy_zapas], columns=sloupce_novy_zapas)
Sparta = pd.concat([Sparta, novy_radek], ignore_index=True)

# 1. Načtení souboru s atraktivitou soupeřů
atraktivita = pd.read_csv("Atraktivita_CSV.csv", sep=";")

# 2. Odstranění neviditelných mezer v názvech sloupců
Sparta.columns = Sparta.columns.str.strip()
atraktivita.columns = atraktivita.columns.str.strip()

# 3. Očištění samotných názvů týmů od mezer.
Sparta["Away team"] = Sparta["Away team"].str.strip()
atraktivita["AWAY_TEAM"] = atraktivita["AWAY_TEAM"].str.strip()

# 4. Propojení tabulek (Merge) do nové proměnné 'merged'
merged = Sparta.merge(
    atraktivita,
    left_on="Away team",   
    right_on="AWAY_TEAM",  
    how="left",            
    validate="m:1"         
)

# 1. Definujeme přesné datum spuštění nového věrnostního programu
Sparta['Date'] = pd.to_datetime(Sparta['Date'])
datum_zahajeni_vernostni_program = pd.to_datetime('2022-07-01')

# 2. Vytvoříme binární sloupec: 1 = s věrnostním systémem, 0 = bez něj
Sparta['vernostni_system'] = (Sparta['Date'] >= datum_zahajeni_vernostni_program).astype(int)


# 1. Nastavení časového rozsahu pro covid data
covid_start = '2020-03-01'
covid_end = '2022-06-30'

# 2. Filtrace pomocí vlnovky (~), která znamená "NEMÁ platit tato podmínka"
Sparta_cista = Sparta[~((Sparta['Date'] >= covid_start) & (Sparta['Date'] <= covid_end))].copy()

# Nezávisla tabulka Sparta_cista

# 1. Den
Sparta_cista['Date'] = pd.to_datetime(Sparta_cista['Date'])
Sparta_cista['day_of_week'] = Sparta_cista['Date'].dt.dayofweek

# 2. Měsíc 
Sparta_cista['month'] = Sparta_cista['Date'].dt.month


# 1. Definice evropských soupeřů 
giganti = ['liverpool', 'milan', 'atletico', 'atlético', 'inter', 'lazio', 'villarreal', 'schalke']

ostatni_uefa_kat2 = [
    'betis', 'galatasaray', 'rangers', 'lyon', 'monaco', 'moscow', 'aris', 'viking',
    'salzburg', 'dinamo zagreb', 'malmö', 'southampton', 'krasnodar', 'copenhagen', 'red star'
]

uefa_kat3 = [
    'shamrock rovers', 'raków', 'aberdeen', 'az alkmaar', 'aktobe', 
    'ararat-armenia', 'riga', 'stade brestois', 'fcsb', 'spartak', 
    'h. beer sheva', 'rostov', 'sönderjyske', 'steaua', 'apoel', 'asteras', 'fc thun'
]

Sparta_cista['UEFA_VAHA'] = 0

# 3.  'Away team' 
away_lower = Sparta_cista['Away team'].str.lower()

# 4. Přiřazení vah pomocí regulárních výrazů
Sparta_cista.loc[away_lower.str.contains('|'.join(uefa_kat3), na=False), 'UEFA_VAHA'] = 3
Sparta_cista.loc[away_lower.str.contains('|'.join(ostatni_uefa_kat2), na=False), 'UEFA_VAHA'] = 2
Sparta_cista.loc[away_lower.str.contains('|'.join(giganti), na=False), 'UEFA_VAHA'] = 1

# 5. Smazání nepotřebných sloupců 
sloupce_ke_smazani = ['IS_TOP_MATCH', 'MATCH_DATETIME', 'LOCATION_ID', 'AWAY_TEAM', 'AWAYTEAM']
Sparta_cista = Sparta_cista.drop(columns=sloupce_ke_smazani, errors='ignore')


# Načtení datového souboru s počasím

Sparta_cista = pd.read_csv("Sparta_pocasi.csv", sep=";", encoding="utf-8")
# 1. Funkce pro kategorizaci teploty 
def temp_category(temp):
    if pd.isna(temp):
        return np.nan  
    if 15 <= temp <= 24:
        return 1       
    elif 10 <= temp < 15 or 24 < temp <= 27:
        return 2       
    elif 5 <= temp < 10 or 27 < temp <= 30:
        return 3       
    elif 0 <= temp < 5 or 30 < temp <= 32:
        return 4      
    else:
        return 5       

# 2. Funkce pro kategorizaci srážek 
def precip_category(precipitation):
    if pd.isna(precipitation):
        return np.nan
    if precipitation == 0:
        return 1      
    elif 0 < precipitation <= 0.2:
        return 2       
    elif 0.2 < precipitation <= 1:
        return 3      
    elif 1 < precipitation <= 3:
        return 4      
    else:
        return 5       

# 3. Aplikace funkcí na sloupce v tabulce

Sparta_cista["TEMP_KATEGORIE"] = Sparta_cista["TEMPERATURE"].apply(temp_category)
Sparta_cista["SRAZKY_KATEGORIE"] = Sparta_cista["PRECIPITATION"].apply(precip_category)

# 4. Spojení do jednoho indexu 
Sparta_cista["POCASI_KATEGORIE"] = Sparta_cista[["TEMP_KATEGORIE", "SRAZKY_KATEGORIE"]].max(axis=1)

# 5. Přemapování čísel na srozumitelné textové štítky
Sparta_cista["POCASI_LABEL"] = Sparta_cista["POCASI_KATEGORIE"].map({
    1: "1_Ideální počasí",
    2: "2_V pohodě",
    3: "3_Hraniční komfort",
    4: "4_Nepříjemné",
    5: "5_Extrém"
})



# 7. Uložení pročištěných dat do finálního souboru
Sparta_cista.to_csv("SPARTA_FINAL1.csv", index=False, encoding="utf-8-sig")