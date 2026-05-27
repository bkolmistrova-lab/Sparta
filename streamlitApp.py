import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import date
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score

# --- PATHS ---
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "SPARTA_FINAL1.csv")

LOGO_PATH = os.path.join(BASE_DIR, "AC-Sparta-LOGO2021.svg.png")
LAT, LON = 50.08787, 14.475525
STADIUM_CAPACITY = 18_262

FEATURES = [
    "VERNOSTNI_SYSTEM",
    "TEMPERATURE",
    "ATRAKTIVITA_FANOUS",
    "ATRAKTIVITA_VSTUPNE",
    "PRECIPITATION",
    'DAY_OF_WEEK',           
    'MONTH',
    "POCASI_KATEGORIE",
    "UEFA_VAHA",
]

UEFA_OPTIONS = {
    "Ne": 0,
    "Ano": 1,
}

KICKOFF_TIMES = [
    "10:00", "12:00", "14:00", "15:00", "16:00",
    "17:00", "18:00", "19:00", "20:00", "21:00",
]


# --- MODEL (trained once, cached for the session) ---
@st.cache_resource(show_spinner="Trénuji model, chvilku strpení...")
def train_model():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["ATTENDANCE"])
    X = df[FEATURES]
    y = df["ATTENDANCE"]

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=500,
            max_depth=7,
            min_samples_leaf=5,
            random_state=42,
        )),
    ])

    # 5-fold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_cv = cross_val_predict(pipeline, X, y, cv=cv)
    mae = mean_absolute_error(y, y_cv)
    r2 = r2_score(y, y_cv)

    pipeline.fit(X, y)
    n_matches = len(y)
    return pipeline, int(round(mae)), round(r2, 2), n_matches


# --- OPPONENTS ---
@st.cache_data
def load_opponents():
    df = pd.read_csv(DATA_PATH)
    return (
        df.groupby("AWAYTEAM")[["ATRAKTIVITA_VSTUPNE", "ATRAKTIVITA_FANOUS"]]
        .first()
        .reset_index()
        .sort_values("AWAYTEAM")
        .rename(columns={"AWAYTEAM": "AWAY_TEAM"})
    )


# --- WEATHER (Open-Meteo API) ---
def compute_pocasi_kategorie(temp: float, precipitation: float, snowfall: float) -> int:
    if snowfall > 0 or temp < 0 or temp > 31:
        return 5
    if temp < 5 or temp > 29:
        return 4
    if temp < 10 or temp > 26:
        return 4 if precipitation >= 1.4 else 3
    if temp < 15 or temp > 24:
        return 4 if precipitation >= 1.0 else 2
    # 15 <= temp <= 24 (ideální teplota)
    if precipitation >= 1.2:
        return 4
    if precipitation >= 0.5:
        return 3
    if precipitation >= 0.1:
        return 2
    return 1


def fetch_weather(match_date: date, hour: int) -> dict | None:
    date_str = match_date.strftime("%Y-%m-%d")
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        if match_date < date.today()
        else "https://api.open-meteo.com/v1/forecast"
    )
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": ["temperature_2m", "precipitation", "rain", "snowfall"],
        "timezone": "Europe/Prague",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        return {
            "temp": data["hourly"]["temperature_2m"][hour],
            "precip": data["hourly"]["precipitation"][hour],
            "rain": data["hourly"]["rain"][hour],
            "snow": data["hourly"]["snowfall"][hour],
        }
    except Exception:
        return None


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Predikce návštěvnosti – Sparta Praha",
    page_icon="⚽",
    layout="centered",
)

col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image(LOGO_PATH, width=90)
with col_title:
    st.title("Sparta Praha – Predikce návštěvnosti")
st.markdown(
    """
    Vyberte soupeře, datum a typ zápasu.  
    Počasí se načte automaticky z **Open-Meteo API**.
    
    ⚠️ **Upozornění:** Vybírat lze **pouze soupeře z nabízeného seznamu**. 
    Jedná se o týmy, které se Spartou reálně hrály v letech 2015–2026.
    """
)

with st.expander("ℹ️ Co ovlivňuje predikci?"):
    st.markdown(
        """
Model se naučil na základě **191 historických zápasů Sparty (2015–2026)** a při předpovědi zohledňuje tyto faktory:

| Faktor | Váha v modelu | Popis |
|---|---|---|
| 💳 **Věrnostní systém** | **49.1 %** | Zvednutí divácké základny a stabilní zájem po zavedení permanentek a věrnostního systému (červenec 2022). |
| 🏆 **Soupeř - fanoušci** | **15.8 %** | Divácká a fanouškovská atraktivita soupeře (derby se Slavií, zápasy s Plzní apod.). |
| 🏅 **UEFA zápas** | **9.4 %** | Prestiž zápasů v evropských pohárech přitahuje stabilně vyšší návštěvy. |
| 🎟️ **Soupeř - vstupenky** | **9.4 %** | Cenová hladina a atraktivita zápasových balíčků / lístků. |
| 🌡️ **Teplota** | **7.2 %** | Teplota vzduchu v době výkopu utkání. |
| 📅 **Den v týdnu** | **4.6 %** | Den v týdnu, kdy se hraje (rozdíly mezi víkendy a pracovními dny). |
| 🗓️ **Měsíc** | **2.5 %** | Kalendářní měsíc (vliv letních prázdnin, zimní pauzy a sezonnosti). |
| 🌧️ **Srážky / sníh** | **1 %** | Celkový úhrn srážek během zápasu. |
| 🌤️ **Kategorie počasí** | **0.5 %** | Souhrnná vypočítaná kategorie celkového komfortu pro diváky. |


⚠️ Váha v modelu ukazuje, jak moc model daný faktor využívá při větvení rozhodovacích stromů — **není to přímá míra vlivu na návštěvnost**.

Počasí (teplota, déšť, sníh, kategorie počasí) se načítá a vypočítává automaticky podle zvoleného data a času výkopu.
        """
    )

pipeline, mae, r2, n_matches = train_model()
opponents_df = load_opponents()
opponent_names = opponents_df["AWAY_TEAM"].tolist()

st.divider()

# --- INPUT FORM ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        opponent = st.selectbox("Soupeř", options=opponent_names)
        match_date = st.date_input(
            "Datum zápasu",
            value=date.today(),
            min_value=date(2015, 1, 1),
        )
        kickoff = st.selectbox(
            "Čas výkopu (případně zaokrouhlený na celou hodinu dolů)",
            options=KICKOFF_TIMES,
            index=8,  # default 20:00
        )

    with col2:
        uefa_label = st.selectbox("UEFA zápas?", options=list(UEFA_OPTIONS.keys()))

    submitted = st.form_submit_button("🔮 Spočítat predikci", use_container_width=True)

# --- PREDICTION ---
if submitted:
    hour = int(kickoff.split(":")[0])
    uefa_vaha = UEFA_OPTIONS[uefa_label]

    opp_row = opponents_df[opponents_df["AWAY_TEAM"] == opponent].iloc[0]
    atraktivita_vstupne = float(opp_row["ATRAKTIVITA_VSTUPNE"])
    atraktivita_fanous = float(opp_row["ATRAKTIVITA_FANOUS"])

    # Automatický výpočet na pozadí bez nutnosti vstupu od uživatele
    day_of_week = float(match_date.weekday())  # 0=Pondělí, 6=Neděle
    month = float(match_date.month)

    with st.spinner("Načítám data o počasí z Open-Meteo..."):
        weather = fetch_weather(match_date, hour)

    if weather is None:
        st.error(
            "Nepodařilo se načíst počasí z Open-Meteo API. "
            "Zkontrolujte připojení k internetu a zkuste znovu."
        )
    else:
        # Vytvoření DataFrame se všemi 11 features potřebnými pro model
        input_df = pd.DataFrame([{
            "VERNOSTNI_SYSTEM": 1.0,
            "TEMPERATURE": weather["temp"],
            "ATRAKTIVITA_FANOUS": atraktivita_fanous,
            "ATRAKTIVITA_VSTUPNE": atraktivita_vstupne,
            "PRECIPITATION": weather["precip"],
            "DAY_OF_WEEK": day_of_week,
            "MONTH": month,
            "UEFA_VAHA": float(uefa_vaha),
            "POCASI_KATEGORIE": float(compute_pocasi_kategorie(weather["temp"], weather["precip"], weather["snow"])),
        }])

        # Seřazení sloupců přesně podle FEATURES listu
        input_df = input_df[FEATURES]

        prediction = int(round(pipeline.predict(input_df)[0]))
        fill_pct = round(prediction / STADIUM_CAPACITY * 100, 1)

        st.divider()
        st.subheader(f"Sparta Praha vs. {opponent}")
        st.caption(
            f"{match_date.strftime('%d. %m. %Y')}  ·  {kickoff}  ·  UEFA: {uefa_label}"
        )

        col_a, col_b, col_c = st.columns(3)
        col_a.metric(
            "Odhadovaná návštěvnost",
            f"{prediction:,} diváků".replace(",", " "),
        )
        col_b.metric("Zaplněnost stadionu", f"{fill_pct} %")
        col_c.metric("Přesnost modelu (MAE)", f"± {mae:,}".replace(",", " "))

        st.progress(min(fill_pct / 100, 1.0))

        with st.expander("Detail počasí (Open-Meteo)"):
            wc1, wc2, wc3, wc4 = st.columns(4)
            wc1.metric("Teplota", f"{weather['temp']} °C")
            wc2.metric("Srážky celkem", f"{weather['precip']} mm")
            wc3.metric("Déšť", f"{weather['rain']} mm")
            wc4.metric("Sníh", f"{weather['snow']} cm")

        with st.expander("Vstupní parametry modelu"):
            st.dataframe(
                input_df.T.rename(columns={0: "Hodnota"}),
                use_container_width=True,
            )

# --- FOOTER ---
st.divider()
st.caption(
    f"Model: Random Forest (500 stromů, max_depth=7)  ·  "
    f"Trénováno na {n_matches} zápasech (2015–2026, bez COVID)  ·  "
    f"R² = {r2}  ·  MAE = ± {mae:,} diváků".replace(",", " ")
)

st.markdown(
    """
    <div style="text-align: center; color: #7f8c8d; font-size: 0.8em; margin-top: 30px; line-height: 1.4; border-top: 1px solid #e0e0e0; padding-top: 15px;">
        <b>Upozornění o autorských právech a užití dat</b><br>
        Tento software je akademickým výstupem v rámci <b>Czechitas Digitální akademie</b>. 
        Veškerá práva k logu, obchodní značce a chráněným názvům náleží výhradnímu vlastníkovi 
        (AC Sparta Praha fotbal, a.s.). Jejich užití v této aplikaci splňuje podmínky <b>férového užití 
        (Fair Use) pro vzdělávací a výzkumné účely</b>. Výstupy modelu jsou pouze orientační a neslouží 
        k žádným komerčním ani oficiálním účelům.
    </div>
    """,
    unsafe_allow_html=True
)