import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

# 1. NAČTENÍ SOUBORU
# Načítáme Spartu - pozor na oddělovač (v CSV bývá čárka nebo středník)
df = pd.read_csv('SPARTA_FINAL1.csv')


# 3. VÝBĚR KATEGORIÍ (FEATURES)
features = [
    'VERNOSTNI_SYSTEM',
    'POCASI_KATEGORIE',    
    'TEMPERATURE',         
    'ATRAKTIVITA_FANOUS',  
    'ATRAKTIVITA_VSTUPNE', 
    'PRECIPITATION',       
    'DAY_OF_WEEK',           
    'MONTH',
    "POCASI_KATEGORIE",               
    'UEFA_VAHA'            
]
target = 'ATTENDANCE'

# Vyčištění dat
df_model = df.dropna(subset=[target]).copy()

X = df_model[features]
y = df_model[target]

# 4. KONSTRUKCE MODELU
model_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")), 
    ("model", RandomForestRegressor(
        n_estimators=500,
        max_depth=7,           
        min_samples_leaf=5,
        random_state=42
    ))
])

# 5. TESTOVÁNÍ (K-Fold Cross-Validation)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(model_pipeline, X, y, cv=cv)

# Statistiky
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"--- VÝSLEDKY MODELU (SPARTA) ---")
print(f"Průměrná chyba (MAE): {int(mae)} diváků")
print(f"Spolehlivost (R2): {r2:.2f}")

# 6. ANALÝZA VLIVU KATEGORIÍ
model_pipeline.fit(X, y)
importances = model_pipeline.named_steps['model'].feature_importances_
importance_df = pd.DataFrame({
    'Kategorie': features, 
    'Vliv (%)': importances * 100
}).sort_values('Vliv (%)', ascending=False)

# 7. VIZUALIZACE VÝSLEDKŮ
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Graf A: Porovnání reality a předpovědi
ax1.scatter(y, y_pred, alpha=0.6, color="#005180") # Sparťanská rudá
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
ax1.set_title('Skutečnost vs. Predikce (Sparta Praha)', fontsize=14)
ax1.set_xlabel('Skutečná návštěvnost')
ax1.set_ylabel('Předpovězená návštěvnost')
ax1.grid(True, alpha=0.2)

# Graf B: Důležitost faktorů
sns.barplot(x='Vliv (%)', y='Kategorie', data=importance_df, ax=ax2, palette='Reds_r')
ax2.set_title('Co nejvíce ovlivňuje návštěvnost na Letné?', fontsize=14)
for i, v in enumerate(importance_df['Vliv (%)']):
    ax2.text(v + 0.5, i, f'{v:.1f}%', va='center', fontweight='bold')

plt.tight_layout()
plt.show()


export_df = pd.DataFrame({
    'Datum': df_model['DATE'].values,
    'Soupeř': df_model['AWAYTEAM'].values,
    'Skutečná návštěvnost': y.values.astype(int),
    'Predikovaná návštěvnost': y_pred.astype(int),
    'Rozdíl (chyba)': y.values.astype(int) - y_pred.astype(int)
})

export_df.to_csv('predikce_navstevnosti.csv', index=False, encoding='utf-8-sig', sep=';')
