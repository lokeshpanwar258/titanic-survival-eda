# =============================================================
#  Titanic Survival Analysis — Exploratory Data Analysis (EDA)
#  Author  : lokeshpanwar258
#  Dataset : Kaggle Titanic (https://www.kaggle.com/c/titanic)
#  Tools   : Python, Pandas, NumPy, Matplotlib, Seaborn
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ── Plot style ────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({'figure.facecolor': '#0f172a',
                     'axes.facecolor':   '#1e293b',
                     'axes.labelcolor':  'white',
                     'xtick.color':      'white',
                     'ytick.color':      'white',
                     'text.color':       'white',
                     'grid.color':       '#334155'})

# ── 1. LOAD DATA ──────────────────────────────────────────────
print("=" * 55)
print("  TITANIC SURVIVAL ANALYSIS — EDA")
print("=" * 55)

# Download dataset automatically via seaborn's built-in loader
df = sns.load_dataset('titanic')

print(f"\n✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ── 2. BASIC INFO ─────────────────────────────────────────────
print("\n── Dataset Info ──────────────────────────────────────")
print(df.dtypes)

print("\n── First 5 Rows ──────────────────────────────────────")
print(df.head())

print("\n── Basic Statistics ──────────────────────────────────")
print(df.describe())

# ── 3. MISSING VALUES ────────────────────────────────────────
print("\n── Missing Values ────────────────────────────────────")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing,
                            'Missing %': missing_pct})
print(missing_df[missing_df['Missing Count'] > 0])

# ── 4. DATA CLEANING ─────────────────────────────────────────
# Impute Age: median per sex + class
df['age'] = df.groupby(['sex', 'pclass'])['age'] \
              .transform(lambda x: x.fillna(x.median()))

# Fill Embarked with mode
df['embark_town'] = df['embark_town'].fillna(df['embark_town'].mode()[0])

# Drop deck (too many nulls)
df.drop(columns=['deck'], inplace=True)

print("\n✅ Missing values handled.")
print(f"   Remaining nulls: {df.isnull().sum().sum()}")

# ── 5. SURVIVAL OVERVIEW ─────────────────────────────────────
print("\n── Survival Overview ─────────────────────────────────")
survival_counts = df['survived'].value_counts()
survival_rate   = df['survived'].mean() * 100
print(f"   Survived     : {survival_counts[1]}  ({survival_rate:.1f}%)")
print(f"   Did Not Survive: {survival_counts[0]}  ({100 - survival_rate:.1f}%)")

# ── 6. ANALYSIS BY KEY FEATURES ──────────────────────────────
print("\n── Survival Rate by Gender ───────────────────────────")
gender_survival = df.groupby('sex')['survived'].mean() * 100
print(gender_survival.round(1).to_string())

print("\n── Survival Rate by Passenger Class ─────────────────")
class_survival = df.groupby('pclass')['survived'].mean() * 100
print(class_survival.round(1).to_string())

print("\n── Survival Rate by Embarkation Port ────────────────")
embark_survival = df.groupby('embark_town')['survived'].mean() * 100
print(embark_survival.round(1).to_string())

print("\n── Survival Rate: Class × Gender ────────────────────")
heatmap_data = df.pivot_table(values='survived',
                               index='sex',
                               columns='pclass',
                               aggfunc='mean') * 100
print(heatmap_data.round(1).to_string())

print("\n── Fare Statistics by Survival ──────────────────────")
fare_stats = df.groupby('survived')['fare'].describe()
print(fare_stats.round(2).to_string())

# ── 7. CORRELATION MATRIX ─────────────────────────────────────
print("\n── Correlation with Survival ─────────────────────────")
corr_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
corr = df[corr_cols].corr()['survived'].drop('survived').sort_values()
print(corr.round(3).to_string())

# ── 8. VISUALIZATIONS ────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(14, 16))
fig.suptitle('Titanic Survival — Exploratory Data Analysis',
             fontsize=18, fontweight='bold', color='white', y=1.01)
fig.patch.set_facecolor('#0f172a')

# ── 8a. Survival Count ────────────────────────────────────────
ax = axes[0, 0]
colors = ['#ef4444', '#3b82f6']
df['survived'].value_counts().plot(kind='bar', ax=ax,
    color=colors, edgecolor='none', width=0.5)
ax.set_title('Overall Survival Count', fontweight='bold')
ax.set_xlabel('Survived (0=No, 1=Yes)')
ax.set_ylabel('Count')
ax.set_xticklabels(['Did Not Survive', 'Survived'], rotation=0)
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 5,
            f'{int(bar.get_height())}',
            ha='center', va='bottom', color='white', fontsize=11)

# ── 8b. Survival by Gender ───────────────────────────────────
ax = axes[0, 1]
gender_pct = df.groupby('sex')['survived'].mean() * 100
bars = ax.bar(gender_pct.index, gender_pct.values,
              color=['#f472b6', '#60a5fa'], edgecolor='none', width=0.4)
ax.set_title('Survival Rate by Gender (%)', fontweight='bold')
ax.set_ylabel('Survival Rate (%)')
ax.set_ylim(0, 100)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1.5,
            f'{bar.get_height():.1f}%',
            ha='center', va='bottom', color='white', fontsize=11)

# ── 8c. Survival by Class ─────────────────────────────────────
ax = axes[1, 0]
class_pct = df.groupby('pclass')['survived'].mean() * 100
bars = ax.bar(['1st Class', '2nd Class', '3rd Class'],
              class_pct.values,
              color=['#10b981', '#f59e0b', '#ef4444'],
              edgecolor='none', width=0.5)
ax.set_title('Survival Rate by Passenger Class (%)', fontweight='bold')
ax.set_ylabel('Survival Rate (%)')
ax.set_ylim(0, 100)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1.5,
            f'{bar.get_height():.1f}%',
            ha='center', va='bottom', color='white', fontsize=11)

# ── 8d. Age Distribution ──────────────────────────────────────
ax = axes[1, 1]
df[df['survived'] == 1]['age'].hist(ax=ax, bins=20, alpha=0.7,
    color='#3b82f6', label='Survived', edgecolor='none')
df[df['survived'] == 0]['age'].hist(ax=ax, bins=20, alpha=0.6,
    color='#ef4444', label='Did Not Survive', edgecolor='none')
ax.set_title('Age Distribution by Survival', fontweight='bold')
ax.set_xlabel('Age')
ax.set_ylabel('Count')
ax.legend(facecolor='#1e293b', edgecolor='none', labelcolor='white')

# ── 8e. Heatmap: Class × Gender ──────────────────────────────
ax = axes[2, 0]
pivot = df.pivot_table(values='survived', index='sex',
                        columns='pclass', aggfunc='mean') * 100
sns.heatmap(pivot, ax=ax, annot=True, fmt='.1f',
            cmap='RdYlGn', linewidths=0.5,
            annot_kws={'size': 13, 'weight': 'bold'},
            cbar_kws={'label': 'Survival Rate (%)'})
ax.set_title('Survival Rate (%) — Class × Gender', fontweight='bold')
ax.set_xlabel('Passenger Class')
ax.set_ylabel('Gender')

# ── 8f. Fare vs Survival (Boxplot) ───────────────────────────
ax = axes[2, 1]
df_fare = df[df['fare'] < 300]   # remove extreme outliers for clarity
sns.boxplot(data=df_fare, x='survived', y='fare', ax=ax,
            palette={0: '#ef4444', 1: '#3b82f6'},
            width=0.4, linewidth=1.2)
ax.set_title('Fare Distribution by Survival', fontweight='bold')
ax.set_xlabel('Survived (0=No, 1=Yes)')
ax.set_ylabel('Fare (£)')
ax.set_xticklabels(['Did Not Survive', 'Survived'])

plt.tight_layout()
plt.savefig('titanic_eda_plots.png', dpi=150, bbox_inches='tight',
            facecolor='#0f172a')
plt.show()
print("\n✅ Plots saved as 'titanic_eda_plots.png'")

# ── 9. KEY FINDINGS SUMMARY ───────────────────────────────────
print("\n" + "=" * 55)
print("  KEY FINDINGS")
print("=" * 55)
print("""
1. GENDER  — Strongest predictor.
   Female survival: 74.2% | Male survival: 18.9%

2. CLASS   — Clear socioeconomic divide.
   1st: 62.9% | 2nd: 47.3% | 3rd: 24.2%

3. AGE     — Children (0-10) had higher survival rates.
   Weak negative correlation (r = -0.077)

4. FARE    — Higher fare = better survival odds.
   Positive correlation (r = +0.257)

5. PORT    — Cherbourg: 55.4% | Southampton: 34.0%
   Likely due to more 1st-class boardings at Cherbourg

6. BEST CASE  — 1st class female: 96.8% survival
   WORST CASE — 3rd class male:   13.5% survival
""")
print("=" * 55)
print("  EDA Complete! ✅")
print("=" * 55)
