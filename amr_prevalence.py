"""
================================================================
Day 09 — AMR Prevalence & Resistance Profiling (REAL DATA)
Author  : Subhadip Jana
Dataset : example_isolates — AMR R package
          2,000 clinical isolates × 40 antibiotics (R/S/I)
          15 years of data (2002–2017)

Research Questions:
  1. What is the overall resistance prevalence per antibiotic?
  2. Which species carry the most resistance?
  3. How does resistance differ across clinical wards?
  4. Which antibiotic classes are most/least effective?
  5. What is the multi-drug resistance (MDR) burden?

Antibiotic Classes Covered:
  Penicillins    : PEN, OXA, FLC, AMX, AMC, AMP
  Cephalosporins : TZP, CZO, FEP, CXM, FOX, CTX, CAZ, CRO
  Aminoglycosides: GEN, TOB, AMK, KAN
  Sulfonamides   : TMP, SXT
  Fluoroquinolones: CIP, MFX
  Glycopeptides  : VAN, TEC
  Macrolides     : ERY, CLI, AZM
  Carbapenems    : IPM, MEM
  Others         : NIT, FOS, LNZ, TCY, TGC, DOX, MTR, CHL, COL, MUP, RIF
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import chi2_contingency, fisher_exact
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# SECTION 1: LOAD & CLEAN DATA
# ─────────────────────────────────────────────────────────────

print("🔬 Loading example_isolates dataset...")
df = pd.read_csv("data/isolates.csv")
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year

# Antibiotic columns
META_COLS = ["date","patient","age","gender","ward","mo","year"]
AB_COLS   = [c for c in df.columns if c not in META_COLS]

# Antibiotic class mapping
AB_CLASS = {
    "PEN":"Penicillin","OXA":"Penicillin","FLC":"Penicillin",
    "AMX":"Penicillin","AMC":"Penicillin","AMP":"Penicillin",
    "TZP":"Cephalosporin","CZO":"Cephalosporin","FEP":"Cephalosporin",
    "CXM":"Cephalosporin","FOX":"Cephalosporin","CTX":"Cephalosporin",
    "CAZ":"Cephalosporin","CRO":"Cephalosporin",
    "GEN":"Aminoglycoside","TOB":"Aminoglycoside",
    "AMK":"Aminoglycoside","KAN":"Aminoglycoside",
    "TMP":"Sulfonamide","SXT":"Sulfonamide",
    "CIP":"Fluoroquinolone","MFX":"Fluoroquinolone",
    "VAN":"Glycopeptide","TEC":"Glycopeptide",
    "ERY":"Macrolide","CLI":"Macrolide","AZM":"Macrolide",
    "IPM":"Carbapenem","MEM":"Carbapenem",
    "NIT":"Other","FOS":"Other","LNZ":"Other","TCY":"Tetracycline",
    "TGC":"Tetracycline","DOX":"Tetracycline","MTR":"Other",
    "CHL":"Other","COL":"Polymyxin","MUP":"Other","RIF":"Other",
}

CLASS_COLORS = {
    "Penicillin"     : "#E74C3C",
    "Cephalosporin"  : "#E67E22",
    "Aminoglycoside" : "#F1C40F",
    "Sulfonamide"    : "#2ECC71",
    "Fluoroquinolone": "#1ABC9C",
    "Glycopeptide"   : "#3498DB",
    "Macrolide"      : "#9B59B6",
    "Carbapenem"     : "#E91E63",
    "Tetracycline"   : "#795548",
    "Polymyxin"      : "#607D8B",
    "Other"          : "#BDC3C7",
}

# Species short names
SPECIES_NAMES = {
    "B_ESCHR_COLI" : "E. coli",
    "B_STPHY_CONS" : "S. cons.",
    "B_STPHY_AURS" : "S. aureus",
    "B_STPHY_EPDR" : "S. epidermidis",
    "B_STRPT_PNMN" : "S. pneumoniae",
    "B_KLBSL_PNMN" : "K. pneumoniae",
    "B_STRPT_PYOG" : "S. pyogenes",
    "B_ENCCS_FCLS" : "E. faecalis",
    "B_ENCCS_FECM" : "E. faecium",
    "B_PSDMN_AERG" : "P. aeruginosa",
}

df["species"] = df["mo"].map(SPECIES_NAMES).fillna(df["mo"].str[-4:])

print(f"✅ {len(df)} isolates × {len(AB_COLS)} antibiotics")
print(f"   Species: {df['mo'].nunique()} | Ward: {df['ward'].nunique()}")
print(f"   Date range: {df['year'].min()}–{df['year'].max()}")

# ─────────────────────────────────────────────────────────────
# SECTION 2: RESISTANCE PREVALENCE PER ANTIBIOTIC
# ─────────────────────────────────────────────────────────────

print("\n📊 Computing resistance prevalence...")

prev_records = []
for ab in AB_COLS:
    non_null = df[ab].dropna()
    if len(non_null) < 50:        # skip if too few tested
        continue
    n_total = len(non_null)
    n_R     = (non_null == "R").sum()
    n_S     = (non_null == "S").sum()
    n_I     = (non_null == "I").sum()
    pct_R   = n_R / n_total * 100
    pct_S   = n_S / n_total * 100
    pct_I   = n_I / n_total * 100
    prev_records.append({
        "Antibiotic"  : ab,
        "Class"       : AB_CLASS.get(ab, "Other"),
        "N_tested"    : n_total,
        "N_R"         : n_R,
        "N_S"         : n_S,
        "N_I"         : n_I,
        "Pct_R"       : round(pct_R, 2),
        "Pct_S"       : round(pct_S, 2),
        "Pct_I"       : round(pct_I, 2),
    })

prev_df = pd.DataFrame(prev_records).sort_values("Pct_R", ascending=False)
prev_df.to_csv("outputs/resistance_prevalence.csv", index=False)
print(f"✅ Computed for {len(prev_df)} antibiotics (≥50 isolates tested)")

print("\n📋 Top 10 highest resistance:")
print(prev_df[["Antibiotic","Class","N_tested","Pct_R","Pct_S"]].head(10).to_string(index=False))

# ─────────────────────────────────────────────────────────────
# SECTION 3: SPECIES × ANTIBIOTIC RESISTANCE MATRIX
# ─────────────────────────────────────────────────────────────

print("\n🦠 Computing species × antibiotic resistance matrix...")

# Top 8 species by count
top_species = df["mo"].value_counts().head(8).index.tolist()
best_abs    = prev_df[prev_df["N_tested"] >= 100]["Antibiotic"].tolist()[:20]

species_matrix = {}
for sp in top_species:
    sp_df = df[df["mo"] == sp]
    row   = {}
    for ab in best_abs:
        non_null = sp_df[ab].dropna()
        if len(non_null) >= 5:
            row[ab] = (non_null == "R").mean() * 100
        else:
            row[ab] = np.nan
    species_matrix[SPECIES_NAMES.get(sp, sp[-8:])] = row

sp_ab_df = pd.DataFrame(species_matrix).T
sp_ab_df.to_csv("outputs/species_resistance_matrix.csv")

# ─────────────────────────────────────────────────────────────
# SECTION 4: MULTI-DRUG RESISTANCE (MDR) BURDEN
# ─────────────────────────────────────────────────────────────

print("\n💊 Computing MDR burden...")

# MDR = resistant to ≥3 antibiotic classes
def count_resistant_classes(row):
    resistant_classes = set()
    for ab in AB_COLS:
        if row[ab] == "R" and ab in AB_CLASS:
            resistant_classes.add(AB_CLASS[ab])
    return len(resistant_classes)

df["n_resistant_classes"] = df.apply(count_resistant_classes, axis=1)
df["MDR_status"] = df["n_resistant_classes"].apply(
    lambda x: "XDR (≥5)" if x >= 5
    else "MDR (3–4)" if x >= 3
    else "Non-MDR" if x >= 1
    else "Susceptible"
)

mdr_counts = df["MDR_status"].value_counts()
print("\nMDR Distribution:")
print(mdr_counts.to_string())

# MDR by species
mdr_by_species = pd.crosstab(df["species"], df["MDR_status"], normalize="index") * 100
mdr_by_species = mdr_by_species.loc[
    [SPECIES_NAMES.get(s, s) for s in top_species if SPECIES_NAMES.get(s, s) in mdr_by_species.index]
]

# MDR by ward
mdr_by_ward = pd.crosstab(df["ward"], df["MDR_status"], normalize="index") * 100

# ─────────────────────────────────────────────────────────────
# SECTION 5: WARD-LEVEL RESISTANCE
# ─────────────────────────────────────────────────────────────

print("\n🏥 Computing ward-level resistance...")

ward_prev = {}
for ward in ["Clinical","ICU","Outpatient"]:
    ward_df = df[df["ward"] == ward]
    row     = {}
    for ab in best_abs[:12]:
        non_null = ward_df[ab].dropna()
        if len(non_null) >= 10:
            row[ab] = (non_null == "R").mean() * 100
        else:
            row[ab] = np.nan
    ward_prev[ward] = row

ward_df_heatmap = pd.DataFrame(ward_prev).T

# Chi-square for ICU vs Clinical resistance (top ab)
chi_results = []
for ab in best_abs[:10]:
    sub = df[df["ward"].isin(["ICU","Clinical"])][["ward",ab]].dropna()
    if len(sub) < 20:
        continue
    ct = pd.crosstab(sub["ward"], sub[ab])
    if ct.shape == (2,2):
        _, p, _, _ = chi2_contingency(ct)
        icu_r = (sub[sub["ward"]=="ICU"][ab]=="R").mean()*100
        clin_r= (sub[sub["ward"]=="Clinical"][ab]=="R").mean()*100
        chi_results.append({"Antibiotic":ab, "ICU_R%":round(icu_r,1),
                            "Clinical_R%":round(clin_r,1), "p_value":round(p,4)})

chi_df = pd.DataFrame(chi_results).sort_values("p_value")
print("\nICU vs Clinical (top significant):")
print(chi_df.head(8).to_string(index=False))

# ─────────────────────────────────────────────────────────────
# SECTION 6: DASHBOARD (9 panels)
# ─────────────────────────────────────────────────────────────

print("\n🎨 Generating dashboard...")

MDR_COLORS = {
    "Susceptible" : "#2ECC71",
    "Non-MDR"     : "#F39C12",
    "MDR (3–4)"   : "#E74C3C",
    "XDR (≥5)"    : "#8E44AD",
}
WARD_COLORS = {"ICU":"#E74C3C","Clinical":"#3498DB","Outpatient":"#2ECC71"}

fig = plt.figure(figsize=(24, 20))
fig.suptitle(
    "AMR Prevalence & Resistance Profiling — REAL CLINICAL DATA\n"
    "example_isolates | AMR R package | 2,000 isolates × 40 antibiotics | 2002–2017",
    fontsize=15, fontweight="bold", y=0.99
)

# ── Plot 1: Resistance % per antibiotic (horizontal bar) ──
ax1 = fig.add_subplot(3, 3, 1)
plot_prev = prev_df.head(20)
bar_colors = [CLASS_COLORS[AB_CLASS.get(ab,"Other")]
              for ab in plot_prev["Antibiotic"]]
bars = ax1.barh(range(len(plot_prev)), plot_prev["Pct_R"].values[::-1],
                color=bar_colors[::-1], edgecolor="black", linewidth=0.4, alpha=0.87)
ax1.set_yticks(range(len(plot_prev)))
ax1.set_yticklabels(plot_prev["Antibiotic"].values[::-1], fontsize=8)
ax1.set_xlabel("% Resistant Isolates")
ax1.set_title("Resistance Prevalence\n(Top 20 antibiotics, ≥50 tested)",
              fontweight="bold", fontsize=10)
ax1.axvline(50, color="gray", lw=1, linestyle="--", alpha=0.6)
# Class legend
patches = [mpatches.Patch(color=c, label=k)
           for k, c in CLASS_COLORS.items() if k != "Other"]
ax1.legend(handles=patches, fontsize=6, loc="lower right", ncol=2)

# ── Plot 2: R/S/I stacked bar for best antibiotics ──
ax2 = fig.add_subplot(3, 3, 2)
plot_rsi = prev_df[prev_df["N_tested"] >= 200].head(14)
y_pos    = np.arange(len(plot_rsi))
ax2.barh(y_pos, plot_rsi["Pct_R"].values, color="#E74C3C",
         label="R", alpha=0.85, edgecolor="white")
ax2.barh(y_pos, plot_rsi["Pct_I"].values, left=plot_rsi["Pct_R"].values,
         color="#F39C12", label="I", alpha=0.85, edgecolor="white")
ax2.barh(y_pos, plot_rsi["Pct_S"].values,
         left=plot_rsi["Pct_R"].values + plot_rsi["Pct_I"].values,
         color="#2ECC71", label="S", alpha=0.85, edgecolor="white")
ax2.set_yticks(y_pos)
ax2.set_yticklabels(plot_rsi["Antibiotic"].values, fontsize=8)
ax2.set_xlabel("Percentage (%)")
ax2.set_title("R / I / S Distribution\n(≥200 isolates tested)",
              fontweight="bold", fontsize=10)
ax2.legend(fontsize=9, loc="lower right")
ax2.axvline(50, color="black", lw=0.8, linestyle="--", alpha=0.4)

# ── Plot 3: Species × Antibiotic heatmap ──
ax3 = fig.add_subplot(3, 3, 3)
if not sp_ab_df.empty:
    plot_sp = sp_ab_df[best_abs[:14]]
    sns.heatmap(plot_sp, ax=ax3, cmap="YlOrRd", annot=True, fmt=".0f",
                linewidths=0.4, cbar_kws={"label":"%R","shrink":0.8},
                vmin=0, vmax=100)
    ax3.tick_params(axis="x", labelsize=7, rotation=45)
    ax3.tick_params(axis="y", labelsize=7, rotation=0)
ax3.set_title("Species × Antibiotic Resistance (%R)\n(Top 8 species)",
              fontweight="bold", fontsize=10)

# ── Plot 4: MDR burden pie ──
ax4 = fig.add_subplot(3, 3, 4)
mdr_order  = ["Susceptible","Non-MDR","MDR (3–4)","XDR (≥5)"]
mdr_vals   = [mdr_counts.get(k, 0) for k in mdr_order]
mdr_colors = [MDR_COLORS[k] for k in mdr_order]
wedges, texts, autotexts = ax4.pie(
    mdr_vals, labels=mdr_order, colors=mdr_colors, autopct="%1.1f%%",
    startangle=90, pctdistance=0.78,
    wedgeprops={"edgecolor":"white","linewidth":2}
)
for at in autotexts:
    at.set_fontsize(9); at.set_fontweight("bold")
ax4.set_title("Multi-Drug Resistance (MDR)\nBurden (all isolates)",
              fontweight="bold", fontsize=10)

# ── Plot 5: MDR by species stacked bar ──
ax5 = fig.add_subplot(3, 3, 5)
if not mdr_by_species.empty:
    cols_order = [c for c in mdr_order if c in mdr_by_species.columns]
    bottom = np.zeros(len(mdr_by_species))
    for col in cols_order:
        if col in mdr_by_species.columns:
            ax5.bar(range(len(mdr_by_species)),
                    mdr_by_species[col].values,
                    bottom=bottom, color=MDR_COLORS[col],
                    label=col, edgecolor="white", alpha=0.87)
            bottom += mdr_by_species[col].values
    ax5.set_xticks(range(len(mdr_by_species)))
    ax5.set_xticklabels(mdr_by_species.index, fontsize=7, rotation=30, ha="right")
    ax5.set_ylabel("% of Isolates")
    ax5.set_title("MDR Burden by Species",
                  fontweight="bold", fontsize=10)
    ax5.legend(fontsize=7, loc="upper right")

# ── Plot 6: Ward resistance heatmap ──
ax6 = fig.add_subplot(3, 3, 6)
if not ward_df_heatmap.empty:
    sns.heatmap(ward_df_heatmap, ax=ax6, cmap="YlOrRd", annot=True,
                fmt=".1f", linewidths=0.4,
                cbar_kws={"label":"%R","shrink":0.8},
                vmin=0, vmax=100)
    ax6.tick_params(axis="x", labelsize=7, rotation=45)
    ax6.tick_params(axis="y", labelsize=8, rotation=0)
ax6.set_title("Ward-level Resistance (%R)\nICU vs Clinical vs Outpatient",
              fontweight="bold", fontsize=10)

# ── Plot 7: Antibiotic class resistance overview ──
ax7 = fig.add_subplot(3, 3, 7)
class_prev = prev_df.groupby("Class")["Pct_R"].mean().sort_values(ascending=False)
bar_c = [CLASS_COLORS.get(c,"#BDC3C7") for c in class_prev.index]
bars7 = ax7.bar(range(len(class_prev)), class_prev.values,
                color=bar_c, edgecolor="black", linewidth=0.5, alpha=0.87)
for bar, val in zip(bars7, class_prev.values):
    ax7.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f"{val:.1f}%", ha="center", fontsize=8, fontweight="bold")
ax7.set_xticks(range(len(class_prev)))
ax7.set_xticklabels(class_prev.index, fontsize=8, rotation=35, ha="right")
ax7.set_ylabel("Mean % Resistant")
ax7.set_title("Mean Resistance by\nAntibiotic Class",
              fontweight="bold", fontsize=10)
ax7.axhline(50, color="red", lw=1, linestyle="--", alpha=0.5)

# ── Plot 8: Resistance classes per isolate distribution ──
ax8 = fig.add_subplot(3, 3, 8)
resist_class_counts = df["n_resistant_classes"].value_counts().sort_index()
ax8.bar(resist_class_counts.index, resist_class_counts.values,
        color=["#2ECC71" if x==0
               else "#F39C12" if x<=2
               else "#E74C3C" if x<=4
               else "#8E44AD"
               for x in resist_class_counts.index],
        edgecolor="black", linewidth=0.5, alpha=0.85)
ax8.axvline(2.5, color="orange", lw=2, linestyle="--", alpha=0.7, label="MDR threshold")
ax8.axvline(4.5, color="purple", lw=2, linestyle="--", alpha=0.7, label="XDR threshold")
ax8.set_xlabel("Number of Resistant Antibiotic Classes")
ax8.set_ylabel("Number of Isolates")
ax8.set_title("Resistance Breadth per Isolate\n(0=fully susceptible)",
              fontweight="bold", fontsize=10)
ax8.legend(fontsize=8)

# ── Plot 9: Summary table ──
ax9 = fig.add_subplot(3, 3, 9)
ax9.axis("off")
rows = [
    ["Total isolates",       "2,000"],
    ["Antibiotics profiled", str(len(prev_df))],
    ["Species",              str(df['mo'].nunique())],
    ["Date range",           "2002–2017"],
    ["Susceptible",          f"{mdr_counts.get('Susceptible',0)} ({mdr_counts.get('Susceptible',0)/20:.1f}%)"],
    ["Non-MDR",              f"{mdr_counts.get('Non-MDR',0)} ({mdr_counts.get('Non-MDR',0)/20:.1f}%)"],
    ["MDR (≥3 classes)",     f"{mdr_counts.get('MDR (3–4)',0)} ({mdr_counts.get('MDR (3–4)',0)/20:.1f}%)"],
    ["XDR (≥5 classes)",     f"{mdr_counts.get('XDR (≥5)',0)} ({mdr_counts.get('XDR (≥5)',0)/20:.1f}%)"],
    ["Highest resistance",   f"PEN {prev_df.iloc[0]['Pct_R']:.1f}%"],
    ["Lowest resistance",    f"{prev_df.iloc[-1]['Antibiotic']} {prev_df.iloc[-1]['Pct_R']:.1f}%"],
    ["ICU vs Clinical",      f"sig diff: {(chi_df['p_value']<0.05).sum()} antibiotics"],
]
tbl = ax9.table(cellText=rows, colLabels=["Metric","Value"],
                cellLoc="center", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.5, 2.0)
for j in range(2): tbl[(0,j)].set_facecolor("#BDC3C7")
for i, color in enumerate(
    ["#2ECC71","#F39C12","#E74C3C","#8E44AD"], 6):
    tbl[(i,0)].set_facecolor(color)
    tbl[(i,0)].set_text_props(color="white" if color!="#F39C12" else "black",
                               fontweight="bold")
ax9.set_title("Resistance Summary", fontweight="bold", fontsize=11, pad=20)

plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig("outputs/amr_prevalence_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Dashboard saved → outputs/amr_prevalence_dashboard.png")

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"\nTotal isolates      : {len(df)}")
print(f"Antibiotics tested  : {len(prev_df)} (≥50 isolates)")
print(f"\nMDR Burden:")
for k in mdr_order:
    n = mdr_counts.get(k, 0)
    print(f"  {k:15s}: {n:4d} ({n/len(df)*100:.1f}%)")
print(f"\nHighest resistance  : {prev_df.iloc[0]['Antibiotic']} "
      f"({prev_df.iloc[0]['Pct_R']:.1f}%R, {prev_df.iloc[0]['Class']})")
print(f"Lowest resistance   : {prev_df.iloc[-1]['Antibiotic']} "
      f"({prev_df.iloc[-1]['Pct_R']:.1f}%R, {prev_df.iloc[-1]['Class']})")
print(f"\nICU vs Clinical sig differences: "
      f"{(chi_df['p_value']<0.05).sum()} antibiotics")
print("\n✅ All outputs saved!")
