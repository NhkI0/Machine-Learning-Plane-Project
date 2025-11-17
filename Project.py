import time
from io import StringIO

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

import process_CSV

# Charger le set de donnée en global
data_path = os.path.join("./plane_ticket_price_original.csv")
df = pd.read_csv(data_path, delimiter=',')

# Liste de couleurs pour cycler
DIVIDER_COLORS = ["blue", "green", "orange", "red", "violet", "yellow", "gray"]
color_index = 0


def get_next_color() -> str:
    global color_index
    color = DIVIDER_COLORS[color_index % len(DIVIDER_COLORS)]
    color_index += 1
    return color


def stream_text(text: str, timer: float = 0.2):
    for word in text.split(' '):
        yield word + " "
        time.sleep(timer)


def clean_csv() -> pd.DataFrame:
    return process_CSV.process("./plane_ticket_price_original.csv")


def one():
    st.write("## 1. Aperçu rapide :")

    # Afficher Head
    st.subheader("head()", divider=get_next_color())
    st.dataframe(df.head(), hide_index=True)


def two():
    st.write("## 2. Informations générales :")
    # Informations générales
    st.subheader("info() & shape", divider=get_next_color())

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Nombre de lignes", df.shape[0])
        st.metric("Nombre de colonnes", df.shape[1])

    with col2:
        st.metric("Mémoire utilisée", f"{df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
        st.metric("Valeurs manquantes", df.isnull().sum().sum())

    with col3:
        st.write("**Types de données:**")
        st.write(df.dtypes.value_counts())

    st.divider()

    with st.expander("Détails complets"):
        buffer = StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

    st.subheader("columns", divider=get_next_color())
    for col in df.columns:
        st.write(f"- **{col}**: {df[col].dtype}")

    st.subheader("describe()", divider=get_next_color())
    st.dataframe(df.describe())


def three(data_df: pd.DataFrame, clean: bool = False, original_df: pd.DataFrame = None) -> pd.DataFrame | None:
    # Show comparison summary if viewing cleaned data
    if clean and original_df is not None:
        with st.container():
            st.info("📊 **Résumé des améliorations après nettoyage**")

            # Calculate improvements
            orig_missing = original_df.isnull().sum().sum()
            clean_missing = data_df.isnull().sum().sum()
            orig_duplicates = original_df.duplicated().sum()
            clean_duplicates = data_df.duplicated().sum()
            orig_rows = len(original_df)
            clean_rows = len(data_df)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                missing_diff = orig_missing - clean_missing
                st.metric(
                    "Valeurs manquantes",
                    f"{clean_missing:,}",
                    delta=f"-{missing_diff:,}" if missing_diff > 0 else "0",
                    delta_color="inverse"
                )
            with col2:
                dup_diff = orig_duplicates - clean_duplicates
                st.metric(
                    "Lignes dupliquées",
                    f"{clean_duplicates:,}",
                    delta=f"-{dup_diff:,}" if dup_diff > 0 else "0",
                    delta_color="inverse"
                )
            with col3:
                rows_diff = orig_rows - clean_rows
                st.metric(
                    "Nombre de lignes",
                    f"{clean_rows:,}",
                    delta=f"-{rows_diff:,}" if rows_diff > 0 else "0",
                    delta_color="off"
                )
            with col4:
                quality_before = ((orig_rows - orig_duplicates - orig_missing)
                                  / orig_rows * 100) if orig_rows > 0 else 0
                quality_after = ((clean_rows - clean_duplicates - clean_missing)
                                 / clean_rows * 100) if clean_rows > 0 else 0
                quality_improvement = quality_after - quality_before
                st.metric(
                    "Qualité des données",
                    f"{quality_after:.1f}%",
                    delta=f"+{quality_improvement:.1f}%" if quality_improvement > 0 else f"{quality_improvement:.1f}%",
                    delta_color="normal"
                )

        st.divider()
    # === VALEURS MANQUANTES ===
    st.subheader("🔍 Valeurs manquantes", divider=get_next_color())

    missing_values = data_df.isnull().sum()
    missing_percentage = (data_df.isnull().sum() / len(data_df)) * 100
    total_missing = missing_values.sum()
    global_missing_pct = (total_missing / (len(data_df) * len(data_df.columns)) * 100)

    # Métriques principales
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de valeurs manquantes", f"{total_missing:,}")
    with col2:
        st.metric("Pourcentage global", f"{global_missing_pct:.2f}%")
        st.progress(min(global_missing_pct / 100, 1.0))
    with col3:
        st.metric("Colonnes affectées", (missing_values > 0).sum())

    # Résumé des valeurs manquantes
    missing_summary = pd.DataFrame({
        'Colonne': data_df.columns.astype(str),
        'Valeurs manquantes': missing_values.values.astype(int),
        'Pourcentage (%)': missing_percentage.values.astype(float)
    })
    missing_summary = missing_summary[missing_summary['Valeurs manquantes'] > 0].sort_values('Valeurs manquantes',
                                                                                             ascending=False)

    if len(missing_summary) > 0:
        st.warning(f"⚠️ {len(missing_summary)} colonne(s) contiennent des valeurs manquantes")
        st.dataframe(missing_summary, hide_index=True)

        # Graphique des valeurs manquantes avec style moderne
        fig, ax = plt.subplots(figsize=(10, 5))
        cmap = plt.colormaps['RdYlGn_r']
        colors = cmap(missing_summary['Pourcentage (%)'] / 100)
        missing_summary.plot(x='Colonne', y='Pourcentage (%)', kind='bar', ax=ax, color=colors, legend=False)
        ax.set_title('Pourcentage de valeurs manquantes par colonne', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Colonnes', fontsize=11)
        ax.set_ylabel('Pourcentage (%)', fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.success("✅ Aucune valeur manquante détectée dans le dataset")

    st.divider()

    # === VALEURS DUPLIQUÉES ===
    st.subheader("🔁 Valeurs dupliquées", divider=get_next_color())

    total_rows = len(data_df)
    duplicate_rows = data_df.duplicated().sum()
    duplicate_percentage = (duplicate_rows / total_rows) * 100
    unique_rows = total_rows - duplicate_rows

    # Métriques avec barres de progression
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de lignes", f"{total_rows:,}")
    with col2:
        st.metric("Lignes uniques", f"{unique_rows:,}")
        st.progress((unique_rows / total_rows) if total_rows > 0 else 0)
    with col3:
        st.metric("Lignes dupliquées", f"{duplicate_rows:,}")
        st.progress(min(duplicate_percentage / 100, 1.0))
    with col4:
        st.metric("Pourcentage dupliqué", f"{duplicate_percentage:.2f}%")

    if duplicate_rows > 0:
        st.warning(f"⚠️ {duplicate_rows} ligne(s) dupliquée(s) détectée(s)")

        with st.expander("Voir des exemples de lignes dupliquées (max 10)"):
            duplicates = data_df[data_df.duplicated(keep=False)].sort_values(by=data_df.columns.tolist()).head(10)
            st.dataframe(duplicates)
    else:
        st.success("✅ Aucune ligne dupliquée détectée dans le dataset")

    st.divider()

    # === VALEURS ABERRANTES ===
    st.subheader("📊 Analyse des valeurs aberrantes (Price)", divider=get_next_color())

    # Calcul des quartiles et de l'IQR
    q1 = data_df['Price'].quantile(0.25)
    q3 = data_df['Price'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Détection des outliers
    outliers = data_df[(data_df['Price'] < lower_bound) | (data_df['Price'] > upper_bound)]
    n_outliers = len(outliers)
    outliers_percentage = (n_outliers / len(data_df)) * 100
    clean_data_pct = 100 - outliers_percentage

    # Métriques outliers avec progression
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Outliers détectés", f"{n_outliers:,}")
        st.progress(min(outliers_percentage / 100, 1.0))
    with col2:
        st.metric("Données propres", f"{clean_data_pct:.2f}%")
        st.progress(clean_data_pct / 100)
    with col3:
        if n_outliers > 0:
            st.metric("Prix min outliers", f"₹{outliers['Price'].min():.2f}")
        else:
            st.metric("Prix min", f"₹{data_df['Price'].min():.2f}")
    with col4:
        if n_outliers > 0:
            st.metric("Prix max outliers", f"₹{outliers['Price'].max():.2f}")
        else:
            st.metric("Prix max", f"₹{data_df['Price'].max():.2f}")

    # Statistiques dans un expander
    with st.expander("📈 Statistiques détaillées"):
        stat_col1, stat_col2 = st.columns(2)

        with stat_col1:
            st.write("**Statistiques descriptives:**")
            stats_df = data_df['Price'].describe().to_frame()
            stats_df.columns = ['Valeur']
            st.dataframe(stats_df)

        with stat_col2:
            st.write("**Analyse IQR:**")
            iqr_stats = pd.DataFrame({
                'Métrique': ['Q1 (25e percentile)', 'Q3 (75e percentile)',
                             'IQR', 'Limite inférieure', 'Limite supérieure'],
                'Valeur': [f"{q1:.2f}", f"{q3:.2f}", f"{iqr:.2f}", f"{lower_bound:.2f}", f"{upper_bound:.2f}"]
            })
            st.dataframe(iqr_stats, hide_index=True)

    # Visualisations avec style moderne (outside container to ensure visibility)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('white')

    # Boxplot avec style moderne
    axes[0].boxplot(data_df['Price'], vert=True, patch_artist=True,
                    boxprops=dict(facecolor='#1c83e1', alpha=0.7),
                    medianprops=dict(color='#ff5252', linewidth=2),
                    whiskerprops=dict(color='#1c83e1', linewidth=1.5),
                    capprops=dict(color='#1c83e1', linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='#ff5252', markersize=5, alpha=0.5))
    axes[0].set_title('Boxplot des Prix', fontsize=14, fontweight='bold', pad=15)
    axes[0].set_ylabel('Prix (₹)', fontsize=11)
    axes[0].axhline(y=lower_bound, color='#ff9800', linestyle='--', linewidth=2, label='Limite inférieure', alpha=0.8)
    axes[0].axhline(y=upper_bound, color='#ff9800', linestyle='--', linewidth=2, label='Limite supérieure', alpha=0.8)
    axes[0].legend(loc='best', framealpha=0.9)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')

    # Histogramme avec style moderne
    n, bins, patches = axes[1].hist(data_df['Price'], bins=50, edgecolor='white', linewidth=0.5, alpha=0.9)

    # Colorer les barres selon qu'elles contiennent des outliers
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center < lower_bound or bin_center > upper_bound:
            patch.set_facecolor('#ff5252')
        else:
            patch.set_facecolor('#1c83e1')

    axes[1].axvline(x=lower_bound, color='#ff9800', linestyle='--', linewidth=2, label='Limite inférieure', alpha=0.8)
    axes[1].axvline(x=upper_bound, color='#ff9800', linestyle='--', linewidth=2, label='Limite supérieure', alpha=0.8)
    axes[1].set_title('Distribution des Prix', fontsize=14, fontweight='bold', pad=15)
    axes[1].set_xlabel('Prix (₹)', fontsize=11)
    axes[1].set_ylabel('Fréquence', fontsize=11)
    axes[1].legend(loc='best', framealpha=0.9)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    st.pyplot(fig)

    if n_outliers > 0:
        with st.expander(f"🔍 Voir les {n_outliers} outliers détectés"):
            st.dataframe(outliers)


def run():
    st.set_page_config(
        page_title="Plane ticket price dashboard",
        page_icon=":flight_departure:",
        # layout="wide",
    )

    # Custom CSS
    st.markdown("""
        <style>
        .card {
            padding: 1.5rem;
            border-radius: 0.5rem;
            background-color: rgba(28, 131, 225, 0.1);
            border-left: 5px solid #1c83e1;
            margin-bottom: 1rem;
        }
        .success-card {
            background-color: rgba(0, 200, 83, 0.1);
            border-left: 5px solid #00c853;
        }
        .warning-card {
            background-color: rgba(255, 152, 0, 0.1);
            border-left: 5px solid #ff9800;
        }
        .error-card {
            background-color: rgba(255, 82, 82, 0.1);
            border-left: 5px solid #ff5252;
        }
        div[data-testid="metric-container"] {
            background-color: rgba(28, 131, 225, 0.05);
            border: 1px solid rgba(28, 131, 225, 0.2);
            padding: 10px;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.write("# Bienvenue sur notre dashboard de machine learning !")
    st.markdown("[Lien de notre dataset sur kaggle]"
                "(https://www.kaggle.com/datasets/ibrahimelsayed182/plane-ticket-price)")

    one()
    st.divider()
    two()
    st.divider()

    # Load cleaned data
    with st.spinner("Chargement des données nettoyées...", show_time=True):
        cleaned_df = clean_csv()

    # Radio button toggle for original vs cleaned data
    st.write("## 3. Analyse exploratoire et diagnostique")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        data_view = st.radio(
            "Sélectionnez la version des données à analyser:",
            ["Données originales", "Données nettoyées"],
            horizontal=True,
            label_visibility="collapsed"
        )

    # st.divider()

    # Display selected analysis
    with st.spinner("Merci de patienter...", show_time=True):
        if data_view == "Données originales":
            three(df, clean=False)
        else:
            three(cleaned_df, clean=True, original_df=df)


if __name__ == "__main__":
    run()
