import time
from io import StringIO

import numpy as np
import pandas
import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy import stats

import process_CSV

# ============================================
# FIX: Matplotlib configuration for proper sizing
# ============================================
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['savefig.dpi'] = 100
matplotlib.rcParams['figure.autolayout'] = True
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['axes.labelsize'] = 12

# Charger le set de donnée en global
data_path = os.path.join("./plane_ticket_price_original.csv")
df = pd.read_csv(data_path, delimiter=',')

# Liste de couleurs pour cycler
DIVIDER_COLORS = ["blue", "green", "orange", "red", "violet", "yellow"]
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


# FIX: Removed @st.fragment decorator
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
    st.subheader(":mag: Valeurs manquantes", divider=get_next_color())

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
        st.warning(f":warning: {len(missing_summary)} colonne(s) contiennent des valeurs manquantes")
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
        st.pyplot(fig, use_container_width=True)
        plt.close()
    else:
        st.success(":white_check_mark: Aucune valeur manquante détectée dans le dataset")

    st.divider()

    # === VALEURS DUPLIQUÉES ===
    st.subheader(":repeat: Valeurs dupliquées", divider=get_next_color())

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
        st.warning(f":warning: {duplicate_rows} ligne(s) dupliquée(s) détectée(s)")

        with st.expander("Voir des exemples de lignes dupliquées (max 10)"):
            duplicates = data_df[data_df.duplicated(keep=False)].sort_values(by=data_df.columns.tolist()).head(10)
            st.dataframe(duplicates)
    else:
        st.success(":white_check_mark: Aucune ligne dupliquée détectée dans le dataset")

    st.divider()

    # === VALEURS ABERRANTES ===
    st.subheader(":bar_chart: Analyse des valeurs aberrantes (Price)", divider=get_next_color())

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
    with st.expander(":chart_with_upwards_trend: Statistiques détaillées"):
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
    col1, col2 = st.columns(2)

    # === Colonne 1: Boxplot ===
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')

        ax.boxplot(data_df['Price'], vert=True, patch_artist=True,
                   boxprops=dict(facecolor='#1c83e1', alpha=0.7),
                   medianprops=dict(color='#ff5252', linewidth=2),
                   whiskerprops=dict(color='#1c83e1', linewidth=1.5),
                   capprops=dict(color='#1c83e1', linewidth=1.5),
                   flierprops=dict(marker='o', markerfacecolor='#ff5252', markersize=5, alpha=0.5))

        ax.set_title('Boxplot des Prix', fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel('Prix (₹)', fontsize=11)
        ax.axhline(y=lower_bound, color='#ff9800', linestyle='--', linewidth=2, label='Limite inférieure', alpha=0.8)
        ax.axhline(y=upper_bound, color='#ff9800', linestyle='--', linewidth=2, label='Limite supérieure', alpha=0.8)
        ax.legend(loc='best', framealpha=0.9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # === Colonne 2: Histogramme ===
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')

        n, bins, patches = ax.hist(data_df['Price'], bins=50, edgecolor='white', linewidth=0.5, alpha=0.9)

        # Colorer les barres selon qu'elles contiennent des outliers
        for i, patch in enumerate(patches):
            bin_center = (bins[i] + bins[i + 1]) / 2
            if bin_center < lower_bound or bin_center > upper_bound:
                patch.set_facecolor('#ff5252')
            else:
                patch.set_facecolor('#1c83e1')

        ax.axvline(x=lower_bound, color='#ff9800', linestyle='--', linewidth=2, label='Limite inférieure', alpha=0.8)
        ax.axvline(x=upper_bound, color='#ff9800', linestyle='--', linewidth=2, label='Limite supérieure', alpha=0.8)
        ax.axvline(data_df['Price'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Moyenne: {data_df["Price"].mean():.0f}')
        ax.axvline(data_df['Price'].median(), color='green', linestyle='--', linewidth=2,
                   label=f'Médiane: {data_df["Price"].median():.0f}')

        ax.set_title('Distribution des Prix', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Prix (₹)', fontsize=11)
        ax.set_ylabel('Fréquence', fontsize=11)
        ax.legend(loc='best', framealpha=0.9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    if n_outliers > 0:
        with st.expander(f":mag: Voir les {n_outliers} outliers détectés"):
            st.dataframe(outliers)


# FIX: Removed @st.fragment decorator
def four(data_df: pd.DataFrame):
    st.write("## 4. Distribution des prix selon différents critères")
    st.subheader(":seat: Compagnies :", divider=get_next_color())
    sns.set_theme(style="white")

    fig, ax = plt.subplots(figsize=(14, 6))

    airline_order = data_df.groupby('Airline')['Price'].median().sort_values().index

    sns.violinplot(
        data=data_df,
        x='Airline',
        y='Price',
        order=airline_order,
        palette='Set2',
        inner='box',
        linewidth=1,
        ax=ax
    )

    ax.set_title('Distribution des prix par compagnie aérienne', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('')
    ax.set_ylabel('Prix (INR)', fontsize=11, color='#555')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(left=False)
    ax.yaxis.grid(True, alpha=0.25, linestyle='-')
    ax.set_axisbelow(True)
    plt.xticks(rotation=45, ha='right', fontsize=10)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    stats = data_df.groupby('Airline')['Price'].describe().round(2)

    col1, col2, col3 = st.columns(3)
    col1.metric("Compagnie la plus chère", f"{stats['mean'].idxmax()} — ₹{stats['mean'].max():,.0f}")
    col2.metric("Compagnie la moins chère", f"{stats['mean'].idxmin()} — ₹{stats['mean'].min():,.0f}")
    col3.metric("Écart de prix moyen", f"₹{stats['mean'].max() - stats['mean'].min():,.0f}")

    stats_display = stats.rename(columns={
        'count': 'Nb vols',
        'mean': 'Moyenne',
        'std': 'Écart-type',
        'min': 'Min',
        '25%': 'Q1',
        '50%': 'Médiane',
        '75%': 'Q3',
        'max': 'Max'
    }).sort_values('Moyenne', ascending=True)

    st.dataframe(
        stats_display.style
        .background_gradient(cmap='RdYlGn_r', subset=['Moyenne'])
        .format('{:,.0f}'),
        use_container_width=True,
        height=(len(stats_display) + 1) * 35 + 2
    )

    st.subheader(":city_sunset: Villes :", divider=get_next_color())
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(":round_pushpin: Par ville de départ")

        source_stats = data_df.groupby('Source')['Price'].agg(['median', 'mean', 'std'])

        m1, m2 = st.columns(2)
        m1.metric("Moins chère", source_stats['median'].idxmin())
        m2.metric("Plus chère", source_stats['median'].idxmax())

        fig, ax = plt.subplots(figsize=(10, 6))
        source_order = source_stats['median'].sort_values().index
        sns.boxplot(data=data_df, x='Source', y='Price', order=source_order, palette='Set3', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_xlabel('')
        ax.set_ylabel('Prix (INR)', fontsize=11, color='#555')
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(left=False)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        source_stats_display = source_stats.rename(columns={
            'median': 'Médiane',
            'mean': 'Moyenne',
            'std': 'Écart-type',
        }).sort_values('Moyenne', ascending=True)

        st.dataframe(
            source_stats_display.style
            .background_gradient(cmap='RdYlGn_r', subset=['Moyenne'])
            .format('{:,.0f}'),
            use_container_width=True,
            height=(len(source_stats_display) + 1) * 35 + 2
        )

    with col2:
        st.subheader(":dart: Par destination")

        dest_stats = data_df.groupby('Destination')['Price'].agg(['median', 'mean', 'std'])

        m1, m2 = st.columns(2)
        m1.metric("Moins chère", dest_stats['median'].idxmin())
        m2.metric("Plus chère", dest_stats['median'].idxmax())

        fig, ax = plt.subplots(figsize=(10, 6))
        dest_order = dest_stats['median'].sort_values().index
        sns.boxplot(data=data_df, x='Destination', y='Price', order=dest_order, palette='Set3', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_xlabel('')
        ax.set_ylabel('Prix (INR)', fontsize=11, color='#555')
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(left=False)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        dest_stats_display = dest_stats.rename(columns={
            'median': 'Médiane',
            'mean': 'Moyenne',
            'std': 'Écart-type',
        }).sort_values('Moyenne', ascending=True)

        st.dataframe(
            dest_stats_display.style
            .background_gradient(cmap='RdYlGn_r', subset=['Moyenne'])
            .format('{:,.0f}'),
            use_container_width=True,
            height=(len(dest_stats_display) + 1) * 35 + 2
        )

    st.subheader(":flight_arrival: Escales :", divider=get_next_color())

    # Nettoyer les valeurs nulles pour Total_Stops
    df_stops = data_df.dropna(subset=['Total_Stops'])

    # Compute statistics
    stops_stats = df_stops.groupby('Total_Stops')['Price'].agg(['median', 'mean', 'std']).round(2)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=df_stops, x='Total_Stops', y='Price', palette='viridis', ax=ax)
    ax.set_xlabel('Nombre d\'escales', fontsize=11)
    ax.set_ylabel('Prix (INR)', fontsize=11, color='#555')
    ax.set_title('Boxplot des prix par nombre d\'escales', fontsize=14, fontweight='bold', pad=15)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(left=False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Rename and display
    stops_stats_display = stops_stats.rename(columns={
        'median': 'Médiane',
        'mean': 'Moyenne',
        'std': 'Écart-type',
    }).sort_values('Moyenne', ascending=True)

    st.dataframe(
        stops_stats_display.style
        .background_gradient(cmap='RdYlGn_r', subset=['Moyenne'])
        .format('{:,.0f}'),
        use_container_width=True,
        height=(len(stops_stats_display) + 1) * 35 + 2
    )

    st.subheader(":clock1: Durée du vol :", divider=get_next_color())

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Relation durée vs prix")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(data_df['Duration'], data_df['Price'], alpha=0.3, s=10, c='#1c83e1')
        ax.set_title("Relation entre la durée du vol et le prix", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Durée du vol (minutes)", fontsize=11)
        ax.set_ylabel("Prix (INR)", fontsize=11, color='#555')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("##### Statistiques de durée")
        duration_stats = data_df.groupby(pd.cut(data_df['Duration'], bins=5))['Price'].agg(
            ['count', 'mean', 'median']).round(2)
        duration_stats.columns = ['Nb vols', 'Prix moyen', 'Prix médian']
        st.dataframe(
            duration_stats.style.background_gradient(cmap='Blues', subset=['Prix moyen']).format('{:,.0f}'),
            use_container_width=True
        )

        # Correlation
        correlation = data_df['Duration'].corr(data_df['Price'])
        st.metric("Corrélation Durée-Prix", f"{correlation:.3f}")


def five(data_df: pandas.DataFrame):
    st.write("## 5. Préparation des données pour la modélisation")

    # Encoder les variables catégorielles avec One-Hot Encoding
    df_model = pd.get_dummies(data_df, columns=['Airline', 'Source', 'Destination', 'Additional_Info'], drop_first=True)

    # Définition des features (X) et de la cible (y)
    feature_cols = ['Duration', 'Total_Stops', 'Journey_day', 'Journey_month', 'Departure_hour']
    encoded_cols = [col for col in df_model.columns if
                    col.startswith(('Airline_', 'Source_', 'Destination_', 'Additional_Info_'))]

    X = df_model[feature_cols + encoded_cols]
    y = df_model['Price']

    st.subheader(":gear: Configuration du modèle", divider=get_next_color())

    # Metrics row
    col1, col2, col3 = st.columns(3)
    col1.metric("Nombre de features", X.shape[1])
    col2.metric("Nombre d'observations", f"{X.shape[0]:,}")
    col3.metric("Variable cible", "Price")

    st.divider()

    # Features breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### :1234: Features numériques")
        num_features = pd.DataFrame({
            'Feature': feature_cols,
            'Type': ['Durée (min)', 'Nombre d\'escales', 'Jour', 'Mois', 'Heure'],
            'Description': [
                'Durée totale du vol',
                'Nombre d\'arrêts intermédiaires',
                'Jour du départ',
                'Mois du départ',
                'Heure de départ'
            ]
        })
        st.dataframe(num_features, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("##### :abc: Features catégorielles (One-Hot)")

        # Count encoded columns by category
        cat_summary = pd.DataFrame({
            'Catégorie': ['Airline', 'Source', 'Destination', 'Additional_Info'],
            'Colonnes créées': [
                len([c for c in encoded_cols if c.startswith('Airline_')]),
                len([c for c in encoded_cols if c.startswith('Source_')]),
                len([c for c in encoded_cols if c.startswith('Destination_')]),
                len([c for c in encoded_cols if c.startswith('Additional_Info_')])
            ]
        })
        cat_summary['% du total'] = (cat_summary['Colonnes créées'] / len(encoded_cols) * 100).round(1)
        st.dataframe(cat_summary, hide_index=True, use_container_width=True)

    # Expander with full feature list
    with st.expander(f":mag: Voir toutes les {len(encoded_cols)} features encodées"):
        # Group by category
        tabs = st.tabs(["Airline", "Source", "Destination", "Additional_Info"])

        for i, prefix in enumerate(['Airline_', 'Source_', 'Destination_', 'Additional_Info_']):
            with tabs[i]:
                cols = [c.replace(prefix, '') for c in encoded_cols if c.startswith(prefix)]
                st.write(", ".join(cols))

    # Data shape summary card
    st.info(f"""
    **Résumé de la préparation:**
    - **X** (features): `{X.shape[0]:,}` lignes × `{X.shape[1]}` colonnes
    - **y** (cible): `{y.shape[0]:,}` valeurs (Prix en INR)
    - **Encodage**: One-Hot avec `drop_first=True` pour éviter la multicolinéarité
    """)

    st.subheader(":zero::one: Normalisation des données", divider=get_next_color())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    col1, col2, col3 = st.columns(3)
    col1.metric("Méthode", "StandardScaler")
    col2.metric("Forme de X_scaled", f"{X_scaled.shape[0]:,} × {X_scaled.shape[1]}")
    col3.metric("Statut", "✓ Normalisé")

    # Show before/after comparison
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Avant normalisation")
        before_stats = pd.DataFrame({
            'Métrique': ['Moyenne', 'Écart-type', 'Min', 'Max'],
            'Valeur': [
                f"{X.mean().mean():.2f}",
                f"{X.std().mean():.2f}",
                f"{X.min().min():.2f}",
                f"{X.max().max():.2f}"
            ]
        })
        st.dataframe(before_stats, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("##### Après normalisation")
        after_stats = pd.DataFrame({
            'Métrique': ['Moyenne', 'Écart-type', 'Min', 'Max'],
            'Valeur': [
                f"{X_scaled.mean():.4f} ≈ 0",
                f"{X_scaled.std():.4f} ≈ 1",
                f"{X_scaled.min():.2f}",
                f"{X_scaled.max():.2f}"
            ]
        })
        st.dataframe(after_stats, hide_index=True, use_container_width=True)

    st.success("""
    **StandardScaler** transforme chaque feature pour avoir:
    - **Moyenne = 0** (centrage)
    - **Écart-type = 1** (réduction)

    Formule: `z = (x - μ) / σ`
    """)
    return X_scaled, y, feature_cols, encoded_cols


def six(X_scaled: np.ndarray, y: pd.Series, feature_cols: list, encoded_cols: list):
    st.write("## 6. Modélisation - Régression Linéaire")

    st.subheader(":scissors: Division des données", divider=get_next_color())

    # Division des données
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    col1, col2, col3 = st.columns(3)
    col1.metric("Ensemble d'entraînement", f"{X_train.shape[0]:,}", "80%", delta_color="off")
    col2.metric("Ensemble de test", f"{X_test.shape[0]:,}", "20%", delta_color="off")
    col3.metric("Random state", "42", "reproductible", delta_color="off")

    # Visual split bar
    train_pct = X_train.shape[0] / (X_train.shape[0] + X_test.shape[0])
    col1, col2 = st.columns([int(train_pct * 100), int((1 - train_pct) * 100)])
    with col1:
        st.markdown(
            f'<div style="background: #1c83e1; padding: 10px; border-radius: '
            f'5px 0 0 5px; text-align: center; color: white;">Train: {X_train.shape[0]:,}</div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f'<div style="background: #ff9800; padding: 10px; border-radius: '
            f'0 5px 5px 0; text-align: center; color: white;">Test: {X_test.shape[0]:,}</div>',
            unsafe_allow_html=True
        )

    st.divider()

    st.subheader(":robot_face: Entraînement du modèle", divider=get_next_color())

    # Entraînement avec spinner
    with st.spinner("Entraînement du modèle en cours..."):
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    st.success("✓ Modèle entraîné avec succès!")

    # Évaluation
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.subheader(":chart_with_upwards_trend: Résultats", divider=get_next_color())

    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{r2:.4f}", f"{r2 * 100:.1f}% de variance expliquée", delta_color="off")
    col2.metric("MAE", f"₹{mae:,.0f}", "Erreur absolue moyenne", delta_color="off")
    col3.metric("RMSE", f"₹{rmse:,.0f}", "Erreur quadratique moyenne", delta_color="off")

    # Interpretation card
    if r2 >= 0.8:
        st.success(f"""
        **Interprétation des résultats:**
        - Le modèle explique **{r2 * 100:.1f}%** de la variance des prix ✓ Excellent
        - En moyenne, les prédictions sont à **±{mae:,.0f} INR** du prix réel
        """)
    elif r2 >= 0.6:
        st.info(f"""
        **Interprétation des résultats:**
        - Le modèle explique **{r2 * 100:.1f}%** de la variance des prix — Correct
        - En moyenne, les prédictions sont à **±{mae:,.0f} INR** du prix réel
        """)
    else:
        st.warning(f"""
        **Interprétation des résultats:**
        - Le modèle explique **{r2 * 100:.1f}%** de la variance des prix — À améliorer
        - En moyenne, les prédictions sont à **±{mae:,.0f} INR** du prix réel
        """)

    # Predictions vs Real
    st.subheader(":dart: Analyse des prédictions", divider=get_next_color())

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Prédictions vs Valeurs réelles")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, alpha=0.3, s=10, c='#1c83e1')

        # Perfect prediction line
        max_val = max(y_test.max(), y_pred.max())
        min_val = min(y_test.min(), y_pred.min())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Prédiction parfaite')

        ax.set_xlabel('Prix réel (INR)', fontsize=11)
        ax.set_ylabel('Prix prédit (INR)', fontsize=11)
        ax.legend()
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("##### Distribution des erreurs (résidus)")
        fig, ax = plt.subplots(figsize=(10, 6))
        residuals = y_test - y_pred
        sns.histplot(residuals, kde=True, ax=ax, color='#1c83e1', edgecolor='white', alpha=0.8)
        ax.axvline(x=0, color='red', linestyle='--', lw=2, label='Erreur = 0')
        ax.axvline(x=residuals.mean(), color='green', linestyle='--', lw=2, label=f'Moyenne: {residuals.mean():.0f}')

        ax.set_xlabel('Erreur (Prix réel - Prix prédit)', fontsize=11)
        ax.set_ylabel('Fréquence', fontsize=11)
        ax.set_title("Distribution des erreurs (résidus)", fontsize=14, fontweight='bold', pad=15)
        ax.legend()
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Residuals analysis
    st.subheader(":mag: Analyse des résidus", divider=get_next_color())

    residuals = y_test - y_pred

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Moyenne des résidus", f"₹{residuals.mean():,.0f}")
    col2.metric("Écart-type", f"₹{residuals.std():,.0f}")
    col3.metric("Min", f"₹{residuals.min():,.0f}")
    col4.metric("Max", f"₹{residuals.max():,.0f}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Résidus vs Valeurs prédites")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_pred, residuals, alpha=0.3, s=10, c='#1c83e1')
        ax.axhline(0, color='red', linestyle='--', lw=2, label='Résidu = 0')

        ax.set_xlabel('Prix prédit (INR)', fontsize=11)
        ax.set_ylabel('Résidu (INR)', fontsize=11)
        ax.set_title("Résidus vs Valeurs prédites", fontsize=14, fontweight='bold', pad=15)
        ax.legend()
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.caption("💡 Un bon modèle devrait avoir des résidus répartis uniformément autour de 0.")

    with col2:
        st.markdown("##### Q-Q Plot des résidus")
        fig, ax = plt.subplots(figsize=(10, 6))
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title("Q-Q Plot (Normalité des résidus)", fontsize=14, fontweight='bold', pad=15)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.caption("💡 Si les points suivent la ligne rouge, les résidus sont normalement distribués.")

    # Coefficients analysis
    st.subheader(":bar_chart: Poids des variables", divider=get_next_color())

    feature_names = feature_cols + encoded_cols
    coef_df = pd.DataFrame({
        'Variable': feature_names,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', ascending=True)

    # Top positive and negative
    col1, col2 = st.columns(2)
    with col1:
        top_positive = coef_df.nlargest(3, 'Coefficient')
        st.markdown("**🔺 Variables augmentant le prix:**")
        for _, row in top_positive.iterrows():
            st.write(f"- **{row['Variable']}**: +{row['Coefficient']:.2f}")

    with col2:
        top_negative = coef_df.nsmallest(3, 'Coefficient')
        st.markdown("**🔻 Variables diminuant le prix:**")
        for _, row in top_negative.iterrows():
            st.write(f"- **{row['Variable']}**: {row['Coefficient']:.2f}")

    # Full coefficients plot
    st.markdown("##### Tous les coefficients de la régression")

    # Calculate appropriate height based on number of features
    num_features = len(coef_df)
    fig_height = max(10, int(num_features * 0.4))

    fig, ax = plt.subplots(figsize=(12, fig_height))
    colors = ['#1c83e1' if c >= 0 else '#ff5252' for c in coef_df['Coefficient']]
    ax.barh(coef_df['Variable'], coef_df['Coefficient'], color=colors)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Coefficient', fontsize=11)
    ax.set_ylabel('')
    ax.set_title("Poids des variables dans la régression linéaire", fontsize=14, fontweight='bold', pad=15)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Coefficients table
    with st.expander(":page_facing_up: Voir le tableau complet des coefficients"):
        coef_display = coef_df.sort_values('Coefficient', key=abs, ascending=False).reset_index(drop=True)
        coef_display['Coefficient'] = coef_display['Coefficient'].round(4)
        coef_display['Impact'] = coef_display['Coefficient'].apply(
            lambda x: '🔺 Augmente' if x > 0 else '🔻 Diminue'
        )
        st.dataframe(
            coef_display,
            hide_index=True,
            use_container_width=True,
            height=400
        )


def run():
    st.set_page_config(
        page_title="Plane ticket price dashboard",
        page_icon=":flight_departure:",
        layout="wide",
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

    # Display selected analysis
    with st.spinner("Merci de patienter...", show_time=True):
        if data_view == "Données originales":
            three(df, clean=False)
        else:
            three(cleaned_df, clean=True, original_df=df)

    st.divider()

    four(cleaned_df)

    st.divider()

    X_scaled, y, feature_cols, encoded_cols = five(cleaned_df)

    st.divider()

    six(X_scaled, y, feature_cols, encoded_cols)


if __name__ == "__main__":
    run()
