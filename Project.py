import time
from io import StringIO

import streamlit as st

import pandas as pd
import os


# Charger le set de donnée en global
data_path = os.path.join("./plane ticket price.csv")
df = pd.read_csv(data_path, delimiter=',')

# Liste de couleurs pour cycler
DIVIDER_COLORS = ["blue", "green", "orange", "red", "violet", "yellow", "gray"]
color_index = 0


def get_next_color()->str:
    global color_index
    color = DIVIDER_COLORS[color_index % len(DIVIDER_COLORS)]
    color_index += 1
    return color


def stream_text(text: str, timer: float = 0.2):
    for word in text.split(' '):
        yield word + " "
        time.sleep(timer)


def one():
    st.write("## 1. Aperçu rapide :")

    # Afficher Head
    st.subheader("head()", divider=get_next_color())
    st.dataframe(df.head(), use_container_width=True, hide_index=True)


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
    st.dataframe(df.describe(), use_container_width=False)


def three():
    st.write("## Analyse exploratoire et diagnostique :")


def run():
    st.set_page_config(
        page_title="Plane ticket price dashboard",
        page_icon=":flight_departure:",
    )

    st.write("# Bienvenue sur notre dashboard de machine learning !")
    st.markdown("[Lien de notre dataset sur kaggle]"
                "(https://www.kaggle.com/datasets/ibrahimelsayed182/plane-ticket-price)")

    one()
    st.divider()
    two()
    st.divider()
    three()


if __name__ == "__main__":
    run()
