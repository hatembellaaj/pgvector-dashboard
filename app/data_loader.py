import psycopg2
import pandas as pd
import numpy as np
import streamlit as st


@st.cache_resource
def get_connection():
    return psycopg2.connect(
        dbname=st.secrets["db_name"],
        user=st.secrets["db_user"],
        password=st.secrets["db_password"],
        host=st.secrets["db_host"],
        port=st.secrets["db_port"],
    )


@st.cache_data
def load_embeddings(table_name="my_embeddings_table"):
    conn = get_connection()
    query = f"""
        SELECT id, text, embedding
        FROM {table_name}
    """
    df = pd.read_sql(query, conn)
    df["embedding"] = df["embedding"].apply(lambda v: np.array(v, dtype=float))
    return df
