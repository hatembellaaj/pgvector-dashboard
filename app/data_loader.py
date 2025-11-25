import os

import numpy as np
import pandas as pd
import psycopg2
import streamlit as st


def _get_setting(*keys, default=None):
    """Pick the first defined setting from environment variables or Streamlit secrets."""

    for key in keys:
        env_val = os.getenv(key)
        if env_val:
            return env_val

        secret_val = st.secrets.get(key)
        if secret_val:
            return secret_val

    return default


def get_db_config():
    return {
        "dbname": _get_setting("DB_NAME", "db_name"),
        "user": _get_setting("DB_USER", "db_user"),
        "password": _get_setting("DB_PASSWORD", "db_password"),
        "host": _get_setting("DB_HOST", "db_host", default="localhost"),
        "port": _get_setting("DB_PORT", "db_port", default=5432),
        "table_name": _get_setting("DB_TABLE", "table_name", default="my_embeddings_table"),
    }


@st.cache_resource(show_spinner=False)
def get_connection():
    config = get_db_config()
    return psycopg2.connect(
        dbname=config["dbname"],
        user=config["user"],
        password=config["password"],
        host=config["host"],
        port=config["port"],
    )


@st.cache_data(show_spinner=False)
def load_embeddings(table_name=None):
    config = get_db_config()
    target_table = table_name or config["table_name"]

    conn = get_connection()
    query = f"""
        SELECT id, text, embedding
        FROM {target_table}
    """
    df = pd.read_sql(query, conn)
    df["embedding"] = df["embedding"].apply(lambda v: np.array(v, dtype=float))
    return df
