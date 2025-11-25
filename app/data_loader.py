import ast
import json
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

    if "." in target_table:
        schema, plain_table = target_table.split(".", 1)
    else:
        schema, plain_table = "public", target_table

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s
              AND table_name = %s
            """,
            (schema, plain_table),
        )
        columns = {row[0] for row in cur.fetchall()}

    text_column = "text" if "text" in columns else "document" if "document" in columns else None
    if text_column is None:
        raise ValueError(
            f"Table '{target_table}' must contain either a 'text' or 'document' column"
        )

    query = f"""
        SELECT id, {text_column} AS text, embedding
        FROM {target_table}
    """
    df = pd.read_sql(query, conn)
    df["embedding"] = df["embedding"].apply(_coerce_embedding_to_float_array)
    return df


def _coerce_embedding_to_float_array(value):
    """Normalize embedding values to float numpy arrays.

    Handles arrays returned as Python lists/tuples/ndarrays as well as string
    representations such as "[-0.1, 0.2]" that may come from certain
    PostgreSQL drivers or serialization layers.
    """

    if isinstance(value, np.ndarray):
        return value.astype(float)

    if isinstance(value, (list, tuple)):
        return np.array(value, dtype=float)

    if isinstance(value, str):
        cleaned = value.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(cleaned)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(
                    "Embedding value could not be parsed from string format"
                ) from exc

        return np.array(parsed, dtype=float)

    raise ValueError(f"Unsupported embedding type: {type(value)!r}")
