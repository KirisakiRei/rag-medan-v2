import os
import mysql.connector
from mysql.connector import Error
from config import CONFIG

def _get_connection():
    return mysql.connector.connect(
        host=CONFIG["db"]["host"],
        port=CONFIG["db"]["port"],
        database=CONFIG["db"]["database"],
        user=CONFIG["db"]["username"],
        password=CONFIG["db"]["password"],
        autocommit=True
    )

def get_variable(name: str) -> str | None:
    """
    Mengambil content dari table `variables` berdasarkan `name`.
    Return None jika tidak ada / gagal koneksi (biar fallback ke default prompt).
    """
    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute("SELECT content FROM variables WHERE name = %s AND deleted_at IS NULL LIMIT 1", (name,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            return row[0]
        return None
    except Error:
        # jangan raise — biarkan caller fallback
        return None
