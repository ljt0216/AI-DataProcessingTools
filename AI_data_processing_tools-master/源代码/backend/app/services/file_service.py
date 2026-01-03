import os
import uuid
import re
import pandas as pd
import sqlite3

BASE_DATA_DIR = "data/temp"


def _sanitize_table_name(name: str, fallback: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]", "_", name.strip().lower()) or fallback
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or fallback


def save_uploaded_file(file) -> tuple[str, str]:
    os.makedirs(BASE_DATA_DIR, exist_ok=True)
    session_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower()
    file_path = os.path.join(BASE_DATA_DIR, f"{session_id}{ext}")
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    return session_id, file_path


def load_to_sqlite(file_path: str, session_id: str) -> tuple[str, list[str]]:
    ext = os.path.splitext(file_path)[1].lower()
    db_path = os.path.join(BASE_DATA_DIR, f"{session_id}.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)

    table_names: list[str] = []

    try:
        if ext in [".xls", ".xlsx"]:
            xl = pd.ExcelFile(file_path)
            for idx, sheet in enumerate(xl.sheet_names):
                df = xl.parse(sheet)
                tname = _sanitize_table_name(sheet, f"sheet_{idx+1}")
                df.to_sql(tname, conn, if_exists="replace", index=False)
                table_names.append(tname)
        else:
            df = pd.read_csv(file_path)
            base = os.path.splitext(os.path.basename(file_path))[0]
            tname = _sanitize_table_name(base or "data", "data")
            df.to_sql(tname, conn, if_exists="replace", index=False)
            table_names.append(tname)
    finally:
        conn.close()

    return db_path, table_names


