"""
Database connection and session management for SQLite via SQLModel.
"""

import os
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

# Ensure the data directory exists
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'evaluations.db'}")

engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})


def init_db() -> None:
    """Create all tables defined in SQLModel metadata and run safe column migrations."""
    SQLModel.metadata.create_all(engine)

    # Safe column migration for SQLite
    import sqlite3
    db_file = DATA_DIR / "evaluations.db"
    if db_file.exists():
        with sqlite3.connect(str(db_file)) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(evaluations)")
            existing_columns = {col[1] for col in cursor.fetchall()}
            
            if "ground_truth" not in existing_columns:
                cursor.execute("ALTER TABLE evaluations ADD COLUMN ground_truth TEXT")
            if "judge_reasoning" not in existing_columns:
                cursor.execute("ALTER TABLE evaluations ADD COLUMN judge_reasoning TEXT")
            conn.commit()


def get_session():
    """FastAPI dependency that yields a database session."""
    with Session(engine) as session:
        yield session
