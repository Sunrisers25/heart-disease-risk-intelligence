"""Database handler module for heart disease patient audit logging.
Initializes SQLite database and manages transaction records.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
import pandas as pd

from src.config import DB_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DatabaseHandler:
    """Manages SQLite database storage for patient audit logs."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and return a database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize audit_logs table if it does not already exist."""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            patient_id TEXT,
            age INTEGER,
            gender TEXT,
            blood_pressure REAL,
            cholesterol REAL,
            bmi REAL,
            blood_sugar REAL,
            heart_rate REAL,
            ecg_result INTEGER,
            risk_classification TEXT NOT NULL,
            risk_probability REAL NOT NULL,
            clinical_notes TEXT
        );
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(create_table_query)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_logs(timestamp);")
            conn.commit()

    def log_patient_record(
        self,
        patient_data: Dict[str, Any],
        risk_classification: str,
        risk_probability: float,
        clinical_notes: Optional[List[str]] = None,
    ) -> int:
        """Log a patient inference prediction and clinical assessment.

        Args:
            patient_data: Dictionary of patient vitals/demographics.
            risk_classification: "High Risk" or "Low Risk".
            risk_probability: Predicted probability (float 0.0 to 1.0).
            clinical_notes: List of medical warning guidelines/recommendations.

        Returns:
            int: Primary key ID of the inserted row.
        """
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        notes_str = json.dumps(clinical_notes if clinical_notes is not None else [])

        insert_query = """
        INSERT INTO audit_logs (
            timestamp,
            patient_id,
            age,
            gender,
            blood_pressure,
            cholesterol,
            bmi,
            blood_sugar,
            heart_rate,
            ecg_result,
            risk_classification,
            risk_probability,
            clinical_notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        params = (
            timestamp,
            str(patient_data.get("Patient_ID", "N/A")),
            int(patient_data.get("Age", 0)),
            str(patient_data.get("Gender", "Unknown")),
            float(patient_data.get("Blood_Pressure", 0.0)),
            float(patient_data.get("Cholesterol", 0.0)),
            float(patient_data.get("BMI", 0.0)),
            float(patient_data.get("Blood_Sugar", 0.0)),
            float(patient_data.get("Heart_Rate", 0.0)),
            int(patient_data.get("ECG_Result", 0)),
            str(risk_classification),
            float(risk_probability),
            notes_str,
        )

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(insert_query, params)
            conn.commit()
            record_id = cursor.lastrowid
            logger.info("Inserted audit log record with ID: %d", record_id)
            return record_id

    def get_recent_records(self, limit: int = 50) -> pd.DataFrame:
        """Fetch the most recent patient audit records as a pandas DataFrame."""
        query = """
        SELECT
            id,
            timestamp,
            patient_id,
            age,
            gender,
            blood_pressure,
            cholesterol,
            bmi,
            blood_sugar,
            heart_rate,
            ecg_result,
            risk_classification,
            risk_probability,
            clinical_notes
        FROM audit_logs
        ORDER BY id DESC
        LIMIT ?
        """
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(limit,))
            return df
