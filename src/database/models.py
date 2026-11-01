"""SQLAlchemy database models for patient registry and prediction history."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base declarative class for all SQLAlchemy entities."""
    pass


class Patient(Base):
    """Patient demographic and baseline registry record."""

    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    gender: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        server_default=func.now(),
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Cascade delete ensures no orphaned prediction history records exist
    predictions: Mapped[List["PredictionHistory"]] = relationship(
        "PredictionHistory",
        back_populates="patient",
        cascade="all, delete-orphan",
        order_by="desc(PredictionHistory.prediction_timestamp)",
    )


class PredictionHistory(Base):
    """Historical record of clinical cardiovascular risk prediction evaluations."""

    __tablename__ = "prediction_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_internal_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    patient_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    predicted_class: Mapped[int] = mapped_column(Integer, nullable=False)  # 1: High Risk, 0: Low Risk
    risk_classification: Mapped[str] = mapped_column(String(32), nullable=False)
    risk_probability: Mapped[float] = mapped_column(Float, nullable=False)
    is_calibrated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    model_metadata: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON-encoded metadata
    prediction_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        server_default=func.now(),
        index=True,
        nullable=False,
    )
    audit_record_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    patient: Mapped["Patient"] = relationship("Patient", back_populates="predictions")

