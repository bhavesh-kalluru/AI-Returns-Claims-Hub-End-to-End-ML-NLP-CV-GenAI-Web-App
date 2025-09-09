# app/models/schema.py
from __future__ import annotations
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    String, Integer, DateTime, ForeignKey, Text, Float, Boolean
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base  # <-- THIS is the missing import

class Customer(Base):
    __tablename__ = "customers"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    claims: Mapped[List["Claim"]] = relationship(back_populates="customer", cascade="all, delete-orphan")

class Product(Base):
    __tablename__ = "products"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    sku: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    category: Mapped[Optional[str]] = mapped_column(String(120))
    price: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    claims: Mapped[List["Claim"]] = relationship(back_populates="product", cascade="all, delete-orphan")

class Claim(Base):
    __tablename__ = "claims"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), nullable=False)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), nullable=False)

    description: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="new", index=True)

    # NLP outputs
    issue_label: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    key_phrases: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float)

    # Vision outputs
    is_photo_attached: Mapped[bool] = mapped_column(Boolean, default=False)
    photo_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    photo_blur: Mapped[Optional[float]] = mapped_column(Float)          # Laplacian variance
    photo_brightness: Mapped[Optional[float]] = mapped_column(Float)    # mean gray
    photo_contrast: Mapped[Optional[float]] = mapped_column(Float)      # std gray
    damage_score: Mapped[Optional[float]] = mapped_column(Float)        # [0..1] heuristic

    # ANN output
    predicted_refund_prob: Mapped[Optional[float]] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    customer: Mapped["Customer"] = relationship(back_populates="claims")
    product:  Mapped["Product"]  = relationship(back_populates="claims")

