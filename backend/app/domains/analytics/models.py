from typing import Any
from sqlalchemy import DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column
from app.core.db.database import Base  # <--- Ispravljena putanja do Base klase iz tvog database.py

class SearchAnalyticsORM(Base):
    """Table to store search analytics data."""
    
    __tablename__ = "search_analytics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    clean_query: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    search_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    last_searched_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)