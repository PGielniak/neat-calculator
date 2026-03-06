import secrets
import hashlib
from datetime import datetime, timezone
from typing import Optional, Tuple
from sqlalchemy import String, Boolean, DateTime, Integer
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase

class Base(DeclarativeBase):
    pass

class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String, primary_key=True) # UUID
    prefix: Mapped[str] = mapped_column(String(16), index=True, unique=True)
    hashed_secret: Mapped[str] = mapped_column(String(64)) 
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    comment: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    rate_limit_req_no: Mapped[int] = mapped_column(Integer, default=30)
    rate_limit_interval_minutes: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    @staticmethod
    def generate(prefix: str = "ak", rate_limit_interval_minutes: int = 1, rate_limit_req_no: int = 30, comment: Optional[str] = None) -> Tuple[str, "ApiKey"]:
        """Generates a raw key for the user and a model instance for the DB.
        
        The stored prefix is made unique by appending a short random hex suffix
        to the caller-supplied prefix, so repeated calls never collide.
        """
        unique_prefix = f"{prefix}{secrets.token_hex(3)}"
        secret = secrets.token_urlsafe(32)
        raw_key = f"{unique_prefix}_{secret}"
        
        # We only hash the 'secret' part
        hashed = hashlib.sha256(secret.encode()).hexdigest()
        
        db_obj = ApiKey(
            id=str(secrets.token_hex(16)),
            prefix=unique_prefix,
            hashed_secret=hashed,
            enabled=True,
            rate_limit_req_no=rate_limit_req_no,
            rate_limit_interval_minutes=rate_limit_interval_minutes,
            comment=comment
        )
        return raw_key, db_obj