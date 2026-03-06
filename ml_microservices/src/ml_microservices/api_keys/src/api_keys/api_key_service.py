from database.database_utils import DatabaseRepository
from api_keys.api_key_models import ApiKey
import hashlib
from typing import Optional


class ApiKeyService:
    def __init__(self, db: DatabaseRepository):
        self.db = db

    def generate_api_key(self, prefix: str = "ak", rate_limit_req_no: int = 30, rate_limit_interval_minutes: int = 1, comment: Optional[str] = None) -> str:
        raw_key, db_obj = ApiKey.generate(prefix, rate_limit_interval_minutes, rate_limit_req_no, comment)
        self.db.save_orm_object(db_obj)
        return raw_key

    def validate_api_key(self, raw_key: str) -> bool:
        """Validate a raw API key by comparing its hash against the stored hash."""
        parts = raw_key.split("_", 1)
        if len(parts) != 2:
            return False
        prefix, secret = parts
        api_key = self.db.get_orm_object(ApiKey, prefix=prefix)
        if api_key is None or not api_key.enabled:
            return False
        hashed = hashlib.sha256(secret.encode()).hexdigest()
        return hashed == api_key.hashed_secret

    def disable_api_key(self, prefix: str) -> bool:
        """Disable an API key by prefix. Returns True if found and disabled."""
        api_key = self.db.get_orm_object(ApiKey, prefix=prefix)
        if api_key is None:
            return False
        self.db.update_record(ApiKey.__tablename__, "prefix", prefix, {"enabled": False})
        return True

    def enable_api_key(self, prefix: str) -> bool:
        """Enable an API key by prefix. Returns True if found and enabled."""
        api_key = self.db.get_orm_object(ApiKey, prefix=prefix)
        if api_key is None:
            return False
        self.db.update_record(ApiKey.__tablename__, "prefix", prefix, {"enabled": True})
        return True

    def get_api_key_by_prefix(self, prefix: str) -> Optional[ApiKey]:
        """Retrieve an ApiKey instance by its prefix."""
        return self.db.get_orm_object(ApiKey, prefix=prefix)