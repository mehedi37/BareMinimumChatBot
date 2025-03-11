import os
from typing import Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Application settings, loaded from environment variables
    """
    # API Configuration
    APP_ENV: str = "development"
    DEBUG: bool = True

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"

    # Chatbot settings
    CHATBOT_MODEL: str = "default"

    # Session settings
    COOKIE_MAX_AGE: int = 60 * 60 * 24 * 90  # 90 days in seconds

    @field_validator("APP_ENV")
    def validate_app_env(cls, v: str) -> str:
        if v not in ["development", "production", "testing"]:
            raise ValueError(f"APP_ENV must be one of 'development', 'production', or 'testing', got '{v}'")
        return v

    @field_validator("LOG_LEVEL")
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}, got '{v}'")
        return v.upper()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create settings instance
settings = Settings()