from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # ----------------------------------------------
    # Database Configuration
    # ----------------------------------------------

    DATABASE_URL: str

    DB_POOL_SIZE: int = Field(default=10, ge=1, le=50)
    DB_MAX_OVERFLOW: int = Field(default=20, ge=0, le=100)
    DB_POOL_RECYCLE: int = Field(default=1800, ge=60, le=86_400)

    # ----------------------------------------------
    # Authentication Configuration
    # ----------------------------------------------

    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int

    # ----------------------------------------------
    # Face Template Encryption
    # ----------------------------------------------

    FACE_TEMPLATE_ENCRYPTION_KEY: str

    # ----------------------------------------------
    # Door Controller Configuration
    # ----------------------------------------------

    DOOR_CONTROLLER: str = "simulation"
    ESP32_IP: str = "http://127.0.0.1:9000"
    ESP32_TIMEOUT: float = 3.0

    # ----------------------------------------------
    # Face Image Security Configuration
    # ----------------------------------------------

    MAX_FACE_IMAGE_SIZE_BYTES: int = 5 * 1024 * 1024
    MAX_FACE_IMAGE_WIDTH: int = 4096
    MAX_FACE_IMAGE_HEIGHT: int = 4096
    MAX_FACE_IMAGE_PIXELS: int = 16_777_216

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()