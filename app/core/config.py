from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    DATABASE_URL: str
    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int

    FACE_TEMPLATE_ENCRYPTION_KEY: str

    # ----------------------------------------------
    # Door Controller Configuration
    # ----------------------------------------------

    DOOR_CONTROLLER: str = "simulation"
    ESP32_IP: str = "http://127.0.0.1:9000"
    ESP32_TIMEOUT: float = 3.0

    model_config = SettingsConfigDict(
        env_file=".env"
    )


settings = Settings()