import enum
import os
from pathlib import Path
from tempfile import gettempdir

from pydantic_settings import BaseSettings, SettingsConfigDict
from yarl import URL

TEMP_DIR = Path(gettempdir())
PATH = os.path.dirname(os.path.abspath(__file__))


class LogLevel(str, enum.Enum):  # noqa: WPS600
    """Possible log levels."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class Settings(BaseSettings):
    """
    Application settings.

    These parameters can be configured
    with environment variables.
    """

    host: str = "xri.helloworld2.net"
    port: int = 8000
    # quantity of workers for uvicorn
    workers_count: int = 1
    # Enable uvicorn reloading
    reload: bool = False

    # Current environment

    environment: str = "dev"

    # Path to the directory with templates

    template_dir: Path = PATH + "/web/templates"

    # Path to the directory with static files

    static_dir: Path = PATH + "/static"

    # Path to the directory with uploaded files

    upload_dir: Path = PATH + "/static/uploads"

    modeltext_dir: Path = PATH + "/static/model/temp.txt"

    modeljon_dir: Path = PATH + "/static/model/model2.json"

    modelm_dir: Path = PATH + "/static/model/model2.h5"

    modeltemp_dir: Path = PATH + "/static/model/temp/tempMain.png"

    modeltemp2_dir: str = PATH + "/static/model/temp/temp.png"

    modelres_dir: Path = PATH + "/static/modelresults"

    log_level: LogLevel = LogLevel.INFO

    # Variables for the database
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = "xriweb"
    db_pass: str = "xriweb"
    db_base: str = "xriweb"
    db_echo: bool = False

    @property
    def db_url(self) -> URL:
        """
        Assemble database URL from settings.

        :return: database URL.
        """
        return URL.build(
            scheme="postgresql",
            host=self.db_host,
            port=self.db_port,
            user=self.db_user,
            password=self.db_pass,
            path=f"/{self.db_base}",
        )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="XRIWEB_",
        env_file_encoding="utf-8",
    )


settings = Settings()
