from dataclasses import dataclass, field
from functools import lru_cache
from glob import glob
from pathlib import Path

from omegaconf import OmegaConf


@dataclass
class Settings:
    ConfigFolder: str = "."
    ExperimentsFolder: str = "."
    ResultsFolder: str = "."

    ActiveExperiments: list[str] = field(default_factory=list)
    ExcludeExperiments: list[str] = field(default_factory=list)


@lru_cache
def get_project_folder() -> Path:
    return Path(__file__).absolute().parent.parent.parent


@lru_cache
def get_settings() -> Settings:
    settings_import = OmegaConf.load(
        get_project_folder().joinpath(
            "config.yaml",
        ),
    )

    return OmegaConf.structured(
        Settings(**settings_import),
    )


@lru_cache
def get_config_folder() -> Path:
    return get_project_folder().joinpath(
        get_settings().ConfigFolder,
    )


@lru_cache
def get_experiments_folder() -> Path:
    return get_project_folder().joinpath(
        get_settings().ExperimentsFolder,
    )


@lru_cache
def get_results_folder() -> Path:
    return get_project_folder().joinpath(
        get_settings().ResultsFolder,
    )


@lru_cache
def get_experiments() -> list[str]:
    experiment_path: Path = Path(__file__).parent.parent.parent.joinpath(
        get_experiments_folder()
    )

    if experiment_path.exists():
        if not len(get_settings().ActiveExperiments):
            return list(
                set(
                    glob(
                        "*.yaml",
                        root_dir=experiment_path,
                        include_hidden=True,
                    )
                ).difference(
                    set(
                        get_settings().ExcludeExperiments,
                    ),
                ),
            )
        else:
            return list(
                set(
                    glob(
                        "*.yaml",
                        root_dir=experiment_path,
                        include_hidden=True,
                    )
                ).intersection(
                    set(
                        get_settings().ActiveExperiments,
                    ),
                ),
            )

    return []
