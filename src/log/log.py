import logging


class Logger:
    def __init__(
        self,
        module_name: str,
        log_filename: str | None = "master.log",
        level: int = logging.DEBUG,
        formatter: str = "[%(asctime)s\t %(levelname)s\t %(name)s] %(message)s",
        **kwargs,
    ) -> None:
        self.module_name = module_name
        self.package_name = None

        if kwargs.get("package_name", ""):
            self.package_name = kwargs.get("package_name")
            self.name = f"{self.package_name}:{self.module_name}"

        self.name = self.module_name

        self.formatter = formatter

        self.logger = logging.getLogger(
            name=self.name,
        )
        self.logger.setLevel(
            level=level,
        )

        if log_filename:
            self.file_handler = logging.FileHandler(
                log_filename,
            )

            self.file_handler.setFormatter(
                fmt=logging.Formatter(
                    formatter,
                ),
            )

            self.file_handler.setLevel(
                level=level,
            )

            self.logger.addHandler(
                self.file_handler,
            )

        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setFormatter(
            fmt=logging.Formatter(
                formatter,
            ),
        )

        self.stream_handler.setLevel(
            level=level,
        )

        self.logger.addHandler(
            hdlr=self.stream_handler,
        )

    def critical(
        self,
        message: str,
    ) -> None:
        self.logger.critical(
            msg=message,
        )

    def debug(
        self,
        message: str,
    ) -> None:
        self.logger.debug(
            msg=message,
        )

    def error(
        self,
        message: str,
    ) -> None:
        self.logger.error(
            msg=message,
        )

    def info(
        self,
        message: str,
    ) -> None:
        self.logger.info(
            msg=message,
        )

    def warn(
        self,
        message: str,
    ) -> None:
        self.logger.warning(
            msg=message,
        )
