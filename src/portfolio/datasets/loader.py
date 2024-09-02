import pandas as pd

from log.log import Logger
from portfolio.datasets.bloomberg.asset import Asset as BAsset
from portfolio.datasets.schemas import DatasetMetadata, DataSourceType
from portfolio.datasets.yahoo_finance.asset import Asset as YFAsset


class DataSource:
    def __init__(
        self,
        metadata: DatasetMetadata | dict,
    ):
        self.logger = Logger("dataset_loader")

        if isinstance(metadata, dict):
            metadata = DatasetMetadata(**metadata)

        self.asset: BAsset | YFAsset | None = None

        self.ticker = metadata.Ticker
        self.origin = metadata.Origin
        self.format_date = metadata.FormatDate
        self.period = metadata.Period
        self.interval = metadata.Interval
        self.start_date = metadata.StartDate
        self.end_date = metadata.EndDate

        self.source_name = metadata.SourceName
        self.file_format = metadata.FileFormat

        match self.origin:
            case DataSourceType.YAHOO_FINANCE:
                self.asset = YFAsset(
                    ticker=self.ticker,
                    period=self.period,
                    time_frame=self.interval,
                    start=self.start_date,
                    end=self.end_date,
                )

            case DataSourceType.BLOOMBERG:
                if not self.source_name:
                    self.logger.error("Unable to properly identify a source file.")
                    return

                self.asset = BAsset(
                    self.source_name,
                    file_format=self.file_format,
                )
            case _:
                self.logger.warn(
                    "Unable to correctly identify the origin for the current dataset"
                    " instance. Please specify it."
                )
                return

        self.dataset = self.asset.data_frame()

        self.logger.info("DataSource Instance correctly created.")

    def asset_instance(
        self,
    ) -> BAsset | YFAsset:
        return self.asset

    def data_frame(
        self,
    ) -> pd.DataFrame:
        return self.dataset
