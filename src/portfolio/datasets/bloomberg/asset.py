from datetime import date

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from backtest.schemas import SamplingMethods
from log.log import Logger
from portfolio.datasets.bloomberg.models import SourceType
from portfolio.datasets.bloomberg.schemas import AvailableRawColumns


class Asset:
    def __init__(
        self,
        source_file: str,
        file_format: SourceType | None = SourceType.EXCEL,
        start_date: date | None = None,
        end_date: date | None = None,
        **kwargs,
    ) -> None:
        self.logger: Logger = Logger("bloomberg.asset")

        self.start_date: date | None = start_date
        self.end_date: date | None = end_date

        self.source_file: str = source_file
        self.file_format: str = file_format
        if not self.file_format:
            self.file_format = SourceType.EXCEL

        if isinstance(source_file, str):
            self.result: pd.DataFrame | None = None

        match file_format:
            case SourceType.EXCEL:
                self.result: pd.DataFrame = pd.read_excel(
                    io=source_file,
                    **kwargs,
                )
            case SourceType.CSV:
                self.result: pd.DataFrame = pd.read_csv(
                    filepath_or_buffer=source_file,
                    **kwargs,
                )
            case _:
                self.logger("Unable to correctly identify a recognized format")
                return

        self.available_columns: list[str] = []

        self.result[AvailableRawColumns.DATES] = pd.to_datetime(
            self.result[AvailableRawColumns.DATES], format="%d/%m/%Y"
        )

        if self.start_date and self.end_date:
            self.result = self.result[
                (
                    self.result[AvailableRawColumns.DATES]
                    >= pd.to_datetime(self.start_date)
                )
                & (
                    self.result[AvailableRawColumns.DATES]
                    <= pd.to_datetime(self.end_date)
                )
            ]

        self._check_nans()
        self.compute_returns()

        self.np_dataset: NDArray = self.result.to_numpy()

    def _check_nans(
        self,
    ) -> None:
        _active_columns: list[str] = list(
            set(self.result.columns).intersection(AvailableRawColumns.list())
        )

        for col in _active_columns:
            self.result[col].ffill(inplace=True)
            self.available_columns.append(col)

    def compute_returns(
        self,
    ) -> None:
        self.result[AvailableRawColumns.RETURNS] = self.result[
            AvailableRawColumns.CLOSE_PRICE
        ].pct_change()

    def sampling(
        self,
        sampling_method: SamplingMethods,
        steps_forward: int,
        column: str = AvailableRawColumns.CLOSE_PRICE,
        samples: int = 1,
    ) -> NDArray | None:
        if not 1 < steps_forward < 255 or not isinstance(steps_forward, int):
            return None

        if not 1 <= samples < 10000 or not isinstance(samples, int):
            return None

        prices = self.result[column].to_numpy()
        last_prices = np.ones(shape=(samples, 1)) * prices[-1]

        match sampling_method:
            case SamplingMethods.NORMAL_DISTRIBUTION:
                sampling = np.random.normal(
                    loc=0,
                    scale=np.std(
                        self.close_prices(),
                    ),
                    size=(samples, steps_forward - 1),
                )

                cum_raw_last_prices = np.hstack((last_prices, sampling))
                cum_prices = np.cumsum(cum_raw_last_prices, axis=1)

                return self.plotting(
                    column,
                    np.hstack(
                        (
                            np.repeat(
                                np.reshape(prices, (prices.shape[0], 1)),
                                repeats=samples,
                                axis=1,
                            ).T,
                            cum_prices,
                        )
                    ),
                )
            case SamplingMethods.LAPLACE_DISTRIBUTION:
                sampling = np.random.laplace(
                    loc=0,
                    scale=np.std(
                        self.close_prices(),
                    ),
                    size=(samples, steps_forward - 1),
                )

                cum_raw_last_prices = np.hstack((last_prices, sampling))
                cum_prices = np.cumsum(cum_raw_last_prices, axis=1)

                return self.plotting(
                    column,
                    np.hstack(
                        (
                            np.repeat(
                                np.reshape(prices, (prices.shape[0], 1)),
                                repeats=samples,
                                axis=1,
                            ).T,
                            cum_prices,
                        )
                    ),
                )
            case SamplingMethods.UNIFORM_DISTRIBUTION:
                sampling = np.random.uniform(
                    low=0,
                    high=np.max(prices) - np.min(prices),
                    size=(samples, steps_forward - 1),
                )
                cum_raw_last_prices = np.hstack((last_prices, sampling))
                cum_prices = np.cumsum(cum_raw_last_prices, axis=1)

                return self.plotting(
                    column,
                    np.hstack(
                        (
                            np.repeat(
                                np.reshape(prices, (prices.shape[0], 1)),
                                repeats=samples,
                                axis=1,
                            ).T,
                            cum_prices,
                        )
                    ),
                )
            case _:
                return None

    def dates(
        self,
    ) -> NDArray:
        return self.result[AvailableRawColumns.DATES].to_numpy()

    def open_prices(self) -> NDArray:
        return self.result[AvailableRawColumns.OPEN_PRICE].to_numpy()

    def high_prices(self) -> NDArray:
        return self.result[AvailableRawColumns.HIGH_PRICE].to_numpy()

    def low_prices(self) -> NDArray:
        return self.result[AvailableRawColumns.LOW_PRICE].to_numpy()

    def close_prices(self) -> NDArray:
        return self.result[AvailableRawColumns.CLOSE_PRICE].to_numpy()

    def returns(
        self,
        remove_nans: bool = True,
    ) -> NDArray:
        _returns: NDArray = self.result[AvailableRawColumns.RETURNS].to_numpy()

        if not remove_nans:
            return _returns

        return _returns[~np.isnan(_returns)]

    def volumes(
        self,
        remove_nans: bool = True,
    ) -> NDArray:
        _volumes: NDArray = self.result[AvailableRawColumns.VOLUME].to_numpy()

        if not remove_nans:
            return _volumes

        return _volumes[~np.isnan(_volumes)]

    def put_call_volume_ratio(
        self,
        remove_nans: bool = True,
    ) -> NDArray:
        _put_call_volume_ratio: NDArray = self.result[
            AvailableRawColumns.PUT_CALL_VOLUME_RATIO_CUR_DAY
        ].to_numpy()

        if not remove_nans:
            return _put_call_volume_ratio

        return _put_call_volume_ratio[~np.isnan(_put_call_volume_ratio)]

    def call_options_volume(
        self,
        remove_nans: bool = True,
    ) -> NDArray:
        _volume_total_call: NDArray = self.result[
            AvailableRawColumns.VOLUME_TOTAL_CALL
        ].to_numpy()

        if not remove_nans:
            return _volume_total_call

        return _volume_total_call[~np.isnan(_volume_total_call)]

    def put_options_volume(
        self,
        remove_nans: bool = True,
    ) -> NDArray:
        _volume_total_put: NDArray = self.result[
            AvailableRawColumns.VOLUME_TOTAL_PUT
        ].to_numpy()

        if not remove_nans:
            return _volume_total_put

        return _volume_total_put[~np.isnan(_volume_total_put)]

    def return_distribution(self) -> NDArray:
        return np.sort(self.result[AvailableRawColumns.RETURNS].to_numpy())

    def data_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.result)

    def plotting(
        self,
        column: AvailableRawColumns = AvailableRawColumns.CLOSE_PRICE,
        sampling: NDArray | None = None,
    ) -> tuple:
        return (
            self.dates() if sampling is None else np.arange(0, sampling.shape[1]),
            self.result[column]
            .to_numpy()
            .reshape(1, self.result[column].to_numpy().shape[0])
            if sampling is None
            else sampling,
        )
