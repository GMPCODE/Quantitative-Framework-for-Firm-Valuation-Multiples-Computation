from datetime import date
from pathlib import Path

import keras as tfk
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from numpy.typing import NDArray
from pypfopt.efficient_frontier import EfficientFrontier

pd.set_option('future.no_silent_downcasting', True)

model_folder: Path = Path("./")
data_folder: Path = Path("../../../../data/")

raw_data_folder: Path = data_folder.joinpath("00_raw/")
draft_data_processing_folder: Path = data_folder.joinpath("10_draft_processing/")
aggregare_draft_data_folder: Path = data_folder.joinpath("20_aggregate_draft/")
prod_data_folder: Path = data_folder.joinpath("30_prod/")
features_data_folder: Path = data_folder.joinpath("40_features/")

excel_dataset_name: str = "12_Industry_Portfolios_Daily.xlsx"
equally_weighted_returns_sheet_name: str = "Average Equal Weighted Returns"

df: pd.DataFrame = pd.read_excel(
    io=raw_data_folder.joinpath(excel_dataset_name),
    sheet_name=equally_weighted_returns_sheet_name,
)
df.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")

sectors: list[str] = [
    industry.strip() for industry in df.columns.difference(["Date"]).to_list()
]

features: list[dict[str, str]] = [
    {
        "name": "Volatility_Index",
        "type": "market_status",
        "source": "yahoo_finance",
        "ticker": "^VIX",
        "reference_variable": "Close",
        "description": "CBOE Volatility Index",
        "added_features": [
            {
                "name": "Returns",
                "reference_variable": "Close",
                "periods": [5, 20, 40],
            },
            {
                "name": "StdDev",
                "reference_variable": "Close",
                "periods": [5, 20, 40],
            },
        ],
    },
    {
        "name": "Momentum",
        "type": "asset_feature",
        "reference_variable": "Returns",
        "reference_period": 1,
        "periods": [15, 40, 70],
    },
    {
        "name": "StdDev",
        "type": "asset_feature",
        "reference_variable": "Returns",
        "reference_period": 1,
        "periods": [15, 40, 70],
    },
]

dataset = pd.read_csv(
    filepath_or_buffer=prod_data_folder.joinpath("dataset.csv"),
    sep=",",
    encoding="UTF-8",
)
dataset["Date"] = pd.to_datetime(dataset["Date"], format="%Y-%m-%d")

inflation_rates = pd.read_csv(
    filepath_or_buffer=prod_data_folder.joinpath("inflation_rates.csv"),
    sep=",",
    encoding="UTF-8",
)
inflation_rates["Date"] = pd.to_datetime(inflation_rates["Date"], format="%Y-%m-%d")

"""sentiment_analysis = pd.read_csv(
    filepath_or_buffer=prod_data_folder.joinpath("sentiment_analysis.csv"),
    sep=",",
    encoding="UTF-8",
)
sentiment_analysis["Date"] = pd.to_datetime(sentiment_analysis["Date"],
                                            format="%Y-%m-%d")"""

starting_date: date = date(2018, 1, 1)
ending_date: date = date(2020, 12, 31)

filtered_dataset = dataset[
    (dataset["Date"] >= pd.to_datetime(starting_date)) &
    (dataset["Date"] <= pd.to_datetime(ending_date))].reset_index(drop=True)
filtered_inflation_rates = inflation_rates[
    (inflation_rates["Date"] >= pd.to_datetime(starting_date)) &
    (inflation_rates["Date"] <= pd.to_datetime(ending_date))
    ].reset_index(drop=True)
"""filtered_sentiment_analysis = sentiment_analysis[
    (sentiment_analysis["Date"] >= pd.to_datetime(starting_date)) &
    (sentiment_analysis["Date"] <= pd.to_datetime(ending_date))
    ].reset_index(drop=True)"""

common_dates: set = set(filtered_dataset["Date"])
common_dates.intersection_update(set(filtered_inflation_rates["Date"]))
# common_dates.intersection_update(set(filtered_sentiment_analysis["Date"]))

mask_dataset = filtered_dataset["Date"].isin(common_dates)
mask_inflation_rates = filtered_inflation_rates["Date"].isin(common_dates)
# mask_sentiment_analysis = filtered_sentiment_analysis["Date"].isin(common_dates)

"""result = pd.concat(
    (filtered_dataset[mask_dataset].reset_index(drop=True),
     filtered_inflation_rates[mask_inflation_rates].reset_index(drop=True),
     filtered_sentiment_analysis[mask_sentiment_analysis].reset_index(drop=True)),
    axis=1)"""
result = pd.concat(
    (filtered_dataset[mask_dataset].reset_index(drop=True),
     filtered_inflation_rates[mask_inflation_rates].reset_index(drop=True)),
    axis=1)
result = result.T.drop_duplicates().T

reward_computation_variable_references: dict[str, str] = {
    "sources": "assets",
    "reference_variable": "Returns",
    "period": 1,
}

observation_computation_variable_references: list[dict[str, str | list[str]]] = [
    {
        "sources": "assets",
        "reference_variables": ["Returns", "Momentum", "StdDev", "Volume"],
    },
    {
        "sources": "market_status",
        "reference_variables": ["Close", "Returns", "Momentum", "StdDev"],
    },
    {
        "sources": "macroeconomics",
        "reference_variables": ["Inflation_Rate"]
    }
]

reward_computation_variable_reference: str = reward_computation_variable_references.get(
    "reference_variable", ""
)

reward_computation_period: int | None = reward_computation_variable_references.get(
    "period"
)

reward_columns: list[str] = ["Date"]
if isinstance(reward_computation_period, int):
    reward_columns += [
        f"{sector}_{reward_computation_variable_reference}_{reward_computation_period}"
        for sector in sectors
    ]
elif reward_computation_period is None:
    reward_columns += [
        f"{sector}_{reward_computation_variable_reference}" for sector in sectors
    ]
else:
    print("Unable to properly identify the variable type")

reward_columns = list(dict.fromkeys(reward_columns))

reward_computation_dataset: pd.DataFrame = result[reward_columns]
reward_computation_dataset = reward_computation_dataset[
    (reward_computation_dataset["Date"] >= pd.to_datetime(starting_date))
    & (reward_computation_dataset["Date"] <= pd.to_datetime(ending_date))
].reset_index(drop=True)

observation_dataset_columns: list[str] = []
dataset_columns: list[str] = result.columns.tolist()
market_status_variable: list[str] = [
    ms.get("name", "N/A") for ms in features if ms.get("type", "N/A") == "market_status"
]

for variable_reference in observation_computation_variable_references:
    sources: str = variable_reference.get("sources", "")
    reference_variables: str = variable_reference.get("reference_variables", "")
    if not sources or not reference_variables:
        continue

    match sources:
        case "assets":
            for sector in sectors:
                for rv in reference_variables:
                    observation_dataset_columns.extend(
                        [
                            column
                            for column in dataset_columns
                            if sector in column and rv in column
                        ]
                    )
        case "market_status":
            for ms in market_status_variable:
                for rv in reference_variables:
                    observation_dataset_columns.extend(
                        [
                            column
                            for column in dataset_columns
                            if ms in column and rv in column
                        ]
                    )
        case "macroeconomics":
            for rv in reference_variables:
                observation_dataset_columns.extend(
                    [
                        column for column in reference_variables
                    ]
                )

observation_dataset_columns = list(dict.fromkeys(observation_dataset_columns))

observations_dataset: pd.DataFrame = result[
    (result["Date"] >= pd.to_datetime(starting_date))
    & (result["Date"] <= pd.to_datetime(ending_date))
][observation_dataset_columns].reset_index(drop=True)

observations_dataset.to_csv(
    path_or_buf=features_data_folder.joinpath("features.csv"),
    sep=",",
    encoding="UTF-8",
    index=False,
)

reward_computation_dataset.to_csv(
    path_or_buf=features_data_folder.joinpath("reward.csv"),
    sep=",",
    encoding="UTF-8",
    index=False,
)

assert pd.isna(observations_dataset).sum().sum() == 0


class TradingEnvironment:
    def __init__(
        self,
        n_assets: int,
        _reward_computation_dataset: pd.DataFrame,
        _observation_dataset: pd.DataFrame,
        _delta_days: int = 5,
        _episode_length: int = 250,
        _risk_free: float = 0.0,
    ):
        assert _reward_computation_dataset.shape[0] == _observation_dataset.shape[0]
        assert n_assets > 1

        self.n_assets: int = n_assets
        self.reward_computation_dataset: pd.DataFrame = _reward_computation_dataset
        self.observation_dataset: pd.DataFrame = _observation_dataset
        self.delta_days: int = _delta_days
        self.episode_length: int = _episode_length
        self.risk_free: float = _risk_free

        self.reset()

    def sharpe_ratio(
        self,
        weights: NDArray,
        mean_returns: float,
        cov_matrix: NDArray,
        risk_free_rate: float = 0.0,
    ) -> float:
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -1 * (portfolio_return - risk_free_rate) / portfolio_vol

    def monte_carlo_portfolio(
        self,
        returns: NDArray,
        n: int = 10000,
    ) -> float:
        weights: NDArray = np.random.rand(self.n_assets, 1)
        weights = weights / weights.sum()
        current_sharpe_ratio = -np.inf

        for _ in range(n):
            realized_returns: NDArray = returns @ weights
            mean = realized_returns.mean()
            std = realized_returns.std()
            sharpe_ratio = mean / std

            if sharpe_ratio > current_sharpe_ratio:
                current_sharpe_ratio = sharpe_ratio

        return weights

    def reward_function(
        self,
        current_period_returns: NDArray,
        matrix_backward_returns: NDArray,
        action_vector: NDArray,
    ) -> tuple:

        _model_realized_returns: NDArray = current_period_returns @ action_vector
        _model_realized_mean: float = _model_realized_returns.mean()
        _model_realized_std: float = _model_realized_returns.std()
        _model_realized_sharpe_ratio: float = (
            _model_realized_mean - self.risk_free
        ) / _model_realized_std

        mean_returns: NDArray = matrix_backward_returns.mean(axis=0)
        cov_matrix: NDArray = np.cov(matrix_backward_returns.T)

        ef = EfficientFrontier(mean_returns, cov_matrix)
        ef.min_volatility()
        clean_weights = ef.clean_weights()

        markowitz_portfolio: NDArray = np.array(
            list(clean_weights.values())
            ).reshape(-1, 1)

        _markowitz_realized_returns: NDArray = (
            current_period_returns @ markowitz_portfolio
        )

        _markowitz_realized_mean: float = _markowitz_realized_returns.mean()
        _markowitz_realized_std: float = _markowitz_realized_returns.std()

        _markowitz_realized_sharpe_ratio: float = (
            _markowitz_realized_mean - self.risk_free
        ) / _markowitz_realized_std

        mc_portfolio: NDArray = self.monte_carlo_portfolio(matrix_backward_returns)

        _mc_realized_returns: NDArray = (
            current_period_returns @ mc_portfolio
        )

        _mc_realized_mean: float = _mc_realized_returns.mean()
        _mc_realized_std: float = _mc_realized_returns.std()

        _mc_realized_sharpe_ratio: float = (
            _mc_realized_mean - self.risk_free
        ) / _mc_realized_std

        """print("DDPG: ", _model_realized_sharpe_ratio)
        print("MC: ", _mc_realized_sharpe_ratio)
        print("MARKOWITZ: ", _markowitz_realized_sharpe_ratio)"""

        if _markowitz_realized_sharpe_ratio > _mc_realized_sharpe_ratio:
            _mc_realized_sharpe_ratio = _markowitz_realized_sharpe_ratio
            _mc_realized_mean = _markowitz_realized_mean
            _mc_realized_std = _markowitz_realized_std
            _mc_realized_returns = _markowitz_realized_mean

        return (
            _model_realized_returns,
            _model_realized_mean,
            _model_realized_std,
            _model_realized_sharpe_ratio,
            _mc_realized_returns,
            _mc_realized_mean,
            _mc_realized_std,
            _mc_realized_sharpe_ratio,
            mc_portfolio,
        )

    def reset(
        self,
    ) -> tuple:
        self._episode_ended: bool = False

        self.model_sharpe_ratio: float = 0.0
        self.model_return: float = 0.0
        self.model_mean_returns: float = 0.0
        self.model_volatility_returns: float = 0.0

        self.mc_sharpe_ratio: float = 0.0
        self.mc_return: float = 0.0
        self.mc_mean_returns: float = 0.0
        self.mc_volatility_returns: float = 0.0

        self.portfolio_weights = np.ones((self.n_assets)) / self.n_assets

        self.observations_df: pd.DataFrame = self.observation_dataset.copy()
        self.observations_df.reset_index(drop=True, inplace=True)

        self.reward_df: pd.DataFrame = self.reward_computation_dataset.copy()
        self.reward_df.reset_index(drop=True, inplace=True)

        self.max_index: int = self.reward_df.shape[0]

        self.start_point: int = np.random.choice(
            np.arange(
                self.delta_days,
                self.max_index - self.episode_length,
            )
        )
        self.end_point: int = self.start_point + self.episode_length + 1

        self.observations_df = self.observations_df.loc[
            self.start_point - self.delta_days : self.end_point
        ]
        self.observations_df.reset_index(drop=True, inplace=True)

        self.reward_df = self.reward_df.loc[
            self.start_point - self.delta_days : self.end_point
        ]
        self.reward_df.reset_index(drop=True, inplace=True)

        self.current_index: int = 0
        self.step_reward: float = 0.0

        self._state: NDArray = self.get_observations()
        self._episode_ended: bool = (
            True
            if self.current_index + self.delta_days == self.episode_length
            else False
        )

        _info: dict = {
            "state": self._state,
            "index": self.current_index,
            "portfolio_weights": self.portfolio_weights,
            "model_return": self.model_return,
            "model_mean_returns": self.model_mean_returns,
            "model_volatility_returns": self.model_volatility_returns,
            "model_sharpe_ratio": self.model_sharpe_ratio,
            "mc_return": self.mc_return,
            "mc_mean_returns": self.mc_mean_returns,
            "mc_volatility_returns": self.mc_volatility_returns,
            "mc_sharpe_ratio": self.mc_sharpe_ratio,
            "reward": self.step_reward,
            "episode_ended": self._episode_ended,
        }

        return (self._state, self.step_reward, self._episode_ended, _info)

    def step(
        self,
        action: NDArray,
    ) -> tuple:
        if self._episode_ended:
            return self.reset()

        self.step_time(action)

        self._state: NDArray = self.get_observations()
        self._episode_ended: bool = (
            True
            if self.current_index + self.delta_days >= self.episode_length
            else False
        )

        _info: dict = {
            "state": self._state,
            "index": self.current_index,
            "portfolio_weights": self.portfolio_weights,
            "model_return": self.model_return,
            "model_mean_returns": self.model_mean_returns,
            "model_volatility_returns": self.model_volatility_returns,
            "model_sharpe_ratio": self.model_sharpe_ratio,
            "mc_return": self.mc_return,
            "mc_mean_returns": self.mc_mean_returns,
            "mc_volatility_returns": self.mc_volatility_returns,
            "mc_sharpe_ratio": self.mc_sharpe_ratio,
            "mc_portfolio": self.mc_portfolio,
            "reward": self.step_reward,
            "episode_ended": self._episode_ended,
        }

        return (self._state, self.step_reward, self._episode_ended, _info)

    def step_time(
        self,
        action: NDArray,
    ):
        matrix_returns: NDArray = self.reward_df.loc[
            self.current_index : self.current_index + self.delta_days - 1,
            self.reward_df.columns != "Date",
        ].to_numpy().astype(np.float32)

        matrix_backward_returns: NDArray = self.reward_df.loc[
            self.current_index - self.delta_days : self.current_index - 1,
            self.reward_df.columns != "Date",
        ].to_numpy().astype(np.float32)

        (
            model_realized_returns,
            model_realized_mean,
            model_realized_std,
            model_realized_sharpe_ratio,
            mc_realized_returns,
            mc_realized_mean,
            mc_realized_std,
            mc_realized_sharpe_ratio,
            mc_portfolio,
        ) = self.reward_function(
            matrix_returns,
            matrix_backward_returns,
            action,
        )

        self.model_return = (model_realized_returns + 1).cumprod()[-1]
        self.model_mean_returns = model_realized_mean
        self.model_volatility_returns = model_realized_std
        self.model_sharpe_ratio = model_realized_sharpe_ratio

        self.mc_return = (mc_realized_returns + 1).cumprod()[-1]
        self.mc_mean_returns = mc_realized_mean
        self.mc_volatility_returns = mc_realized_std
        self.mc_sharpe_ratio = mc_realized_sharpe_ratio
        self.mc_portfolio = mc_portfolio

        self.step_reward = model_realized_sharpe_ratio - mc_realized_sharpe_ratio

    def get_observations(
        self,
    ):
        self.current_index += self.delta_days
        return self.observations_df.loc[
            self.current_index - self.delta_days : self.current_index - 1, :
        ].values.astype(np.float32)


class OUActionNoise:
    def __init__(
        self,
        mean: float,
        std_dev: float,
        theta: float = 0.2,
        dt: float = 1 / 252,
        x_initial: float | None = None,
    ):
        self.theta: float = theta
        self.mean: float = mean
        self.std_dev: float = std_dev
        self.dt: float = dt
        self.x_initial: float | None = x_initial
        self.reset()

    def __call__(self):
        x: float = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(
        self,
        n_delta_days: int,
        n_features: int,
        n_assets: int,
        buffer_capacity: int = 50_000,
        batch_size: int = 32,
    ):
        self.n_delta_days: int = n_delta_days
        self.n_features: int = n_features
        self.n_assets: int = n_assets

        self.buffer_capacity: int = buffer_capacity
        self.batch_size: int = batch_size

        self.buffer_counter: int = 0

        self.state_buffer = np.zeros(
            (self.buffer_capacity, self.n_delta_days, self.n_features)
        )
        self.action_buffer = np.zeros((self.buffer_capacity, self.n_assets))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros(
            (self.buffer_capacity, self.n_delta_days, self.n_features)
        )

    def record(
        self,
        obs_tuple: tuple,
    ):
        index: int = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index, :, :] = np.array(obs_tuple[0])
        self.action_buffer[index, :] = obs_tuple[1]
        self.reward_buffer[index, :] = obs_tuple[2]
        self.next_state_buffer[index, :] = np.array(obs_tuple[3])

        self.buffer_counter += 1

    @tf.function
    def update(
        self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
        target_actor,
        target_critic,
        actor_model,
        actor_optimizer,
        critic_model,
        critic_optimizer,
        gamma,
    ):
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    def learn(
        self,
        target_actor,
        target_critic,
        actor_model,
        actor_optimizer,
        critic_model,
        critic_optimizer,
        gamma,
    ):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.expand_dims(
            tf.convert_to_tensor(self.reward_buffer[batch_indices].astype(np.float32)),
            1,
        )
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            target_actor,
            target_critic,
            actor_model,
            actor_optimizer,
            critic_model,
            critic_optimizer,
            gamma,
        )


@tf.function
def update_target(target_weights, weights, tau):
    for a, b in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


def get_actor(
    delta_days: int,
    n_features: int,
    n_actions: int,
):
    inputs = layers.Input(shape=(delta_days, n_features))
    conv1 = layers.Conv1D(
        filters=512,
        kernel_size=3,
        padding="same",
    )(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.ReLU()(conv1)

    conv2 = layers.Conv1D(filters=256, kernel_size=3, padding="same")(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.ReLU()(conv2)

    conv3 = layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.ReLU()(conv3)

    gap = layers.GlobalAveragePooling1D()(conv3)
    gap = layers.Flatten()(gap)
    gap = layers.Dense(32)(gap)
    outputs = layers.Dense(n_actions, activation="softmax")(gap)

    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic(
    delta_days: int,
    n_features: int,
    n_actions: int,
):
    inputs = layers.Input((delta_days, n_features))

    conv1 = layers.Conv1D(filters=64, kernel_size=3, padding="same")(inputs)
    conv1 = layers.BatchNormalization()(conv1)

    gap = layers.GlobalAveragePooling1D()(conv1)
    gap = layers.Flatten()(gap)
    outputs = layers.Dense(32, activation="linear")(gap)

    action_input = layers.Input(shape=(n_actions))
    action_out = layers.Dense(32, activation="softmax")(action_input)
    concat = layers.Concatenate()([outputs, action_out])

    out = layers.Dense(256, activation="ReLU")(concat)
    out = layers.Dense(128, activation="ReLU")(out)
    out = layers.Dense(64, activation="ReLU")(out)
    out = layers.Dense(32, activation="linear")(out)
    outputs = layers.Dense(1)(out)

    model = tf.keras.Model([inputs, action_input], outputs)

    return model


def policy(
    actor_model,
    state,
    noise_object,
):
    sampled_actions = actor_model(np.expand_dims(state, 0))
    sampled_actions = np.array(sampled_actions).reshape(-1, 1)
    # noise = np.abs(noise_object().reshape(-1, 1))
    noise = np.zeros(len(sectors)).reshape(-1, 1)

    sampled_actions = (sampled_actions + noise) / (sampled_actions + noise).sum()

    return sampled_actions


delta_d: int = 15
std_dev: float = 0.05
ou_noise = OUActionNoise(
    mean=np.zeros(len(sectors)),
    std_dev=float(std_dev) * np.ones(len(sectors)),
    dt=1 / 252,
)

n_features: int = len(observation_dataset_columns)
n_actions: int = len(sectors)

actor_model = get_actor(delta_days=delta_d, n_features=n_features, n_actions=n_actions)
critic_model = get_critic(
    delta_days=delta_d, n_features=n_features, n_actions=n_actions
)

target_actor = get_actor(delta_days=delta_d, n_features=n_features, n_actions=n_actions)
target_critic = get_critic(
    delta_days=delta_d, n_features=n_features, n_actions=n_actions
)

target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

critic_lr: float = 0.0001
actor_lr: float = 0.0001

critic_optimizer = tfk.optimizers.RMSprop(critic_lr)
actor_optimizer = tfk.optimizers.RMSprop(actor_lr)

total_episodes: int = 4_500
gamma: float = 0.95
tau: float = 0.15

buffer: Buffer = Buffer(n_delta_days=delta_d, n_features=n_features, n_assets=n_actions)

environment = TradingEnvironment(
    len(sectors),
    reward_computation_dataset,
    observations_dataset,
    _delta_days=15,
    _episode_length=200,
)

episode_reward: dict[int, list[float]] = {}
states_: dict[int, list[float]] = {}
actions_: dict[int, list[float]] = {}
info_: dict[int, list[dict[str, float | list | int]]] = {}

for ep in range(1, total_episodes):
    prev_tuple = environment.reset()
    prev_state = prev_tuple[0]

    while True:
        action = policy(
            actor_model=actor_model,
            state=prev_state,
            noise_object=ou_noise,
        )
        state, reward, done, info = environment.step(action)

        buffer.record((prev_state, action.reshape(1, -1), reward, state))
        rewards_list: list[float] = episode_reward.get(ep, [])
        rewards_list.append(info.get("reward", 0.0))
        episode_reward[ep] = rewards_list

        states_list: list[float] = states_.get(ep, [])
        states_list.append(state)
        states_[ep] = states_list

        actions_list: list[float] = actions_.get(ep, [])
        actions_list.append(action)
        actions_[ep] = actions_list

        info_list: list[float] = info_.get(ep, [])
        info_list.append(info)
        info_[ep] = info_list

        buffer.learn(
            target_actor=target_actor,
            target_critic=target_critic,
            actor_model=actor_model,
            actor_optimizer=actor_optimizer,
            critic_model=critic_model,
            critic_optimizer=critic_optimizer,
            gamma=gamma,
        )

        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)

        if done:
            break

        prev_state = state

    print(f"Episode #{ep} -> Cumulative Reward: {sum(episode_reward.get(ep, [0]))}")

actor_model.save_weights(model_folder.joinpath("actor.h5"))
critic_model.save_weights(model_folder.joinpath("critic.h5"))

target_actor.save_weights(model_folder.joinpath("target_actor.h5"))
target_critic.save_weights(model_folder.joinpath("target_critic.h5"))
