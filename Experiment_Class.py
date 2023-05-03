from dataclasses import dataclass, field
import pandas as pd
import tensorflow

import myLibrary as mL
import matplotlib.pyplot as plt
import re
from sklearn.metrics import mean_squared_error, mean_absolute_error

# MOVED TO myLibrary.py
# TODO: DELETE THIS FILE AS SOON AS EVERYTHING WORKS!!


# @dataclass
# class Experiment():
#     name: str
#     description: str
#
#     stations: list
#     years: list
#     nan_threshold: float
#     features: list
#     era5: bool
#     stationary_shift: int
#     lag: int
#     n_test_hours: int
#
#     #Preprocessing:
#     #stationary: bool
#     scaler: object
#
#     #Model:
#     model_name: str
#     model_summary: str
#
#     one_shot_forecast: pd.DataFrame
#     recursive_forecast: pd.DataFrame
#
#     one_shot_forecast_MSE: float  = -1.0
#     one_shot_forecast_MAE: float  = -1.0
#     recursive_forecast_MAE: float = -1.0
#     recursive_forecast_MSE: float = -1.0
#
#     def __post_init__(self):
#         self.__calc_metrics()
#
#     def get_raw_data(self):
#         data = mL.get_data(
#             stations=self.stations,
#             years=self.years,
#             nan_threshold=self.nan_threshold,
#             features=self.features,
#             era5=self.era5
#         )
#         return data
#
#     def print_settings(self):
#         print(f"Experiment: {self.name}")
#         print(f"{self.description}")
#         print(f"---------------------------------------")
#         print(f"Stations: {self.stations}")
#         print(f"Years: {self.years}")
#         print(f"NaN_Threshold: {self.nan_threshold}")
#         print(f"Features: {self.features}")
#         print(f"ERA5: {self.era5}, Stationary Shift: {self.stationary_shift}, lag: {self.lag}, Test-Hours:{self.n_test_hours}")
#         print(f"\n---------------------------------------")
#         # print(f"Preprocessing: Stationary: {self.stationary}")
#         print(f"Normalized: {True if self.scaler is not None else False}")
#         print(f"\n---------------------------------------")
#         print(f"Internal Model name: {self.model_name}")
#         print(self.model_summary)
#
#     def __calc_metrics(self):
#         # One shot forecasting:
#         wtmp_true = [col for col in self.one_shot_forecast.columns if col.startswith("WTMP")][0]
#
#         self.one_shot_forecast_MAE = mean_absolute_error(
#             self.one_shot_forecast[wtmp_true],
#             self.one_shot_forecast[f"{wtmp_true}_pred"]
#         )
#
#         self.one_shot_forecast_MSE = mean_squared_error(
#             self.one_shot_forecast[wtmp_true],
#             self.one_shot_forecast[f"{wtmp_true}_pred"]
#         )
#         # recurent forecasting:
#         wtmp_true = [col for col in self.recursive_forecast.columns if col.startswith("WTMP")][0]
#
#         self.recursive_forecast_MAE = mean_absolute_error(
#             self.recursive_forecast[wtmp_true],
#             self.recursive_forecast[f"{wtmp_true}_pred"]
#         )
#
#         self.recursive_forecast_MSE = mean_squared_error(
#             self.recursive_forecast[wtmp_true],
#             self.recursive_forecast[f"{wtmp_true}_pred"]
#         )
#
#     def print_metrics(self):
#         print("One-Shot-Forecasting:")
#         print(f"MAE: {self.one_shot_forecast_MAE} \tMSE: {self.one_shot_forecast_MSE}")
#         print("\nRecurrent-Forecasting:")
#         print(f"MAE: {self.recursive_forecast_MAE} \tMSE: {self.recursive_forecast_MSE}")
#
#     # PLOT CHARTS
#     # __ is the prefix for private functions!
#     def __print_chart_helper(self, df):
#         pred_cols = df.columns[df.columns.str.contains('_pred')]
#
#         # Extract the feature names from the column names using regular expression
#         features = [re.sub('_pred$', '', col) for col in pred_cols]
#
#         # Plot the ground truth and prediction using the selected columns
#         fig, ax = plt.subplots()
#         df.plot(y=[f'{feature}_pred' for feature in features] + [f'{feature}' for feature in features], ax=ax)
#
#         # Set the title and legend
#         plt.title('Ground Truth vs Prediction')
#         plt.legend([f'{feature}_pred' for feature in features] + [f'{feature}' for feature in features])
#         plt.show()
#
#     def print_one_shot_forecast(self):
#         self.__print_chart_helper(self.one_shot_forecast)
#
#     def print_recursive_forecast(self):
#         self.__print_chart_helper(self.recursive_forecast)
#
#     def print_one_shot_WTMP(self):
#         wtmp_true = [col for col in self.one_shot_forecast.columns if col.startswith("WTMP")][0]
#
#         # Plot the ground truth and prediction using the selected columns
#         fig, ax = plt.subplots()
#         self.one_shot_forecast.plot(y=[wtmp_true, f"{wtmp_true}_pred"], ax=ax)
#
#         # Set the title and legend
#         plt.title('Ground Truth vs Prediction')
#         plt.legend([wtmp_true, f"{wtmp_true}_pred"])
#         plt.show()
#
#     def print_recursive_WTMP(self):
#         wtmp_true = [col for col in self.one_shot_forecast.columns if col.startswith("WTMP")][0]
#
#         # Plot the ground truth and prediction using the selected columns
#         fig, ax = plt.subplots()
#         self.recursive_forecast.plot(y=[wtmp_true, f"{wtmp_true}_pred"], ax=ax)
#
#         # Set the title and legend
#         plt.title('Ground Truth vs Prediction')
#         plt.legend([wtmp_true, f"{wtmp_true}_pred"])
#         plt.show()
#
