import re
from copy import deepcopy

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import numpy as np


class FeaturePreprocessorBasic(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self, X, y=None):

        X = deepcopy(X)
        X["mileage"] = X["mileage"].str.split(" ").str[0].astype(float)
        X["engine"] = X["engine"].str.split(" ").str[0].astype(float)
        X["max_power"] = (
            X["max_power"].str.split(" ").str[0].replace("", None).astype(float)
        )

        X["max_torque_rpm"] = X["torque"].apply(self.find_max_torque_rpm)
        X["torque"] = X["torque"].apply(self.find_torque)

        self.cols_with_nans = X.columns[X.isna().any(axis=0)].tolist()
        self.imp = SimpleImputer(strategy="median")
        self.imp.fit(X[self.cols_with_nans])
        self.imp.set_output(transform="pandas")

        return self

    @staticmethod
    def find_torque(value):

        if value is np.nan:
            return None

        torque = float(
            re.findall("\d+\.*,*\d*", value.split(" ")[0])[0].replace(",", "")
        )

        if len(re.findall("Nm|nm", value)) > 0:
            # torque is in Nm
            # nothing to do
            pass
        else:
            # torque is in Kgm
            # convert to Nm
            torque = torque * 9.80665

        return torque

    @staticmethod
    def find_max_torque_rpm(value):

        if value is np.nan:
            return None

        max_torque_rpm = None
        for k in value.split(" ")[1:]:
            # iterate untill not finding next digits
            try:
                max_torque_rpm = float(
                    re.findall("\d+\.*,*\d*", k)[-1].replace(",", "")
                )
                break
            except IndexError:
                continue

        return max_torque_rpm

    def transform(self, X):
        X = deepcopy(X)
        X["mileage"] = X["mileage"].str.split(" ").str[0].astype(float)
        X["engine"] = X["engine"].str.split(" ").str[0].astype(float)
        X["max_power"] = (
            X["max_power"].str.split(" ").str[0].replace("", None).astype(float)
        )

        X["max_torque_rpm"] = X["torque"].apply(self.find_max_torque_rpm)
        X["torque"] = X["torque"].apply(self.find_torque)

        X[self.cols_with_nans] = self.imp.transform(X[self.cols_with_nans])

        X[["engine", "seats"]] = X[["engine", "seats"]].astype(int)

        X["name"] = X.name.str.split(" ").str[0]

        return X

    def set_output(self, *args, **kwargs):
        return
