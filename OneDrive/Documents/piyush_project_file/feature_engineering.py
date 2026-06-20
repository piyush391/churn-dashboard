from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Feature 1
        X["BalanceSalaryRatio"] = (
            X["Balance"] /
            (X["EstimatedSalary"] + 1)
        )

        # Feature 2
        X["ProductDensity"] = (
            X["NumOfProducts"] /
            (X["Age"] + 1)
        )

        # Feature 3
        X["AgeTenureInteraction"] = (
            X["Age"] *
            X["Tenure"]
        )

        return X