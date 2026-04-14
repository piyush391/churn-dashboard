from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X['BalanceSalaryRatio'] = X['Balance'] / (X['EstimatedSalary'] + 1)
        X['ProductDensity'] = X['NumOfProducts'] / (X['Tenure'] + 1)
        X['EngagementScore'] = X['IsActiveMember'] * X['NumOfProducts']
        X['AgeTenure'] = X['Age'] * X['Tenure']

        return X