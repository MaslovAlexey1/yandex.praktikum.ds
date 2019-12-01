import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('/Users/alexeymaslov/Git/github/praktikum/Sprint 5/datasets/flights_preprocessed.csv')
features = df.drop(columns=['Arrival Delay'], axis=1)
target = df['Arrival Delay']

features_train, features_valid, target_train, target_valid = train_test_split(features, target, test_size=0.25, random_state=12345)


model = LinearRegression()
model.fit(features_train, target_train)
print(model.score(features_valid, target_valid))

for min_samples_leaf in range(2, 20, 1):
    model = RandomForestRegressor(n_estimators=20, max_depth=12, random_state=12345, min_samples_leaf=min_samples_leaf)
    model.fit(features_train, target_train)
    predicted_valid = model.predict(features_valid)
    print('min_samples_leaf=', min_samples_leaf, '| mae=', mean_absolute_error(predicted_valid, target_valid))

# model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=12345, min_samples_leaf=7)
# model.fit(features_train, target_train)
# print(model.score(features_valid, target_valid))

joblib.dump(model, 
    '/Users/alexeymaslov/Git/github/praktikum/Sprint 5/best_model.joblib')