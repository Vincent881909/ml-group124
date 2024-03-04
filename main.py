import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor


df = pd.read_csv('data/train.csv')
df = df.drop(columns=['Id'])

# getting only numerical columns
df_num = df.select_dtypes(include=['float64', 'int64'])
df = df_num[
    ['LotArea', 'MasVnrArea', 'TotalBsmtSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
     'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
     'SalePrice']]

# drop rows with missing values as random forest cannot directly handle missing values
df = df.dropna()

# split data set into train and validation
train = df.sample(frac=0.8, random_state=200)
validation = df.drop(train.index)

# split the columns into features and target; ie x y split
target = 'SalePrice'
train_x = train.drop(target, axis=1)
train_y = train[target]

validation_x = validation.drop(target, axis=1)
validation_y = validation[target]

# handle categorical data #TODO not currently functional
# encoder = OrdinalEncoder()
# for column in train[features].columns:
#     if train[column].dtype == 'object':
#         train[column] = encoder.fit_transform(train[column].values.reshape(-1, 1))
#         validation[column] = encoder.fit_transform(validation[column].values.reshape(-1, 1))

# create the model
rfr_model = RandomForestRegressor(n_estimators=100, max_depth=10)

# fit the model
rfr_model.fit(train_x, train_y)

# make predictions
validation_predictions = rfr_model.predict(validation_x)

# print the mean squared error
print(mean_squared_error(validation_y, validation_predictions))
print(r2_score(validation_y, validation_predictions))

# create the model
gbr_model = GradientBoostingRegressor(loss='squared_error')

gbr_model.fit(train_x, train_y)

print(gbr_model.score(validation_x, validation_y))

# print the names of the features
# print(features)

# print names of columns
# print(df.columns)
