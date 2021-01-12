## example of correlation feature selection for numerical data
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from matplotlib import pyplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

## feature selection
def select_features(X_train, y_train, X_test):
	# configure to select all features
	fs = SelectKBest(score_func=f_regression, k='all')
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

# load the dataset

#X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1, random_state=1)
dataset = pd.read_csv('life_expectancy.csv')
#dataset = dataset.drop(['Population'], axis=1)
X = dataset.iloc[:,2:].values
X = np.delete(X,1,1)
y = dataset.iloc[:, 3].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer2 = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, :])
y = y.reshape(-1,1)
imputer2.fit(y)
X[:, :] = imputer.transform(X[:, :])
y = imputer2.transform(y)



from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct1 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct1.fit_transform(X))


# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train[:, :] = sc.fit_transform(X_train[:, :])
# X_test[:, :] = sc.transform(X_test[:, :])

# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# what are scores for the features
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()

# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# regressor.fit(X_train_fs, y_train)

# # Training the Decision Tree Regression model on the Training set
# from sklearn.tree import DecisionTreeRegressor
# regressor = DecisionTreeRegressor(random_state = 0)
# regressor.fit(X_train_fs, y_train)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 50, random_state = 0)
regressor.fit(X_train, y_train)



#Predicting the Test set results
y_pred = regressor.predict(X_test_fs)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluating the Model Performance
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
print(r2_score(y_test, y_pred))

# plt.scatter(X_train[:,0], y_train, color = 'red')
# #plt.plot(X_train[:,16], regressor.predict(X_train[:,16]), color = 'blue')
# plt.title('Salary vs Experience (Training set)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()


from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print('MAE: %.3f' % mae)

