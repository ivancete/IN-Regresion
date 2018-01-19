# -*- coding: utf-8 -*-

#https://www.kaggle.com/neviadomski/how-to-get-to-top-25-with-simple-model-sklearn/notebook

# Adding needed libraries and reading data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head()

#Checking for missing data
NAs = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1, keys=['Train', 'Test'])
print(NAs[NAs.sum(axis=1) > 0])

# Prints R2 and RMSE scores
def get_score(prediction, lables):    
    print('R2: {}'.format(r2_score(prediction, lables)))
    print('RMSE: {}'.format(np.sqrt(mean_squared_error(prediction, lables))))

# Shows scores for train and validation sets    
def train_test(estimator, x_trn, x_tst, y_trn, y_tst):
    prediction_train = estimator.predict(x_trn)
    # Printing estimator
    print(estimator)
    # Printing train scores
    get_score(prediction_train, y_trn)
    prediction_test = estimator.predict(x_tst)
    # Printing test scores
    print("Test")
    get_score(prediction_test, y_tst)
    
# Spliting to features and lables and deleting variable I don't need
# Aquí le quitamos la columna del precio de venta al dataset train.
train_labels = train.pop('SalePrice')


features = pd.concat([train, test], keys=['train', 'test'])

# Eliminamos las columnas que tienen más de la mitad de los valores perdidos.
# I decided to get rid of features that have more than half of missing information or do not correlate to SalePrice
#features.drop(['Utilities', 'RoofMatl', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'Heating', 'LowQualFinSF',
#               'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'WoodDeckSF',
#               'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal'],
#              axis=1, inplace=True)

features.drop(['Alley', 'PoolQC', 'Fence', 'FireplaceQu', 'MiscVal', 'MiscFeature', 'LotFrontage','GarageYrBlt', 'GarageArea', 'GarageCond',
               'MasVnrArea','MasVnrType', 'GarageQual'],
              axis=1, inplace=True)

features['Utilities'] = features['Utilities'].fillna('AllPub')

features['RoofMatl'] = features['RoofMatl'].fillna('CompShg')

features['BsmtFinSF1'] = features['BsmtFinSF1'].fillna(0)

features['BsmtFinSF2'] = features['BsmtFinSF2'].fillna(0)

features['BsmtUnfSF'] = features['BsmtUnfSF'].fillna(0)

features['Heating'] = features['Heating'].fillna('GasA')

features['LowQualFinSF'] = features['LowQualFinSF'].fillna(0)

features['BsmtFullBath'] = features['BsmtFullBath'].fillna(0)

features['BsmtHalfBath'] = features['BsmtHalfBath'].fillna(0)

features['Functional'] = features['Functional'].fillna('Typ')

features['WoodDeckSF'] = features['WoodDeckSF'].fillna(0)

features['OpenPorchSF'] = features['OpenPorchSF'].fillna(0)

features['EnclosedPorch'] = features['EnclosedPorch'].fillna(0)

features['3SsnPorch'] = features['3SsnPorch'].fillna(0)

features['ScreenPorch'] = features['ScreenPorch'].fillna(0)

features['PoolArea'] = features['PoolArea'].fillna(0)

features['PoolArea'] = features['PoolArea'].fillna(0)

# MSSubClass as str
features['MSSubClass'] = features['MSSubClass'].astype(str)

# MSZoning NA in pred. filling with most popular values
features['MSZoning'] = features['MSZoning'].fillna(features['MSZoning'].mode()[0])

# LotFrontage  NA in all. I suppose NA means 0
#features['LotFrontage'] = features['LotFrontage'].fillna(features['LotFrontage'].mean())

# Alley  NA in all. NA means no access
#features['Alley'] = features['Alley'].fillna('NOACCESS')

# Converting OverallCond to str
features.OverallCond = features.OverallCond.astype(str)

# MasVnrType NA in all. filling with most popular values
#features['MasVnrType'] = features['MasVnrType'].fillna(features['MasVnrType'].mode()[0])

# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2
# NA in all. NA means No basement
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    features[col] = features[col].fillna('NoBSMT')

# TotalBsmtSF  NA in pred. I suppose NA means 0
features['TotalBsmtSF'] = features['TotalBsmtSF'].fillna(0)

# Electrical NA in pred. filling with most popular values
features['Electrical'] = features['Electrical'].fillna(features['Electrical'].mode()[0])

# KitchenAbvGr to categorical
features['KitchenAbvGr'] = features['KitchenAbvGr'].astype(str)

# KitchenQual NA in pred. filling with most popular values
features['KitchenQual'] = features['KitchenQual'].fillna(features['KitchenQual'].mode()[0])

# FireplaceQu  NA in all. NA means No Fireplace
#features['FireplaceQu'] = features['FireplaceQu'].fillna('NoFP')

# GarageType, GarageFinish, GarageQual  NA in all. NA means No Garage
#for col in ('GarageType', 'GarageFinish', 'GarageQual'):
#    features[col] = features[col].fillna('NoGRG')

# GarageCars  NA in pred. I suppose NA means 0
features['GarageCars'] = features['GarageCars'].fillna(0.0)

# SaleType NA in pred. filling with most popular values
features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])

# Year and Month to categorical
features['YrSold'] = features['YrSold'].astype(str)
features['MoSold'] = features['MoSold'].astype(str)

# Adding total sqfootage feature and removing Basement, 1st and 2nd floor features
features['TotalSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']
features.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)
# Hasta aquí lo que hacemos es reemplazar valores de cada una de las características.

# Our SalesPrice is skewed right (check plot below). I'm logtransforming it. 
plt.figure(1)
plt.clf()
ax = sns.distplot(train_labels)

## Log transformation of labels
train_labels = np.log(train_labels)

## Now it looks much better
plt.figure(2)
plt.clf()
ax = sns.distplot(train_labels)

## Standardizing numeric features
numeric_features = features.loc[:,['LotFrontage', 'LotArea', 'GrLivArea', 'TotalSF']]
numeric_features_standardized = (numeric_features - numeric_features.mean())/numeric_features.std()


#ax = sns.pairplot(numeric_features_standardized)

# Getting Dummies from Condition1 and Condition2
conditions = set([x for x in features['Condition1']] + [x for x in features['Condition2']])
dummies = pd.DataFrame(data=np.zeros((len(features.index), len(conditions))),
                       index=features.index, columns=conditions)
for i, cond in enumerate(zip(features['Condition1'], features['Condition2'])):
    dummies.ix[i, cond] = 1
features = pd.concat([features, dummies.add_prefix('Condition_')], axis=1)
features.drop(['Condition1', 'Condition2'], axis=1, inplace=True)

# Getting Dummies from Exterior1st and Exterior2nd
exteriors = set([x for x in features['Exterior1st']] + [x for x in features['Exterior2nd']])
dummies = pd.DataFrame(data=np.zeros((len(features.index), len(exteriors))),
                       index=features.index, columns=exteriors)
for i, ext in enumerate(zip(features['Exterior1st'], features['Exterior2nd'])):
    dummies.ix[i, ext] = 1
features = pd.concat([features, dummies.add_prefix('Exterior_')], axis=1)
features.drop(['Exterior1st', 'Exterior2nd', 'Exterior_nan'], axis=1, inplace=True)

# Getting Dummies from all other categorical vars
for col in features.dtypes[features.dtypes == 'object'].index:
    for_dummy = features.pop(col)
    features = pd.concat([features, pd.get_dummies(for_dummy, prefix=col)], axis=1)
    
### Copying features
features_standardized = features.copy()

### Replacing numeric features by standardized values
features_standardized.update(numeric_features_standardized)

### Splitting features
train_features = features.loc['train'].drop('Id', axis=1).select_dtypes(include=[np.number]).values
test_features = features.loc['test'].drop('Id', axis=1).select_dtypes(include=[np.number]).values

### Splitting standardized features
train_features_st = features_standardized.loc['train'].drop('Id', axis=1).select_dtypes(include=[np.number]).values
test_features_st = features_standardized.loc['test'].drop('Id', axis=1).select_dtypes(include=[np.number]).values

### Shuffling train sets
train_features_st, train_features, train_labels = shuffle(train_features_st, train_features, train_labels, random_state = 5)

### Splitting
x_train, x_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.1, random_state=200)
x_train_st, x_test_st, y_train_st, y_test_st = train_test_split(train_features_st, train_labels, test_size=0.1, random_state=200)

'''
Elastic Net
'''

ENSTest = linear_model.ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10], l1_ratio=[.01, .1, .5, .9, .99], max_iter=5000).fit(x_train_st, y_train_st)
train_test(ENSTest, x_train_st, x_test_st, y_train_st, y_test_st)    

# Average R2 score and standard deviation of 5-fold cross-validation
scores = cross_val_score(ENSTest, train_features_st, train_labels, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

'''
Gradient Boosting
'''

GBest = ensemble.GradientBoostingRegressor(n_estimators=3500, learning_rate=0.05, max_depth=5, max_features='sqrt',
                                               min_samples_leaf=15, min_samples_split=10, loss='huber').fit(x_train, y_train)
train_test(GBest, x_train, x_test, y_train, y_test)

# Average R2 score and standart deviation of 5-fold cross-validation
scores = cross_val_score(GBest, train_features_st, train_labels, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Retraining models
GB_model = GBest.fit(train_features, train_labels)
ENST_model = ENSTest.fit(train_features_st, train_labels)

## Getting our SalePrice estimation
Final_labels = (np.exp(GB_model.predict(test_features)) + np.exp(ENST_model.predict(test_features_st))) / 2

## Saving to CSV
pd.DataFrame({'Id': test.Id, 'SalePrice': Final_labels}).to_csv('submissionGradientBoosted.csv', index =False)

'''
Random Forest
'''
'''
RFest = ensemble.RandomForestRegressor(n_estimators=3000, max_depth=4, max_features='sqrt',
                                               min_samples_leaf=20, min_samples_split=20).fit(x_train, y_train)

train_test(RFest, x_train, x_test, y_train, y_test)

# Average R2 score and standart deviation of 5-fold cross-validation
scores = cross_val_score(RFest, train_features_st, train_labels, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Retraining models
RF_model = RFest.fit(train_features, train_labels)
ENST_model = ENSTest.fit(train_features_st, train_labels)

## Getting our SalePrice estimation
Final_labels = (np.exp(RF_model.predict(test_features)) + np.exp(ENST_model.predict(test_features_st))) / 2

## Saving to CSV
pd.DataFrame({'Id': test.Id, 'SalePrice': Final_labels}).to_csv('submissionRandomForest.csv', index =False)
'''

'''
Decision Tree
'''
'''
DTest = tree.DecisionTreeRegressor(max_depth=10).fit(x_train, y_train)

train_test(DTest, x_train, x_test, y_train, y_test)

# Average R2 score and standart deviation of 5-fold cross-validation
scores = cross_val_score(DTest, train_features_st, train_labels, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Retraining models
DT_model = DTest.fit(train_features, train_labels)
ENST_model = ENSTest.fit(train_features_st, train_labels)

## Getting our SalePrice estimation
Final_labels = (np.exp(DT_model.predict(test_features)) + np.exp(ENST_model.predict(test_features_st))) / 2

## Saving to CSV
pd.DataFrame({'Id': test.Id, 'SalePrice': Final_labels}).to_csv('submissionDecisionTree.csv', index =False)
'''

'''
Neural Network
'''
'''
Nnest = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1).fit(x_train, y_train)

train_test(Nnest, x_train, x_test, y_train, y_test)

# Average R2 score and standart deviation of 5-fold cross-validation
scores = cross_val_score(Nnest, train_features_st, train_labels, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Retraining models
NN_model = Nnest.fit(train_features, train_labels)
ENST_model = ENSTest.fit(train_features_st, train_labels)

## Getting our SalePrice estimation
Final_labels = (np.exp(NN_model.predict(test_features)) + np.exp(ENST_model.predict(test_features_st))) / 2

## Saving to CSV
pd.DataFrame({'Id': test.Id, 'SalePrice': Final_labels}).to_csv('submissionNeuralNetwork.csv', index =False)
'''

'''
SVR
'''
'''
SVRest = SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False).fit(x_train, y_train)

train_test(SVRest, x_train, x_test, y_train, y_test)

# Average R2 score and standart deviation of 5-fold cross-validation
scores = cross_val_score(SVRest, train_features_st, train_labels, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Retraining models
SVR_model = SVRest.fit(train_features, train_labels)
ENST_model = ENSTest.fit(train_features_st, train_labels)

## Getting our SalePrice estimation
Final_labels = (np.exp(SVR_model.predict(test_features)) + np.exp(ENST_model.predict(test_features_st))) / 2

## Saving to CSV
pd.DataFrame({'Id': test.Id, 'SalePrice': Final_labels}).to_csv('submissionSVR.csv', index =False)
'''
