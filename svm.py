#------------------------------------------------- svm for titanic (kaggle) --------------------------------------------
import pandas
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import numpy

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score

#import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------------------------------------------------
# load data
#-----------------------------------------------------------------------------------------------------------------------
titanic = pandas.read_csv("titanic/titanic_clean.csv")

convert_dict = {"PassengerId":int, 'class':int, "prefix":int,
                'sex':int, 'age':int, "sib_sp":int, "parch":int,
                'family_size':int, 'age_by_class':int, 'fare_by_class':int,
                'group_size':int, 'group_type':int, 'with_children':int, 'set':str}


titanic = titanic.astype(convert_dict)

#-----------------------------------------------------------------------------------------------------------------------
# split into X, Y and fit models
#-----------------------------------------------------------------------------------------------------------------------
X_train = titanic[titanic.set == "train"].drop(["survived", "set", "PassengerId"], axis = 1)
Y_train = titanic[titanic.set == "train"]["survived"].copy()
Y_train = Y_train.astype({"survived":int})


# support vector machine:
svm = SVC(kernel = 'rbf')
svm.fit(X_train, Y_train)
acc_svm = round(svm.score(X_train, Y_train)*100, 2)

print("sgd")
# Stocastic Gradient Decent:
sgd = linear_model.SGDClassifier(max_iter=5, tol = None)
sgd.fit(X_train, Y_train)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

print("rf")
# Random Forest:
rf = RandomForestClassifier(n_estimators=100, oob_score=True)
rf.fit(X_train, Y_train)
acc_rf = round(rf.score(X_train, Y_train)*100, 2)


print("lr")
# Logistic regression: errors - max iter reached
logreg = LogisticRegression(max_iter=10000)
logreg.fit(X_train, Y_train)
acc_logreg = round(logreg.score(X_train, Y_train)*100, 2)

print("knn")
# K Nearest Neighbor:
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)
acc_knn = round(knn.score(X_train, Y_train)*100, 2)

print("gnb")
# Gaussian Naive Bayes:
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
acc_gnb = round(gnb.score(X_train, Y_train)*100, 2)

print("p")
# Perceptron
prcy = Perceptron(max_iter=5)
prcy.fit(X_train, Y_train)
acc_prcy = round(prcy.score(X_train, Y_train)*100, 2)

print("svml")
# linear svm: # errors - max_iter reached
lsvm = LinearSVC(max_iter = 100000)
lsvm.fit(X_train, Y_train)
acc_lsvm = round(lsvm.score(X_train, Y_train)*100, 2)

print("dt")
# Decision Tree
dtree = DecisionTreeClassifier()
dtree.fit(X_train, Y_train)
acc_dtree = round(dtree.score(X_train, Y_train)*100, 2)

#-----------------------------------------------------------------------------------------------------------------------
# print results:
#-----------------------------------------------------------------------------------------------------------------------
results = pandas.DataFrame({
    'Model': ["SVM", "Stockastic Gradient Decent", "Random Forest",
              "Logistic Regression", "K Nearest Neighbors", "Gaussian Naive Bays",
              "Perceptron", "Linear SVM", "Decision Tree"],
    "Score": [acc_svm, acc_sgd, acc_rf,
              acc_logreg, acc_knn, acc_gnb,
              acc_prcy, acc_lsvm, acc_dtree]
}).sort_values(by = 'Score', ascending=False)

print(results.to_string())

#-----------------------------------------------------------------------------------------------------------------------
# relative importance
#-----------------------------------------------------------------------------------------------------------------------

rel_importance = pd.DataFrame({
    'feature':X_train.columns,
    'importance':numpy.round(rf.feature_importances_, 3)
}).sort_values('importance', ascending=False)

print(rel_importance.to_string())

#-----------------------------------------------------------------------------------------------------------------------
# k-folds cross validation
#-----------------------------------------------------------------------------------------------------------------------

rf_scores = cross_val_score(rf, X_train, Y_train, cv = 10, scoring="accuracy")
print(numpy.round(rf_scores, 4))
#[0.8667 0.7    0.9    0.7667 0.8333 0.7667 0.8333 0.7333 0.7    0.7
# 0.9    0.8667 0.8667 0.7333 0.8667 0.8333 0.7667 0.8333 0.7667 0.8333
# 0.8333 0.7931 0.8621 0.8276 0.7931 0.8621 0.8966 0.8276 0.8276 0.8966]
print("Mean:", round(rf_scores.mean(), 3), " SD:", round(rf_scores.std(), 3) ) # Mean: 0.816  SD: 0.06

#-----------------------------------------------------------------------------------------------------------------------
# try random forrest with fewer params
#-----------------------------------------------------------------------------------------------------------------------
#X_train_red = X_train.drop(["parch", 'with_children'], axis=1) # Mean: 0.8  SD: 0.04

#rf_red = RandomForestClassifier(n_estimators=100, oob_score=True)
#rf_red.fit(X_train_red, Y_train)
#
#rf_red_scores = cross_val_score(rf_red, X_train_red, Y_train, cv = 10, scoring="accuracy")
#print(print(numpy.round(rf_red_scores, 4)))
#print("Mean:", round(rf_red_scores.mean(), 3), " SD:", round(rf_red_scores.std(), 2))#

#acc_rf_red = round(rf_red.score(X_train_red, Y_train)*100, 2)
#print("complete rf: ", acc_rf, "reduced rf:",acc_rf_red)
#print('complete oob score:',round(rf.oob_score_, 4)*100, "reduced oob score: ", round(rf_red.oob_score_, 4)*100)

#rel_importance = pd.DataFrame({
#    'feature':X_train_red.columns,
#    'importance':numpy.round(rf_red.feature_importances_, 3)
#}).sort_values('importance', ascending=False)

#print(rel_importance.to_string())

#print(rf.get_params())

#-----------------------------------------------------------------------------------------------------------------------
# tune the hyper-parameters:
#-----------------------------------------------------------------------------------------------------------------------
#from sklearn.model_selection import GridSearchCV, cross_val_score
#param_grid = {"criterion":["gini", "entropy"],
#              "max_features":['auto', 'sqrt'],
#             "bootstrap":[True, False],
#              "max_depth":[int(x) for x in numpy.linspace(10, 110, num = 11)],
#              "min_samples_leaf": [1, 2, 4],
#              "min_samples_split":[2, 5, 10],
#              "n_estimators":[100, 400, 700, 1000, 1500]}

#rf = RandomForestClassifier()
#rf_random = RandomizedSearchCV(estimator = rf,param_distributions = param_grid,
#                   n_iter = 1980, cv = 5, verbose=2,
#                   random_state=42, n_jobs = -1)

#rf_random.fit(X_train, Y_train)
#print(rf_random.best_params_)

#{'n_estimators': 100,
# 'min_samples_split': 2,
# 'min_samples_leaf': 4,
# 'max_features': 'auto',
# 'max_depth': 70,
# 'criterion': 'gini',
# 'bootstrap': True}

#{'n_estimators': 100,
# 'min_samples_split': 10,
# 'min_samples_leaf': 2,
# 'max_features': 'sqrt',
# 'max_depth': 70,
# 'criterion': 'entropy',
# 'bootstrap': True}


# {'n_estimators': 100,
# 'min_samples_split': 10,
# 'min_samples_leaf': 4,
# 'max_features': 'auto',
# 'max_depth': 60,
# 'criterion': 'gini',
# 'bootstrap': True}

#-----------------------------------------------------------------------------------------------------------------------
# run model against test data and save output against passenger id
#-----------------------------------------------------------------------------------------------------------------------
rf = RandomForestClassifier(n_estimators=100,
                                min_samples_split = 10,
                                min_samples_leaf = 4,
                                max_features='auto',
                                max_depth=60,
                                criterion='gini',
                                bootstrap=True)
rf.fit(X_train, Y_train)

rf_scores = cross_val_score(rf, X_train, Y_train, cv = 10, scoring="accuracy")
print(print(numpy.round(rf_scores, 4)))
print("Mean:", round(rf_scores.mean(), 3), " SD:", round(rf_scores.std(), 2))

test_X = titanic[titanic.set == "test"].drop(["survived", "set", "PassengerId"], axis=1)

out = titanic[["PassengerId"]][titanic.set == "test"].copy()
out.loc[:,"Survived"] = rf.predict(test_X)
out = out.astype({"Survived":int})

out.to_csv("titanic/titanic_submission.csv", index = False)


#print(out.to_string())
#print(sum(titanic.set == "test"))
#print(len(mod.predict(test_X)))





#-----------------------------------------------------------------------------------------------------------------------
# split into X, Y, train and test
#-----------------------------------------------------------------------------------------------------------------------

#print(titanic.head())
#print(ohe.categories_[0][:8])
#print( numpy.delete(deck_trans.toarray().astype(int), 8, axis = 1) )


#print( type(titanic[["class"]]) )
#print( class_trans.toarray().astype(int) )
#print(ohe.categories_[0])



#print(titanic.head())


#print(type(numpy.array([2,3,1,0,2,7,8,2])))
#print(numpy.array([[2,3],[1,0],[2,7],[8,2]]) )

# one hot
#print(type(train))

#print(train["embarked"])

#enc = OneHotEncoder()
#transformed = enc.fit_transform(train[["embarked"]])
#print( transformed.toarray() )
#print(enc.categories_[0])
#print(onehot.get_feature_names())


#print(train.to_string())
#train_X = test.iloc[]
#train_Y
#test_X
#test_Y

#print(train)

#dtype = {
#    'survived': pandas.Int64Dtype(),
#    'class': int,
#    'sex': int,
#    'age': float,
#    'sib_sp': int,
#    'parch': int,
#    'fare': float,
#    'deck': str,
#    'cabin_number': str,
#    'embarked': str,
#    'set': str
#}


#-----------------------------------------------------------------------------------------------------------------------
# preprocess dataset
#-----------------------------------------------------------------------------------------------------------------------

# make array for titanic data: (include cols that don't need pre-processing)
#titanic = titanic[["survived", "sex", "set"]].copy()

# one hot encodeing - prefix
#ohe = OneHotEncoder()
#prefix_trans = ohe.fit_transform(titanic[["prefix"]])
#titanic[ohe.categories_[0]] = prefix_trans.toarray().astype(int)

# one hot encodeing - class
#ohe = OneHotEncoder()
#class_trans = ohe.fit_transform(titanic[["class"]])
#titanic[["class_1", "class_2", "class_3"]] = class_trans.toarray().astype(int)

# feature scaleing - age
#ss = StandardScaler()
#age_trans = ss.fit_transform(titanic[["age"]])
#titanic[["age"]] = age_trans

# feature scaleing - fare
#ss = StandardScaler()
#fare_trans = ss.fit_transform(titanic[["fare"]])
#titanic[["fare"]] = fare_trans

# one hot encodeing - age_bracket
#ohe = OneHotEncoder()
#ab_trans = ohe.fit_transform(titanic[["age_bracket"]])
#titanic[ohe.categories_[0]] = ab_trans.toarray().astype(int)

# one hot encodeing - fare_bracket
#ohe = OneHotEncoder()
#fare_trans = ohe.fit_transform(titanic[["fare_bracket"]])
#titanic[ohe.categories_[0]] = fare_trans.toarray().astype(int)

# one hot encodeing - deck
#ohe = OneHotEncoder()
#deck_trans = ohe.fit_transform(titanic[["deck"]])
#titanic["deck_"+ohe.categories_[0][:8]] = numpy.delete(deck_trans.toarray().astype(int), 8, axis = 1)

# one hot encodeing - embarked
#ohe = OneHotEncoder()
#embarked_trans = ohe.fit_transform(titanic[["embarked"]])
#titanic["port_"+ohe.categories_[0][:3]] = numpy.delete(embarked_trans.toarray().astype(int), 3, axis = 1)

#print(titanic.to_string())
