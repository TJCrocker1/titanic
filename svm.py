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

#import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------------------------------------------------
# load data
#-----------------------------------------------------------------------------------------------------------------------
titanic = pandas.read_csv("titanic/titanic_clean.csv")

#[print(col) for col in titanic.columns]
#print( titanic.to_string() )
#-----------------------------------------------------------------------------------------------------------------------
# preprocess dataset
#-----------------------------------------------------------------------------------------------------------------------

# make array for processed data: (include cols that don't need pre-processing)
processed = titanic[["survived", "sex", "set"]].copy()

# one hot encodeing - prefix
ohe = OneHotEncoder()
prefix_trans = ohe.fit_transform(titanic[["prefix"]])
processed[ohe.categories_[0]] = prefix_trans.toarray().astype(int)

# one hot encodeing - class
ohe = OneHotEncoder()
class_trans = ohe.fit_transform(titanic[["class"]])
processed[["class_1", "class_2", "class_3"]] = class_trans.toarray().astype(int)

# feature scaleing - age
#ss = StandardScaler()
#age_trans = ss.fit_transform(titanic[["age"]])
#processed[["age"]] = age_trans

# feature scaleing - fare
#ss = StandardScaler()
#fare_trans = ss.fit_transform(titanic[["fare"]])
#processed[["fare"]] = fare_trans

# one hot encodeing - age_bracket
ohe = OneHotEncoder()
ab_trans = ohe.fit_transform(titanic[["age_bracket"]])
processed[ohe.categories_[0]] = ab_trans.toarray().astype(int)

# one hot encodeing - fare_bracket
ohe = OneHotEncoder()
fare_trans = ohe.fit_transform(titanic[["fare_bracket"]])
processed[ohe.categories_[0]] = fare_trans.toarray().astype(int)

# one hot encodeing - deck
ohe = OneHotEncoder()
deck_trans = ohe.fit_transform(titanic[["deck"]])
processed["deck_"+ohe.categories_[0][:8]] = numpy.delete(deck_trans.toarray().astype(int), 8, axis = 1)

# one hot encodeing - embarked
ohe = OneHotEncoder()
embarked_trans = ohe.fit_transform(titanic[["embarked"]])
processed["port_"+ohe.categories_[0][:3]] = numpy.delete(embarked_trans.toarray().astype(int), 3, axis = 1)

#print(processed.to_string())

#-----------------------------------------------------------------------------------------------------------------------
# split into X, Y and fit models
#-----------------------------------------------------------------------------------------------------------------------
X_train = processed[processed.set == "train"].drop(["survived", "set"], axis = 1)
Y_train = processed[processed.set == "train"]["survived"].copy()


# support vector machine:
svm = SVC(kernel = 'poly')
svm.fit(X_train, Y_train)
acc_svm = round(svm.score(X_train, Y_train)*100, 2)
#print("Accuracy:", svm.score(X_train, Y_train))

# Stocastic Gradient Decent:
#sgd = linear_model.SGDClassifier(max_iter=5, tol = None)
#sgd.fit(X_train, Y_train)
#sgd.score(X_train, Y_train)

#acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

# Random Forest:
rf = RandomForestClassifier(n_estimators=1000)
rf.fit(X_train, Y_train)
rf.score(X_train, Y_train)

#acc_rf = round(rf.score(X_train, Y_train)*100, 2)

# Logistic regression:
#logreg = LogisticRegression()
#logreg.fit(X_train, Y_train)

#acc_logreg = round(logreg.score(X_train, Y_train)*100, 2)

# K Nearest Neighbor:
#knn = KNeighborsClassifier(n_neighbors=3)
#knn.fit(X_train,Y_train)

#acc_knn = round(knn.score(X_train, Y_train)*100, 2)

# Gaussian Naive Bayes:
#gnb = GaussianNB()
#gnb.fit(X_train, Y_train)
#acc_gnb = round(gnb.score(X_train, Y_train)*100, 2)

# Perceptron
#prcy = Perceptron(max_iter=5)
#prcy.fit(X_train, Y_train)

#acc_prcy = round(prcy.score(X_train, Y_train)*100, 2)

# linear svm:
#lsvm = LinearSVC()
#lsvm.fit(X_train, Y_train)

#acc_lsvm = round(lsvm.score(X_train, Y_train)*100, 2)

# Decision Tree
#dtree = DecisionTreeClassifier()
#dtree.fit(X_train, Y_train)

#acc_dtree = round(dtree.score(X_train, Y_train)*100, 2)

#-----------------------------------------------------------------------------------------------------------------------
# print results:
#-----------------------------------------------------------------------------------------------------------------------
#results = pandas.DataFrame({
#    'Model': ["SVM", "Stockastic Gradient Decent", "Random Forest",
#              "Logistic Regression", "K Nearest Neighbors", "Gaussian Naive Bays",
#              "Perceptron", "Linear SVM", "Decision Tree"],
#    "Score": [acc_svm, acc_sgd, acc_rf,
#              acc_logreg, acc_knn, acc_gnb,
#              acc_prcy, acc_lsvm, acc_dtree]
#}).sort_values(by = 'Score', ascending=False)

#print(results.to_string())

#-----------------------------------------------------------------------------------------------------------------------
# k-folds cross validation
#-----------------------------------------------------------------------------------------------------------------------
#from sklearn.model_selection import cross_val_score
#rf_scores = cross_val_score(rf, X_train, Y_train, cv = 10, scoring="accuracy")
#print(rf_scores)
#print("Mean:", rf_scores.mean(), " SD:", rf_scores.std())

#svm_scores = cross_val_score(rf, X_train, Y_train, cv = 10, scoring="accuracy")
#print(svm_scores)
#print("Mean:", svm_scores.mean(), " SD:", svm_scores.std())


rel_importance = pd.DataFrame({
    'feature':X_train.columns,
    'importance':numpy.round(rf.feature_importances_, 3)
}).sort_values('importance', ascending=False)

print(rel_importance.to_string())

#-----------------------------------------------------------------------------------------------------------------------
# try random forrest with fewer params
#-----------------------------------------------------------------------------------------------------------------------
X_train_red = X_train.drop(["Fare_high", "Fare_highest", "Ms", "deck_T", "deck_F", "deck_G"], axis=1)

rf_red = RandomForestClassifier(n_estimators=100, oob_score=True)
rf_red.fit(X_train_red, Y_train)
acc_rf_red = round(rf_red.score(X_train_red, Y_train)*100, 2)
print(acc_rf_red)
print('oob score:', round(rf_red.oob_score_, 4)*100)

#-----------------------------------------------------------------------------------------------------------------------
# tune the hyper parameters:
#-----------------------------------------------------------------------------------------------------------------------
#from sklearn.model_selection import GridSearchCV, cross_val_score
#param_grid = {"criterion":["gini", "entropy"],
#              "min_samples_leaf":[1,5,10,25,50,70],
#              "min_samples_split":[2,4,10,12,16,18,25,35],
#              "n_estimators":[100, 400, 700, 1000, 1500]}

#rf_clf = RandomForestClassifier(n_estimators=100, max_features='auto', oob_score=True,
#                                random_state=1, n_jobs=-1)

#clf = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1)
#clf.fit((X_train, Y_train))
#print(clf.best_params_)
#-----------------------------------------------------------------------------------------------------------------------
# run model against test data and save output against passenger id
#-----------------------------------------------------------------------------------------------------------------------
test_X = processed[processed.set == "test"].drop(["survived", "set", "Fare_high",
                                                  "Fare_highest", "Ms", "deck_T",
                                                  "deck_F", "deck_G"], axis = 1)
#print(type(titanic))

out = titanic[["PassengerId"]][titanic.set == "test"].copy()
out.loc[:,"Survived"] = rf_red.predict(test_X)

out.to_csv("titanic/titanic_submission.csv", index = False)


#print(out.to_string())
#print(sum(titanic.set == "test"))
#print(len(mod.predict(test_X)))





#-----------------------------------------------------------------------------------------------------------------------
# split into X, Y, train and test
#-----------------------------------------------------------------------------------------------------------------------

#print(processed.head())
#print(ohe.categories_[0][:8])
#print( numpy.delete(deck_trans.toarray().astype(int), 8, axis = 1) )


#print( type(titanic[["class"]]) )
#print( class_trans.toarray().astype(int) )
#print(ohe.categories_[0])



#print(processed.head())


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