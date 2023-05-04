#------------------------------------------------- svm for titanic (kaggle) --------------------------------------------
import pandas
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import metrics
import numpy

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
processed = titanic[["survived", "sex", "married", "with_spouse", "with_dependants", "socialite", "set"]].copy()

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
# split into X, Y and fit model
#-----------------------------------------------------------------------------------------------------------------------
train_X = processed[processed.set == "train"].drop(["survived", "set"], axis = 1)
train_Y = processed[processed.set == "train"]["survived"].copy()

mod = svm.SVC(kernel = 'poly', gamma = 1, C = 1)
mod.fit(train_X, train_Y)
pred = mod.predict(train_X)

print("Accuracy:",metrics.accuracy_score(train_Y, pred))

#-----------------------------------------------------------------------------------------------------------------------
# run model against test data and save output against passenger id
#-----------------------------------------------------------------------------------------------------------------------
test_X = processed[processed.set == "test"].drop(["survived", "set"], axis = 1)
print(type(titanic))

out = titanic[["PassengerId"]][titanic.set == "test"].copy()
out.loc[:,"Survived"] = mod.predict(test_X)

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