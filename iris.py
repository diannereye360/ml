# The Iris dataset is referred to as a “toy dataset” because it has only 150 samples and four features. 
# The dataset describes 50 samples for each of three Iris flower species—Iris setosa, Iris versicolor and Iris 
# virginica. Each sample’s features are the sepal length, sepal width, petal 
# length and petal width, all measured in centimeters. The sepals are the larger outer parts of each flower 
# that protect the smaller inside petals before the flower buds bloom.

#EXERCISE
# load the iris dataset and use classification
# to see if the expected and predicted species
# match up

# display the shape of the data, target and target_names

# display the first 10 predicted and expected results using
# the species names not the number (using target_names)

# display the values that the model got wrong

# visualize the data using the confusion matrix


from sklearn.datasets import load_iris

iris = load_iris()
# print(iris.DESCR)

#print(iris.data[13])
#print(iris.target[13])

print(iris.data.shape)
print(iris.target.shape)
print(iris.target_names)

import matplotlib.pyplot as plt

#figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(6, 4))

#for item in zip(axes.ravel(), iris.images, iris.target):
 #   axes, image, target = item
  #  axes.imshow(image, cmap=plt.cm.gray_r)
   # axes.set_xticks([])
    #axes.set_yticks([])
    #axes.set_title(target)

#plt.tight_layout()
#plt.show()

from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    iris.data, iris.target, random_state=11
)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X=data_train, y=target_train)

predicted = knn.predict(X=data_test)
expected = target_test

print(iris.target_names[predicted[:10]])
print(iris.target_names[expected[:10]])

wrong = [([iris.target_names[p], iris.target_names[e]]) for (p, e) in zip(predicted, expected) if p != e]
print(wrong)

print(f"score: {knn.score(data_test, target_test):.2%}")

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_true=expected, y_pred=predicted)

print (confusion)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt2

confusion_df = pd.DataFrame(confusion, 
    index=iris.target_names,
    columns=iris.target_names)
figure = plt2.figure(figsize=(7,6))
axes = sns.heatmap(confusion_df, annot=True, cmap=plt2.cm.nipy_spectral_r)
plt2.xlabel("Expected")
plt2.ylabel("Predicted")
plt2.show()