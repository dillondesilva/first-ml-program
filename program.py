from sklearn import tree

# features is the training data which is
# a list of lists containing the weight and then
# whether it is bumpy. 1 is bumpy and 0 is smooth
features = [[140,1],[130,1],[150,0],[170,0]]

# labels corresponds to features training data.
# a 0 is an apple and a 1 is an orange
labels = [0,0,1,1]

# Making a decision tree classifier for our machine learning
clf = tree.DecisionTreeClassifier()
# Using the classifier with our training data
clf = clf.fit(features, labels)

print(clf.predict([[150,0]]))