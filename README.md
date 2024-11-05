# Importing Dependencies
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
iris = pd.read_csv('/content/Iris.csv')

print(iris.head())
print(iris.describe())
print("Target Labels", iris["Species"].unique())
import plotly.express as px
fig = px.scatter(iris, x="SepalWidthCm", y="SepalLengthCm", color="Species") # Change x and y to actual column names
fig.show()
x = iris.drop("Species", axis=1) # Changed "species" to "Species"
y = iris["Species"] # Changed "species" to "Species"
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2,
                                                    random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
# Original:
# x_new = np.array([[5, 2.9, 1, 0.2]])

# Modified:
# Assuming the missing feature is 'Id' which often might be the first feature
# and needs to be included for consistency.
# If a different feature is missing, replace 'Id_value' with the actual value
# and adjust the position accordingly.

Id_value = 1  # Replace with the actual Id value if known
x_new = np.array([[Id_value, 5, 2.9, 1, 0.2]])

# Alternatively, if the 'Id' is not needed and your original data
# was intended to have 4 features, consider re-training the model.

# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=1)
# x = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
# y = iris["Species"]
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y,
#                                                     test_size=0.2,
#                                                     random_state=0)

# knn.fit(x_train, y_train)
# prediction = knn.predict(x_new) # Now with 4 features data to the model
# print("Prediction: {}".format(prediction))

prediction = knn.predict(x_new)
print("Prediction: {}".format(prediction))

