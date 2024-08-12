## Import Libraries
~~~
import csv
import math
~~~

-   **`import csv`**: Imports Python's CSV module to handle CSV file reading.
-   **`import math`**: Imports the math module to use mathematical functions like `sqrt` for distance calculations

##  Load the Dataset

~~~
def load_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            dataset.append([float(value) for value in row])
    return dataset

~~~

-   **`def load_csv(filename):`**: Defines a function `load_csv` to load data from a CSV file.
-   **`dataset = []`**: Initializes an empty list to store the dataset.
-   **`with open(filename, 'r') as file:`**: Opens the CSV file in read mode.
-   **`csv_reader = csv.reader(file)`**: Creates a CSV reader object to read the file.
-   **`next(csv_reader)`**: Skips the header row of the CSV file.
-   **`for row in csv_reader:`**: Iterates over each row in the CSV file.
-   **`dataset.append([float(value) for value in row])`**: Converts each value in the row to a float and appends it to the dataset list.
-   **`return dataset`**: Returns the loaded dataset.

## Min-Max Normalization

~~~
def normalize(dataset):
    minmax = [[min(column), max(column)] for column in zip(*dataset)]
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
~~~

-   **`def normalize(dataset):`**: Defines a function `normalize` to normalize dataset features.
-   **`minmax = [[min(column), max(column)] for column in zip(*dataset)]`**: Computes the min and max for each column (feature) in the dataset. `zip(*dataset)` transposes the dataset to iterate over columns.
-   **`for row in dataset:`**: Iterates over each row in the dataset.
-   **`for i in range(len(row)-1):`**: Iterates over each feature (excluding the last column, assumed to be the label).
-   **`row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])`**: Applies min-max normalization to each feature. It scales the feature value to a range between 0 and 1.

## Splitting the Data

~~~
def train_test_split(dataset, split_ratio):
    train_size = int(len(dataset) * split_ratio)
    train_set = dataset[:train_size]
    test_set = dataset[train_size:]
    return train_set, test_set

~~~

-   **`def train_test_split(dataset, split_ratio):`**: Defines a function `train_test_split` to split the dataset into training and testing sets.
-   **`train_size = int(len(dataset) * split_ratio)`**: Calculates the size of the training set based on the split ratio.
-   **`train_set = dataset[:train_size]`**: Slices the dataset to get the training set.
-   **`test_set = dataset[train_size:]`**: Slices the dataset to get the testing set.
-   **`return train_set, test_set`**: Returns the training and testing sets.

## Calculate Euclidean Distance

~~~
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)

~~~

-   **`def euclidean_distance(row1, row2):`**: Defines a function to compute the Euclidean distance between two data points.
-   **`distance = 0.0`**: Initializes the distance variable.
-   **`for i in range(len(row1)-1):`**: Iterates over each feature (excluding the last column).
-   **`distance += (row1[i] - row2[i]) ** 2`**: Adds the squared difference between corresponding feature values to the distance.
-   **`return math.sqrt(distance)`**: Returns the square root of the accumulated distance to get the Euclidean distance.

## Get Neighbors

~~~
def get_neighbors(train, test_row, num_neighbors):
    distances = []
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = []
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

~~~

-   **`def get_neighbors(train, test_row, num_neighbors):`**: Defines a function to find the nearest neighbors for a test row.
-   **`distances = []`**: Initializes a list to store distances between the test row and training rows.
-   **`for train_row in train:`**: Iterates over each row in the training set.
-   **`dist = euclidean_distance(test_row, train_row)`**: Calculates the distance between the test row and the training row.
-   **`distances.append((train_row, dist))`**: Appends the training row and its distance to the list.
-   **`distances.sort(key=lambda tup: tup[1])`**: Sorts the list by distance (ascending).
-   **`neighbors = []`**: Initializes a list to store the nearest neighbors.
-   **`for i in range(num_neighbors):`**: Iterates over the specified number of nearest neighbors.
-   **`neighbors.append(distances[i][0])`**: Appends the nearest neighbor rows to the list.
-   **`return neighbors`**: Returns the list of nearest neighbors.

## Predict Classification

~~~
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

~~~

-   **`def predict_classification(train, test_row, num_neighbors):`**: Defines a function to predict the classification of a test row based on its neighbors.
-   **`neighbors = get_neighbors(train, test_row, num_neighbors)`**: Gets the nearest neighbors of the test row.
-   **`output_values = [row[-1] for row in neighbors]`**: Extracts the labels of the nearest neighbors.
-   **`prediction = max(set(output_values), key=output_values.count)`**: Determines the most common label (mode) among the neighbors.
-   **`return prediction`**: Returns the predicted label.

## Evaluate Accuracy

~~~
def evaluate_accuracy(test_set, train_set, num_neighbors):
    correct = 0
    for row in test_set:
        prediction = predict_classification(train_set, row, num_neighbors)
        if row[-1] == prediction:
            correct += 1
    return correct / float(len(test_set)) * 100.0

~~~
-   **`def evaluate_accuracy(test_set, train_set, num_neighbors):`**: Defines a function to evaluate the accuracy of the model.
-   **`correct = 0`**: Initializes a counter for correct predictions.
-   **`for row in test_set:`**: Iterates over each row in the test set.
-   **`prediction = predict_classification(train_set, row, num_neighbors)`**: Predicts the label for the current test row.
-   **`if row[-1] == prediction:`**: Checks if the predicted label matches the actual label.
-   **`correct += 1`**: Increments the correct prediction counter if the labels match.
-   **`return correct / float(len(test_set)) * 100.0`**: Calculates and returns the accuracy as a percentage.

## Final Steps

~~~
num_neighbors = 5
accuracy = evaluate_accuracy(test_set, train_set, num_neighbors)
print(f'Model Accuracy: {accuracy:.2f}%')
~~~

-   **`num_neighbors = 5`**: Sets the number of neighbors to consider for the KNN algorithm.
-   **`accuracy = evaluate_accuracy(test_set, train_set, num_neighbors)`**: Evaluates the model accuracy using the specified number of neighbors.
-   **`print(f'Model Accuracy: {accuracy:.2f}%')`**: Prints the accuracy of the model formatted to two decimal places.