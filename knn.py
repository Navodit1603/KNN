import csv
import math

# Load CSV file into a dataset
def load_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            dataset.append([float(value) for value in row])
    return dataset

# Normalize the dataset
def normalize(dataset):
    minmax = [[min(column), max(column)] for column in zip(*dataset)]
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Split dataset into training and test sets
def train_test_split(dataset, split_ratio):
    train_size = int(len(dataset) * split_ratio)
    train_set = dataset[:train_size]
    test_set = dataset[train_size:]
    return train_set, test_set

# Calculate Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)

# Locate the most similar neighbors
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

# Make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

# Evaluate the accuracy of predictions
def evaluate_accuracy(test_set, train_set, num_neighbors):
    correct = 0
    for row in test_set:
        prediction = predict_classification(train_set, row, num_neighbors)
        if row[-1] == prediction:
            correct += 1
    return correct / float(len(test_set)) * 100.0

# Main function to load the dataset and execute the KNN algorithm
filename = 'smallcc.csv'  # Replace with your file name
split_ratio = 0.8
num_neighbors = 5

# Load and prepare data
dataset = load_csv(filename)
normalize(dataset)
train_set, test_set = train_test_split(dataset, split_ratio)

# Evaluate accuracy
accuracy = evaluate_accuracy(test_set, train_set, num_neighbors)
print(f'Model Accuracy: {accuracy:.2f}%')
