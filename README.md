# K-Nearest Neighbors (KNN) and K-Means Clustering from Scratch in Python

## Overview

This project implements two fundamental machine learning algorithms completely **from scratch using pure Python**, without relying on libraries such as NumPy, Pandas, or Scikit-Learn.

The purpose of this project is not simply to obtain predictions, but to understand the internal working of these algorithms and implement every step manually.

Implemented algorithms:

- **K-Nearest Neighbors (KNN)** for classification of labeled data.
- **K-Means Clustering** for grouping unlabeled data.
- Multiple distance measures.
- Weighted and unweighted voting.
- Random train-test splitting.
- Confusion matrix generation.
- Silhouette score calculation.

# Machine Learning Background

Machine learning algorithms are generally divided into two major categories:

| Supervised Learning  | Unsupervised Learning     |
| -------------------- | ------------------------- |
| Uses labeled data    | Uses unlabeled data       |
| Learns from examples | Discovers hidden patterns |
| Performs prediction  | Performs clustering       |
| KNN                  | K-Means                   |

# Project Objectives

The objective of this project was to implement both algorithms manually and understand:

- How distances are calculated.
- How neighbors are selected.
- How voting determines predictions.
- How clusters are formed.
- How centroids move during iterations.
- How convergence occurs.
- How clustering quality is measured.

Everything was implemented manually without using machine learning libraries.

# Distance Measures

Both KNN and K-Means rely heavily on distance calculations.

## 1\. Euclidean Distance

Euclidean distance represents the straight-line distance between two points.

Example:

Point A:

(1,2)

Point B:

(4,6)

Distance:

√\[(4−1)² + (6−2)²\]  
<br/>\= √(9 + 16)  
<br/>\= 5

This is similar to measuring distance using a ruler.

## 2\. Manhattan Distance

Manhattan distance measures movement along horizontal and vertical directions.

Example:

|4−1| + |6−2|  
<br/>\= 3 + 4  
<br/>\= 7

It is called Manhattan distance because movement resembles travelling through city streets arranged in a grid.

## 3\. Minkowski Distance

Minkowski distance is a general form of distance.

p = 1 → Manhattan Distance  
<br/>p = 2 → Euclidean Distance  
<br/>p = 3 → Higher-order Minkowski Distance

In this project, Minkowski distance with:

p = 3

has been implemented.

# K-Nearest Neighbors (KNN)

KNN is a supervised learning algorithm used for classification.

Unlike many machine learning algorithms, KNN performs almost no training. It simply stores the training data and uses it when predictions are required.

## How KNN Works

Suppose we have:

K = 3

Test point:

(4,5)

Training points:

(3,4) → Red  
(5,5) → Red  
(7,8) → Blue

Distances:

1.41  
1  
4.24

After sorting:

1  
1.41  
4.24

The three nearest neighbors become:

Red  
Red  
Blue

Votes:

Red = 2  
<br/>Blue = 1

Prediction:

Red

# Weighted KNN

In ordinary KNN, every neighbor contributes equally.

However, closer neighbors should usually have greater influence.

Example:

| Neighbor | Weight |
| -------- | ------ |
| Closest  | 5      |
| Second   | 4      |
| Third    | 3      |
| Fourth   | 2      |
| Fifth    | 1      |

Suppose:

Red  
Red  
Blue  
Blue  
Blue

Without weighting:

Blue wins (3 votes)

With weighting:

Red = 5 + 4 = 9  
<br/>Blue = 3 + 2 + 1 = 6

Prediction:

Red

This project supports both weighted and unweighted neighbors.

# Train-Test Split

The dataset is divided into:

- Training data
- Testing data

Examples:

70-30 split  
80-20 split  
60-40 split

Testing samples are selected randomly.

This prevents evaluating the model on data it has already seen.

# Confusion Matrix

A confusion matrix compares predictions with actual labels.

Example:

| Actual Predicted | Cat | Dog |
| ---------------- | --- | --- |
| Cat              | 10  | 2   |
| Dog              | 1   | 8   |

This means:

- 10 cats were correctly classified.
- 8 dogs were correctly classified.
- 2 cats were wrongly classified as dogs.
- 1 dog was wrongly classified as a cat.

Confusion matrices help visualize the strengths and weaknesses of a classifier.

# K-Means Clustering

K-Means is an unsupervised learning algorithm.

Unlike KNN, K-Means works with unlabeled data.

Its purpose is to discover natural groups present in the dataset.

# Initial Centroids

K random points are selected as centroids.

Example:

Centroid 1 = (2,3)  
<br/>Centroid 2 = (8,7)

These centroids act as the centers of clusters.

# Cluster Assignment

Every point calculates its distance to each centroid.

The point joins the cluster whose centroid is closest.

# Updating Centroids

After clusters are formed, the centroid is moved to the average location of all points inside that cluster.

Suppose a cluster contains:

(1,2)  
<br/>(2,4)  
<br/>(3,6)

Average coordinates:

x = (1+2+3)/3 = 2  
<br/>y = (2+4+6)/3 = 4

New centroid:

(2,4)

# Iterative Process

K-Means repeatedly performs:

Assign points  
<br/>↓  
<br/>Update centroids  
<br/>↓  
<br/>Assign points again  
<br/>↓  
<br/>Update centroids again

until convergence occurs.

# Convergence

Eventually, centroids stop changing.

When the old centroids and new centroids become identical, the algorithm has converged.

At this stage, further iterations produce no change.

# Empty Clusters

Sometimes a cluster may contain no points.

In that situation, calculating an average becomes impossible.

Therefore, the previous centroid is retained.

This prevents errors and keeps the algorithm stable.

# Silhouette Score

The silhouette score evaluates clustering quality.

For every point:

### a

Average distance to points inside the same cluster.

### b

Average distance to points in the nearest neighboring cluster.

Silhouette score:

s = (b − a) / max(a,b)

Interpretation:

| Score    | Meaning              |
| -------- | -------------------- |
| Near +1  | Excellent clustering |
| Near 0   | Overlapping clusters |
| Negative | Poor clustering      |

A larger silhouette score indicates better separation between clusters.

# Algorithm Complexity

## KNN

Training:

O(1)

Prediction:

O(n)

Memory:

O(n)

where:

n = number of training samples

## K-Means

Time complexity:

O(n × k × iterations)

where:

- n = number of data points
- k = number of clusters
- iterations = maximum number of repetitions

# Design Decisions

Several design choices were made during implementation.

### Distance Functions Outside Classes

Distance functions were implemented separately because both KNN and K-Means require them.

This improves reusability.

### Dictionaries for Voting

Votes are stored inside dictionaries.

This makes counting labels simple and efficient.

### Dictionaries for Clusters

Clusters are represented using dictionaries.

Cluster numbers serve as keys.

### Handling Empty Clusters

If a cluster becomes empty, the old centroid is preserved.

This avoids division by zero and maintains algorithm stability.

# Pipeline Used in This Project

For unlabeled datasets:

Raw Data  
<br/>↓  
<br/>K-Means Clustering  
<br/>↓  
<br/>Assign Cluster IDs  
<br/>↓  
<br/>Temporary Labeled Dataset  
<br/>↓  
<br/>KNN Classification  
<br/>↓  
<br/>Confusion Matrix

This demonstrates how unsupervised and supervised learning can be combined.

# Features Implemented

### KNN

✔ Random train-test split

✔ Adjustable value of K

✔ Euclidean distance

✔ Manhattan distance

✔ Minkowski distance

✔ Weighted neighbors

✔ Unweighted neighbors

✔ Prediction mechanism

✔ Confusion matrix generation

### K-Means

✔ Random centroid initialization

✔ Adjustable number of clusters

✔ Adjustable maximum iterations

✔ Multiple distance measures

✔ Centroid updates

✔ Convergence detection

✔ Empty cluster handling

✔ Silhouette score calculation

# Possible Improvements

Future enhancements may include:

- K-Means++ initialization.
- KD-Trees for faster KNN.
- Precision, Recall and F1-score.
- CSV module for file handling.
- Data visualization using Matplotlib.
- Additional distance measures.
- Graphical user interface.
- Support for higher-dimensional datasets.

# Conclusion

This project demonstrates the internal workings of two important machine learning algorithms by implementing them entirely from scratch.

Rather than relying on existing libraries, every major component-including distance calculations, voting, centroid updates, confusion matrices, and silhouette scores-was developed manually. This provides a deeper understanding of both supervised and unsupervised learning and highlights the mathematical principles that govern these algorithms.

Understanding how these methods work internally is often more valuable than simply using ready-made implementations, since it builds intuition that can later be applied to more advanced machine learning techniques.
