import random
import math

class KNN:
    '''KNN is a class which can be used to train a KNN algorithm that saves the training data and and find the k number of nearest neighbours to predict the results on a test data.'''

    def __init__(self, k, filename, split, dm, w):
        self.data = self.load_data(filename)
        self.train_data = []
        self.test_data = []
        self.k = k
        self.split = split/100
        self.disstance_measure = dm
        self.weighted = w

    def load_data(self, filename):
        '''load the data into a list of lists'''
        
        data = []

        file = open(filename, 'r') #open the file for reading

        # read each line in the file, strip whitespace, split by comma, convert to float and append to data list
        for line in file:
            line = line.strip()
            row = line.split(',')

            try:
                float(row[0])
            except ValueError:
                continue    # skip header row

            # convert all values except the last one (the label) to float
            for i in range(len(row) - 1):
                row[i] = float(row[i])

            data.append(row)

        # remeber to close the file
        file.close() 

        return data

    def split_data(self):
        '''splitting the data into training and testing data, 2 different lists'''

        # create a copy of the data to avoid modifying the original list while selecting test samples
        data_copy = self.data.copy()

        # set test size to given percentage by user of the total data length
        test_size = int(len(self.data) * self.split)

        # take random samples from the data copy until we have enough for the test set, removing them from the copy as we go using pop.
        while len(self.test_data) < test_size:
            index = random.randint(0, len(data_copy)-1)
            self.test_data.append(data_copy.pop(index))
        
        # remaining data in the copy is used for training
        self.train_data = data_copy 

    def get_neighbors(self, test_point):
        '''find k number of nearest neighbors to test point'''

        
        # need a list to store all distances
        distances = []

        # find distance between test and train points and append it to the list as a tuple containing distance to the train point and its lable
        for train_point in self.train_data:

            if self.disstance_measure == 1:
                dist = euclidean_distance(test_point, train_point[:-1])
            elif self.disstance_measure == 2:
                dist = manhattan_distance(test_point, train_point[:-1])
            elif self.disstance_measure == 3:
                dist = minkowski_distance(test_point, train_point[:-1])


            distances.append((dist, train_point[-1]))  # (distance, label)

        # this sorts the list, distances, according to the key, which is first item of tuple, the distance 
        distances.sort(key=lambda x: x[0])

        # only keep first k muber of neighbors
        neighbors = distances[:self.k]

        return neighbors

    def predict(self, test_point):
        '''This is the heart of KNN. this function actually calles other function to get k nearest neighbours and decides where test point belongs to'''

        # fins the nieghbours
        neighbors = self.get_neighbors(test_point)

        # need a list to store how many times a lable is repeated in neighbors, simialr to voting. consider each item in neighbours is voting, to the lable they have
        votes = {}

        # cast the votes
        for i, (dist, label) in enumerate(neighbors):

            # neighbours list is already sorted, so 1st element is closest to test point , and last one is farthest, if k is 5, weight of 1st one is k-0 = 5 and so on till last one is k-4 = 1 . 
            weight = 1

            if self.weighted == 1:
                weight = self.k - i


            if label in votes:
                votes[label] += weight
            else:
                votes[label] = weight

        # find label with maximum votes which is winner and this lable is now assigned to test point, hence predicted
        prediction = max(votes, key=votes.get)

        return prediction

    def get_labels(self):
        '''retrive all the labels from data, not always necessary, but can be used'''

        # set is used so that repeating values automatically vanishes
        labels = set()

        # get the last item (label) from each row, set removes dulicate by itself, do not need to do that
        for row in self.data:
            labels.add(row[-1])

        return list(labels)

    def confusion_matrix(self):
        '''prints the confusion matrix'''

        # lebels we have in data
        labels = self.get_labels()

        # we will use a dictionary to show matrix.
        matrix = {}

        # we have lables as keys and values as dictionaries. which contains vlues to show in confusion matrix
        for actual in labels:
            matrix[actual] = {}
            for predicted in labels:
                matrix[actual][predicted] = 0

        for test_point in self.test_data:
            actual = test_point[-1]
            prediction = self.predict(test_point[:-1])

            matrix[actual][prediction] += 1

        return matrix

    def print_confusion_matrix(self, matrix):
        '''better to write seperate fuction to print confusion matrix properly'''

        print("\nConfusion Matrix:\n")

        labels = self.get_labels()

        print(f"{'':<8}", end="")
        for l in labels:
            print(f"{l:<8}", end="   ")
        print()
        print("-" * (8 + len(labels) * 12))

        for actual in labels:
            print(f"{actual:<8}", end="")
            for predicted in labels:
                print(f"{matrix[actual][predicted]:<8}", end="   ")
            print()


class KMeans:
    '''k-means is used when data is unlabeld, so we dont know how to clssify them, so we take few points (centriods) in data and cluster, closest points to them, aroud them. adustust centriods to average of their cluster and repeat the same process again and again untill we reach a point there is no more change in centeriods when we adjust them, or untill maximum allowed repetitions are done. '''

    def __init__(self, filename, dm, k=2, max_iterations=100 ):
        self.k = k
        self.max_iterations = max_iterations

        self.data = self.load_data(filename)
        self.centroids = []
        self.clusters = {}
        self.distance_measure = dm

    def load_data(self, filename):
        '''load all the data into a list of list'''

        data = []

        file = open(filename, "r")

        for line in file:
            line = line.strip()
            row = line.split(",")

            # convert all values to float (NO label now)
            row = [float(x) for x in row]

            data.append(row)

        file.close()

        return data

    def initialize_centroids(self):
        '''create k number of random points (centriods) to clusster data arounf them'''

        self.centroids = random.sample(self.data, self.k)

    def assign_clusters(self):
        '''measure the distances between all reamining points  and, centriods to find nearest cluster and assign the point to that cluster '''

        # Step 1: create empty clusters, named (indexed) as 0, 1, 2, ....
        for i in range(self.k):
            self.clusters[i] = []

        # Step 2: assign each point to nearest centroid
        for point in self.data:

            distances = []

            for centroid in self.centroids:
                if self.distance_measure == 1:
                    dist = euclidean_distance(point, centroid)
                elif self.distance_measure == 2:
                    dist = manhattan_distance(point, centroid)
                elif self.distance_measure == 3:
                    dist = minkowski_distance(point, centroid)

                distances.append(dist)

            # we are using actual index as cluster name, say if 3 clusters, they are 0, 1 and 2. for exaple if 2nd centriod has minimum distance, closet centeriod is 2nd, indexed 1. 
            closest_centroid = distances.index(min(distances))

            # append the point to related cluster, say 2nd cluster or clusterat index 1. 
            self.clusters[closest_centroid].append(point)

    def update_centroids(self):
        '''after each point is assinget to some clusters, we update the centriods accordingly (to the average point of all the cluster) and then again assign each point to newly formed clusters arounf that centerionds, and keep doing that for sef.max_iteratrions'''

        # need a new list for centeriods
        new_centroids = []

        # this loops retrieve each cluster one by one
        for i in range(self.k):
            cluster = self.clusters[i] # retriving clurster

            # if cluster is already empty , we cannot do anything to get new centriod, so we just assigne the old one to new centriod
            if len(cluster) == 0:
                new_centroids.append(self.centroids[i])
                continue

            # new_centroid = [0,0]
            new_centroid = [0] * len(cluster[0])

            # for each point in that cluster
            for point in cluster:
                
                # for each index in that point (remember point can be (x,y,z,...))
                for j in range(len(point)):
                    new_centroid[j] += point[j] #add each index number of that pont with same index of cetriod. x1 + x2 and y1 + y2 so on. for all points

            # again for each index in ceteriod, we have total of all points's that index
            for j in range(len(new_centroid)): 
                new_centroid[j] /= len(cluster) #for exaple after adding all points in line 221 ans 222, cluster points are say, (2,5,9) , we divide each of these indexes, by total legth of cluster. if length was to be 5, 2/5, 5/5, 9/5 ...so to get average. 

            # after each iteration , we get one more centeriod
            new_centroids.append(new_centroid)

        self.centroids = new_centroids

    def fit(self):
        '''here we run actuall k-means'''   

        # make centriods
        self.initialize_centroids()

        # do this for max iterations allowed, or untill centeriods stop to change on updation
        for i in range(self.max_iterations):
            
            # make clusters around centriods
            self.assign_clusters()

            # remeber the old centriods before updating so we can compare
            old_centroids = [centroid[:] for centroid in self.centroids]

            # update cteriods
            self.update_centroids()

            # compare centriods, and if they did not change, no need to work furtur, break the loop
            if old_centroids == self.centroids:
                break

        # when centriods have finaly converged we make a temprary file to run knn on it, as knn needs lables, we assign centriod/cluster number as lables
        file = open("temp.csv", "w")

        for cluster_id in self.clusters:
            for point in self.clusters[cluster_id]:

                # write features
                for value in point:
                    file.write(str(value) + ",")

                # write cluster number as label
                file.write("p" + str(cluster_id))

                file.write("\n")

        file.close() 

        temp_file = "temp.csv"
        return temp_file

    def average_distance(self, point, cluster):
        '''find average distance from a point to  a cluster (all points of some cluster), that is used in calculating silhouette score'''

        total = 0

        if len(cluster) <= 1:
            return 0
    
        for other_point in cluster:

            if point != other_point:
                total += self.euclidean_distance(point, other_point)

        return total / (len(cluster) - 1)            

    def silhouette_score(self):
        """Calculte silhouette score, just impelemnting the formula"""

        scores = []

        # go through each cluster
        for cluster_id in self.clusters:

            current_cluster = self.clusters[cluster_id]

            # go through each point in that cluster
            for point in current_cluster:

                # a = average distance to points in same cluster
                a = self.average_distance(point, current_cluster)

                # b = average distance to nearest other cluster
                b = float("inf")

                for other_cluster_id in self.clusters:

                    # skip its own cluster
                    if other_cluster_id != cluster_id:

                        other_cluster = self.clusters[other_cluster_id]

                        # ignore empty clusters
                        if len(other_cluster) == 0:
                            continue

                        distance_sum = 0

                        for other_point in other_cluster:
                            distance_sum += self.euclidean_distance(point, other_point)

                        average = distance_sum / len(other_cluster)

                        # keep smallest average distance
                        if average < b:
                            b = average

                # calculate silhouette score for this point
                if max(a, b) == 0:
                    s = 0
                else:
                    s = (b - a) / max(a, b)

                scores.append(s)

        # average score over all points
        if len(scores) == 0:
            return 0

        return sum(scores) / len(scores)

    def show_centroids(self):
        '''made a seperate function to show the centeriods'''

        for i, centroid in enumerate(self.centroids):
            print(f"Centroid {i}: {centroid}")

    def show_clusters(self):
        '''printing the full clusterd formed on screen'''

        for cluster_id in self.clusters:
            print(f"\nCluster {cluster_id}:")
            for point in self.clusters[cluster_id]:
                print(point)


def euclidean_distance(p1, p2):
        '''Measure euclidean distance between points, defined outside classes as there is no need to assing tem to one class, and also, this way we can use it for both classes'''

        distance = 0

        # simply appliyin euclidean distance formula
        for i in range(len(p1)):
            distance += (p1[i] - p2[i]) ** 2

        return math.sqrt(distance)

def manhattan_distance(p1, p2):
    '''Measure Manhattan distance between two points'''

    distance = 0

    for i in range(len(p1)):
        distance += abs(p1[i] - p2[i])

    return distance

def minkowski_distance(p1, p2, p=3):
    '''Measure Minkowski distance between two points'''

    distance = 0

    for i in range(len(p1)):
        distance += abs(p1[i] - p2[i]) ** p

    return distance ** (1 / p)

def run(filename, k , w, c, split, n):
    
    new_file = filename    

    if n == 0:
        print("=" * 15 + " Runnig K-Means " + "=" * 15)
        print("This file has no lables so runnig K-means first")
        m = int(input("Enter max number of iterations: "))

        km = KMeans(filename, c, k, m)
        new_file = km.fit()
        km.show_centroids()

        cl = input("Do you want to see clusters made? (y/n)")

        if cl == "y":
            km.show_clusters()

    
    print("=" * 15 + " Runnig KNN " + "=" * 15)
    knn = KNN(k, new_file, split, c, w)
    knn.split_data()
    matrix = knn.confusion_matrix()
    knn.print_confusion_matrix(matrix)

        






print("=" * 20 + " KNN & K-Means " + "=" * 20 )


filename = input("Enter filename or address of file you want to run KNN algorithm: "
)

k = int(input("Enter the value of K: "))

w = input("Do you want to use weighted neighbors? (y/n): ")

if w == "y":
    w = 1
elif w == "n":
    w = 0
else:
    print("Error! wrong choice")
    exit()

print ("Chose the distance measure: ")
print("1. Euclidean Distance \n 2. Manhatten Distance \n 3. Minkowski Distance")
c = int(input("Enter your choice : "))

if (c <= 0 or c>= 4):
    print("Error! wrong choice")
    exit()

split = int(input("Enter test data percentage: "))
if (split <= 0 or split >= 100):
    print("Percentage should be less tahn 100 and grater than 0")
    exit()


file = open(filename, 'r')
    
n = 0

with open(filename, 'r') as file:


    for i, line in enumerate(file):

        if (i >= 3):
            break

        row = line.strip().split(",")

        try:
            float(row[-1])
        except ValueError:
            if i > 0:
                n = 1
                break
            continue

run(filename, k , w, c, split, n)



