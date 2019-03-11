import numpy as np

# Neural network
#       o  Flower color
#   w1 / \  w2  b
#     o   o  length, width


# Each point is length, width, type (0, 1)
data = [[3, 1.5, 1],
        [2, 1, 0],
        [4, 1.5, 1],
        [3, 1, 0],
        [3.5, .5, 1],
        [2, .5, 0],
        [5.5, 1, 1],
        [1, 1, 0]]

mystery_flower = [3, 1.5]
# It's red

# Activation function


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Sigmoid prime

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Train

def train():
    # Random initial weights
    w1 = np.random.randn()
    w2 = np.random.randn()
    b = np.random.randn()

    iterations = 100000
    learning_rate = 0.1
    costs = []  # keep costs during training, see if they go down

    for i in range(iterations):
        # get a random point
        ri = np.random.randint(len(data))
        point = data[ri]

        z = point[0] * w1 + point[1] * w2 + b
        pred = sigmoid(z)  # Network prediction

        target = point[2]

        # Cost for current random point
        cost = np.square(pred - target)

        if i % 1000 == 0:
            c = 0
            for j in range(len(data)):
                p = data[j]
                p_pred = sigmoid(w1 * p[0] + w2 * p[1] + b)
                c += np.square(p_pred - p[2])
            costs.append(c)

        dcost_dpred = 2 * (pred - target)
        dpred_dz = d_sigmoid(z)

        dz_dw1 = point[0]
        dz_dw2 = point[1]
        dz_db = 1

        dcost_dz = dcost_dpred * dpred_dz

        dcost_dw1 = dcost_dz * dz_dw1
        dcost_dw2 = dcost_dz * dz_dw2
        dcost_db = dcost_dz * dz_db

        w1 = w1 - learning_rate * dcost_dw1
        w2 = w2 - learning_rate * dcost_dw2
        b = b - learning_rate * dcost_db

    return costs, w1, w2, b


costs, w1, w2, b = train()
print("----------------------------------")
print("Cost every 1000 iterations")
for i in range(len(costs)):
    print(costs[i])

print("----------------------------------")
print("w1 = " + str(w1))
print("w2 = " + str(w2))
print("b = " + str(b))
print("----------------------------------")

# Predict what the mystery flower is!

z = w1 * mystery_flower[0] + w2 * mystery_flower[1] + b
pred = sigmoid(z)

print("Close to 0 -> blue, close to 1 -> red")
print("Result = " + str(pred))
print("----------------------------------")
