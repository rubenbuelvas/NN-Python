import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def NN(m1, m2, w1, w2, b):
    z = (m1 * w1) + (m2 * w2) + b
    return sigmoid(z)

w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

# Weights and bias after 500k iterations
w1 = 14.580152208324947
w2 = 7.07037621405957
b = -52.21946241074214

data = [[3, 1.5, 1], [2, 1, 0], [4, 1.5, 1], [3, 1, 0], [3.5, 0.5, 1], [2, 0.5, 0], [5.5, 1, 1], [1, 1, 0]]

colors = ["blue", "red"]
guessed = 0

print("----------------------------------")
for i in range(len(data)):
    print("Flower #" + str(i+1))

    m1 = data[i][0]
    m2 = data[i][1]

    prediction = NN(m1, m2, w1, w2, b)
    predictionText = colors[int(np.round(prediction))]

    # print("Result = " + str(prediction))
    if np.round(prediction) == 0:
        howSure = (1 - prediction) * 100
    else:
        howSure = prediction * 100
    print("I think it's " + predictionText + ", I'm " + str(int(howSure)) +
          "% sure")
    print("And it's actually " + colors[data[i][2]])
    print("----------------------------------")
    if predictionText == colors[data[i][2]]:
        guessed += 1


print("Mystery flower")
# Default red (3, 1.5)
m1 = 3
m2 = 1.5

prediction = NN(m1, m2, w1, w2, b)
predictionText = colors[int(np.round(prediction))]

if np.round(prediction) == 0:
    howSure = (1 - prediction) * 100
else:
    howSure = prediction * 100
print("I think it's " + predictionText + ", I'm " + str(int(howSure)) +
      "% sure")
print("----------------------------------")
print("Success rate = " + str(guessed*100/len(data)) + "%")
print("----------------------------------")
