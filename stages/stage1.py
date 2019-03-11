import numpy


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def NN(m1, m2, w1, w2, b):
    z = (m1 * w1) + (m2 * w2) + b
    return sigmoid(z)


w1 = numpy.random.randn()
w2 = numpy.random.randn()
b = numpy.random.randn()

data = [[3, 1.5, 1], [2, 1, 0], [4, 1.5, 1], [3, 1, 0], [3.5, 0.5, 1], [2, 0.5, 0], [5.5, 1, 1], [1, 1, 0]]

colors = ["blue", "red"]

pos = numpy.random.randint(len(data))
randData = data[pos]

print("----------------------------------")
print("Flower #" + str(pos + 1))

m1 = randData[0]
m2 = randData[1]

prediction = NN(m1, m2, w1, w2, b)
predictionText = colors[int(numpy.round(prediction))]

print("Result = " + str(prediction))
print("----------------------------------")
if numpy.round(prediction) == 0:
    howSure = (1 - prediction) * 100
else:
    howSure = prediction * 100
print("I think it's " + predictionText + ", I'm " + str(int(howSure)) +
      "% sure")
print("And it's actually " + str(colors[randData[2]]))
print("----------------------------------")
