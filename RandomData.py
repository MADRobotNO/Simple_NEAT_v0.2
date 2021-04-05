import random


class Xor:
    def __init__(self):
        self.data = [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
        self.targets = [[1.0], [1.0], [0.0], [0.0]]

    def getRandomXorData(self):
        random_number = random.randint(0, 4)
        return self.data[random_number]
