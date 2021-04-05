from NEAT import Neat
from RandomData import Xor

xor = Xor()
neat = Neat(2, 1, 1, 0.1)
neat.fit(xor.data, xor.targets, 1)
