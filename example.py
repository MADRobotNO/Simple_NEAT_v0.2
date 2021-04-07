from NEAT import Neat
from RandomData import Xor

xor = Xor()
neat = Neat(2, 1, 10, 0.4)
for model in neat.list_of_all_models:
    model.mutate(neat.list_of_innovations)
# print(neat)
neat.fit(xor.data, xor.targets, 50)
print("done")
