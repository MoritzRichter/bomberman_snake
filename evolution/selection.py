import random
import math

def powerSelection(population, selectionPower) :
    index = math.floor((random.random() ** selectionPower) * len(population))
    return population[index]