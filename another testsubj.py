

from typing import List, Callable, Tuple
from random import randint, randrange, random
from pyamaze import maze, agent, COLOR
import matplotlib.pyplot as plt

# this code is supposed to solve a n by n maze that we will create using pyamaze package from python. the diea is to generate a population, evaluate fitness parameter andsort the population in such a manner that the highest fitness values will be the first few enteries of the population. Once we got the sorted population we will use elitism and take the first two enteries of population then do crossover mutation and use these two in the next generation the ngenerate the rest of the enteries it will keep repeating this till max generation number is reached and a solution is found. The solution  needs to have fitness>250 and infeasible steps = 0. it will then print the path of the solution. reverse the solution list and then display it on the maze.
n = 10
Num_col = n
Num_rows = n

m = maze() #change to n by n and take n from user
m.CreateMaze(Num_col, Num_col,loopPercent = 100, theme = COLOR.dark)

POPULATION_SIZE = 50

Genome = List[int]
Population = list(Genome)
pathlength = []
paths = []
numberofturns = []
infsteps = []
fitness = []
sol = []
b_path = []


# generating different posibilties for different genomes == Popsize
def generate_genome(num_col: int) -> Genome:
    genome = [randint(2, num_col-2) for _ in range(num_col-2)]
    genome.insert(0, 1)
    genome.append(num_col-1)
    while len(genome) < num_col:
        genome.append(randint(2, num_col-2))
    return genome

# all the different types of combinations of genome that will be stored in the population == Pop size
def generate_population(size: POPULATION_SIZE, num_col: Num_col)->Population:
    return[generate_genome(num_col) for _ in range(size)]


# for fitness function just use the old implementation of fitness
# fitness function:
def fit(Population, POPULATION_SIZE):
    global infsteps
    global paths
    global numberofturns
    global fitness
    global pathlength

    # check that all individuals have the correct length
    assert all(len(individual) == Num_col for individual in Population), "Not all individuals have the correct length"
    k = 1 # k is controlling the column 
    for idx in range(POPULATION_SIZE):
        path = []
        infsteps= [] # empty list to store current paths
        for i in range(Num_col-1):
            # print(f"idx={idx}, i={i}, Num_col={Num_col}, len(Population[idx])={len(Population[idx])}")
            if i < Num_col-2 and i < len(Population[idx])-1 and Population[idx][i] - Population[idx][i+1] > 0:
                for j in range(Population[idx][i], Population[idx][i+1]-1, -1):
                    # print(f"({j},{k}), ", end="")
                    path.append((j,k))
                k += 1 
            elif i < Num_col-2 and i < len(Population[idx])-1 and Population[idx][i] - Population[idx][i+1] < 0:
                for j in range(Population[idx][i], Population[idx][i+1]+1):
                    # print(f"({j},{k}), ", end="")
                    path.append((j,k))
                k += 1   
            else:
                    # print(f"({Population[idx][i]},{k}), ", end="")
                    path.append((Population[idx][i],k))
                    k += 1
        paths.append(path)  
        k = 1
    pathlength = [len(path) for path in paths]
    minlength = min(pathlength)
    maxlength = max(pathlength)
    flength = [ 1 - ((pathlength[i] - minlength) / (maxlength - minlength)) for i in range(POPULATION_SIZE)]
    for path in paths:
        infeasible_steps = 0
        for i in range(len(path)-1):
            if m.maze_map[path[i]]['E'] == 0 and path[i][1] - path[i+1][1] == 1:
                infeasible_steps += 1
            if m.maze_map[path[i]]['W'] == 0 and path[i][1] - path[i-1][1] == 1:
                infeasible_steps += 1
            if m.maze_map[path[i]]['N'] == 0 and path[i][0] - path[i-1][0] == 1:
                infeasible_steps += 1
            if m.maze_map[path[i]]['S'] == 0 and path[i][0] - path[i+1][0] == 1:
                infeasible_steps += 1
        infsteps.append(infeasible_steps) 
    maxinf = max(infsteps)
    mininf = min(infsteps)
    finf = [1 - ((infsteps[i] - mininf) / (maxinf - mininf)) for i in range(POPULATION_SIZE)]
    for i in range(POPULATION_SIZE):
        turns = 0
        for j in range(Num_col):
            if j < Num_col - 1 and Population[i][j] != Population[i][j+1]:
                turns += 1
        numberofturns.append(turns)
    maxturn = 0
    minturn = 0
    if numberofturns:
        maxturn = max(numberofturns)
        minturn = min(numberofturns) 
        if maxturn == minturn:
            maxturn += 1 #fixing zero division error
    fturns = [1 - ((numberofturns[i] - minturn) / (maxturn - minturn)) for i in range(POPULATION_SIZE)]       
    fitness = [100*3*finf[i]*((2*flength[i]+2*fturns[i])/4) for i in range(POPULATION_SIZE)]
    pop_fitness = zip(Population, fitness)
    pop_fitness_sorted = sorted(pop_fitness, key=lambda x: x[1], reverse=True)
    Population, fitness = zip(*pop_fitness_sorted)
    Population = list(Population)
    fitness = list(fitness)

    return Population, fitness

def selection_pair(population, fitness) -> Tuple[Tuple[int], Tuple[int]]:
    population_fitness = [(population[i], fitness[i]) for i in range(len(population))] 
    # sorted by descending order
    sorted_population = sorted(population_fitness, key=lambda x: x[1], reverse=True)
    parent1 = sorted_population[0][0]
    parent2 = sorted_population[1][0] # kind of an extra step could return directly
    return parent1, parent2
def single_point_crossover( a:Genome, b:Genome) -> Tuple[Genome, Genome]:
    if len(a) != len(b):
        raise ValueError('Genome a and b must be equal')
    length = len(a)
    if length< 2:
        return a, b
    p = randint(1, length-1)
    return a[0:p]+b[p:], b[0:p]+a[p:]
def mutation(genome: Genome, num: int =1, probabiltiy: float =0.5)-> Genome:
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random()> probabiltiy else abs(genome[index]-1)
        return genome
def run_evolution(Genome, population, num_col, generation_limit, fitness, infsteps, POPULATION_SIZE, solution) -> Tuple[Population,int]:

    def is_solution(genome):
        # if infsteps[genome] == 0 and fitness(genome) >= 290:
        if infsteps[population.index(genome)] == 0 and fitness(population.index(genome)) >= 250:

            return True
        return False
    Genome = generate_genome(Num_col)
    population = generate_population(POPULATION_SIZE, Num_col)
    fit(Population, POPULATION_SIZE)
    for i in range(generation_limit):
        # population = sorted(population, fitness, reverse=True)
        # population = sorted( population, key = lambda x:fitness[x], reverse=True)
        
        solution= []
        if is_solution(population[0]):
            solution.append(Population[0])
            break

        next_generation = population[:2]  # elitism where we just keep the top two pairs for next generation

        for j in range(int(len(population) / 2) - 1):  # saves 1 loop because we have saved 2 parents from past generation
            parents = selection_pair(population, fitness)
            offspring_a, offspring_b = single_point_crossover(parents[0], parents[1])
            offspring_a = mutation(offspring_a)
            offspring_b = mutation(offspring_b)
            next_generation += [offspring_a, offspring_b]

        population = next_generation
    return population, i+1 if is_solution(population[0]) else "sol not found"

def path_from_genome(solu, Num_col):
    path = []
    k = 1
    for i in range(Num_col - 1):
        if solu[i] < solu[i+1]:
            for j in range(solu[i], solu[i+1]+1):
                path.append((j, k))
            k += 1
        else:
            for j in range(solu[i], solu[i+1]-1, -1):
                path.append((j, k))
            k += 1
    return path
# last piece
gen = 50
maxgen = 100
Genome = generate_genome(Num_col)
Population = generate_population(POPULATION_SIZE, Num_col)
# print(Population)
best_fitness_values = []
generations = []

fit(Population, POPULATION_SIZE)
print()
# print(paths)
# print(fitness)
pair1, pair2 = selection_pair(Population, fitness)
# print(pair1, pair2)
cross1, cross2 = single_point_crossover(pair1, pair2)
# print(cross1, cross2)
child = mutation(Genome, 1, 0.5)
# print(child)
print("Initial fitness scores:", fitness)
print()
print("Searching for solution...")
print("Generation\tBest Fitness Score")
solu = []
for i in range(1, maxgen+1):
    population, generation = run_evolution(Genome, Population, Num_col, gen, fitness, infsteps, POPULATION_SIZE, sol)
    print(f"{generation}\t\t{max(fitness)}") 
    best_fitness =max(fitness)
    best_fitness_values.append(best_fitness)
    generations.append(generation)

    if population and infsteps[population.index(population[0])] == 0 and fitness(population.index(population[0])) >= 250:
        solu.append(population[0])
        break
    else:
        Population = generate_population(POPULATION_SIZE, Num_col)
        fit(Population, POPULATION_SIZE)

print('solution',solu)

if solu:
    solution_list = path_from_genome(solu[0], Num_col)
    print("Solution found: ", solution_list)
else:
    print("No solution found.") 
reversed_sol = solution_list[::-1]
print(reversed_sol)

plt.plot(generations, best_fitness_values)
plt.xlabel('Generation')
plt.ylabel('Best fitness')
plt.title('Genetic Algorithm')
plt.show()

a = agent(m, footprints=True)
m.tracePath({a: reversed_sol})
m.run()
# plot graph





 