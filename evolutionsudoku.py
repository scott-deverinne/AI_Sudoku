import random
import numpy as np
import copy

# Problem Parameters

list_of_population_size = [10,100,1000,10000]
MUTATION_RATE = 1.0 / 8
TRUNCATION_RATE = 0.5
NUMBER_GENERATION = 100
# The total maximum solution space wouldbe 9! * 9 = 3265920
grid1 = np.array([[3,0,0,0,0,5,0,4,7]
        ,[0,0,6,0,4,2,0,0,1]
        ,[0,0,0,0,0,7,8,9,0]
        ,[0,5,0,0,1,6,0,0,2]
        ,[0,0,3,0,0,0,0,0,4]
        ,[8,1,0,0,0,0,7,0,0]
        ,[0,0,2,0,0,0,4,0,0]
        ,[5,6,0,8,7,0,1,0,0]
        ,[0,0,0,3,0,0,6,0,0]])

grid2 = np.array([[0,0,2,0,0,0,6,3,4]
        ,[1,0,6,0,0,0,5,8,0]
        ,[0,0,7,3,0,0,2,9,0]
        ,[0,8,5,0,0,1,0,0,6]
        ,[0,0,0,7,5,0,0,2,3]
        ,[0,0,3,0,0,0,0,5,0]
        ,[3,1,4,0,0,2,0,0,0]
        ,[0,0,9,0,8,0,4,0,0]
        ,[7,2,0,0,4,0,0,0,9]])

grid3 = np.array([[0,0,4,0,1,0,0,6,0]
        ,[9,0,0,0,0,0,0,3,0]
        ,[0,5,0,7,9,6,0,0,0]
        ,[0,0,2,5,0,4,9,0,0]
        ,[0,8,3,0,6,0,0,0,0]
        ,[0,0,0,0,0,0,6,0,7]
        ,[0,0,0,9,0,3,6,0,7]
        ,[0,0,0,0,0,0,0,0,0]
        ,[0,0,6,0,0,0,0,1,0]])

## Evolutionary Algorithm ##

def evolve():
    grid_number = 0
    for grid in [grid1,grid2,grid3]:
        grid_number += 1
        for population_size in list_of_population_size: # Run 4 times for population size 10,100,1000,10000
            for run in range(1,6): # Run an average of 5 times for each experiment
                random.seed(random.randint(1,10000)) # Each run is ran with a new random seed.
                population = create_pop(population_size,grid)
                fitness_population = evaluate_pop(population)
                for gen in range(NUMBER_GENERATION):
                    mating_pool = select_pop(population, fitness_population, population_size)
                    offspring_population = crossover_pop(mating_pool, population_size, grid)
                    population = mutate_pop(offspring_population, grid)
                    fitness_population = evaluate_pop(population)
                    best_ind, best_fit = best_pop(population, fitness_population)
                print("Grid %d, Run #%d, Population size: %d." % (grid_number, run, population_size))
                print("#%3d" % gen, "fit:%3d" % best_fit, best_ind)


## Population level operators

def create_pop(population_size,grid):
    return [ create_ind(grid) for _ in range(population_size) ]

def evaluate_pop(population):
    return [evaluate_ind(individual) for individual in population]

def select_pop(population, fitness_population, population_size):
    sorted_population = sorted(zip(population, fitness_population), key = lambda ind_fit: ind_fit[1])
    return [ individual for individual, fitness in sorted_population[:int(population_size * TRUNCATION_RATE)] ]

def crossover_pop(population, population_size, grid):
    return [ crossover_ind(random.choice(population), random.choice(population), grid) for _ in range(population_size) ]

def mutate_pop(population,grid):
    return [ mutate_ind(individual,grid) for individual in population ]

def best_pop(population, fitness_population):
    return sorted(zip(population, fitness_population), key = lambda ind_fit: ind_fit[1])[0]

## Create individual

def create_ind(grid):
    individual = copy.deepcopy(grid)
    for row in individual:
        unused_numbers = [1,2,3,4,5,6,7,8,9]
        for number in row:
            if number != 0:
                unused_numbers.remove(number)

        for i in range(9):
            if row[i] == 0:
                number = random.choice(unused_numbers)
                row[i] = number
                unused_numbers.remove(number)
    return individual

## Fitness function
# appropriate fitness function calculates the number of misalignments in the 9x9 grid
def evaluate_ind(individual):
    return evaluate_column(individual) + evaluate_subgrids(individual)

# evaluate number of rows with repeated digits in a 9x9 grid.
def evaluate_row(individual):
    repeated_rows = 0
    for row in individual:
        if len(set(row)) != len(row):
            repeated_rows += 1
    return repeated_rows

# evaluate number of columns with repeated digits in a 9x9 grid.
def evaluate_column(individual):
    repeated_columns = 0
    for i in range(0,9):
        column = individual[:,i]
        if len(set(column)) != len(column):
            repeated_columns += 1
    return repeated_columns

# evaluate number of subgrids with repeated digits in a 9x9 grid.
def evaluate_subgrids(individual):
    repeated_subgrid = 0
    for subgrid in create_subgrids(individual):
        if len(set(subgrid)) != len(subgrid):
            repeated_subgrid += 1
    return repeated_subgrid

def create_subgrids(individual):
    answer = []
    for r in range(3):
        for c in range(3):
            block = []
            for i in range(3):
                for j in range(3):
                    block.append(individual[3*r + i][3*c + j])
            answer.append(block)
    return answer

## Mutation and Crossover

# Check position in the 2d array is a valid mutation / crossover choice
# Single point crossover used on the same row of two different candidates swapped

def crossover_ind(individual1,individual2,grid):
    row_cross = random.randint(0,8) # Random row position to crossover for indv1 and indv2.
    cross_indv1 = copy.deepcopy(individual1)
    cross_indv2 = copy.deepcopy(individual2)
    row_indv2 = copy.deepcopy(cross_indv2[row_cross])
    cross_indv1[row_cross] = row_indv2
    return cross_indv1


def mutate_ind(individual,grid):
    individual = copy.deepcopy(individual)
    random_row = random.randint(0,8)
    fixed_number_position = [] # List containg column positions with fixed numbers already specified in coursework. 
    
    # Append all fixed number positions to list.
    for i in range(9):
        if grid[random_row][i] != 0:
            fixed_number_position.append(i)

    # Each integer in the row has a chance to mutate if the mutation probability is met.
    for i in range(0,9):
        if random.random() < MUTATION_RATE and i not in fixed_number_position:
            j = random.randint(0,8)
            while j == i or j in fixed_number_position:
                j = random.randint(0,8)
            individual[random_row][i], individual[random_row][j] = individual[random_row][j], individual[random_row][i]
    return individual


if __name__ == "__main__":
    evolve()

                    
