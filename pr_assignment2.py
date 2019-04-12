import numpy as np
import copy
DNA_SIZE = 4             # DNA length
MUTATION_RATE = 0.2    # mutation probability
N_GENERATIONS = 20
DNA_final = []
final = 0.0
All_DNA = []
DNA1 = [str(i) for i in np.random.randint(0,2,4).tolist()]
DNA2 = [str(i) for i in np.random.randint(0,2,4).tolist()]
num = 0

# to find the maximum of this function
def F(x):
    return 0.2*float(x[0]) + 0.3*float(x[1]) + 0.5*float(x[2]) + 0.1*float(x[3])

# to constraint the input
def constraint(pred):
    if 0.5*float(pred[0]) + 1.0*float(pred[1]) + 1.5*float(pred[2]) + 0.1*float(pred[3]) <= 3.1 \
            and 0.3*float(pred[0]) + 0.8*float(pred[1]) + 1.5*float(pred[2]) + 0.4*float(pred[3]) <= 2.5 \
            and 0.2*float(pred[0]) + 0.2*float(pred[1]) + 0.3*float(pred[2]) + 0.1*float(pred[3]) <= 0.4:
        return True
    else:
        return False

# select detect and add DNA
def select_detec_add_DNA(parent):
    if constraint(parent) is True and d_recur(parent) is True:
        All_DNA.append(parent)
    else:
        pass


# crossover the parents
def crossover(parent1,parent2):
    child = parent1[0:1] + parent2[1:]
    return child

# mutate the child
def mutate(child):
    dna = copy.copy(child)
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = "1" if child[point] == "0" else "0"
    if dna != child:
        print dna, " mutation to ", child
    return child

#detect recur
def d_recur(DNA):
    if DNA not in All_DNA:
        return True
    else:
        return False

# crossover, mutation and add to All list
def ge_al(ALL_DNA):
    for i in range(0,len(All_DNA)-1):
        print "Selected for crossover as parent1:", All_DNA[i]
        for j in range(i + 1, len(All_DNA)):
            print "Selected for crossover as parent2:", All_DNA[j]
            child1 = crossover(All_DNA[i],All_DNA[j])
            child2 = crossover(All_DNA[j],All_DNA[i])
            child1 = mutate(child1)
            child2 = mutate(child2)
            select_detec_add_DNA(child1)
            select_detec_add_DNA(child2)

# calculate return
def calculate(All_DNA):
    global num
    for k in All_DNA[num:len(All_DNA)]:
        print "Each Population: ",k
        result = F(k)
        global final
        print "calculate final:",final
        num = num + 1
        if result > final:
            final = result
            global DNA_final
            DNA_final = k
            print "Current Max return DNA:", k
            print "Current Max return:", final
            print "###############    Finish one for loop    ###############"
        else:
            final = final

# first two DNA
def ft(DNA):
    if constraint(DNA) is True:
        All_DNA.append(DNA)
    else:
        while constraint(DNA) is False:
            DNA = [str(i) for i in np.random.randint(0, 2, 4).tolist()]
        All_DNA.append(DNA)


ft(DNA1)
print "First DNA in ALL_DNA :",All_DNA
ft(DNA2)
print "Second DNA in ALL_DNA:",All_DNA
print "Mutation propability :",MUTATION_RATE
print "Population Size:", len(All_DNA)
print "###############"
for x in range(N_GENERATIONS):
    ge_al(All_DNA)
    calculate(All_DNA)
    print "Population Size:",len(All_DNA)

print "################        Final Step      ################"
print "Final Max DNA:", DNA_final
print "Max return:", final

