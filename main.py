import leoAI, json, random

#training data
random.seed(4)
learning = [[-1, 1], [-0.1, 0.1]]
network = leoAI.generate(2, [3, 4, 2], 1, [-1, 1], [-1, 1])

data = [
    [[0, 1], 1],
    [[1, 1], 0],
    [[0, 0], 0],
    [[1, 0], 1]
]

#calculate errorSum neural network
def errorSum(network):
    error_sum = 0
    for test in data:
        result = leoAI.calculate(network, test[0], 'sigmoid')[0]
        answer = test[1]
        error_sum += (result - answer) ** 2
        
    return error_sum

#training
keep = True
while keep:
    error_sum = errorSum(network)
    
    saved_network = json.loads(json.dumps(network))

    #chancing weights
    for _ in range(random.randint(1, 3)):
        random_point = random.randint(1, len(network)-2)
        random_point_weight = random.randint(0, len(network[random_point]['weight'])-1)
        network[random_point]['weight'][random_point_weight] += random.randint(learning[0][0]*100, learning[0][1]*100)/100

    #chancing bias
    random_point = random.randint(1, len(network)-2)
    random_point_biass = random.randint(0, len(network[random_point]['biass'])-1)
    network[random_point]['biass'][random_point_biass] += random.randint(learning[1][0]*100, learning[1][1]*100)/100

    test_error_sum = errorSum(network)

    if error_sum < test_error_sum:
        network = saved_network

    print('Error sum: ' + str(error_sum))

    if error_sum < 0.15:
        keep = False

#showing off
print('\nDumping network')
print(network)

print('\nDone learning, answers here!')
for test in data:
    answer = leoAI.calculate(network, test[0], 'sigmoid')[0]
    print(test[0], answer)
