import math
import random
import time
import matplotlib.pyplot as plt
import Gamelogic


def addPoint(frames):
    framesVector.append(frames)
    posVector.append(len(framesVector))
    plt.plot(posVector, framesVector)
    plt.show()
    plt.pause(0.0001)


# Setting up the plot
framesVector = []  # y-coodrinates
posVector = []  # x-coordinates
plt.ion()
figure = plt.figure()

find_action_time=0


class Neuron:
    def __init__(self, cross_value, connections, number, input_layer=False, output_layer=False):
        self.cross_value = cross_value
        self.value = 0
        self.connections = connections
        self.number = number
        self.input_layer = input_layer
        self.output_layer = output_layer

    def add_connection(self, connection):
        self.connections.append(connection)

    def remove_connection(self, connection):
        self.connections.remove(connection)

    # Feeding the activation forward in the network
    def feed_forward(self):
        activation_value = self._sigmoid() if not self.input_layer else self.value
        for connection in self.connections:
            if connection.enabled:
                connection.out_node.value += activation_value * connection.weight

        return_value = self.value
        self.value = 0
        if not self.output_layer:
            return "null"
        return return_value

    # Calculating and returning activation
    def _sigmoid(self):
        exp = -self.value
        # print(exp)
        activated = 1 / (1 + math.pow(math.e, exp))
        if not self.output_layer:
            self.value = 0
        return activated


class Connection:
    def __init__(self, in_node, out_node, weight, enabled, innovation_number):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled
        self.innovation_number = innovation_number


class Agent:
    def __init__(self, topology, global_mutation_neuron_dict, global_connection_dict, agent_num, add_connections=True):
        # Initialization parameters
        self.max_weight = 2
        self.min_weight = -2
        self.bias = False

        self.agent_num=agent_num

        self.fitness = -25
        self.connections = []
        self.local_connections_dict = {}
        self.global_mutation_neuron_dict = global_mutation_neuron_dict
        self.global_connection_dict = global_connection_dict

        # Initializing output-neurons
        self.sum_default_neurons = sum(topology)
        self.neural_network = [Neuron(1, [], x, output_layer=True) for x in range(topology[1])]
        self.network_dict = {}
        for neuron in self.neural_network:
            self.network_dict[neuron.number] = neuron

        # Initializing input-neurons
        for x in range(topology[0]):
            new_node = Neuron(0, [], len(self.neural_network))
            # Adding connections from input-neurons to output-neurons
            if add_connections:
                for y in range(topology[1]):
                    innovation_number = len(self.connections)
                    new_connection = Connection(new_node, self.neural_network[y],
                                                random.uniform(self.min_weight, self.max_weight), True,
                                                innovation_number)
                    new_node.add_connection(new_connection)
                    self.connections.append(new_connection)

                    # Adding the connection to local and global dictionary
                    connection_name = str(new_node.number) + '-' + str(self.neural_network[y].number)
                    self.local_connections_dict[connection_name] = new_connection
                    if self.global_connection_dict.get(connection_name, -1) == -1:
                        self.global_connection_dict[connection_name] = innovation_number

            self.neural_network.append(new_node)
            self.network_dict[new_node.number] = new_node
        self.sort_agent()

    def get_neuron(self, number):
        return self.network_dict.get(number, -1)

    # Adding an neuron to the neural network list and -dict
    def add_neuron(self, cross_value, number):
        self.neural_network.append(Neuron(cross_value, [], number))
        self.network_dict[number] = self.neural_network[-1]
        return self.neural_network[-1]

    # Adding a new connection from a model connection to the network and creating new nodes if they don't exist
    def add_connection(self, connection):
        from_neuron = self.get_neuron(connection.in_node.number)
        to_neuron = self.get_neuron(connection.out_node.number)

        if from_neuron == -1:
            from_neuron = self.add_neuron(connection.in_node.cross_value, connection.in_node.number)
            self.neural_network.append(from_neuron)
            self.network_dict[from_neuron.number] = from_neuron

        if to_neuron == -1:
            to_neuron = self.add_neuron(connection.out_node.cross_value, connection.out_node.number)
            self.neural_network.append(to_neuron)
            self.network_dict[to_neuron.number] = to_neuron

        new_connection = Connection(from_neuron, to_neuron, connection.weight, connection.enabled,
                                    connection.innovation_number)
        self.connections.append(new_connection)
        connection_name = str(from_neuron.number) + '-' + str(to_neuron.number)
        self.local_connections_dict[connection_name] = new_connection

        if self.global_connection_dict.get(connection_name, -1) == -1:
            self.global_connection_dict[
                str(from_neuron.number) + '-' + str(to_neuron.number)] = connection.innovation_number  # new_connection
        from_neuron.add_connection(new_connection)

    @staticmethod
    def disable_connection(connection):
        connection.enabled = False

    # Adding a new neuron to the network by disabling an old connection
    def mutate_neuron(self):
        connection_num = random.randint(0, len(self.connections))
        x = 0
        while True:
            if connection_num == 0:
                break
            if self.connections[x].enabled:
                connection_num -= 1
            x += 1
            x %= len(self.connections)
        self.connections[x].enabled = False
        mutation_global_connection = self.global_mutation_neuron_dict.get(
            str(self.connections[x].in_node.number) + '-' + str(self.connections[x].out_node.number), -1)

        # The mutation has never occurred before, making a new one from scratch
        if mutation_global_connection == -1:

            first_neuron = self.connections[x].in_node
            second_neuron = self.connections[x].out_node
            num_global_connections = len(self.global_connection_dict.keys())
            cross_value = (first_neuron.cross_value + second_neuron.cross_value) / 2

            # New mutated neuron
            mutation_neuron = Neuron(cross_value, [],
                                     self.sum_default_neurons + len(self.global_mutation_neuron_dict.keys()))

            # New connection to the newly mutated neuron
            to_connection = Connection(first_neuron, mutation_neuron, 1, True, num_global_connections)
            to_connection_name = str(first_neuron.number) + '-' + str(mutation_neuron.number)

            # Updating the global dictionary of connections
            self.global_connection_dict[to_connection_name] = num_global_connections
            num_global_connections += 1

            # New connection from the newly mutated neuron
            from_connection = Connection(mutation_neuron, second_neuron, self.connections[x].weight,
                                         True, num_global_connections)
            from_connection_name = str(mutation_neuron.number) + '-' + str(second_neuron.number)

            # Updating the global dictionary of connections
            self.global_connection_dict[from_connection_name] = num_global_connections

            # Updating the global dictionary of neurons
            new_mutation_name = str(first_neuron.number) + '-' + str(second_neuron.number)
            self.global_mutation_neuron_dict[new_mutation_name] = GlobalNeuronMutation(mutation_neuron,
                                                                                       to_connection,
                                                                                       from_connection)

            # Implementing the neuron in the network
            self.neural_network.append(mutation_neuron)
            self.network_dict[mutation_neuron.number] = mutation_neuron

            first_neuron.add_connection(to_connection)
            self.connections.append(to_connection)
            self.local_connections_dict[to_connection_name] = to_connection

            mutation_neuron.add_connection(from_connection)
            self.connections.append(from_connection)
            self.local_connections_dict[from_connection_name] = from_connection

        # The mutation has happened locally before
        elif self.get_neuron(mutation_global_connection.neuron.number) != -1:
            return

        # The mutation has occurred before, only copying this mutation
        else:
            self.add_connection(mutation_global_connection.connection_to)
            self.add_connection(mutation_global_connection.connection_from)

    # Adding a new connection to the agent
    def mutate_connection(self, min_weight_inp, max_weight_inp):
        min_weight = max_weight_inp
        max_weight = min_weight_inp

        neuron_one_num = random.randint(0, len(self.neural_network) - 1)
        neuron_two_num = random.randint(0, len(self.neural_network) - 1)
        neuron_one = self.neural_network[neuron_one_num]
        neuron_two = self.neural_network[neuron_two_num]

        # The neurons are incompatible
        if neuron_one.cross_value == neuron_two.cross_value:
            return

        # The connection is the other way
        if neuron_one.cross_value > neuron_two.cross_value:
            neuron_one = neuron_two
            neuron_two = self.neural_network[neuron_one_num]

        connection_name = str(neuron_one.number) + '-' + str(neuron_two.number)

        # The connection is already in the agent
        if self.local_connections_dict.get(connection_name, -1) != -1:
            self.local_connections_dict.get(connection_name).enabled = True
            return
        innovation_number = self.global_connection_dict.get(connection_name, -1)
        if innovation_number == -1:
            innovation_number = len(self.global_connection_dict.keys())
            self.global_connection_dict[connection_name] = innovation_number

        new_connection = Connection(neuron_one, neuron_two, random.uniform(min_weight, max_weight), True,
                                    innovation_number)
        self.local_connections_dict[connection_name] = new_connection
        self.connections.append(new_connection)
        neuron_one.add_connection(new_connection)

    def mutate_reweight(self, prob_change_weight, prob_small_weight_change, min_add_inp, max_add_inp, min_random_inp,
                        max_random_inp):
        min_add = min_add_inp
        max_add = max_add_inp
        min_random = min_random_inp
        max_random = max_random_inp
        for connection in self.connections:
            # Change weight
            if random.random() < prob_change_weight:
                # Small weight change
                if random.random() < prob_small_weight_change:
                    connection.weight += random.uniform(min_add, max_add)
                # New weight
                else:
                    connection.weight = random.uniform(min_random, max_random)

    def sort_agent(self):
        self.neural_network.sort(key=lambda x: (x.cross_value, x.number), reverse=False)
        self.connections.sort(key=lambda x: x.innovation_number, reverse=False)

    # Execute action
    def action(self, input):
        global find_action_time
        now=time.time()
        self.fitness += 1

        # Giving the input to the input-neurons
        for x in range(len(input)):
            self.neural_network[x].value = input[x]
            self.neural_network[x].feed_forward()

        # Feeding the activations through the network
        output = []
        for x in range(len(input), len(self.neural_network)):
            value = self.neural_network[x].feed_forward()
            if value != "null":
                output.append(value)
        find_action_time+=time.time()-now
        return output.index(max(output))


class GlobalNeuronMutation:
    def __init__(self, neuron, connection_to, connection_from):
        self.neuron = neuron
        self.connection_to = connection_to
        self.connection_from = connection_from


class Population:
    def __init__(self, number):
        # Initializing all agents
        self.generation_num = 0
        self.last_run = 0
        self.population_number = number
        self.global_neuron_mutation_dict = {}  # from-to:GlobalNeuronMutation
        self.global_connection_dict = {}  # from-to:innovation_number
        self.global_agent_num=0

        self.agents = [Agent([4, 2], self.global_neuron_mutation_dict, self.global_connection_dict, x) for x in
                       range(number)]

        self.global_agent_num=number

    # Evaluate generation and breed a new
    def generation(self, gamelogic_inp):
        global find_action_time
        find_action_time=0
        self.generation_num += 1
        to_render = (self.last_run > 5000)
        game = gamelogic_inp  # to_render

        # # # # Parameters
        # Divide into groups
        threshold = 4
        c1 = 1
        c2 = 1
        c3 = 3
        # Kill
        kill_precentage = 80
        # # Mutation
        # Neuron-mutation
        prob_new_node = 0.03
        # Connection-mutation
        prob_new_connection = 0.05
        new_connection_min_weight = -2
        new_connection_max_weight = 2
        # Reweigting-mutation
        prob_change_weight = 0.9
        min_add_weight = -0.5
        max_add_weight = 0.5
        prob_small_weight_change = 0.9
        min_random = -2
        max_random = 2

        dist = 0
        for player in self.agents:
            game.add_player(human=False, player=player, render=dist < 10)
            dist += 1

        # Fitness
        game.run_game(0, self.generation_num)
        print("find_action_time:", find_action_time)
        self.last_run = max(a.fitness for a in self.agents)
        print("most fit:", self.last_run)

        # addPoint(max(a.fitness for a in self.agents))
        print("Average:", sum(a.fitness for a in self.agents) / len(self.agents))

        # Divide to groups
        groups = []
        for organism in self.agents:
            added = False
            for x in range(len(groups)):
                if threshold >= self.compatibility_distance(organism, groups[x][0], c1, c2, c3):
                    groups[x].append(organism)
                    added = True
                    break
            if not added:
                groups.append([organism])

        # Kill
        relative_fitness = []
        for x in range(len(groups)):
            groups[x].sort(key=lambda a: a.fitness, reverse=True)
            groups[x] = groups[x][:int(math.floor(len(groups[x])*(100-kill_precentage)/100))]
            relative_fitness.append(sum(temp_organism.fitness for temp_organism in groups[x]) / max(len(groups[x]), 1))
        print("groups:", len(groups))

        # Selection
        total_relative_fitness = sum(relative_fitness)
        num_next_gen = [math.floor(self.population_number * x / total_relative_fitness) for x in relative_fitness]
        next_gen = []
        for x in range(self.population_number - sum(num_next_gen)):
            group_num = random.randint(0, len(groups) - 1)
            if len(groups[group_num]) != 0:
                num_next_gen[group_num] += 1
            else:
                x -= 1

        # Survival of the fittest from each group
        num_next_gen = [a - 1 for a in num_next_gen]

        # Iterating through all species
        now = time.time()
        for x in range(len(groups)):
            sum_fitness = sum(e.fitness for e in groups[x])
            # Adding new organisms for current species
            for y in range(num_next_gen[x]):
                parent_num_a = random.randint(0, sum_fitness)
                parent_num_b = random.randint(0, sum_fitness)
                parent_a = -1
                parent_b = -1
                temp_sum = 0
                # Finding the parents for the new organism
                for z in range(len(groups[x])):
                    if parent_num_a == parent_num_b == -1:
                        break
                    temp_sum += groups[x][z].fitness
                    if parent_num_a != -1 and parent_num_a <= temp_sum:
                        parent_num_a = -1
                        parent_a = z
                    if parent_num_b != -1 and parent_num_b <= temp_sum:
                        parent_num_b = -1
                        parent_b = z

                next_gen.append(self.breed(groups[x][parent_a], groups[x][parent_b]))
        print("breedtime: ", time.time() - now)

        # Mutation
        now = time.time()
        for agent in next_gen:
            # New nodes
            if random.random() < prob_new_node:
                agent.mutate_neuron()
            # New links
            if random.random() < prob_new_connection:
                agent.mutate_connection(new_connection_min_weight, new_connection_max_weight)
            # Re_weighting
            agent.mutate_reweight(prob_change_weight, prob_small_weight_change, min_add_weight, max_add_weight,
                                  min_random, max_random)
            agent.sort_agent()
        # Add the fittest agent from each group
        for agent in groups:
            if len(agent) > 0:
                agent[0].fitness = -25
                next_gen.append(agent[0])
        self.agents = next_gen
        print("num next gen:", sum(num_next_gen))
        print("mutationtime:", time.time() - now)

    # Making a new agent by combining two agents
    def breed(self, agent_a, agent_b):
        new_agent = Agent([4, 2], self.global_neuron_mutation_dict, self.global_connection_dict, self.global_agent_num, add_connections=False)
        self.global_agent_num+=1
        first_more_fit = agent_a.fitness > agent_b.fitness
        pos_a = 0
        pos_b = 0
        probability = agent_a.fitness / (agent_a.fitness + agent_b.fitness)
        while pos_a < len(agent_a.connections) and pos_b < len(agent_b.connections):
            inn_num_a = agent_a.connections[pos_a].innovation_number
            inn_num_b = agent_b.connections[pos_b].innovation_number

            # If they are on the same connection
            if inn_num_a == inn_num_b:
                original_connection = agent_a.connections[pos_a] if random.random() < probability else \
                    agent_b.connections[pos_b]
                new_agent.add_connection(original_connection)
                pos_a += 1
                pos_b += 1
                continue

            # If disjoint a
            elif inn_num_a < inn_num_b:
                # a is more fit
                if first_more_fit:
                    new_agent.add_connection(agent_a.connections[pos_a])
                pos_a += 1
                continue

            # If disjoint b
            elif inn_num_a > inn_num_b:
                # b is more fit
                if not first_more_fit:
                    new_agent.add_connection(agent_b.connections[pos_b])
                pos_b += 1
                continue

        # Adding excess-genes for a
        if pos_a < len(agent_a.connections) and first_more_fit:
            for x in range(pos_a, len(agent_a.connections)):
                new_agent.add_connection(agent_a.connections[pos_a])
        # Adding excess-genes for b
        if pos_b < len(agent_b.connections) and not first_more_fit:
            for x in range(pos_b, len(agent_b.connections)):
                new_agent.add_connection(agent_b.connections[pos_b])

        return new_agent

    @staticmethod
    def compatibility_distance(agent_a, agent_b, c1, c2, c3):
        nn_a = agent_a.connections
        nn_b = agent_b.connections
        pos_a = 0
        pos_b = 0
        disjoint = 0
        excess = 0
        sum_difference_equals = 0
        num_matching = 0
        num_genome = max(len(nn_a), len(nn_b))
        while pos_a < len(nn_a) and pos_b < len(nn_b):
            if nn_a[pos_a].innovation_number == nn_b[pos_b].innovation_number:
                sum_difference_equals += abs(nn_a[pos_a].weight - nn_b[pos_b].weight)
                num_matching += 1
                pos_a += 1
                pos_b += 1
            elif nn_a[pos_a].innovation_number < nn_b[pos_b].innovation_number:
                pos_a += 1
                disjoint += 1
            elif nn_a[pos_a].innovation_number > nn_b[pos_b].innovation_number:
                pos_b += 1
                disjoint += 1
        if pos_b == len(nn_b) and pos_a < len(nn_a):
            excess += len(nn_a) - pos_a
        elif pos_a == len(nn_a) and pos_b < len(nn_b):
            excess += len(nn_b) - pos_b

        # To prevent math error if no matches occurred
        num_matching = num_matching if num_matching != 0 else 1

        return (c1 * excess + c2 * disjoint) / num_genome + c3 * sum_difference_equals / num_matching


generation_num = 0
pop = Population(1000)
gamelogic = Gamelogic.Gamelogic(new_window=True, rendering=True)
while True:
    now = time.time()
    generation_num += 1
    pop.generation(gamelogic)
    gamelogic.reset_game(delete_players=True)
    print(generation_num, time.time() - now)
    print()

# TODO: add bias
