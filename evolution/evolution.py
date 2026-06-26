import random
import math
import copy
from network import (
    Network, Node, build_node, buildNetwork,
    connectNodes, disconnect_nodes, has_connection,
    remove_node_from_network, reindex_network,
)
from constants import config, ALL_ACTIVATIONS

from enum import Enum, auto

class MutationType(Enum):
    ADD_NODE = auto()
    REMOVE_NODE = auto()
    ADD_CONN = auto()
    REMOVE_CONN = auto()
    MOD_WEIGHT = auto()
    MOD_BIAS = auto()
    MOD_ACTIVATION = auto()
    SWAP_NODES = auto()

#Changes activation function to a random different one
def mutateNodeActivationFunction(node: Node):
    allowed = ALL_ACTIVATIONS

    squash = node.squash
    while squash == node.squash:
        squash = random.choice(allowed)

    node.squash = squash
    return node


#Adds a node inbetween an existing connection
def mutateAddNode(network: Network) :
    if len(network.connections) == 0 :
        if config.warnings :
            print("No connections to mutate", network)
        return

    connection = network.connections[random.randint(0, len(network.connections) - 1)]

    disconnect_nodes(network, connection.from_node, connection.to_node)

    toIndex = network.nodes.index(connection.to_node)
    node = build_node("hidden")

    mutatedNode = mutateNodeActivationFunction(node)

    minBound = min(toIndex, len(network.nodes) - network.output_size)
    network.nodes.insert(minBound, mutatedNode)

    connectNodes(network, connection.from_node, mutatedNode, random.uniform(-0.1, 0.1))
    connectNodes(network, mutatedNode, connection.to_node, random.uniform(-0.1, 0.1))

    return network

#Removes inner random node
def mutateRemoveNode(network: Network) :
    if len(network.nodes) <= network.input_size + network.output_size :
        if config.warnings :
            print("no nodes to remove")
        return network

    hiddenNodes = [node for node in network.nodes if node.type == "hidden"]

    if not hiddenNodes :
        return network

    nodeToRemove = random.choice(hiddenNodes)

    mutatedNetwork = remove_node_from_network(network, nodeToRemove)

    return mutatedNetwork


#Add random new connection
def mutateAddConnection(network: Network) :
    available = []

    for index1, node1 in enumerate(network.nodes) :
        if node1.type == "output" :
            continue

        for index2, node2 in enumerate(network.nodes) :
            if node2.type == "input" :
                continue
            if node1 is node2 :
                continue
            if index1 > index2 :
                continue
            if has_connection(node1, node2) :
                continue

            available.append([node1, node2])

    if len(available) == 0 :
        if config.warnings :
            print("no more connections can be built")
        return network

    pair = random.choice(available)
    connectNodes(network, pair[0], pair[1], random.uniform(-0.1, 0.1))

    return network


#Removes a random connection of a node (as long as it still has at least 2 left after)
def mutateRemoveConnection(network: Network) :
    possible = []

    for conn in network.connections[:] :
        if (
            len(conn.from_node.connections_out) > 1 and
            len(conn.to_node.connections_in) > 1 and
            network.nodes.index(conn.to_node) > network.nodes.index(conn.from_node)
        ) :
            possible.append(conn)

    if len(possible) == 0 :
        if config.warnings :
            print("No connections to remove!")
        return network

    randomConn = random.choice(possible)
    mutatedNetwork = disconnect_nodes(network, randomConn.from_node, randomConn.to_node)

    return mutatedNetwork

#Randomly changes connection weight of a random connection
def mutateConnectionWeight(network: Network) :
    minVal = config.mutations.connectionWeight.min
    maxVal = config.mutations.connectionWeight.max

    allConnections = network.connections[:]

    if len(allConnections) == 0 :
        if config.warnings :
            print("no connection to mutate")
        return network

    connection = random.choice(allConnections)
    modification = random.uniform(minVal, maxVal)

    connection.weight += modification

    return network

#Randomly mutates activation function of hidden (/output) node
def mutateActivationFunction(network: Network) :
    mutateOutput = config.mutations.activationFunction.mutateOutput

    if not mutateOutput and network.input_size + network.output_size == len(network.nodes) :
        return network

    rangeSize = len(network.nodes) - (0 if mutateOutput else network.output_size) - network.input_size
    index = random.randint(network.input_size, network.input_size + rangeSize - 1)
    node = network.nodes[index]

    mutateNodeActivationFunction(node)

    return network

#Mutates bias of a random hidden/output node
def mutateBias(network: Network) :
    minVal = config.mutations.bias.min
    maxVal = config.mutations.bias.max

    index = random.randint(network.input_size, len(network.nodes) - 1)
    node = network.nodes[index]

    modification = random.uniform(minVal, maxVal)
    node.bias += modification

    return network


#Swaps bias and activation of 2 random hidden/output nodes
def mutateSwapNodes(network: Network) :
    mutateOutput = config.mutations.swapNodes.mutateOutput

    if (
        (mutateOutput and len(network.nodes) - network.input_size < 2) or
        (not mutateOutput and len(network.nodes) - network.input_size - network.output_size < 2)
    ) :
        return network

    rangeSize = len(network.nodes) - (0 if mutateOutput else network.output_size) - network.input_size

    index1 = random.randint(network.input_size, network.input_size + rangeSize - 1)
    node1 = network.nodes[index1]

    index2 = random.randint(network.input_size, network.input_size + rangeSize - 1)
    node2 = network.nodes[index2]

    biasTemp = node1.bias
    squashTemp = node1.squash

    node1.bias = node2.bias
    node1.squash = node2.squash
    node2.bias = biasTemp
    node2.squash = squashTemp

    return network


mutation = {
    MutationType.ADD_NODE: mutateAddNode,
    MutationType.REMOVE_NODE: mutateRemoveNode,
    MutationType.ADD_CONN: mutateAddConnection,
    MutationType.REMOVE_CONN: mutateRemoveConnection,
    MutationType.MOD_WEIGHT: mutateConnectionWeight,
    MutationType.MOD_BIAS: mutateBias,
    MutationType.MOD_ACTIVATION: mutateActivationFunction,
    MutationType.SWAP_NODES: mutateSwapNodes,
}

ALL_MUTATIONS = list(mutation.keys())


# -------------------------------------------------------
# Population operations
# -------------------------------------------------------

def selectMutationMethod(possible_mutations) :
    return random.choice(possible_mutations)


def mutateNetwork(network: Network, method: MutationType) -> Network :
    if method not in mutation :
        raise ValueError(f"Unknown mutation method: {method}")

    return mutation[method](network)


def performMutation(genome: Network, mutation_amount=None, possible_mutations=None) -> Network :
    mutation_amount    = mutation_amount    if mutation_amount    is not None else config.mutation_amount
    possible_mutations = possible_mutations if possible_mutations is not None else ALL_MUTATIONS

    # Deep copy so the original (e.g. an elitist) is never modified in place
    target = copy.deepcopy(genome)

    for _ in range(mutation_amount) :
        method = selectMutationMethod(possible_mutations)
        target = mutateNetwork(target, method)

    return target


def mutatePopulation(population, mutation_rate=None, mutation_amount=None) :
    mutation_rate   = mutation_rate   if mutation_rate   is not None else config.mutation_rate
    mutation_amount = mutation_amount if mutation_amount is not None else config.mutation_amount

    result = []
    for genome in population :
        if random.random() <= mutation_rate :
            result.append(performMutation(genome, mutation_amount))
        else :
            result.append(genome)

    return result


def sortPopulation(population) :
    return sorted(population, key=lambda n: n.score if n.score is not None else float('-inf'), reverse=True)


def getOffspring(population, strategy: str = None, **selection_kwargs):
    from selection import select_parent
    from crossover import crossover

    if strategy is None:
        strategy = "power"
        selection_kwargs.setdefault("power", config.selection_power)

    parent1 = select_parent(population, strategy, **selection_kwargs)
    parent2 = select_parent(population, strategy, **selection_kwargs)

    return crossover(parent1, parent2, equal=False)


def createPopulation(template: Network, count: int) -> list :
    """Create count fresh networks with the same input/output size as template."""
    return [buildNetwork(template.input_size, template.output_size) for _ in range(count)]