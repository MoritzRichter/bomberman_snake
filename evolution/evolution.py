import random
import math
from network import (
    Network, Node, build_node, buildNetwork,
    connectNodes, disconnect_nodes, has_connection,
    remove_node_from_network, reindex_network, clone_network,
)
from constants import config, ALL_ACTIVATIONS

from enum import Enum, auto

class MutationType(Enum):
    ADD_NODE = auto()
    REMOVE_NODE = auto()
    ADD_CONN = auto()
    REMOVE_CONN = auto()
    MOD_WEIGHT = auto()
    MOD_WEIGHT_LARGE = auto()
    MOD_BIAS = auto()
    MOD_ACTIVATION = auto()
    SWAP_NODES = auto()
    ADD_LAYER = auto()

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

#Randomly changes connection weight of a random connection (small perturbation: ±0.1)
def mutateConnectionWeight(network: Network) :
    minVal = config.mutations.connectionWeight.min
    maxVal = config.mutations.connectionWeight.max

    allConnections = network.connections[:]

    if len(allConnections) == 0 :
        if config.warnings :
            print("no connection to mutate")
        return network

    connection = random.choice(allConnections)
    connection.weight += random.uniform(minVal, maxVal)

    return network


#Resets a random connection weight to a new value in [-2.0, 2.0] — escapes local optima
def mutateConnectionWeightLarge(network: Network) :
    minVal = config.mutations.connectionWeightLarge.min
    maxVal = config.mutations.connectionWeightLarge.max

    allConnections = network.connections[:]

    if len(allConnections) == 0 :
        if config.warnings :
            print("no connection to mutate")
        return network

    connection = random.choice(allConnections)
    connection.weight = random.uniform(minVal, maxVal)

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


# Inserts a full hidden layer just before the output nodes.
# All connections that previously led into output nodes are removed and
# replaced by: (former sources) → (new layer) → (output nodes).
# Layer size is random in [2, max(2, input_size // 2)].
def mutateAddLayer(network: Network) -> Network:
    output_connections = [conn for conn in network.connections if conn.to_node.type == "output"]

    if not output_connections:
        return network

    layer_size = random.randint(2, max(2, network.input_size // 2))

    # Collect unique source nodes (preserve order via dict)
    source_nodes = list({id(conn.from_node): conn.from_node for conn in output_connections}.values())

    for conn in output_connections[:]:
        disconnect_nodes(network, conn.from_node, conn.to_node)

    insert_pos = len(network.nodes) - network.output_size
    new_nodes = []
    for i in range(layer_size):
        node = build_node("hidden")
        mutateNodeActivationFunction(node)
        network.nodes.insert(insert_pos + i, node)
        new_nodes.append(node)

    scale_in  = math.sqrt(2 / max(len(source_nodes), 1))
    scale_out = math.sqrt(2 / layer_size)

    for src in source_nodes:
        for new_node in new_nodes:
            connectNodes(network, src, new_node, random.uniform(-1, 1) * scale_in)

    for new_node in new_nodes:
        for out_node in network.output_nodes:
            connectNodes(network, new_node, out_node, random.uniform(-1, 1) * scale_out)

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
    MutationType.ADD_NODE        : mutateAddNode,
    MutationType.REMOVE_NODE     : mutateRemoveNode,
    MutationType.ADD_CONN        : mutateAddConnection,
    MutationType.REMOVE_CONN     : mutateRemoveConnection,
    MutationType.MOD_WEIGHT      : mutateConnectionWeight,
    MutationType.MOD_WEIGHT_LARGE: mutateConnectionWeightLarge,
    MutationType.MOD_BIAS        : mutateBias,
    MutationType.MOD_ACTIVATION  : mutateActivationFunction,
    MutationType.SWAP_NODES      : mutateSwapNodes,
    MutationType.ADD_LAYER       : mutateAddLayer,
}

ALL_MUTATIONS = list(mutation.keys())

# Relative probability per mutation type.
# Fine-tuning (weight/bias) dominates; structural changes are rare.
# MOD_WEIGHT (small ±0.1) is the most common; MOD_WEIGHT_LARGE (reset ±2.0) is rare.
# ADD_CONN preferred over ADD_NODE (finer-grained structural change).
# MOD_ACTIVATION is disruptive like structural, so weighted accordingly.
MUTATION_WEIGHTS = {
    MutationType.ADD_NODE        : 1,
    MutationType.REMOVE_NODE     : 1,
    MutationType.ADD_CONN        : 2,
    MutationType.REMOVE_CONN     : 1,
    MutationType.MOD_WEIGHT      : 4,
    MutationType.MOD_WEIGHT_LARGE: 1,
    MutationType.MOD_BIAS        : 3,
    MutationType.MOD_ACTIVATION  : 2,
    MutationType.SWAP_NODES      : 3,
    MutationType.ADD_LAYER       : 1,
}


# -------------------------------------------------------
# Population operations
# -------------------------------------------------------

def selectMutationMethod(possible_mutations) :
    weights = [MUTATION_WEIGHTS[m] for m in possible_mutations]
    return random.choices(possible_mutations, weights=weights, k=1)[0]


def mutateNetwork(network: Network, method: MutationType) -> Network :
    if method not in mutation :
        raise ValueError(f"Unknown mutation method: {method}")

    return mutation[method](network)


def performMutation(genome: Network, mutation_amount=None, possible_mutations=None) -> Network :
    mutation_amount    = mutation_amount    if mutation_amount    is not None else config.mutation_amount
    possible_mutations = possible_mutations if possible_mutations is not None else ALL_MUTATIONS

    # Deep copy so the original (e.g. an elitist) is never modified in place
    target = clone_network(genome)

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