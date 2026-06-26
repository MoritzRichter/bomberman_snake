import random
import math
from network import (
    Network, build_node, buildNetwork,
    connectNodes, disconnect_nodes, has_connection,
    remove_node_from_network, reindex_network,
)

# --------------------------------------------------------

#Crossover

#---------------------------------------------------------

def crossover(network1: Network, network2: Network, equal) :
   
    if network1.score is None or network2.score is None:
        raise ValueError("Can't crossover networks without scores!")
    
    is_equal = equal

    #Fitter parent always 1
    if network2.score > network1.score:
        network1, network2 = network2, network1

    if network1.score == network2.score:
        is_equal = True


    #Create baseline Child based on Parents' size
    empty_offspring_foundation = createOffspringFoundation(network1)

    offspring_size = determineOffspringSize(network1, network2, is_equal)

    empty_offspring_foundation.size = offspring_size

    #Re-assign indexes of nodes
    reindex_network(network1)
    reindex_network(network2)

    #Fill child with nodes
    offspring_foundation_with_nodes = assignNodesFromParents(
        empty_offspring_foundation, network1, network2
    )

    #Create a connection-merge from parents
    n1_conns = extractConnectionGenes(network1)
    n2_conns = extractConnectionGenes(network2)

    connections = mergeConnectionGenes(n1_conns, n2_conns, is_equal)

    #Fill child with connections
    offspring = integrateConnectionsIntoOffspringFoundation(
        offspring_foundation_with_nodes, connections
    )

    return offspring
    


# --- Helper-Functions --------------------------------------------------------------

#Creates empty child with input/output size equal to fittest parent
def createOffspringFoundation(network: Network) :
    offspringFoundation = buildNetwork(network.input_size, network.output_size)
    offspringFoundation.connections = []
    offspringFoundation.nodes = []
    offspringFoundation.input_nodes = set()
    offspringFoundation.output_nodes = set()

    return offspringFoundation


#Determines amount of hidden nodes for child. Takes size of fitter parent. 
#If both equally fit, random inbetween
def determineOffspringSize(network1: Network, network2: Network, equal) :
    if network1.input_size != network2.input_size or network1.output_size != network2.output_size :
        raise Exception("Networks don't have the same input/output size!")

    if equal :
        maxNodes = max(len(network1.nodes), len(network2.nodes))
        minNodes = min(len(network1.nodes), len(network2.nodes))

        return random.randint(minNodes, maxNodes)

    return len(network1.nodes)


#Goes through all connection genes and processed into a merger
def mergeConnectionGenes(n1conns, n2conns, equal) :
    mergedConnections = []
    keys1 = list(n1conns.keys())
    keys2 = list(n2conns.keys())

    # Merge common and disjoint genes
    for key in keys1 :
        if key in n2conns :
            mergedConnections.append(n1conns[key] if random.random() >= 0.5 else n2conns[key])

            del n2conns[key]  # Mark as processed
        else :
            mergedConnections.append(n1conns[key])

    if equal :
        for key in keys2 :
            if key in n2conns :
                mergedConnections.append(n2conns[key])

    return mergedConnections


#Apply the connection merger to the body of the child
def integrateConnectionsIntoOffspringFoundation(offspringFoundation: Network, connections) :
    size = offspringFoundation.size
    nodes = offspringFoundation.nodes

    for i in range (len(connections)) :
        connData = connections[i]
        if connData["to"] < size and connData["from"] < size :
            fromNode = nodes[connData["from"]]
            toNode = nodes[connData["to"]]
            connectNodes(offspringFoundation, fromNode, toNode, connData["weight"])

    return offspringFoundation


# Depending on the size of the child create new nodes based on the parents' nodes
def safe_get(lst, index):
    return lst[index] if 0 <= index < len(lst) else None

def assignNodesFromParents(offspring: Network, network1: Network, network2: Network) :
    outputSize = network1.output_size
    size = offspring.size

    for i in range (size) :
        node = None

        if i < size - outputSize :
            # Non-output nodes
            randomValue = random.random()
            node = safe_get(network1.nodes, i) if randomValue >= 0.5 else safe_get(network2.nodes, i)
            other = safe_get(network1.nodes, i) if randomValue < 0.5 else safe_get(network2.nodes, i)

            if node is None or node.type == "output" :
                node = other
        else :
            # Output nodes
            parent1Index = len(network1.nodes) + i - size
            parent2Index = len(network2.nodes) + i - size
            node = safe_get(network1.nodes, parent1Index) if random.random() >= 0.5 else safe_get(network2.nodes, parent2Index)

        newNode = build_node(node.type, node.bias, node.squash)

        if newNode.type == "input" :
            offspring.input_nodes.add(newNode)
        elif newNode.type == "output" :
            offspring.output_nodes.add(newNode)

        offspring.nodes.append(newNode)

    outputCount = len([x for x in offspring.nodes if x.type == "output"])
    if outputCount > network1.output_size :
        raise Exception("The amount of output nodes in the offspring exceeds the amount in the parents!")

    return offspring


#Extract Connection information in a more processable manner
def extractConnectionGenes(network: Network) :
    return addConnectionData(network.connections)

def addConnectionData(connections) :
    connData = {}

    for conn in connections :
        data = {
            "weight": conn.weight,
            "from": conn.from_node.index,
            "to": conn.to_node.index
        }
        connData[getInnovationId(data["from"], data["to"])] = data

    return connData

def getInnovationId(a, b) :
    return (1 / 2) * (a + b) * (a + b + 1) + b


