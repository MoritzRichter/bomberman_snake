import random
import math
from constants import sigmoid

#---------------------------------------------------------------------
#A NEAT-Style neural network consisting of a network objects with node objects and connection objects

#---------------------------------------------------------------------

class Node:
    """One neuron in the network."""

    def __init__(self, node_type, bias=None, squash=None):
        self.type  = node_type   # 'input', 'hidden', or 'output'

        # Bias shifts the activation threshold of this node.
        # Initialised to a small random value so networks start with variety.
        self.bias  = bias   #if bias   is not None else random.uniform(-0.1, 0.1)

        # The activation function that squashes this node's weighted sum.
        # Defaults to sigmoid.
        self.squash = squash #if squash is not None else sigmoid

        self.activation = 0.0  # output value produced by the last forward pass
        self.state      = 0.0  # raw weighted sum before squashing

        # Lists of Connection objects — one list for incoming, one for outgoing.
        self.connections_in  = []
        self.connections_out = []

        # Integer position in network.nodes; set by reindex_network().
        # Used during crossover to match nodes between two parents.
        self.index = 0



class Connection:
    """One directed edge between two nodes, carrying a weight."""

    def __init__(self, from_node, to_node, weight=None):
        self.from_node = from_node
        self.to_node   = to_node
        # Weight scales how much the from_node's activation influences to_node.
        self.weight = weight if weight is not None else random.uniform(-0.1, 0.1)



class Network:
    """The full neural network: a list of nodes and a list of connections."""

    def __init__(self, input_size, output_size):
        self.input_size  = input_size
        self.output_size = output_size
        self.nodes       = []   # all nodes in activation order
        self.connections = []   # all connections (edges)
        self.input_nodes  = []  # subset: only input nodes
        self.output_nodes = []  # subset: only output nodes
        self.score = None        # fitness score assigned during a game



# -------- Network Initiallization ---------------------------------------

def buildNetwork(inputSize, outputSize) :
    network = Network(inputSize, outputSize)

    createInitialNodes(network)
    connectInitialNodes(network)

    return network

def createInitialNodes(network: Network) :
    for i in range (network.input_size) :
        node = build_node("input")
        network.nodes.append(node)
        network.input_nodes.append(node)

    for i in range (network.output_size) :
        node = build_node("output")
        network.nodes.append(node)
        network.output_nodes.append(node)


def connectInitialNodes(network: Network) :
    for i in range (network.input_size) :
        for j in range (network.input_size, (network.output_size + network.input_size), 1) :
            weight = random.uniform(-1, 1) * math.sqrt(2 / network.input_size)
            connectNodes(network, network.nodes[i], network.nodes[j], weight)




# --- Network Building Operations ------------------------------------------------

#Activation function value to experiment with later
def build_node(type="hidden", bias=None, squash=sigmoid):
    if bias is None:
        bias = random.random() * 0.2 - 0.1
    node = Node(type, bias, squash)
    return node
    
def buildConnection(fromNode: Node, toNode: Node, weight) :
    connection = Connection(fromNode, toNode, weight)
    return connection 

def connectNodes(network: Network, from_node: Node, to_node: Node, weight):
    connection = buildConnection(from_node, to_node, weight)
    network.connections.append(connection)
    from_node.connections_out.append(connection)
    to_node.connections_in.append(connection)
    return connection



def reindex_network(network: Network) :
    for i, node in enumerate(network.nodes) :
        node.index = i


def disconnect_nodes(network: Network, from_node: Node, to_node: Node) :
    for conn in network.connections[:] :
        if conn.from_node is from_node and conn.to_node is to_node :
            network.connections.remove(conn)
            from_node.connections_out.remove(conn)
            to_node.connections_in.remove(conn)
            return network

    return network


def has_connection(from_node: Node, to_node: Node) -> bool :
    for conn in from_node.connections_out :
        if conn.to_node is to_node :
            return True

    return False


def remove_node_from_network(network: Network, node: Node) -> Network :
    for conn in node.connections_in[:] :
        disconnect_nodes(network, conn.from_node, conn.to_node)

    for conn in node.connections_out[:] :
        disconnect_nodes(network, conn.from_node, conn.to_node)

    network.nodes.remove(node)

    return network





# ---------- Forward Pass ---------------------------------------------------

def activate_node(node: Node, input_value=None):
    if input_value != None:
        node.activation = input_value
        return node.activation

    state = node.bias
    #Collects all inputs multiplied by the connections weight
    for connection in node.connections_in:
        state += connection.from_node.activation * connection.weight
    node.state = state

    #Activation function
    node.activation = node.squash(node.state)

    return node.activation


def activate_network(network: Network, input_values):
    output = []
    for i, node in enumerate(network.nodes):
        if node.type == "input":
            activate_node(node, input_values[i])
        elif node.type == "output":
            output.append(activate_node(node))
        else:
            activate_node(node)
    return output
