"""Tests für mutateAddLayer in evolution/evolution.py"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'evolution'))

import unittest
from network import buildNetwork, activate_network, Network
from evolution import mutateAddLayer, mutateAddNode, performMutation, MutationType


# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def fresh_net(input_size=6, output_size=2) -> Network:
    return buildNetwork(input_size, output_size)

def count_hidden(net: Network) -> int:
    return sum(1 for n in net.nodes if n.type == "hidden")

def all_conn_refs_valid(net: Network) -> bool:
    """Alle Connections müssen auf Nodes zeigen, die wirklich in net.nodes sind."""
    ids = {id(n) for n in net.nodes}
    return all(id(c.from_node) in ids and id(c.to_node) in ids for c in net.connections)

def outputs_have_inputs(net: Network) -> bool:
    """Jeder Output-Node muss mindestens eine eingehende Verbindung haben."""
    return all(len(n.connections_in) > 0 for n in net.output_nodes)

def nodes_in_correct_order(net: Network) -> bool:
    """Inputs zuerst, dann Hidden, dann Outputs — keine Durchmischung."""
    types = [n.type for n in net.nodes]
    seen_hidden = seen_output = False
    for t in types:
        if t == "hidden":
            if seen_output:
                return False
            seen_hidden = True
        elif t == "output":
            seen_output = True
        elif t == "input":
            if seen_hidden or seen_output:
                return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Struktur nach der Mutation
# ─────────────────────────────────────────────────────────────────────────────

class TestAddLayerStructure(unittest.TestCase):

    def test_hidden_count_increases(self):
        """Nach ADD_LAYER muss es mehr Hidden-Nodes geben als davor."""
        net = fresh_net()
        before = count_hidden(net)
        mutateAddLayer(net)
        self.assertGreater(count_hidden(net), before)

    def test_layer_size_in_valid_range(self):
        """Neue Nodes: mindestens 2, maximal max(2, input_size // 2)."""
        input_size = 6
        net = fresh_net(input_size=input_size)
        before = count_hidden(net)
        mutateAddLayer(net)
        added = count_hidden(net) - before
        self.assertGreaterEqual(added, 2)
        self.assertLessEqual(added, max(2, input_size // 2))

    def test_new_nodes_are_hidden_type(self):
        """Alle neu eingefügten Nodes müssen vom Typ 'hidden' sein."""
        net = fresh_net()
        before_ids = {id(n) for n in net.nodes if n.type == "hidden"}
        mutateAddLayer(net)
        new_nodes = [n for n in net.nodes if n.type == "hidden" and id(n) not in before_ids]
        self.assertTrue(all(n.type == "hidden" for n in new_nodes))

    def test_new_nodes_inserted_before_outputs(self):
        """Hidden-Nodes dürfen nie nach Output-Nodes in net.nodes erscheinen."""
        net = fresh_net()
        mutateAddLayer(net)
        self.assertTrue(nodes_in_correct_order(net))

    def test_outputs_still_have_incoming_connections(self):
        """Kein Output-Node darf nach der Mutation isoliert sein."""
        net = fresh_net()
        mutateAddLayer(net)
        self.assertTrue(outputs_have_inputs(net))

    def test_all_connection_refs_point_to_existing_nodes(self):
        """Keine Connection darf auf einen Node zeigen, der nicht in net.nodes ist."""
        net = fresh_net()
        mutateAddLayer(net)
        self.assertTrue(all_conn_refs_valid(net))

    def test_new_nodes_connected_to_all_outputs(self):
        """Jeder neue Hidden-Node muss eine Verbindung zu jedem Output haben."""
        net = fresh_net()
        before_ids = {id(n) for n in net.nodes if n.type == "hidden"}
        mutateAddLayer(net)
        new_nodes = [n for n in net.nodes if n.type == "hidden" and id(n) not in before_ids]
        output_ids = {id(n) for n in net.output_nodes}
        for new_node in new_nodes:
            targets = {id(c.to_node) for c in new_node.connections_out}
            self.assertTrue(output_ids.issubset(targets),
                            "Neuer Hidden-Node hat keine Verbindung zu allen Outputs")

    def test_total_node_count_correct(self):
        """net.nodes muss genau input + output + alle hidden enthalten."""
        net = fresh_net(input_size=6, output_size=2)
        mutateAddLayer(net)
        self.assertEqual(len(net.nodes), net.input_size + net.output_size + count_hidden(net))


# ─────────────────────────────────────────────────────────────────────────────
# Forward Pass
# ─────────────────────────────────────────────────────────────────────────────

class TestAddLayerForwardPass(unittest.TestCase):

    def test_forward_pass_does_not_crash(self):
        net = fresh_net(input_size=6, output_size=2)
        mutateAddLayer(net)
        result = activate_network(net, [0.5] * net.input_size)
        self.assertIsNotNone(result)

    def test_forward_pass_output_count(self):
        """Forward Pass muss genau output_size Werte zurückgeben."""
        net = fresh_net(input_size=6, output_size=2)
        mutateAddLayer(net)
        result = activate_network(net, [0.5] * net.input_size)
        self.assertEqual(len(result), net.output_size)

    def test_output_values_are_finite(self):
        """Kein Output darf NaN oder inf sein."""
        import math
        net = fresh_net(input_size=6, output_size=2)
        mutateAddLayer(net)
        for val in activate_network(net, [1.0] * net.input_size):
            self.assertTrue(math.isfinite(val), f"Output ist nicht finite: {val}")


# ─────────────────────────────────────────────────────────────────────────────
# Mehrfache Anwendung & Interaktion mit anderen Mutationen
# ─────────────────────────────────────────────────────────────────────────────

class TestAddLayerRepeated(unittest.TestCase):

    def test_two_layers_in_sequence(self):
        """ADD_LAYER zweimal hintereinander produziert ein valides Netz."""
        net = fresh_net()
        mutateAddLayer(net)
        mutateAddLayer(net)
        self.assertTrue(all_conn_refs_valid(net))
        self.assertTrue(outputs_have_inputs(net))
        self.assertTrue(nodes_in_correct_order(net))

    def test_add_node_after_add_layer(self):
        """ADD_NODE nach ADD_LAYER darf das Netz nicht korrumpieren."""
        net = fresh_net()
        mutateAddLayer(net)
        mutateAddNode(net)
        self.assertTrue(all_conn_refs_valid(net))
        self.assertTrue(outputs_have_inputs(net))

    def test_forward_pass_after_two_layers(self):
        """Forward Pass funktioniert nach zwei ADD_LAYER-Mutationen."""
        import math
        net = fresh_net(input_size=6, output_size=2)
        mutateAddLayer(net)
        mutateAddLayer(net)
        result = activate_network(net, [0.5] * net.input_size)
        self.assertEqual(len(result), net.output_size)
        for val in result:
            self.assertTrue(math.isfinite(val))

    def test_via_performMutation(self):
        """ADD_LAYER über den normalen Mutations-Dispatcher ausführbar."""
        net = fresh_net()
        mutated = performMutation(net, mutation_amount=1,
                                  possible_mutations=[MutationType.ADD_LAYER])
        self.assertIsNotNone(mutated)
        self.assertTrue(all_conn_refs_valid(mutated))
        self.assertTrue(outputs_have_inputs(mutated))


# ─────────────────────────────────────────────────────────────────────────────
# Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestAddLayerEdgeCases(unittest.TestCase):

    def test_minimal_network_input_size_2(self):
        """Netz mit input_size=2 → layer_size mindestens 2."""
        net = fresh_net(input_size=2, output_size=2)
        before = count_hidden(net)
        mutateAddLayer(net)
        self.assertGreaterEqual(count_hidden(net) - before, 2)

    def test_large_input_network(self):
        """Netz mit 17 Inputs (realer Fall: full-Profil) bleibt valide."""
        net = fresh_net(input_size=17, output_size=2)
        mutateAddLayer(net)
        self.assertTrue(all_conn_refs_valid(net))
        self.assertTrue(outputs_have_inputs(net))
        self.assertEqual(len(activate_network(net, [0.5] * 17)), 2)

    def test_network_with_existing_hidden_nodes(self):
        """Auch wenn bereits Hidden-Nodes existieren, bleibt das Netz nach ADD_LAYER valide."""
        net = fresh_net(input_size=6, output_size=2)
        mutateAddNode(net)
        mutateAddNode(net)
        hidden_before = count_hidden(net)
        mutateAddLayer(net)
        self.assertGreater(count_hidden(net), hidden_before)
        self.assertTrue(all_conn_refs_valid(net))
        self.assertTrue(outputs_have_inputs(net))
        self.assertTrue(nodes_in_correct_order(net))


if __name__ == '__main__':
    unittest.main(verbosity=2)
