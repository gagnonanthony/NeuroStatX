import unittest
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from CCPM.network.viz import (creating_node_colormap,
                              create_cmap_from_list,
                              set_nodes_position)


class TestNodeColormapFunctions(unittest.TestCase):

    def test_creating_node_colormap(self):
        percentile_dict = {'cluster1': [0, 1, 0, 1, 0],
                           'cluster2': [0, 0, 1, 0, 0]}
        expected_output = ['black', np.array(plt.cm.tab10(1)),
                           np.array(plt.cm.tab10(1)),
                           np.array(plt.cm.tab10(1)), 'black']
        for i in range(0, len(expected_output)):
            if isinstance(expected_output[i], np.ndarray):
                self.assertEqual(
                    expected_output[i].all(),
                    creating_node_colormap(percentile_dict)[i].all())
            else:
                self.assertEqual(expected_output[i],
                                 creating_node_colormap(percentile_dict)[i])

    def test_create_cmap_from_list_integer_array(self):
        integer_array = np.array([0, 1, 2, 3, 4])
        expected_output = ['black',
                           np.array(plt.cm.tab10(1)),
                           np.array(plt.cm.tab10(2)),
                           np.array(plt.cm.tab10(3)),
                           np.array(plt.cm.tab10(4))]
        for i in range(0, len(expected_output)):
            if isinstance(expected_output[i], np.ndarray):
                self.assertEqual(
                    expected_output[i].all(),
                    create_cmap_from_list(integer_array)[i].all())
            else:
                self.assertEqual(expected_output[i],
                                 create_cmap_from_list(integer_array)[i])

    def test_create_cmap_from_list_float_array(self):
        float_array = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        expected_output = [np.array(plt.cm.plasma(0)),
                           np.array(plt.cm.plasma(1)),
                           np.array(plt.cm.plasma(2)),
                           np.array(plt.cm.plasma(3)),
                           np.array(plt.cm.plasma(4))]
        for i in range(0, len(expected_output)):
            if isinstance(expected_output[i], np.ndarray):
                self.assertEqual(
                    expected_output[i].all(),
                    create_cmap_from_list(float_array)[i].all())
            else:
                self.assertEqual(expected_output[i],
                                 create_cmap_from_list(float_array)[i])


class TestSetNodesPosition(unittest.TestCase):
    def test_set_nodes_position(self):
        # Create a graph
        G = nx.Graph()
        G.add_nodes_from([1, 2, 3])

        # Define positions
        pos = {1: (0, 0), 2: (1, 1), 3: (2, 2)}

        # Call the function
        set_nodes_position(G, pos)

        # Check if node positions are set correctly
        self.assertTrue(all("pos" in G.nodes[node] for node in G.nodes))
        for node, position in pos.items():
            self.assertAlmostEqual(G.nodes[node]["pos"][0],
                                   float(position[0]))
            self.assertAlmostEqual(G.nodes[node]["pos"][1],
                                   float(position[1]))


if __name__ == '__main__':
    unittest.main()
