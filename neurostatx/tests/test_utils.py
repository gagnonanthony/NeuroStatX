import unittest
import pandas as pd
import networkx as nx

from neurostatx.network.utils import (get_nodes_and_edges,
                                      filter_node_centroids,
                                      filter_node_subjects,
                                      extract_subject_percentile,
                                      construct_attributes_dict,
                                      fetch_attributes_df,
                                      fetch_edge_data)


class TestFunctions(unittest.TestCase):

    def setUp(self):
        # Generate sample DataFrame
        data = {
            'Subject': ['A', 'B', 'C'],
            'c1': [0.05, 0.1, 0.7],
            'c2': [0.9, 0.1, 0.2],
            'c3': [0.05, 0.8, 0.1],
        }
        self.df = pd.DataFrame(data)

    def test_get_nodes_and_edges(self):
        result_df, subject_list, center_list = get_nodes_and_edges(self.df)
        # Check if returned DataFrame has the correct columns
        self.assertEqual(list(result_df.columns), ['node1', 'node2',
                                                   'membership'])
        # Check if subject_list and center_list are generated correctly
        self.assertEqual(list(subject_list), ['A', 'B', 'C'])
        self.assertEqual(center_list, ['c1', 'c2', 'c3'])

    def test_filter_node_centroids(self):
        self.assertTrue(filter_node_centroids('c1'))
        self.assertFalse(filter_node_centroids('A'))

    def test_filter_node_subjects(self):
        self.assertFalse(filter_node_subjects('c1'))
        self.assertTrue(filter_node_subjects('A'))

    def test_extract_subject_percentile(self):
        mat = self.df.drop('Subject', axis=1).values.T
        label_dict = extract_subject_percentile(mat, 40)
        self.assertEqual(label_dict['c1'].tolist(), [0, 0, 1])
        self.assertEqual(label_dict['c2'].tolist(), [2, 0, 0])

    def test_construct_attributes_dict(self):
        labels = ['c1']
        attributes_dict = construct_attributes_dict(self.df, labels, 'Subject')
        self.assertEqual(attributes_dict, {'A': {'c1': 0.05}, 'B': {'c1': 0.1},
                                           'C': {'c1': 0.7}})

    def test_fetch_attributes_df(self):
        G = nx.Graph()
        G.add_node('A', c1=0.1, c2=0.2)
        G.add_node('B', c1=0.3, c2=0.4)
        G.add_node('C', c1=0.5, c2=0.6)
        df = fetch_attributes_df(G, attributes=['c1'])
        self.assertEqual(df.columns.tolist(), ['c1'])
        self.assertEqual(df.index.tolist(), ['A', 'B', 'C'])
        self.assertEqual(df['c1'].tolist(), [0.1, 0.3, 0.5])

    def test_fetch_edge_data(self):
        result_df, subject_list, center_list = get_nodes_and_edges(self.df)
        G = nx.Graph()
        G = nx.from_pandas_edgelist(result_df, "node1", "node2",
                                    edge_attr="membership")
        df = fetch_edge_data(G)
        expected = pd.DataFrame([[0.05, 0.9, 0.05],
                                 [0.1, 0.1, 0.8],
                                 [0.7, 0.2, 0.1]],
                                index=['A', 'B', 'C'],
                                columns=['Cluster 1', 'Cluster 2',
                                         'Cluster 3'])
        pd.testing.assert_frame_equal(df, expected)


if __name__ == '__main__':
    unittest.main()
