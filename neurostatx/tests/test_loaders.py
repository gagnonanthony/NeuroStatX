import unittest
import networkx as nx
import numpy as np
import pandas as pd
import tempfile
import os

from neurostatx.io.loader import DatasetLoader, GraphLoader


class TestDatasetLoader(unittest.TestCase):
    def setUp(self):
        self.loader = DatasetLoader()
        self.sample_data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9]
        })
        self.sample_dict = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
        self.sample_array = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])

    def test_load_data(self):
        self.loader.import_data(self.sample_data)
        self.assertEqual(self.loader.nb_subjects, 3)
        self.assertEqual(self.loader.nb_variables, 3)

    def test_import_data_from_dataframe(self):
        self.loader.import_data(self.sample_data)
        pd.testing.assert_frame_equal(self.loader.data, self.sample_data)

    def test_import_data_from_dict(self):
        self.loader.import_data(self.sample_dict)
        expected_df = pd.DataFrame(self.sample_dict)
        pd.testing.assert_frame_equal(self.loader.data, expected_df)

    def test_import_data_from_array(self):
        self.loader.import_data(self.sample_array, columns=['A', 'B', 'C'])
        pd.testing.assert_frame_equal(self.loader.data, self.sample_data)

    def test_get_descriptive_columns(self):
        self.loader.import_data(self.sample_data)
        result = self.loader.get_descriptive_columns([0, 2])
        expected = self.sample_data[['A', 'C']]
        pd.testing.assert_frame_equal(result, expected)

    def test_drop_columns_int(self):
        self.loader.import_data(self.sample_data.copy())
        self.loader.drop_columns([1])
        expected = self.sample_data.drop(columns=['B'])
        pd.testing.assert_frame_equal(self.loader.data, expected)

    def test_drop_columns_str(self):
        self.loader.import_data(self.sample_data.copy())
        self.loader.drop_columns(['B'])
        expected = self.sample_data.drop(columns=['B'])
        pd.testing.assert_frame_equal(self.loader.data, expected)

    def test_join_right(self):
        self.loader.import_data(self.sample_data[['A', 'B']])
        df_to_join = self.sample_data[['C']]
        self.loader.join(df_to_join, left=False)
        pd.testing.assert_frame_equal(self.loader.data, self.sample_data)

    def test_join_left(self):
        self.loader.import_data(self.sample_data[['B', 'C']])
        df_to_join = self.sample_data[['A']]
        self.loader.join(df_to_join, left=True)
        expected = pd.concat([df_to_join, self.sample_data[['B', 'C']]],
                             axis=1)
        pd.testing.assert_frame_equal(self.loader.data, expected)

    def test_reset_index(self):
        self.loader.import_data(self.sample_data)
        self.loader.data.index = [10, 11, 12]
        self.loader.reset_index()
        self.assertTrue((self.loader.data.index == [0, 1, 2]).all())

    def test_set_type(self):
        self.loader.import_data(self.sample_data)
        self.loader.set_type(str, columns=[0, 1])
        self.assertTrue(self.loader.data['A'].dtype == 'object')
        self.assertTrue(self.loader.data['B'].dtype == 'object')

    def test_transpose(self):
        self.loader.import_data(self.sample_data)
        self.loader.transpose()
        pd.testing.assert_frame_equal(self.loader.data, self.sample_data.T)

    def test_get_metadata(self):
        self.loader.import_data(self.sample_data)
        metadata = self.loader.get_metadata()
        self.assertEqual(metadata['nb_subjects'], 3)
        self.assertEqual(metadata['nb_variables'], 3)

    def test_get_data(self):
        self.loader.import_data(self.sample_data)
        pd.testing.assert_frame_equal(self.loader.get_data(), self.sample_data)

    def test_save_data(self):
        self.loader.import_data(self.sample_data)
        self.loader.save_data("test_output.csv")
        loaded_df = pd.read_csv("test_output.csv", index_col=0)
        pd.testing.assert_frame_equal(loaded_df, self.sample_data)
        os.remove("test_output.csv")

    def test_custom_function(self):
        self.loader.import_data(self.sample_data)
        result = self.loader.custom_function(lambda df: df * 2)
        expected = self.sample_data * 2
        pd.testing.assert_frame_equal(result, expected)


class TestGraphLoader(unittest.TestCase):

    def setUp(self):
        """Initialize the GraphLoader before each test."""
        self.loader = GraphLoader()
        self.test_graph = nx.Graph()
        self.test_graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'a')])
        self.test_df = pd.DataFrame({'source': ['a', 'b', 'c'],
                                     'target': ['b', 'c', 'a'],
                                     'weight': [1, 2, 3]})
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        """Clean up temporary files and objects after tests."""
        self.temp_dir.cleanup()

    def test_load_graph_valid_formats(self):
        """Test loading graphs from supported formats."""
        for ext in ['gml', 'graphml', 'gexf']:
            file_path = os.path.join(self.temp_dir.name, f'test.{ext}')
            nx.write_gml(self.test_graph, file_path) if ext == 'gml' else (
                nx.write_graphml(self.test_graph, file_path)
                if ext == 'graphml' else
                nx.write_gexf(self.test_graph, file_path))
            self.loader.load_graph(file_path)
            self.assertIsInstance(self.loader.graph, nx.Graph)
            self.assertEqual(self.loader.nb_nodes, 3)
            self.assertEqual(self.loader.nb_edges, 3)

    def test_load_graph_invalid_format(self):
        """Test loading an unsupported file format."""
        with self.assertRaises(ValueError):
            self.loader.load_graph("invalid.txt")

    def test_build_graph(self):
        """Test building a graph from a pandas DataFrame."""
        self.loader.build_graph(self.test_df)
        self.assertEqual(self.loader.nb_nodes, 3)
        self.assertEqual(self.loader.nb_edges, 3)

    def test_build_graph_invalid_input(self):
        """Test error handling for non-DataFrame input."""
        with self.assertRaises(ValueError):
            self.loader.build_graph("invalid_data")

    def test_add_node_attribute(self):
        """Test adding attributes to nodes."""
        self.loader.build_graph(self.test_df)
        attributes = {'a': {"attr": "A"}, 'b': {"attr": "B"},
                      'c': {"attr": "C"}}
        self.loader.add_node_attribute(attributes)
        self.assertEqual(self.loader.graph.nodes['a']["attr"], "A")

    def test_add_edge_attribute(self):
        """Test adding attributes to edges."""
        self.loader.build_graph(self.test_df)
        attributes = {('a', 'b'): {"weight": 5}, ('b', 'c'): {"weight": 10}}
        self.loader.add_edge_attribute(attributes)
        self.assertEqual(self.loader.graph.edges['a', 'b']["weight"], 5)

    def test_get_metadata(self):
        """Test retrieving graph metadata."""
        self.loader.build_graph(self.test_df)
        metadata = self.loader.get_metadata()
        self.assertEqual(metadata, {"nb_nodes": 3, "nb_edges": 3})

    def test_get_graph(self):
        """Test retrieving the loaded graph."""
        self.loader.build_graph(self.test_df)
        graph = self.loader.get_graph()
        self.assertIsInstance(graph, nx.Graph)

    def test_save_graph(self):
        """Test saving the graph in supported formats."""
        self.loader.build_graph(self.test_df)
        for ext in ['gml', 'graphml', 'gexf']:
            file_path = os.path.join(self.temp_dir.name, f'test.{ext}')
            self.loader.save_graph(file_path)
            self.assertTrue(os.path.exists(file_path))

    def test_layout(self):
        """Test generating a layout for the graph."""
        self.loader.build_graph(self.test_df,
                                "source",
                                "target",
                                edge_attr="weight")
        self.loader.layout(weight="weight")
        pos = self.loader.fetch_attributes_df(["pos"])
        self.assertEqual(len(pos.data), 2)
        self.assertEqual(pos.data.shape[1], 1)

    def test_custom_function(self):
        """Test applying a custom function to the graph."""
        self.loader.build_graph(self.test_df)

        def custom_func(graph):
            return graph.number_of_nodes() + graph.number_of_edges()

        result = self.loader.custom_function(custom_func)
        self.assertEqual(result, 6)

    def test_custom_function_invalid(self):
        """Test passing a non-callable object to custom_function."""
        self.loader.build_graph(self.test_df)
        with self.assertRaises(ValueError):
            self.loader.custom_function("not a function")


if __name__ == '__main__':
    unittest.main()
