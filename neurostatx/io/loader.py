import networkx as nx

from neurostatx.io.utils import load_df_in_any_format
from neurostatx.network.viz import visualize_network


class DatasetLoader:
    def __init__(self):
        self.data = None
        self.nb_subjects = None
        self.nb_variables = None

    def load_data(self, file, **kwargs):
        """
        Load tabular data in any format (.txt, .csv, .xlsx).

        Parameters
        ----------
        file: str
            Input file to load.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        df: pandas.DataFrame
        """
        self.data = load_df_in_any_format(file, **kwargs)
        self.nb_subjects, self.nb_variables = self.data.shape
        return self.data

    def get_metadata(self):
        """
        Get metadata of the loaded data.

        Returns
        -------
        metadata: dict
            Dictionary containing the number of subjects and variables.
        """
        return {
            "nb_subjects": self.nb_subjects,
            "nb_variables": self.nb_variables
        }

    def get_data(self):
        """
        Get the loaded data.

        Returns
        -------
        data: pandas.DataFrame
            The loaded data.
        """
        return self.data


class GraphLoader:
    def __init__(self):
        self.graph = None
        self.nb_nodes = None
        self.nb_edges = None

    def load_graph(self, file, **kwargs):
        """
        Load graph data.

        Parameters
        ----------
        file: str
            Input file to load.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        graph: networkx.Graph
        """
        if file.endswith(".gml"):
            self.graph = nx.read_gml(file, **kwargs)
        elif file.endswith(".graphml"):
            self.graph = nx.read_graphml(file, **kwargs)
        elif file.endswith(".gexf"):
            self.graph = nx.read_gexf(file, **kwargs)
        else:
            raise ValueError("File format not supported. Currently supported "
                             "formats are .gml, .graphml, .gexf.")

        self.nb_nodes = self.graph.number_of_nodes()
        self.nb_edges = self.graph.number_of_edges()
        return self.graph

    def add_node_attribute(self, attributes):
        """
        Add a node attribute to the graph.

        Parameters
        ----------
        attributes: dict
            Dictionary containing the values of the attribute for each node.
        """
        nx.set_node_attributes(self.graph, attributes)

    def add_edge_attribute(self, attributes):
        """
        Add an edge attribute to the graph.

        Parameters
        ----------
        attribute: dict
            Dictionary containing the values of the attribute for each edge.
        """
        nx.set_edge_attributes(self.graph, attributes)

    def visualize(self, output,
                  weight="weight",
                  centroids_labelling=True,
                  subjects_labelling=False,
                  centroid_node_shape=500,
                  centroid_alpha=1,
                  centroid_node_color="white",
                  centroid_edge_color="black",
                  subject_node_shape=5,
                  subject_alpha=0.3,
                  subject_node_color="black",
                  subject_edge_color=None,
                  colormap="plasma",
                  title="Graph Network",
                  legend_title="Membership values"):
        """
        Visualize the graph network.

        Parameters
        ----------
        output: str
            Output file name.
        weight: str, optional
            Edge attribute to use as weights for the edges.
        centroids_labelling: bool, optional
            If true, label the centroid nodes.
        subjects_labelling: bool, optional
            If true, label the subject nodes.
        centroid_node_shape: int, optional
            Shape of the centroid nodes.
        centroid_alpha: float, optional
            Alpha value of the centroid nodes.
        centroid_node_color: str, optional
            Color of the centroid nodes.
        centroid_edge_color: str, optional
            Color of the centroid edges.
        subject_node_shape: int, optional
            Shape of the subject nodes.
        subject_alpha: float, optional
            Alpha value of the subject nodes.
        subject_node_color: str, optional
            Color of the subject nodes.
        subject_edge_color: str, optional
            Color of the subject edges.
        colormap: str, optional
            Colormap to use for the edges.
        title: str, optional
            Title of the plot.
        legend_title: str, optional
            Title of the legend.
        """
        visualize_network(self.graph,
                          output,
                          weight=weight,
                          centroids_labelling=centroids_labelling,
                          subjects_labelling=subjects_labelling,
                          centroid_node_shape=centroid_node_shape,
                          centroid_alpha=centroid_alpha,
                          centroid_node_color=centroid_node_color,
                          centroid_edge_color=centroid_edge_color,
                          subject_node_shape=subject_node_shape,
                          subject_alpha=subject_alpha,
                          subject_node_color=subject_node_color,
                          subject_edge_color=subject_edge_color,
                          colormap=colormap,
                          title=title,
                          legend_title=legend_title)

    def get_metadata(self):
        """
        Get metadata of the loaded graph.

        Returns
        -------
        metadata: dict
            Dictionary containing the number of nodes and edges.
        """
        return {
            "nb_nodes": self.nb_nodes,
            "nb_edges": self.nb_edges
        }

    def get_graph(self):
        """
        Get the loaded graph.

        Returns
        -------
        graph: networkx.Graph
            The loaded graph.
        """
        return self.graph

    def save_graph(self, file, **kwargs):
        """
        Save the graph data.

        Parameters
        ----------
        file: str
            Output file to save.
        **kwargs
            Additional keyword arguments.
        """
        if file.endswith(".gml"):
            nx.write_gml(self.graph, file, **kwargs)
        elif file.endswith(".graphml"):
            nx.write_graphml(self.graph, file, **kwargs)
        elif file.endswith(".gexf"):
            nx.write_gexf(self.graph, file, **kwargs)
        else:
            raise ValueError("File format not supported. Currently supported "
                             "formats are .gml, .graphml, .gexf.")
