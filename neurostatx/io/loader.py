import os

from detect_delimiter import detect
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import pandas as pd

from neurostatx.network.viz import NetworkLayout


def filter_node_centroids(n):
    """
    Function to filter cluster nodes from subject's nodes.

    Parameters
    ----------
    n : str
        Node label.

    Returns
    -------
    bool
        True or False
    """
    return "c" in n


def filter_node_subjects(n):
    """
    Function to filter subject nodes from cluster's nodes.

    Parameters
    ----------
    n : str
        Node label.

    Returns
    -------
    bool
        True or False
    """
    return "c" not in n


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
        _, ext = os.path.splitext(file)
        if ext == ".csv":
            self.data = pd.read_csv(file, **kwargs)
        elif ext == ".xlsx":
            self.data = pd.read_excel(file, **kwargs)
        elif ext == ".tsv":
            self.data = pd.read_csv(file, sep="\t", **kwargs)
        elif ext == ".txt":
            with open(file, "r") as f:
                f = f.read()
                delimiter = detect(f, whitelist=["\t", ":", ";", " ", ","])
            self.data = pd.read_csv(file, sep=delimiter, **kwargs)
        else:
            raise ValueError("File format not supported. Currently supported "
                             "formats are .csv, .xlsx, .tsv, .txt.")

        self.nb_subjects, self.nb_variables = self.data.shape
        return self

    def import_data(self, data, columns=None, index=None, **kwargs):
        """
        Import data from a DataFrame or a array-like object.

        Parameters
        ----------
        data: pandas.DataFrame or array-like
            Data to import.
        columns: list, optional
            List of columns to use. If None, all columns will be used.

        Returns
        -------
        DatasetLoader
            DatasetLoader object with the imported data.
        """
        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, dict):
            self.data = pd.DataFrame.from_dict(data,
                                               columns=columns,
                                               **kwargs)
        else:
            self.data = pd.DataFrame(data, columns=columns, index=index)
        self.nb_subjects, self.nb_variables = self.data.shape
        return self

    def get_descriptive_columns(self, columns):
        """
        Get descriptive columns from the data.

        Parameters
        ----------
        columns: list
            List of descriptive columns.

        Returns
        -------
        data: pandas.DataFrame
            DataFrame containing the descriptive columns.
        """
        if not hasattr(self, 'data'):
            raise ValueError("Data not loaded. Please load data first.")

        if not isinstance(columns, list):
            raise ValueError("Provided columns is not a list.")

        return self.data[self.data.columns[columns]]

    def drop_columns(self, columns):
        """
        Drop specified columns from the data.

        Parameters
        ----------
        columns: list
            List of columns index or names to drop.

        Returns
        -------
        DatasetLoader
            DatasetLoader object with the specified columns dropped.
        """
        if not hasattr(self, 'data'):
            raise ValueError("Data not loaded. Please load data first.")

        if not isinstance(columns, list):
            raise ValueError("Provided columns is not a list.")

        if isinstance(columns[0], int):
            self.data.drop(self.data.columns[columns], axis=1, inplace=True)
        else:
            self.data.drop(columns, axis=1, inplace=True)

        self.nb_subjects, self.nb_variables = self.data.shape
        return self

    def join(self, df, left=True, **kwargs):
        """
        Join two DataFrames.

        Parameters
        ----------
        df: pandas.DataFrame
            DataFrame to join with.
        left: bool, optional
            If true, provided DataFrame is the left DataFrame.
            If false, provided DataFrame is the right DataFrame.
            Default is True.
        **kwargs
            Additional keyword arguments for the pd.concat function.

        Returns
        -------
        data: pandas.DataFrame
            Joined DataFrame.
        """
        if not isinstance(df, pd.DataFrame | pd.Series):
            raise ValueError(
                "Provided data is not a pandas DataFrame or Series.")
        if not hasattr(self, 'data'):
            raise ValueError("Data not loaded. Please load data first.")

        if left:
            self.data = pd.concat([df, self.data], axis=1, **kwargs)
        else:
            self.data = pd.concat([self.data, df], axis=1, **kwargs)
        self.nb_subjects, self.nb_variables = self.data.shape
        return self

    def reset_index(self, **kwargs):
        """
        Reset the index of the DataFrame.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments for the pd.DataFrame.reset_index
            function.
        """
        if not hasattr(self, 'data'):
            raise ValueError("Data not loaded. Please load data first.")

        self.data.reset_index(drop=True, inplace=True, **kwargs)
        return self

    def set_type(self, dtype, columns=None):
        """
        Set the type of the specified columns.

        Parameters
        ----------
        dtype: str
            Type to set.
        columns: list, optional
            List of columns to set the type for. If None, all columns will be
            converted.

        Returns
        -------
        data: pandas.DataFrame
            DataFrame with the specified type set.
        """
        if not hasattr(self, 'data'):
            raise ValueError("Data not loaded. Please load data first.")

        if columns is None:
            self.data = self.data.astype(dtype)
        else:
            self.data[self.data.columns[columns]] = self.data[
                self.data.columns[columns]].astype(dtype)

        return self

    def transpose(self):
        """
        Transpose the data.

        Returns
        -------
        DatasetLoader
            DatasetLoader object with the transposed data.
        """
        if not hasattr(self, 'data'):
            raise ValueError("Data not loaded. Please load data first.")

        self.data = self.data.T
        self.nb_subjects, self.nb_variables = self.data.shape
        return self

    def get_metadata(self):
        """
        Get metadata of the loaded data.

        Returns
        -------
        metadata: dict
            Dictionary containing the number of subjects and variables.
        """
        if not hasattr(self, 'data'):
            raise ValueError("Data not loaded. Please load data first.")

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
        if not hasattr(self, 'data'):
            raise ValueError("Data not loaded. Please load data first.")

        return self.data

    def save_data(self, file, **kwargs):
        """
        Save the data to a file.

        Parameters
        ----------
        file: str
            Output file name.
        **kwargs
            Additional keyword arguments.
        """
        if not hasattr(self, 'data'):
            raise ValueError("Data not loaded. Please load data first.")

        _, ext = os.path.splitext(file)
        if ext == ".csv":
            self.data.to_csv(file, **kwargs)
        elif ext == ".xlsx":
            self.data.to_excel(file, **kwargs)
        elif ext == ".tsv":
            self.data.to_csv(file, sep="\t", **kwargs)
        elif ext == ".txt":
            self.data.to_csv(file, sep="\t", **kwargs)
        else:
            raise ValueError("File format not supported. Currently supported "
                             "formats are .csv, .xlsx, .tsv, .txt.")

    def custom_function(self, func, **kwargs):
        """
        Apply a custom function to the data.

        Parameters
        ----------
        func: callable
            Custom function to apply.
        **kwargs
            Additional keyword arguments for the custom function.

        Returns
        -------
        data: pandas.DataFrame
            The modified data.
        """
        if not callable(func):
            raise ValueError("Provided function is not callable.")
        if not hasattr(self, 'data'):
            raise ValueError("Data not loaded. Please load data first.")

        return func(self.data, **kwargs)


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
        return self

    def build_graph(self, data, source='source', target='target', **kwargs):
        """
        Build a graph from the provided data.

        Parameters
        ----------
        data: pandas.DataFrame
            DataFrame containing the data to build the graph.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        graph: networkx.Graph
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Provided data is not a pandas DataFrame.")

        self.graph = nx.from_pandas_edgelist(data,
                                             source=source,
                                             target=target,
                                             **kwargs)
        self.nb_nodes = self.graph.number_of_nodes()
        self.nb_edges = self.graph.number_of_edges()
        return self

    def layout(self, layout=NetworkLayout.Spring, weight="membership",
               **kwargs):
        """
        Compute the layout of the graph.
        Parameters
        ----------
        layout: NetworkLayout
            Layout algorithm to use.
        weight: str, optional
            Edge attribute to use as weights for the layout.
        **kwargs
            Additional keyword arguments for the layout algorithm.
        Returns
        -------
        pos: dict
            Dictionary containing the positions of the nodes.
        """
        if not hasattr(self, 'graph'):
            raise ValueError("Graph not loaded. Please load a graph first.")
        if not isinstance(layout, NetworkLayout):
            raise ValueError("Provided layout is not a valid NetworkLayout.")
        if not any(weight in data for _, _,
                   data in self.graph.edges(data=True)):
            raise ValueError(
                f"Weight '{weight}' not found in the graph edges.")

        pos = getattr(nx, layout)(self.graph, weight=weight, **kwargs)
        pos = {k: list(map(float, pos[k])) for k in pos}
        nx.set_node_attributes(self.graph, pos, "pos")

    def add_node_attribute(self, attributes):
        """
        Add a node attribute to the graph.

        Parameters
        ----------
        attributes: dict
            Dictionary containing the values of the attribute for each node.
        """
        if not hasattr(self, 'graph'):
            raise ValueError("Graph not loaded. Please load a graph first.")

        nx.set_node_attributes(self.graph, attributes)

    def add_edge_attribute(self, attributes):
        """
        Add an edge attribute to the graph.

        Parameters
        ----------
        attribute: dict
            Dictionary containing the values of the attribute for each edge.
        """
        if not hasattr(self, 'graph'):
            raise ValueError("Graph not loaded. Please load a graph first.")

        nx.set_edge_attributes(self.graph, attributes)

    def fetch_attributes_df(self, attributes=None):
        """
        Fetch nodes' attributes from the graph as a DataFrame.

        Parameters
        ----------
        attributes: List, optional
            List of attributes to fetch.

        Returns
        -------
        DatasetLoader
            DatasetLoader object containing the nodes' attributes.
        """
        if not hasattr(self, 'graph'):
            raise ValueError("Graph not loaded. Please load a graph first.")

        # Filter out nodes that are not subjects.
        sub_node = nx.subgraph_view(self.graph,
                                    filter_node=filter_node_subjects)
        d = {n: self.graph.nodes[n] for n in sub_node}

        # Filter for selected attributes.
        if attributes is not None:
            d = {k: {k2: v2 for k2, v2 in v.items() if k2 in attributes}
                 for k, v in d.items()}
        else:
            d = {k: {k2: v2 for k2, v2 in v.items() if k2 != 'label'}
                 for k, v in d.items()}

        # Create df.
        df = pd.DataFrame.from_dict(d, orient="index")

        return DatasetLoader().import_data(df)

    def fetch_edge_data(self, weight="membership"):
        """
        Fetch edge data from the graph.

        Parameters
        ----------
        weight: str, optional
            Edge attribute to use as weights for the edges.

        Returns
        -------
        DatasetLoader
            DatasetLoader object containing the edge data.
        """
        if not hasattr(self, 'graph'):
            raise ValueError("Graph not loaded. Please load a graph first.")

        if not any(weight in data for _, _,
                   data in self.graph.edges(data=True)):
            raise ValueError(
                f"Weight '{weight}' not found in the graph edges.")

        # Fetching edges data.
        cntr_node = nx.subgraph_view(self.graph,
                                     filter_node=filter_node_centroids)
        sub_node = nx.subgraph_view(self.graph,
                                    filter_node=filter_node_subjects)

        # Get adjacency matrix.
        adj = np.delete(
            nx.to_numpy_array(self.graph, weight=weight),
            [i for i in range(1, len(cntr_node) + 1)],
            axis=0
        )
        df = pd.DataFrame(
            adj[:, 1:(len(cntr_node) + 1)], index=sub_node,
            columns=[f'Cluster {i+1}' for i in range(len(cntr_node))]
        )

        return DatasetLoader().import_data(df)

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
        if not hasattr(self, 'graph'):
            raise ValueError("Graph not loaded. Please load a graph first.")

        # Fetching nodes position.
        pos = nx.get_node_attributes(self.graph, "pos")

        # Fetching edges widths.
        widths = nx.get_edge_attributes(self.graph, weight)

        # Sorting which nodes to label.
        labels = {}
        if centroids_labelling:
            for node in self.graph.nodes():
                if "c" in node:
                    labels[node] = node
        elif centroids_labelling and subjects_labelling:
            for node in self.graph.nodes():
                labels[node] = node
        else:
            for node in self.graph.nodes():
                labels[node] = ""

        # Setting z-order of nodes.
        cntr_node = nx.subgraph_view(self.graph,
                                     filter_node=filter_node_centroids)
        sub_node = nx.subgraph_view(self.graph,
                                    filter_node=filter_node_subjects)

        # Centroids customization lists.
        cntr_shape = np.array([centroid_node_shape] * len(cntr_node.nodes()))
        cntr_alpha = np.array([centroid_alpha] * len(cntr_node.nodes()))

        # Subjects customization lists.
        sub_shape = np.array([subject_node_shape] * len(sub_node.nodes()))
        # sub_alpha = np.array([subject_alpha] * len(sub_node.nodes()))

        # Plotting the graph.
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot()

        nodes1 = nx.draw_networkx_nodes(
            self.graph,
            pos,
            nodelist=sub_node.nodes(),
            node_size=sub_shape,
            node_color=subject_node_color,
            alpha=subject_alpha,
            ax=ax,
        )
        nodes2 = nx.draw_networkx_nodes(
            self.graph,
            pos,
            nodelist=cntr_node.nodes(),
            node_size=cntr_shape,
            node_color=centroid_node_color,
            alpha=cntr_alpha,
            ax=ax,
        )

        # Drawing edges.
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=widths.keys(),
            width=(list(widths.values()) * 10),
            edge_color=list(widths.values()),
            edge_cmap=getattr(plt.cm, colormap),
            alpha=list(widths.values()),
            ax=ax,
        )

        # Setting z-order.
        nodes1.set_zorder(2)
        if subject_edge_color is not None:
            nodes1.set_edgecolor(subject_edge_color)
        nodes2.set_zorder(3)
        nodes2.set_edgecolor(centroid_edge_color)

        # Plotting labels if set.
        nx.draw_networkx_labels(self.graph,
                                pos,
                                labels=labels,
                                font_color="black",
                                ax=ax)

        # Adding colorbar, titles, etc.
        cmappable = ScalarMappable(Normalize(0, 1),
                                   getattr(plt.cm, colormap))
        cbar = plt.colorbar(cmappable, ax=ax, location="right", shrink=0.5)

        plt.box(False)
        ax.set_title(title)
        cbar.ax.set_title(legend_title)

        plt.tight_layout()
        plt.savefig(output)
        plt.close()

    def get_metadata(self):
        """
        Get metadata of the loaded graph.

        Returns
        -------
        metadata: dict
            Dictionary containing the number of nodes and edges.
        """
        if not hasattr(self, 'graph'):
            raise ValueError("Graph not loaded. Please load a graph first.")

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
        if not hasattr(self, 'graph'):
            raise ValueError("Graph not loaded. Please load a graph first.")

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
        if not hasattr(self, 'graph'):
            raise ValueError("Graph not loaded. Please load a graph first.")

        if file.endswith(".gml"):
            nx.write_gml(self.graph, file, **kwargs)
        elif file.endswith(".graphml"):
            nx.write_graphml(self.graph, file, **kwargs)
        elif file.endswith(".gexf"):
            nx.write_gexf(self.graph, file, **kwargs)
        else:
            raise ValueError("File format not supported. Currently supported "
                             "formats are .gml, .graphml, .gexf.")

    def custom_function(self, func, **kwargs):
        """
        Apply a custom function to the graph.

        Parameters
        ----------
        func: callable
            Custom function to apply.
        **kwargs
            Additional keyword arguments for the custom function.

        Returns
        -------
        graph: networkx.Graph
            The modified graph.
        """
        if not callable(func):
            raise ValueError("Provided function is not callable.")
        if not hasattr(self, 'graph'):
            raise ValueError("Graph not loaded. Please load a graph first.")

        return func(self.graph, **kwargs)
