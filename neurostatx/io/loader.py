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
    """Return True if ``n`` is a cluster-centroid node label.

    Parameters
    ----------
    n : str
        Node label.

    Returns
    -------
    match : bool
        True when the label contains ``"c"``.

    Examples
    --------
    >>> from neurostatx.io.loader import filter_node_centroids
    >>> filter_node_centroids("c1")
    True
    """
    return "c" in n


def filter_node_subjects(n):
    """Return True if ``n`` is a subject node label.

    Parameters
    ----------
    n : str
        Node label.

    Returns
    -------
    match : bool
        True when the label does not contain ``"c"``.

    Examples
    --------
    >>> from neurostatx.io.loader import filter_node_subjects
    >>> filter_node_subjects("s01")
    True
    """
    return "c" not in n


class DatasetLoader:
    """Load, reshape, and persist tabular datasets."""

    def __init__(self):
        self.data = None
        self.nb_subjects = None
        self.nb_variables = None

    def load_data(self, file, **kwargs):
        """Load tabular data from ``.txt``, ``.csv``, ``.tsv``, or ``.xlsx``.

        Parameters
        ----------
        file : str
            Input file to load.
        **kwargs
            Additional keyword arguments forwarded to pandas.

        Returns
        -------
        self : DatasetLoader
            Loader with ``data`` populated.

        Examples
        --------
        >>> from neurostatx.io.loader import DatasetLoader
        >>> DatasetLoader().load_data("data.csv")
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
        """Import data from a DataFrame, mapping, or array-like object.

        Parameters
        ----------
        data : pandas.DataFrame or array-like
            Data to import.
        columns : list, optional
            Column names to use. If None, pandas default names are kept.
        index : array-like, optional
            Index used when constructing a DataFrame from array-like data.
        **kwargs
            Additional keyword arguments forwarded to
            ``pandas.DataFrame.from_dict``.

        Returns
        -------
        self : DatasetLoader
            Loader with the imported data.

        Examples
        --------
        >>> from neurostatx.io.loader import DatasetLoader
        >>> loader = DatasetLoader().import_data([[1, 2], [3, 4]],
        ...                                      columns=["a", "b"])
        >>> loader.get_data().shape
        (2, 2)
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
        """Return selected descriptive columns by integer index.

        Parameters
        ----------
        columns : list
            Integer column indices to extract.

        Returns
        -------
        data : pandas.DataFrame
            DataFrame containing the selected columns.

        Examples
        --------
        >>> from neurostatx.io.loader import DatasetLoader
        >>> loader = DatasetLoader().import_data([[1, 2], [3, 4]],
        ...                                      columns=["id", "x"])
        >>> loader.get_descriptive_columns([0]).columns.tolist()
        ['id']
        """
        if not hasattr(self, 'data'):
            raise ValueError("Data not loaded. Please load data first.")

        if not isinstance(columns, list):
            raise ValueError("Provided columns is not a list.")

        return self.data[self.data.columns[columns]]

    def drop_columns(self, columns):
        """Drop columns by integer index or name.

        Parameters
        ----------
        columns : list
            Column indices or names to drop.

        Returns
        -------
        self : DatasetLoader
            Loader with the specified columns removed.

        Examples
        --------
        >>> from neurostatx.io.loader import DatasetLoader
        >>> loader = DatasetLoader().import_data([[1, 2], [3, 4]],
        ...                                      columns=["id", "x"])
        >>> loader.drop_columns(["id"]).get_data().columns.tolist()
        ['x']
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
        """Concatenate ``df`` with the loaded table along columns.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame or Series to concatenate.
        left : bool, optional
            If True, ``df`` is placed on the left. Defaults to True.
        **kwargs
            Additional keyword arguments forwarded to ``pandas.concat``.

        Returns
        -------
        self : DatasetLoader
            Loader with the concatenated table.

        Examples
        --------
        >>> import pandas as pd
        >>> from neurostatx.io.loader import DatasetLoader
        >>> loader = DatasetLoader().import_data([[1], [2]], columns=["a"])
        >>> extra = pd.DataFrame({"b": [3, 4]})
        >>> loader.join(extra, left=False).get_data().columns.tolist()
        ['a', 'b']
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
        """Reset the DataFrame index in place.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments forwarded to
            ``pandas.DataFrame.reset_index``.

        Returns
        -------
        self : DatasetLoader
            Loader with a reset index.

        Examples
        --------
        >>> from neurostatx.io.loader import DatasetLoader
        >>> loader = DatasetLoader().import_data([[1], [2]], columns=["a"],
        ...                                      index=[10, 20])
        >>> loader.reset_index().get_data().index.tolist()
        [0, 1]
        """
        if not hasattr(self, 'data'):
            raise ValueError("Data not loaded. Please load data first.")

        self.data.reset_index(drop=True, inplace=True, **kwargs)
        return self

    def set_type(self, dtype, columns=None):
        """Cast selected columns, or the full table, to ``dtype``.

        Parameters
        ----------
        dtype : str
            Target dtype.
        columns : list, optional
            Integer column indices to convert. If None, all columns are
            converted.

        Returns
        -------
        self : DatasetLoader
            Loader with updated dtypes.

        Examples
        --------
        >>> from neurostatx.io.loader import DatasetLoader
        >>> loader = DatasetLoader().import_data([[1, 2], [3, 4]],
        ...                                      columns=["a", "b"])
        >>> loader.set_type("float").get_data()["a"].dtype.kind
        'f'
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
        """Transpose the loaded table.

        Returns
        -------
        self : DatasetLoader
            Loader with transposed data.

        Examples
        --------
        >>> from neurostatx.io.loader import DatasetLoader
        >>> loader = DatasetLoader().import_data([[1, 2], [3, 4]],
        ...                                      columns=["a", "b"])
        >>> loader.transpose().get_data().shape
        (2, 2)
        """
        if not hasattr(self, 'data'):
            raise ValueError("Data not loaded. Please load data first.")

        self.data = self.data.T
        self.nb_subjects, self.nb_variables = self.data.shape
        return self

    def get_metadata(self):
        """Return the number of rows and columns.

        Returns
        -------
        metadata : dict
            Mapping with ``nb_subjects`` and ``nb_variables``.

        Examples
        --------
        >>> from neurostatx.io.loader import DatasetLoader
        >>> loader = DatasetLoader().import_data([[1, 2], [3, 4]])
        >>> loader.get_metadata()["nb_subjects"]
        2
        """
        if not hasattr(self, 'data'):
            raise ValueError("Data not loaded. Please load data first.")

        return {
            "nb_subjects": self.nb_subjects,
            "nb_variables": self.nb_variables
        }

    def get_data(self):
        """Return the loaded DataFrame.

        Returns
        -------
        data : pandas.DataFrame
            The loaded data.

        Examples
        --------
        >>> from neurostatx.io.loader import DatasetLoader
        >>> DatasetLoader().import_data([[1, 2]]).get_data().shape
        (1, 2)
        """
        if not hasattr(self, 'data'):
            raise ValueError("Data not loaded. Please load data first.")

        return self.data

    def save_data(self, file, **kwargs):
        """Save the loaded table to ``.csv``, ``.tsv``, ``.txt``, or ``.xlsx``.

        Parameters
        ----------
        file : str
            Output file name.
        **kwargs
            Additional keyword arguments forwarded to pandas.

        Examples
        --------
        >>> from neurostatx.io.loader import DatasetLoader
        >>> loader = DatasetLoader().import_data([[1, 2]], columns=["a", "b"])
        >>> loader.save_data("out.csv", index=False)
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
        """Apply ``func`` to the loaded DataFrame.

        Parameters
        ----------
        func : callable
            Function called as ``func(data, **kwargs)``.
        **kwargs
            Additional keyword arguments forwarded to ``func``.

        Returns
        -------
        result
            Return value of ``func``.

        Examples
        --------
        >>> from neurostatx.io.loader import DatasetLoader
        >>> loader = DatasetLoader().import_data([[1, 2]], columns=["a", "b"])
        >>> loader.custom_function(lambda df: df.shape)
        (1, 2)
        """
        if not callable(func):
            raise ValueError("Provided function is not callable.")
        if not hasattr(self, 'data'):
            raise ValueError("Data not loaded. Please load data first.")

        return func(self.data, **kwargs)


class GraphLoader:
    """Load, build, annotate, and visualize NetworkX graphs."""

    def __init__(self):
        self.graph = None
        self.nb_nodes = None
        self.nb_edges = None

    def load_graph(self, file, **kwargs):
        """Load a graph from ``.gml``, ``.graphml``, or ``.gexf``.

        Parameters
        ----------
        file : str
            Input file to load.
        **kwargs
            Additional keyword arguments forwarded to NetworkX.

        Returns
        -------
        self : GraphLoader
            Loader with ``graph`` populated.

        Examples
        --------
        >>> from neurostatx.io.loader import GraphLoader
        >>> GraphLoader().load_graph("network.gml")
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
        """Build a graph from an edgelist DataFrame.

        Parameters
        ----------
        data : pandas.DataFrame
            Edgelist used to construct the graph.
        source : str, optional
            Source-node column name. Defaults to ``"source"``.
        target : str, optional
            Target-node column name. Defaults to ``"target"``.
        **kwargs
            Additional keyword arguments forwarded to
            ``networkx.from_pandas_edgelist``.

        Returns
        -------
        self : GraphLoader
            Loader with the constructed graph.

        Examples
        --------
        >>> import pandas as pd
        >>> from neurostatx.io.loader import GraphLoader
        >>> edges = pd.DataFrame({"source": ["s1"], "target": ["c1"],
        ...                       "membership": [0.8]})
        >>> GraphLoader().build_graph(edges, edge_attr="membership").nb_nodes
        2
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
        """Compute node positions and store them as a ``pos`` attribute.

        Parameters
        ----------
        layout : NetworkLayout
            Layout algorithm to use.
        weight : str, optional
            Edge attribute used as layout weights. Defaults to
            ``"membership"``.
        **kwargs
            Additional keyword arguments forwarded to the NetworkX layout
            function.

        Examples
        --------
        >>> import pandas as pd
        >>> from neurostatx.io.loader import GraphLoader
        >>> from neurostatx.network.viz import NetworkLayout
        >>> edges = pd.DataFrame({"source": ["s1", "s2"], "target": ["c1", "c1"],
        ...                       "membership": [0.8, 0.4]})
        >>> g = GraphLoader().build_graph(edges, edge_attr="membership")
        >>> g.layout(NetworkLayout.Spring)
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
        """Set node attributes from a nested dictionary.

        Parameters
        ----------
        attributes : dict
            Mapping of node to attribute dictionary.

        Examples
        --------
        >>> import pandas as pd
        >>> from neurostatx.io.loader import GraphLoader
        >>> edges = pd.DataFrame({"source": ["s1"], "target": ["c1"]})
        >>> g = GraphLoader().build_graph(edges)
        >>> g.add_node_attribute({"s1": {"age": 20}})
        """
        if not hasattr(self, 'graph'):
            raise ValueError("Graph not loaded. Please load a graph first.")

        nx.set_node_attributes(self.graph, attributes)

    def add_edge_attribute(self, attributes):
        """Set edge attributes from a nested dictionary.

        Parameters
        ----------
        attributes : dict
            Mapping of edge to attribute dictionary.

        Examples
        --------
        >>> import pandas as pd
        >>> from neurostatx.io.loader import GraphLoader
        >>> edges = pd.DataFrame({"source": ["s1"], "target": ["c1"]})
        >>> g = GraphLoader().build_graph(edges)
        >>> g.add_edge_attribute({("s1", "c1"): {"membership": 0.8}})
        """
        if not hasattr(self, 'graph'):
            raise ValueError("Graph not loaded. Please load a graph first.")

        nx.set_edge_attributes(self.graph, attributes)

    def fetch_attributes_df(self, attributes=None):
        """Return subject-node attributes as a DatasetLoader.

        Parameters
        ----------
        attributes : list, optional
            Attribute names to fetch. If None, all attributes except
            ``label`` are returned.

        Returns
        -------
        data : DatasetLoader
            Loader containing subject-node attributes.

        Examples
        --------
        >>> import pandas as pd
        >>> from neurostatx.io.loader import GraphLoader
        >>> edges = pd.DataFrame({"source": ["s1"], "target": ["c1"]})
        >>> g = GraphLoader().build_graph(edges)
        >>> g.add_node_attribute({"s1": {"age": 20}})
        >>> g.fetch_attributes_df(["age"]).get_data().loc["s1", "age"]
        20
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
        """Return subject-to-cluster edge weights as a DatasetLoader.

        Parameters
        ----------
        weight : str, optional
            Edge attribute used as weights. Defaults to ``"membership"``.

        Returns
        -------
        data : DatasetLoader
            Loader containing one column per cluster.

        Examples
        --------
        >>> import pandas as pd
        >>> from neurostatx.io.loader import GraphLoader
        >>> edges = pd.DataFrame({"source": ["s1"], "target": ["c1"],
        ...                       "membership": [0.8]})
        >>> g = GraphLoader().build_graph(edges, edge_attr="membership")
        >>> g.fetch_edge_data().get_data().shape[1]
        1
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
                  edge_width_multiplier=1,
                  colormap="plasma",
                  title="Graph Network",
                  legend_title="Membership values"):
        """Draw the graph network and write it to ``output``.

        Parameters
        ----------
        output : str
            Output file name.
        weight : str, optional
            Edge attribute used as edge weights. Defaults to ``"weight"``.
        centroids_labelling : bool, optional
            If True, label centroid nodes. Defaults to True.
        subjects_labelling : bool, optional
            If True, label subject nodes. Defaults to False.
        centroid_node_shape : int, optional
            Size of centroid nodes. Defaults to 500.
        centroid_alpha : float, optional
            Alpha of centroid nodes. Defaults to 1.
        centroid_node_color : str, optional
            Face color of centroid nodes. Defaults to ``"white"``.
        centroid_edge_color : str, optional
            Edge color of centroid nodes. Defaults to ``"black"``.
        subject_node_shape : int, optional
            Size of subject nodes. Defaults to 5.
        subject_alpha : float, optional
            Alpha of subject nodes. Defaults to 0.3.
        subject_node_color : str, optional
            Face color of subject nodes. Defaults to ``"black"``.
        subject_edge_color : str, optional
            Edge color of subject nodes. Defaults to None.
        edge_width_multiplier : float, optional
            Scale factor applied to edge widths. Defaults to 1.
        colormap : str, optional
            Matplotlib colormap name for edges. Defaults to ``"plasma"``.
        title : str, optional
            Title of the plot. Defaults to ``"Graph Network"``.
        legend_title : str, optional
            Title of the colorbar. Defaults to ``"Membership values"``.

        Examples
        --------
        >>> import pandas as pd
        >>> from neurostatx.io.loader import GraphLoader
        >>> from neurostatx.network.viz import NetworkLayout
        >>> edges = pd.DataFrame({"source": ["s1"], "target": ["c1"],
        ...                       "weight": [0.8]})
        >>> g = GraphLoader().build_graph(edges, edge_attr="weight")
        >>> g.layout(NetworkLayout.Spring, weight="weight")
        >>> g.visualize("graph.png")
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

        # Assess alpha values are between 0 and 1. If not, normalize them.
        if max(widths.values()) > 1 or min(widths.values()) < 0:
            widths = {k: (v - 0) / (max(widths.values()) - 0)
                      for k, v in widths.items()}

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
            width=1 + np.array(list(widths.values())) * edge_width_multiplier,
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
        """Return the number of nodes and edges.

        Returns
        -------
        metadata : dict
            Mapping with ``nb_nodes`` and ``nb_edges``.

        Examples
        --------
        >>> import pandas as pd
        >>> from neurostatx.io.loader import GraphLoader
        >>> edges = pd.DataFrame({"source": ["s1"], "target": ["c1"]})
        >>> GraphLoader().build_graph(edges).get_metadata()["nb_nodes"]
        2
        """
        if not hasattr(self, 'graph'):
            raise ValueError("Graph not loaded. Please load a graph first.")

        return {
            "nb_nodes": self.nb_nodes,
            "nb_edges": self.nb_edges
        }

    def get_graph(self):
        """Return the loaded NetworkX graph.

        Returns
        -------
        graph : networkx.Graph
            The loaded graph.

        Examples
        --------
        >>> import pandas as pd
        >>> from neurostatx.io.loader import GraphLoader
        >>> edges = pd.DataFrame({"source": ["s1"], "target": ["c1"]})
        >>> GraphLoader().build_graph(edges).get_graph().number_of_nodes()
        2
        """
        if not hasattr(self, 'graph'):
            raise ValueError("Graph not loaded. Please load a graph first.")

        return self.graph

    def save_graph(self, file, **kwargs):
        """Save the graph to ``.gml``, ``.graphml``, or ``.gexf``.

        Parameters
        ----------
        file : str
            Output file to save.
        **kwargs
            Additional keyword arguments forwarded to NetworkX.

        Examples
        --------
        >>> import pandas as pd
        >>> from neurostatx.io.loader import GraphLoader
        >>> edges = pd.DataFrame({"source": ["s1"], "target": ["c1"]})
        >>> GraphLoader().build_graph(edges).save_graph("network.gml")
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
        """Apply ``func`` to the loaded graph.

        Parameters
        ----------
        func : callable
            Function called as ``func(graph, **kwargs)``.
        **kwargs
            Additional keyword arguments forwarded to ``func``.

        Returns
        -------
        result
            Return value of ``func``.

        Examples
        --------
        >>> import pandas as pd
        >>> from neurostatx.io.loader import GraphLoader
        >>> edges = pd.DataFrame({"source": ["s1"], "target": ["c1"]})
        >>> GraphLoader().build_graph(edges).custom_function(
        ...     lambda g: g.number_of_nodes())
        2
        """
        if not callable(func):
            raise ValueError("Provided function is not callable.")
        if not hasattr(self, 'graph'):
            raise ValueError("Graph not loaded. Please load a graph first.")

        return func(self.graph, **kwargs)
