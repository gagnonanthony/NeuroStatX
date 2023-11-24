# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from typer.testing import CliRunner

from CCPM.io.download import get_home, download_data, save_files_dict
from Scripts.CCPM_graph_metrics import app


download_data(save_files_dict(), keys=["data.zip"])
tmp_dir = tempfile.TemporaryDirectory()

runner = CliRunner()


def test_help():
    ret = runner.invoke(app, ["--help"])

    assert ret.exit_code == 0


def test_eigencentrality():
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_graph = os.path.join(get_home(), "data/graph_file.gexf")

    ret = runner.invoke(app, ["eigencentrality", in_graph, "membership", "-f"])

    assert ret.exit_code == 0


def test_closenesscentrality():
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_graph = os.path.join(get_home(), "data/graph_file.gexf")

    ret = runner.invoke(app, ["closenesscentrality", in_graph, "membership",
                              "-f"])

    assert ret.exit_code == 0


def test_betweennesscentrality():
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_graph = os.path.join(get_home(), "data/graph_file.gexf")

    ret = runner.invoke(app, ["betweennesscentrality", in_graph, "membership",
                              "-f"])

    assert ret.exit_code == 0


def test_informationcentrality():
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_graph = os.path.join(get_home(), "data/graph_file.gexf")

    ret = runner.invoke(app, ["informationcentrality", in_graph, "membership",
                              "-f"])

    assert ret.exit_code == 0


def test_currentflowbc():
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_graph = os.path.join(get_home(), "data/graph_file.gexf")

    ret = runner.invoke(app, ["currentflowbc", in_graph, "membership", "-f"])

    assert ret.exit_code == 0


def test_loadcentrality():
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_graph = os.path.join(get_home(), "data/graph_file.gexf")

    ret = runner.invoke(app, ["loadcentrality", in_graph, "membership", "-f"])

    assert ret.exit_code == 0


def test_harmoniccentrality():
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_graph = os.path.join(get_home(), "data/graph_file.gexf")

    ret = runner.invoke(app, ["harmoniccentrality", in_graph, "membership",
                              "-f"])

    assert ret.exit_code == 0


def test_eccentricity():
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_graph = os.path.join(get_home(), "data/graph_file.gexf")

    ret = runner.invoke(app, ["eccentricity", in_graph, "membership", "-f"])

    assert ret.exit_code == 0


def test_clustering():
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_graph = os.path.join(get_home(), "data/graph_file.gexf")

    ret = runner.invoke(app, ["clustering", in_graph, "membership", "-f"])

    assert ret.exit_code == 0


# def test_constraint():
#    os.chdir(os.path.expanduser(tmp_dir.name))
#    in_graph = os.path.join(get_home(), "data/graph_file.gexf")

#    ret = runner.invoke(app, ["constraint", in_graph, "c1", "membership",
#                              "-f"])

#    assert ret.exit_code == 0


# def test_effectivesize():
#    os.chdir(os.path.expanduser(tmp_dir.name))
#    in_graph = os.path.join(get_home(), "data/graph_file.gexf")

#    ret = runner.invoke(app, ["effectivesize", in_graph, "c1", "membership",
#                              "-f"])

#    assert ret.exit_code == 0


def test_closenessvitality():
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_graph = os.path.join(get_home(), "data/graph_file.gexf")

    ret = runner.invoke(app, ["closenessvitality", in_graph, "c1",
                              "membership", "-f"])

    assert ret.exit_code == 0


def test_degree():
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_graph = os.path.join(get_home(), "data/graph_file.gexf")

    ret = runner.invoke(app, ["degree", in_graph, "membership", "-f"])

    assert ret.exit_code == 0
