# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from CCPM.io.download import get_home, download_data, save_files_dict


download_data(save_files_dict(), keys=["data.zip"])
tmp_dir = tempfile.TemporaryDirectory()


def test_help(script_runner):
    ret = script_runner.run([
        "GraphMetrics", "-h"])

    assert ret.success


def test_eigencentrality(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_graph = os.path.join(get_home(), "data/graph_file.gml")

    ret = script_runner.run([
        "GraphMetrics",
        "eigencentrality", in_graph,
        "membership", "-f"])

    assert ret.success


def test_closenesscentrality(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_graph = os.path.join(get_home(), "data/graph_file.gml")

    ret = script_runner.run([
        "GraphMetrics",
        "closenesscentrality", in_graph,
        "membership", "-f"])

    assert ret.success


def test_betweennesscentrality(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_graph = os.path.join(get_home(), "data/graph_file.gml")

    ret = script_runner.run([
        "GraphMetrics",
        "betweennesscentrality", in_graph,
        "membership", "-f"])

    assert ret.success


def test_informationcentrality(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_graph = os.path.join(get_home(), "data/graph_file.gml")

    ret = script_runner.run([
        "GraphMetrics",
        "informationcentrality", in_graph,
        "membership", "-f"])

    assert ret.success


def test_currentflowbc(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_graph = os.path.join(get_home(), "data/graph_file.gml")

    ret = script_runner.run([
        "GraphMetrics",
        "currentflowbc", in_graph,
        "membership", "-f"])

    assert ret.success


def test_loadcentrality(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_graph = os.path.join(get_home(), "data/graph_file.gml")

    ret = script_runner.run([
        "GraphMetrics",
        "loadcentrality", in_graph,
        "membership", "-f"])

    assert ret.success


def test_harmoniccentrality(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_graph = os.path.join(get_home(), "data/graph_file.gml")

    ret = script_runner.run([
        "GraphMetrics",
        "harmoniccentrality", in_graph,
        "membership", "-f"])

    assert ret.success


def test_eccentricity(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_graph = os.path.join(get_home(), "data/graph_file.gml")

    ret = script_runner.run([
        "GraphMetrics",
        "eccentricity", in_graph,
        "membership", "-f"])

    assert ret.success


def test_clustering(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_graph = os.path.join(get_home(), "data/graph_file.gml")

    ret = script_runner.run([
        "GraphMetrics",
        "clustering", in_graph,
        "membership", "-f"])

    assert ret.success


# def test_constraint(script_runner):
#    os.chdir(os.path.expanduser(tmp_dir.name))
#    in_graph = os.path.join(get_home(), "data/graph_file.gml")

#    ret = script_runner.run([
    # "GraphMetrics",
    # "constraint", in_graph, "c1",
    # "membership", "-f"])

#    assert ret.success


# def test_effectivesize(script_runner):
#    os.chdir(os.path.expanduser(tmp_dir.name))
#    in_graph = os.path.join(get_home(), "data/graph_file.gml")

#    ret = script_runner.run([
    # "GraphMetrics",
    # "effectivesize", in_graph, "c1",
    # "membership", "-f"])

#    assert ret.success


def test_closenessvitality(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_graph = os.path.join(get_home(), "data/graph_file.gml")

    ret = script_runner.run([
        "GraphMetrics",
        "closenessvitality", in_graph, "c1",
        "membership", "-f"])

    assert ret.success


def test_degree(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_graph = os.path.join(get_home(), "data/graph_file.gml")

    ret = script_runner.run([
        "GraphMetrics",
        "degree", in_graph,
        "membership", "-f"])

    assert ret.success
