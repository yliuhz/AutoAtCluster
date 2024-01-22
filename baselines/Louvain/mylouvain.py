# -*- coding: utf-8 -*-
"""
This module implements community detection.
"""
from __future__ import print_function

import array

import numbers
import warnings

import networkx as nx
import numpy as np

import time
from datetime import datetime
import torch
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.special import softmax

import os
import random
import setproctitle
from louvain import load_data
import traceback
import argparse

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # - %(name)s
logger = logging.getLogger(__name__)

"""Community_status.py"""
class Status(object):
    """
    To handle several data in one struct.

    Could be replaced by named tuple, but don't want to depend on python 2.6
    """
    node2com = {}
    total_weight = 0
    internals = {}
    degrees = {}
    gdegrees = {}

    def __init__(self):
        self.node2com = dict([])
        self.total_weight = 0
        self.degrees = dict([])
        self.gdegrees = dict([])
        self.internals = dict([])
        self.loops = dict([])

    def __str__(self):
        return ("node2com : " + str(self.node2com) + " degrees : "
                + str(self.degrees) + " internals : " + str(self.internals)
                + " total_weight : " + str(self.total_weight))

    def copy(self):
        """Perform a deep copy of status"""
        new_status = Status()
        new_status.node2com = self.node2com.copy()
        new_status.internals = self.internals.copy()
        new_status.degrees = self.degrees.copy()
        new_status.gdegrees = self.gdegrees.copy()
        new_status.total_weight = self.total_weight

    def init(self, graph, weight, part=None):
        """Initialize the status of a graph with every node in one community"""
        count = 0
        self.node2com = dict([])
        self.total_weight = 0
        self.degrees = dict([])
        self.gdegrees = dict([])
        self.internals = dict([])
        self.total_weight = graph.size(weight=weight)
        if part is None:
            for node in graph.nodes():
                self.node2com[node] = count
                deg = float(graph.degree(node, weight=weight)) # weighted degree
                if deg < 0:
                    error = "Bad node degree ({})".format(deg)
                    raise ValueError(error)
                self.degrees[count] = deg
                self.gdegrees[node] = deg
                edge_data = graph.get_edge_data(node, node, default={weight: 0})
                self.loops[node] = float(edge_data.get(weight, 1))
                self.internals[count] = self.loops[node]
                count += 1

        else:
            for node in graph.nodes():
                com = part[node]
                self.node2com[node] = com
                deg = float(graph.degree(node, weight=weight))
                self.degrees[com] = self.degrees.get(com, 0) + deg
                self.gdegrees[node] = deg
                inc = 0.
                for neighbor, datas in graph[node].items():
                    edge_weight = datas.get(weight, 1)
                    if edge_weight <= 0:
                        error = "Bad graph type ({})".format(type(graph))
                        raise ValueError(error)
                    if part[neighbor] == com:
                        if neighbor == node:
                            inc += float(edge_weight)
                        else:
                            inc += float(edge_weight) / 2.
                self.internals[com] = self.internals.get(com, 0) + inc



"""Community_louvain.py"""
__author__ = """Thomas Aynaud (thomas.aynaud@lip6.fr)"""
#    Copyright (C) 2009 by
#    Thomas Aynaud <thomas.aynaud@lip6.fr>
#    All rights reserved.
#    BSD license.

__PASS_MAX = -1
__MIN = 0.0000001


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r cannot be used to seed a numpy.random.RandomState"
                     " instance" % seed)


def partition_at_level(dendrogram, level):
    """Return the partition of the nodes at the given level

    A dendrogram is a tree and each level is a partition of the graph nodes.
    Level 0 is the first partition, which contains the smallest communities,
    and the best is len(dendrogram) - 1.
    The higher the level is, the bigger are the communities

    Parameters
    ----------
    dendrogram : list of dict
       a list of partitions, ie dictionnaries where keys of the i+1 are the
       values of the i.
    level : int
       the level which belongs to [0..len(dendrogram)-1]

    Returns
    -------
    partition : dictionnary
       A dictionary where keys are the nodes and the values are the set it
       belongs to

    Raises
    ------
    KeyError
       If the dendrogram is not well formed or the level is too high

    See Also
    --------
    best_partition : which directly combines partition_at_level and
    generate_dendrogram : to obtain the partition of highest modularity

    Examples
    --------
    >>> G=nx.erdos_renyi_graph(100, 0.01)
    >>> dendrogram = generate_dendrogram(G)
    >>> for level in range(len(dendrogram) - 1) :
    >>>     print("partition at level", level, "is", partition_at_level(dendrogram, level))  # NOQA
    """
    partition = dendrogram[0].copy()
    for index in range(1, level + 1):
        for node, community in partition.items():
            partition[node] = dendrogram[index][community]
    return partition


def modularity(partition, graph, weight='weight'):
    """Compute the modularity of a partition of a graph

    Parameters
    ----------
    partition : dict
       the partition of the nodes, i.e a dictionary where keys are their nodes
       and values the communities
    graph : networkx.Graph
       the networkx graph which is decomposed
    weight : str, optional
        the key in graph to use as weight. Default to 'weight'


    Returns
    -------
    modularity : float
       The modularity

    Raises
    ------
    KeyError
       If the partition is not a partition of all graph nodes
    ValueError
        If the graph has no link
    TypeError
        If graph is not a networkx.Graph

    References
    ----------
    .. 1. Newman, M.E.J. & Girvan, M. Finding and evaluating community
    structure in networks. Physical Review E 69, 26113(2004).

    Examples
    --------
    >>> import community as community_louvain
    >>> import networkx as nx
    >>> G = nx.erdos_renyi_graph(100, 0.01)
    >>> partition = community_louvain.best_partition(G)
    >>> modularity(partition, G)
    """
    if graph.is_directed():
        raise TypeError("Bad graph type, use only non directed graph")

    inc = dict([])
    deg = dict([])
    links = graph.size(weight=weight)
    if links == 0:
        raise ValueError("A graph without link has an undefined modularity")

    for node in graph:
        com = partition[node]
        deg[com] = deg.get(com, 0.) + graph.degree(node, weight=weight)
        for neighbor, datas in graph[node].items():
            edge_weight = datas.get(weight, 1)
            if partition[neighbor] == com:
                if neighbor == node:
                    inc[com] = inc.get(com, 0.) + float(edge_weight)
                else:
                    inc[com] = inc.get(com, 0.) + float(edge_weight) / 2.

    res = 0.
    for com in set(partition.values()):
        res += (inc.get(com, 0.) / links) - \
               (deg.get(com, 0.) / (2. * links)) ** 2
    return res


def best_partition(graph,
                   partition=None,
                   weight='weight',
                   resolution=1.,
                   randomize=None,
                   random_state=None,
                   neighbor_mode="all"):
    """Compute the partition of the graph nodes which maximises the modularity
    (or try..) using the Louvain heuristices

    This is the partition of highest modularity, i.e. the highest partition
    of the dendrogram generated by the Louvain algorithm.

    Parameters
    ----------
    graph : networkx.Graph
       the networkx graph which is decomposed
    partition : dict, optional
       the algorithm will start using this partition of the nodes.
       It's a dictionary where keys are their nodes and values the communities
    weight : str, optional
        the key in graph to use as weight. Default to 'weight'
    resolution :  double, optional
        Will change the size of the communities, default to 1.
        represents the time described in
        "Laplacian Dynamics and Multiscale Modular Structure in Networks",
        R. Lambiotte, J.-C. Delvenne, M. Barahona
    randomize : boolean, optional
        Will randomize the node evaluation order and the community evaluation
        order to get different partitions at each call
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    partition : dictionnary
       The partition, with communities numbered from 0 to number of communities

    Raises
    ------
    NetworkXError
       If the graph is not undirected.

    See Also
    --------
    generate_dendrogram : to obtain all the decompositions levels

    Notes
    -----
    Uses Louvain algorithm

    References
    ----------
    .. 1. Blondel, V.D. et al. Fast unfolding of communities in
    large networks. J. Stat. Mech 10008, 1-12(2008).

    Examples
    --------
    >>> # basic usage
    >>> import community as community_louvain
    >>> import networkx as nx
    >>> G = nx.erdos_renyi_graph(100, 0.01)
    >>> partion = community_louvain.best_partition(G)

    >>> # display a graph with its communities:
    >>> # as Erdos-Renyi graphs don't have true community structure,
    >>> # instead load the karate club graph
    >>> import community as community_louvain
    >>> import matplotlib.cm as cm
    >>> import matplotlib.pyplot as plt
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> # compute the best partition
    >>> partition = community_louvain.best_partition(G)

    >>> # draw the graph
    >>> pos = nx.spring_layout(G)
    >>> # color the nodes according to their partition
    >>> cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    >>> nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
    >>>                        cmap=cmap, node_color=list(partition.values()))
    >>> nx.draw_networkx_edges(G, pos, alpha=0.5)
    >>> plt.show()
    """
    dendo = generate_dendrogram(graph,
                                partition,
                                weight,
                                resolution,
                                randomize,
                                random_state,
                                neighbor_mode)
    return partition_at_level(dendo, len(dendo) - 1)


def generate_dendrogram(graph,
                        part_init=None,
                        weight='weight',
                        resolution=1.,
                        randomize=None,
                        random_state=None,
                        neighbor_mode="all"):
    """Find communities in the graph and return the associated dendrogram

    A dendrogram is a tree and each level is a partition of the graph nodes.
    Level 0 is the first partition, which contains the smallest communities,
    and the best is len(dendrogram) - 1. The higher the level is, the bigger
    are the communities


    Parameters
    ----------
    graph : networkx.Graph
        the networkx graph which will be decomposed
    part_init : dict, optional
        the algorithm will start using this partition of the nodes. It's a
        dictionary where keys are their nodes and values the communities
    weight : str, optional
        the key in graph to use as weight. Default to 'weight'
    resolution :  double, optional
        Will change the size of the communities, default to 1.
        represents the time described in
        "Laplacian Dynamics and Multiscale Modular Structure in Networks",
        R. Lambiotte, J.-C. Delvenne, M. Barahona

    Returns
    -------
    dendrogram : list of dictionaries
        a list of partitions, ie dictionnaries where keys of the i+1 are the
        values of the i. and where keys of the first are the nodes of graph

    Raises
    ------
    TypeError
        If the graph is not a networkx.Graph

    See Also
    --------
    best_partition

    Notes
    -----
    Uses Louvain algorithm

    References
    ----------
    .. 1. Blondel, V.D. et al. Fast unfolding of communities in large
    networks. J. Stat. Mech 10008, 1-12(2008).

    Examples
    --------
    >>> G=nx.erdos_renyi_graph(100, 0.01)
    >>> dendo = generate_dendrogram(G)
    >>> for level in range(len(dendo) - 1) :
    >>>     print("partition at level", level,
    >>>           "is", partition_at_level(dendo, level))
    :param weight:
    :type weight:
    """

    if graph.is_directed():
        raise TypeError("Bad graph type, use only non directed graph")

    # Properly handle random state, eventually remove old `randomize` parameter
    # NOTE: when `randomize` is removed, delete code up to random_state = ...
    if randomize is not None:
        warnings.warn("The `randomize` parameter will be deprecated in future "
                      "versions. Use `random_state` instead.", DeprecationWarning)
        # If shouldn't randomize, we set a fixed seed to get determinisitc results
        if randomize is False:
            random_state = 0

    # We don't know what to do if both `randomize` and `random_state` are defined
    if randomize and random_state is not None:
        raise ValueError("`randomize` and `random_state` cannot be used at the "
                         "same time")

    random_state = check_random_state(random_state)

    # special case, when there is no link
    # the best partition is everyone in its community
    if graph.number_of_edges() == 0:
        part = dict([])
        for i, node in enumerate(graph.nodes()):
            part[node] = i
        return [part]

    current_graph = graph.copy()
    status = Status()
    status.init(current_graph, weight, part_init)
    status_list = list()
    num_loops = __one_level(current_graph, status, weight, resolution, random_state, neighbor_mode)
    new_mod = __modularity(status, resolution)
    partition = __renumber(status.node2com)
    status_list.append(partition)
    mod = new_mod
    current_graph = induced_graph(partition, current_graph, weight)
    status.init(current_graph, weight)

    while True:
        __one_level(current_graph, status, weight, resolution, random_state, neighbor_mode)
        new_mod = __modularity(status, resolution)
        if new_mod - mod < __MIN:
            break
        partition = __renumber(status.node2com)
        status_list.append(partition)
        mod = new_mod
        current_graph = induced_graph(partition, current_graph, weight)
        status.init(current_graph, weight)
    
    return status_list[:]


def induced_graph(partition, graph, weight="weight"):
    """Produce the graph where nodes are the communities

    there is a link of weight w between communities if the sum of the weights
    of the links between their elements is w

    Parameters
    ----------
    partition : dict
       a dictionary where keys are graph nodes and  values the part the node
       belongs to
    graph : networkx.Graph
        the initial graph
    weight : str, optional
        the key in graph to use as weight. Default to 'weight'


    Returns
    -------
    g : networkx.Graph
       a networkx graph where nodes are the parts

    Examples
    --------
    >>> n = 5
    >>> g = nx.complete_graph(2*n)
    >>> part = dict([])
    >>> for node in g.nodes() :
    >>>     part[node] = node % 2
    >>> ind = induced_graph(part, g)
    >>> goal = nx.Graph()
    >>> goal.add_weighted_edges_from([(0,1,n*n),(0,0,n*(n-1)/2), (1, 1, n*(n-1)/2)])  # NOQA
    >>> nx.is_isomorphic(ind, goal)
    True
    """
    ret = nx.Graph()
    ret.add_nodes_from(partition.values())

    for node1, node2, datas in graph.edges(data=True):
        edge_weight = datas.get(weight, 1)
        com1 = partition[node1]
        com2 = partition[node2]
        w_prec = ret.get_edge_data(com1, com2, {weight: 0}).get(weight, 1)
        ret.add_edge(com1, com2, **{weight: w_prec + edge_weight})

    return ret


def __renumber(dictionary):
    """Renumber the values of the dictionary from 0 to n
    """
    values = set(dictionary.values())
    target = set(range(len(values)))

    if values == target:
        # no renumbering necessary
        ret = dictionary.copy()
    else:
        # add the values that won't be renumbered
        renumbering = dict(zip(target.intersection(values),
                               target.intersection(values)))
        # add the values that will be renumbered
        renumbering.update(dict(zip(values.difference(target),
                                    target.difference(values))))
        ret = {k: renumbering[v] for k, v in dictionary.items()}

    return ret


def load_binary(data):
    """Load binary graph as used by the cpp implementation of this algorithm
    """
    data = open(data, "rb")

    reader = array.array("I")
    reader.fromfile(data, 1)
    num_nodes = reader.pop()
    reader = array.array("I")
    reader.fromfile(data, num_nodes)
    cum_deg = reader.tolist()
    num_links = reader.pop()
    reader = array.array("I")
    reader.fromfile(data, num_links)
    links = reader.tolist()
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    prec_deg = 0

    for index in range(num_nodes):
        last_deg = cum_deg[index]
        neighbors = links[prec_deg:last_deg]
        graph.add_edges_from([(index, int(neigh)) for neigh in neighbors])
        prec_deg = last_deg

    return graph


def __one_level(graph, status, weight_key, resolution, random_state, neighbor_mode="all"):
    """Compute one level of communities
    neighbor_mode: ways of selecting neighbor communities
        all: tranverse all neighbor communities
        unweighted: uniformly sample a neighbor 
        weighted: sample a neighbor based on edge probs
        queue: leiden
    """
    modified = True
    nb_pass_done = 0
    cur_mod = __modularity(status, resolution)
    new_mod = cur_mod

    num_loops = 0
    avg_time_per_loop = 0.0

    while modified and nb_pass_done != __PASS_MAX:
        num_loops += 1

        cur_mod = new_mod
        modified = False
        nb_pass_done += 1

        alg_st = time.time()
        if neighbor_mode == "all":
            for node in __randomize(graph.nodes(), random_state):
                com_node = status.node2com[node]
                degc_totw = status.gdegrees.get(node, 0.) / (status.total_weight * 2.)  # NOQA
                neigh_communities = __neighcom(node, graph, status, weight_key)
                remove_cost = - neigh_communities.get(com_node,0) + \
                    resolution * (status.degrees.get(com_node, 0.) - status.gdegrees.get(node, 0.)) * degc_totw
                __remove(node, com_node,
                        neigh_communities.get(com_node, 0.), status)
                best_com = com_node
                best_increase = 0
                for com, dnc in __randomize(neigh_communities.items(), random_state):
                    incr = remove_cost + dnc - \
                        resolution * status.degrees.get(com, 0.) * degc_totw
                    if incr > best_increase:
                        best_increase = incr
                        best_com = com
                if best_increase > __MIN:
                    __insert(node, best_com,
                        neigh_communities.get(best_com, 0.), status)
                    if best_com != com_node:
                        modified = True
                else:
                    __insert(node, com_node,
                        neigh_communities.get(com_node, 0.), status)
                
        elif neighbor_mode == "unweighted":
            for node in __randomize(graph.nodes(), random_state):
                com_node = status.node2com[node]
                degc_totw = status.gdegrees.get(node, 0.) / (status.total_weight * 2.)  # NOQA
                neigh_communities = __neighcom(node, graph, status, weight_key)
                remove_cost = - neigh_communities.get(com_node,0) + \
                    resolution * (status.degrees.get(com_node, 0.) - status.gdegrees.get(node, 0.)) * degc_totw
                __remove(node, com_node,
                        neigh_communities.get(com_node, 0.), status)
                
                neighbors = list(graph[node].keys())
                deg_of_node = len(neighbors) # unweighted degree
                r = random_state.randint(deg_of_node) # Sample a neighbor id
                r_neighbor = neighbors[r]
                best_com = status.node2com.get(r_neighbor)
                if best_com == -1: # r_neighbor == node
                    __insert(node, com_node,
                            neigh_communities.get(best_com, 0.), status) # change back
                    continue

                dnc = neigh_communities[best_com]
                incr = remove_cost + dnc - \
                            resolution * status.degrees.get(best_com, 0.) * degc_totw
                if incr > __MIN:
                    __insert(node, best_com,
                            neigh_communities.get(best_com, 0.), status) # strict increase
                    # print(f"{node}: {com_node}->{best_com} {incr}") 
                    if best_com != com_node:
                        modified = True
                else:
                    __insert(node, com_node,
                            neigh_communities.get(best_com, 0.), status) # change back
                
                
        elif neighbor_mode == "weighted":
            for node in __randomize(graph.nodes(), random_state):
                com_node = status.node2com[node]
                degc_totw = status.gdegrees.get(node, 0.) / (status.total_weight * 2.)  # NOQA
                neigh_communities = __neighcom(node, graph, status, weight_key)
                remove_cost = - neigh_communities.get(com_node,0) + \
                    resolution * (status.degrees.get(com_node, 0.) - status.gdegrees.get(node, 0.)) * degc_totw
                __remove(node, com_node,
                        neigh_communities.get(com_node, 0.), status)
                
                ps = np.array([x.get(weight_key, 0) for x in graph[node].values()])
                # ps = np.exp(ps) / sum(np.exp(ps)) # Transform neighbor weights into distribution
                ps = softmax(ps)
                # ps = ps/sum(ps)

                deg_of_node = len(graph[node]) # unweighted degree
                r = random_state.choice(deg_of_node, size=1, p=ps)[0] # Sample a neighbor id
                r_neighbor = list(graph[node].keys())[r] # TODO: 确保keys的顺序和ps的顺序一致
                best_com = status.node2com[r_neighbor]

                if best_com != -1:
                    dnc = neigh_communities[best_com]
                    incr = remove_cost + dnc - \
                                resolution * status.degrees.get(best_com, 0.) * degc_totw
                    if incr > __MIN:
                        __insert(node, best_com,
                            neigh_communities.get(best_com, 0.), status) # strict increase 
                        if best_com != com_node:
                            modified = True
                    else:
                        __insert(node, com_node,
                            neigh_communities.get(com_node, 0.), status) # change back
                else:
                    __insert(node, com_node,
                            neigh_communities.get(com_node, 0.), status) # change back
        elif neighbor_mode == "queue":
            q = list(__randomize(graph.nodes(), random_state)) # Initialize a queue
            q_mask = np.ones_like(q, dtype=int)
            while len(q) > 0:
                node = q.pop(0)
                q_mask[node] = 0
                com_node = status.node2com[node]
                degc_totw = status.gdegrees.get(node, 0.) / (status.total_weight * 2.)  # NOQA
                neigh_communities = __neighcom(node, graph, status, weight_key)
                remove_cost = - neigh_communities.get(com_node,0) + \
                    resolution * (status.degrees.get(com_node, 0.) - status.gdegrees.get(node, 0.)) * degc_totw
                __remove(node, com_node,
                        neigh_communities.get(com_node, 0.), status)
                
                best_com = com_node
                best_increase = 0
                for com, dnc in __randomize(neigh_communities.items(), random_state):
                    incr = remove_cost + dnc - \
                        resolution * status.degrees.get(com, 0.) * degc_totw
                    if incr > best_increase:
                        best_increase = incr
                        best_com = com
                if best_increase > __MIN:
                    __insert(node, best_com,
                            neigh_communities.get(best_com, 0.), status)
                    
                    if best_com != com_node:
                        # Push node's neighbors to queue
                        node_list = np.array([x for x in range(graph.number_of_nodes())])
                        neighbors = list(graph[node].keys())

                        com_mask = np.zeros_like(node_list, dtype=int)
                        for neighbor in neighbors:
                            if neighbor != node and status.node2com[neighbor] != best_com:
                                com_mask[neighbor] = 1 # neighbor not in best_com will be pushed
                        
                        finalized_mask = (1 - q_mask) * com_mask
                        finalized_mask = np.array(finalized_mask, dtype=bool)
                        finalized_node_list = node_list[finalized_mask]
                        # print(f"{len(q)} {len(finalized_node_list)} Node {node} {com_node} --> {best_com} {best_increase}")

                        for neighbor in finalized_node_list:
                            q.append(neighbor) # Push neighbors into queue
                            q_mask[neighbor] = 1
                        # assert sum(q) - sum(np.unique(q)) == 0
                else:
                    __insert(node, com_node,
                        neigh_communities.get(com_node, 0.), status)
        alg_ed = time.time()
        avg_time_per_loop += (alg_ed-alg_st)

        new_mod = __modularity(status, resolution)
        if new_mod - cur_mod < __MIN:
            break
    
    # print(f"{neighbor_mode} Num loops: {num_loops} Avg. Time per loop: {avg_time_per_loop/num_loops:.2f}")
    return num_loops

def __neighcom(node, graph, status, weight_key):
    """
    Compute the communities in the neighborhood of node in the graph given
    with the decomposition node2com
    """
    weights = {}
    for neighbor, datas in graph[node].items(): # TODO: still visit all neighbors
        if neighbor != node:
            edge_weight = datas.get(weight_key, 1)
            neighborcom = status.node2com[neighbor]
            weights[neighborcom] = weights.get(neighborcom, 0) + edge_weight

    return weights


def __remove(node, com, weight, status):
    """ Remove node from community com and modify status"""
    status.degrees[com] = (status.degrees.get(com, 0.)
                           - status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) -
                                  weight - status.loops.get(node, 0.))
    status.node2com[node] = -1


def __insert(node, com, weight, status):
    """ Insert node into community and modify status"""
    status.node2com[node] = com
    status.degrees[com] = (status.degrees.get(com, 0.) +
                           status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) +
                                  weight + status.loops.get(node, 0.))


def __modularity(status, resolution):
    """
    Fast compute the modularity of the partition of the graph using
    status precomputed
    """
    links = float(status.total_weight)
    result = 0.
    for community in set(status.node2com.values()):
        in_degree = status.internals.get(community, 0.)
        degree = status.degrees.get(community, 0.)
        if links > 0:
            result += in_degree * resolution / links -  ((degree / (2. * links)) ** 2)
    return result


def __randomize(items, random_state):
    """Returns a List containing a random permutation of items"""
    randomized_items = list(items)
    random_state.shuffle(randomized_items)
    return randomized_items


def load_complete_graph(dataset, seed):
    """dataset in cora,citeseer
    """
    emb_path = f"/home/yliumh/github/AutoAtCluster/emb_models/GGD/manual_version/outputs/GGD_{dataset}_emb_{seed}.npz"
    try:
        data = np.load(emb_path)
        emb = data["emb"]
        emb = torch.FloatTensor(emb)
        emb = F.normalize(emb)
        sim = torch.mm(emb, emb.t())
        sim = F.relu(sim).numpy()
        graph = nx.from_numpy_array(sim)
        return graph
    except Exception as e:
        print(e)

def load_knn_graph(dataset, k, seed):
    try:
        knn_graph_path = f"/home/yliumh/github/AutoAtCluster/baselines/KNN/outputs/knn_adj_{dataset}_{seed}_{k}.npz"
        data = np.load(knn_graph_path)
        adj_data, adj_row, adj_col = data["data"], data["row"], data["col"]
        knn_adj = sp.coo_matrix((adj_data, (adj_row, adj_col)))
        graph = nx.from_scipy_sparse_array(knn_adj)
        return graph
    except Exception as e:
        print(e)

def simple_test():
    graph = nx.karate_club_graph()
    graph = nx.chordal_cycle_graph(p=1700)
    # graph = nx.complete_graph(n=1700)
    # graph = nx.read_weighted_edgelist("syndata/data2.txt", create_using=nx.Graph)
    # graph = load_complete_graph("citeseer", 0)
    graph = load_knn_graph("cora", 1, 0)
    print(f"#Nodes= {graph.number_of_nodes()} #Edges= {graph.number_of_edges()}")
    partition = best_partition(graph, random_state=0, neighbor_mode="queue")
    preds = list(partition.values())

def louvain_cluster(adj, labels, random_state=None, neighbor_mode="all"):
    graph = nx.from_scipy_sparse_array(adj)
    partition = best_partition(graph, random_state=random_state, neighbor_mode=neighbor_mode) # Use louvain defined in this file
    preds = list(partition.values())
    return preds

"""
Faster unfolding of communities: speeding up the Louvain algorithm
    - nc: number of nodes in each clique
    - r: number of cliques
    - connected: if true, connect each pair of clique using one edge
"""
def test_ring_of_cliques():
    nc = 10
    r = 1000
    n = nc*r

    data = np.load(f"dataset/ring_of_cliques_{nc}_{r}.npz")
    adj_data, adj_row, adj_col = data["data"], data["row"], data["col"]
    adj = sp.coo_matrix((adj_data, (adj_row, adj_col)), shape=(n,n))

    alg_st = time.time()
    louvain_cluster(adj, None, 0, "all")
    alg_ed = time.time()
    time_cost_all = alg_ed-alg_st
    print(f"Time cost all: {time_cost_all:.2f}")

    alg_st = time.time()
    louvain_cluster(adj, None, 0, "unweighted")
    alg_ed = time.time()
    time_cost_unweighted = alg_ed-alg_st
    print(f"Time cost unweighted: {time_cost_unweighted:.2f}")

    alg_st = time.time()
    louvain_cluster(adj, None, 0, "queue")
    alg_ed = time.time()
    time_cost_queue = alg_ed-alg_st
    print(f"Time cost queue: {time_cost_queue:.2f}")

    print(f"Time cost: {time_cost_all:.2f}\t{time_cost_unweighted:.2f}\t{time_cost_queue:.2f}")

    


if __name__ == "__main__":

    # simple_test()
    # test_ring_of_cliques()

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["all", "unweighted", "weighted", "queue"])
    args = parser.parse_args()

    datasets = [
        "cora",
        "citeseer",
        "wiki",
        "pubmed",
        "amazon-photo",
        "amazon-computers",
        "cora-full",
        "ogbn-arxiv"
    ]

    # neighbor_modes = [
    #     "unweighted",
    #     "queue",
    #     "all",
    #     "weighted"
    # ]
    neighbor_modes = [
        f"{args.mode}"
    ]
    
    seeds = np.arange(0, 3, dtype=int)

    os.makedirs("outputs", exist_ok=True)
    expdir = f"myLouvain-formal"


    for neighbor_mode in neighbor_modes:
        for dataset in datasets:
            for seed in seeds:
                logger.info(f"{dataset}, {seed}")

                np.random.seed(seed)
                random.seed(seed)

                setproctitle.setproctitle("Louvain-{}-{}-{}".format(neighbor_mode, dataset, seed))

                adj, features, labels = load_data(dataset)

                n = adj.shape[0]
                m = adj.sum()
                edges = np.arange(m, 10*m, m, dtype=int)
                edges = np.concatenate([edges, np.arange(10*m, 101*m, 10*m, dtype=int)])

                for m2 in edges:
                    if os.path.exists("outputs/{}/Louvain_{}_{}_{:.0f}_{}.npz".format(expdir, dataset, seed, m2/m, neighbor_mode)):
                        logger.info(f"Skip: knn_adj_{dataset}_{seed}_{m2/m:.0f}_{neighbor_mode}.npz")
                        continue

                    knn_graph_path = f"/home/yliumh/github/AutoAtCluster/baselines/KNN/outputs/knn_adj_{dataset}_{seed}_{m2/m:.0f}.npz"

                    try:
                        data = np.load(knn_graph_path)
                        adj_data, adj_row, adj_col = data["data"], data["row"], data["col"]
                        knn_adj = sp.coo_matrix((adj_data, (adj_row, adj_col)), shape=(n,n))
                        
                        # knn_adj = knn_graph(emb, k, non_linear, i=6)
                        alg_st = time.time()
                        preds = louvain_cluster(knn_adj, labels, random_state=seed, neighbor_mode=neighbor_mode)
                        alg_end = time.time()
                    
                        time_now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

                        os.makedirs(f"outputs/{expdir}", exist_ok=True)
                        with open(f"outputs/{expdir}/time.txt", "a+") as f:
                            f.write(f"{dataset}\t{seed}\t{m2/m:.0f}\t{neighbor_mode}\t{alg_end-alg_st}\t{time_now}\n")
                        np.savez("outputs/{}/Louvain_{}_{}_{:.0f}_{}.npz".format(expdir, dataset, seed, m2/m, neighbor_mode), preds=preds, labels=labels)
                    except Exception as e:
                        logger.error(f"Error: knn_adj_{dataset}_{seed}_{m2/m:.0f}_{neighbor_mode}.npz\n{traceback.format_exc()}")
                    else:
                        logger.info(f"Success: knn_adj_{dataset}_{seed}_{m2/m:.0f}_{neighbor_mode}.npz")
            
    pass