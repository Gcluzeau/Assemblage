#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""Perform assembly based on debruijn graph."""

import argparse
import os
import sys
from pathlib import Path
from networkx import (
    DiGraph,
    all_simple_paths,
    lowest_common_ancestor,
    has_path,
    random_layout,
    draw,
    spring_layout,
)
import matplotlib
from operator import itemgetter
import random

random.seed(9001)
from random import randint
import statistics
import textwrap
import matplotlib.pyplot as plt
from typing import Iterator, Dict, List

matplotlib.use("Agg")

__author__ = "Your Name"
__copyright__ = "Universite Paris Diderot"
__credits__ = ["Your Name"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Your Name"
__email__ = "your@email.fr"
__status__ = "Developpement"


def isfile(path: str) -> Path:  # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file

    :raises ArgumentTypeError: If file does not exist

    :return: (Path) Path object of the input file
    """
    myfile = Path(path)
    if not myfile.is_file():
        if myfile.is_dir():
            msg = f"{myfile.name} is a directory."
        else:
            msg = f"{myfile.name} does not exist."
        raise argparse.ArgumentTypeError(msg)
    return myfile


def get_arguments():  # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description=__doc__, usage=f"{0} -h".format(sys.argv[0])
    )
    parser.add_argument(
        "-i", dest="fastq_file", type=isfile, required=True, help="Fastq file"
    )
    parser.add_argument(
        "-k", dest="kmer_size", type=int, default=22, help="k-mer size (default 22)"
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=Path,
        default=Path(os.curdir + os.sep + "contigs.fasta"),
        help="Output contigs in fasta file (default contigs.fasta)",
    )
    parser.add_argument(
        "-f", dest="graphimg_file", type=Path, help="Save graph as an image (png)"
    )
    if len(sys.argv) == 1:
        sys.argv.extend(["-i", "example.fastq"])  # Example default FASTQ file
        sys.argv.extend(["-k", "22"])  # Default k-mer size
    return parser.parse_args()


def read_fastq(fastq_file: Path) -> Iterator[str]:
    """Extract reads from fastq files.

    :param fastq_file: (Path) Path to the fastq file.
    :return: A generator object that iterate the read sequences.
    """
    with fastq_file.open('r') as f:
        # Read the file 4 lines at a time (since each read in FASTQ is 4 lines)
        while True:
            f.readline()  # Skip the identifier line (e.g. "@SEQ_ID")
            seq = f.readline().strip()  # Read sequence (second line)
            f.readline()  # Skip the plus line ("+")
            f.readline()  # Skip the quality score line
            if not seq:
                break
            yield seq  # Yield the sequence for further processing


def cut_kmer(read: str, kmer_size: int) -> Iterator[str]:
    """Cut read into kmers of size kmer_size.

    :param read: (str) Sequence of a read.
    :return: A generator object that provides the kmers (str) of size kmer_size.
    """
    for i in range(len(read) - kmer_size + 1):
        yield read[i:i + kmer_size]  # Generate each k-mer from the sequence

def build_kmer_dict(fastq_file: Path, kmer_size: int) -> Dict[str, int]:
    """Build a dictionnary object of all kmer occurrences in the fastq file

    :param fastq_file: (str) Path to the fastq file.
    :return: A dictionnary object that identify all kmer occurrences.
    """
    kmer_dict = {}
    # Read sequences from the FASTQ file
    for read in read_fastq(fastq_file): # Cut the read into k-mers and count occurrences
        for kmer in cut_kmer(read, kmer_size):
            if kmer in kmer_dict:
                kmer_dict[kmer] += 1
            else:
                kmer_dict[kmer] = 1
    
    return kmer_dict


def build_graph(kmer_dict: Dict[str, int]) -> DiGraph:
    """Build the debruijn graph

    :param kmer_dict: A dictionnary object that identify all kmer occurrences.
    :return: A directed graph (nx) of all kmer substring and weight (occurrence).
    """
    graph = DiGraph()
    
    for kmer, weight in kmer_dict.items():
        prefix = kmer[:-1]  #  prefixe
        suffix = kmer[1:]  # suffixe
    
        graph.add_edge(prefix, suffix, weight=weight)

    return graph

def remove_paths(
    graph: DiGraph,
    path_list: List[List[str]],
    delete_entry_node: bool,
    delete_sink_node: bool,
) -> DiGraph:
    """Remove a list of path in a graph. A path is set of connected node in
    the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    for path in path_list:
               
        # Remove the intermediate edges of the path (without entry and sink if specified)
        if delete_entry_node and delete_sink_node:
            # Remove the entire path including entry and sink nodes
            nodes_to_remove = path
        elif delete_entry_node:
            # Remove the entry node and intermediate edges
            nodes_to_remove = path[:-1]
        elif delete_sink_node:
            # Remove the intermediate and sink node
            nodes_to_remove = path[1:]
        else:
            # Remove only the intermediate nodes (keep entry and sink)
            nodes_to_remove = path[1:-1]  # Exclude first and last nodes

        # Remove nodes and their associated edges from the graph
        graph.remove_nodes_from(nodes_to_remove)
    
    return graph



def select_best_path(
    graph: DiGraph,
    path_list: List[List[str]],
    path_length: List[int],
    weight_avg_list: List[float],
    delete_entry_node: bool = False,
    delete_sink_node: bool = False,
) -> DiGraph:
    """Select the best path between different paths

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param path_length_list: (list) A list of length of each path
    :param weight_avg_list: (list) A list of average weight of each path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    # 1. Verification des ecarts sur les poids
    if statistics.stdev(weight_avg_list) > 0:
        # Si l'ecart type des poids est superieur a 0, on prend le chemin avec le poids le plus élevé
        best_path_index = weight_avg_list.index(max(weight_avg_list))
    elif statistics.stdev(path_length) > 0:
        # Si les poids sont egaux, on verifie la longueur des chemins
        # Si l'ecart type des longueurs est superieur a 0, on prend le chemin le plus long
        best_path_index = path_length.index(max(path_length))
    else:
        # Si tous les chemins ont le meme poids et la meme longueur, on choisit au hasard
        best_path_index = random.randint(0, len(path_list) - 1)

    # 2. Supprimer les autres chemins du graphe
    paths_to_remove = [path for i, path in enumerate(path_list) if i != best_path_index]

    # Appel a une fonction pour supprimer les chemins non selectionnes
    graph = remove_paths(graph, paths_to_remove, delete_entry_node, delete_sink_node)

    return graph


def path_average_weight(graph: DiGraph, path: List[str]) -> float:
    """Compute the weight of a path

    :param graph: (nx.DiGraph) A directed graph object
    :param path: (list) A path consist of a list of nodes
    :return: (float) The average weight of a path
    """
    return statistics.mean(
        [d["weight"] for (u, v, d) in graph.subgraph(path).edges(data=True)]
    )


def solve_bubble(graph: DiGraph, ancestor_node: str, descendant_node: str) -> DiGraph:
    """Explore and solve bubble issue

    :param graph: (nx.DiGraph) A directed graph object
    :param ancestor_node: (str) An upstream node in the graph
    :param descendant_node: (str) A downstream node in the graph
    :return: (nx.DiGraph) A directed graph object
    """
    # Find all simple paths between the ancestor and descendant
    all_paths = list(all_simple_paths(graph, source=ancestor_node,target=descendant_node))

    path_lengths = [len(path) for path in all_paths]
    weight_avgs = [path_average_weight(graph, path) for path in all_paths]
    
    graph = select_best_path(graph, all_paths, path_lengths, weight_avgs, delete_entry_node=False, delete_sink_node=False)

    return graph


def simplify_bubbles(graph: DiGraph) -> DiGraph:
    """Detect and explode bubbles

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    bubble_found = False  # Indicateur pour savoir si une bulle est detectee

    # Iterer sur chaque noeud du graphe
    for node in list(graph.nodes):
        # Obtenir la liste des predecesseurs du noeud actuel
        predecessors = list(graph.predecessors(node))

        # Si le noeud a plus d'un predecesseur, cela peut etre une bulle
        if len(predecessors) > 1:
            # Verifier les combinaisons uniques de predecesseurs
            for i in range(len(predecessors)):
                for j in range(i + 1, len(predecessors)):
                    pred_i = predecessors[i]
                    pred_j = predecessors[j]

                    # Trouver l'ancetre commun le plus bas entre les deux predecesseurs
                    ancestor_node = lowest_common_ancestor(graph, pred_i, pred_j)

                    if ancestor_node is not None:
                        # Bulle detectee entre ancestor_node et node
                        bubble_found = True
                        break  # On sort de la boucle des qu'une bulle est trouvee
                if bubble_found:
                    break

        if bubble_found:
            break

    # Si une bulle a ete detectee, la resoudre et rappeler simplify_bubbles de maniere recursive
    if bubble_found:
        # Appel a solve_bubble pour resoudre la bulle
        graph = solve_bubble(graph, ancestor_node, node)

        # Reappeler simplify_bubbles pour traiter les autres bulles potentiellement presentes
        return simplify_bubbles(graph)
    
    # Retourner le graphe simplifie si aucune bulle n'a ete trouvee
    return graph


def solve_entry_tips(graph: DiGraph, starting_nodes: List[str]) -> DiGraph:
    """Remove entry tips

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of starting nodes
    :return: (nx.DiGraph) A directed graph object
    """
    for node in graph.nodes:
        # Obtenir les predecesseurs de chaque noeud
        predecessors = list(graph.predecessors(node))
        
        if len(predecessors) > 1:

            path_list = []
            path_lengths = []
            weight_avg_list = []

            for start_node in starting_nodes:
                if has_path(graph, start_node, node):
                    # Obtenir tous les chemins simples entre le noeud de depart et le noeud d'arrivee
                    for path in all_simple_paths(graph, start_node, node):
                        if len(path) >= 2:
                            path_list.append(path)
                            path_lengths.append(len(path))
                            weight_avg_list.append(path_average_weight(graph, path))

            # Selectionnez le meilleur chemin si plusieur
            if len(path_list) > 1:
                graph = select_best_path(graph, path_list, path_lengths, weight_avg_list, delete_entry_node=True, delete_sink_node=False)
                return solve_entry_tips(graph, get_starting_nodes(graph))
    return graph


def solve_out_tips(graph: DiGraph, ending_nodes: List[str]) -> DiGraph:
    """Remove out tips

    :param graph: (nx.DiGraph) A directed graph object
    :param ending_nodes: (list) A list of ending nodes
    :return: (nx.DiGraph) A directed graph object
    """
    for node in graph.nodes:
        # Obtenir les predecesseurs de chaque noeud
        successors = list(graph.successors(node))
        
        if len(successors) > 1:

            path_list = []
            path_lengths = []
            weight_avg_list = []

            for end_node in ending_nodes:
                if has_path(graph, node, end_node):
                    # Obtenir tous les chemins simples entre le noeud de depart et le noeud d'arrivee
                    for path in all_simple_paths(graph, node, end_node):
                        if len(path) >= 2:
                            path_list.append(path)
                            path_lengths.append(len(path))
                            weight_avg_list.append(path_average_weight(graph, path))

            # Selectionnez le meilleur chemin si plusieur
            if len(path_list) > 1:
                graph = select_best_path(graph, path_list, path_lengths, weight_avg_list, delete_entry_node=False, delete_sink_node=True)
                return solve_out_tips(graph, get_sink_nodes(graph))
    return graph


def get_starting_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without predecessors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without predecessors
    """
    starting_nodes = []
    for node in graph.nodes:
        # Si le noeud n'a aucun predecesseur, c'est un noeud d'entree
        if not any(graph.predecessors(node)):  # Utilise directement l'iterateur
            starting_nodes.append(node)
    
    return starting_nodes


def get_sink_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without successors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without successors
    """
    sink_nodes = []
    # Liste des noeuds sans successeurs
    for node in graph.nodes:
        # Si le noeud n'a pas de sucesseur, il est considere comme un noeud de sortie
        if not any(graph.successors(node)):  # Utilise directement l'iterateur
            sink_nodes.append(node)
        
    return sink_nodes

def get_contigs( graph: DiGraph, starting_nodes: List[str], ending_nodes: List[str]) -> List:
    """Extract the contigs from the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of nodes without predecessors
    :param ending_nodes: (list) A list of nodes without successors
    :return: (list) List of [contiguous sequence and their length]
    """
    contigs = []

    # Parcours de chaque noeud d'entrze
    for start_node in starting_nodes:
        # Parcours de chaque noeud de sortie
        for end_node in ending_nodes:
            # Verifie s'il existe un chemin entre le noeud de depart et le noeud d'arrivee
            if has_path(graph, start_node, end_node):
                # Pour chaque chemin simple entre start_node et end_node
                for path in all_simple_paths(graph, start_node, end_node):
                    # Construire le contig a partir du chemin
                    contig = path[0]  # Initialisation avec le premier noeud
                    for node in path[1:]:
                        contig += node[-1]  # Ajoute seulement la derniere lettre de chaque noeud suivant
                    # Ajouter le contig et sa longueur a la liste des resultats
                    contigs.append((contig, len(contig)))
    
    return contigs


def save_contigs(contigs_list: List[str], output_file: Path) -> None:
    """Write all contigs in fasta format

    :param contig_list: (list) List of [contiguous sequence and their length]
    :param output_file: (Path) Path to the output file
    """
    #output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open('w') as f:
        # Parcours de tous les contigs
        for i, (contig, length) in enumerate(contigs_list):
            # Ecriture d'un en-tete FASTA pour chaque contig
            f.write(f">contig_{i} len={length}\n")
            # Formatage et ecriture du contig avec un wrapping a 80 caracteres par ligne
            f.write(textwrap.fill(contig, 80) + "\n")


def draw_graph(graph: DiGraph, graphimg_file: Path) -> None:  # pragma: no cover
    """Draw the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param graphimg_file: (Path) Path to the output file
    """
    fig, ax = plt.subplots()
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] > 3]
    # print(elarge)
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] <= 3]
    # print(elarge)
    # Draw the graph with networkx
    # pos=nx.spring_layout(graph)
    pos = random_layout(graph)
    draw(graph, pos, node_size=6)
    draw(graph, pos, edgelist=elarge, width=6)
    draw(graph, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed")
    # nx.draw_networkx(graph, pos, node_size=10, with_labels=False)
    # save image
    plt.savefig(graphimg_file.resolve())


# ==============================================================
# Main program
# ==============================================================
def main() -> None:  # pragma: no cover
    """
    Main program function
    """
    # Get arguments
    args = get_arguments()

    # Fonctions de dessin du graphe
    # A decommenter si vous souhaitez visualiser un petit
    # graphe
    # Plot the graph
    # if args.graphimg_file:
    #     draw_graph(graph, args.graphimg_file)


    fastq_file = args.fastq_file
    kmer_size = args.kmer_size
    output_file = args.output_file
    


    kmer_dict = build_kmer_dict(fastq_file, kmer_size)
    graph = build_graph(kmer_dict)

    
    graph = simplify_bubbles(graph)


    graph = solve_entry_tips(graph, get_starting_nodes(graph))
    graph = solve_out_tips(graph, get_sink_nodes(graph))

    start_nodes = get_starting_nodes(graph)
    sink_nodes = get_sink_nodes(graph)

    contigs = get_contigs(graph, start_nodes, sink_nodes)
    save_contigs(contigs, output_file)

    draw_graph(graph, Path("image-graph.jpeg") )
    
if __name__ == "__main__":  # pragma: no cover
    main()






