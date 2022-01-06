#!/usr/bin/env python3
from math import log
from typing import List, Tuple, Dict
import sys
from collections import Counter


# Used for typing
Data = List[List]

def _parse_value(value: str):
    try:
        return float(value)
    except ValueError:
        return value

def read(file_name: str) -> Tuple[List[str], Data]:
    """
    t3: Load the data into a bidimensional list.
    Return the headers as a list, and the data
    """
    data = list()
    with open(file_name, "r") as file_:
        headers = file_.readline().strip("\n").split(",")
        for row in file_:
            values = row.strip("\n").split(",")
            values = [_parse_value(v) for v in values]
            data.append(values)
    return headers,data



def unique_counts(part: Data):
    """
    t4: Create counts of possible results
    (the last column of each row is the
    result)
    """     
    results = Counter()
    for row in part:
        label = row[-1]
        results[label]+= 1
    return results


def gini_impurity(part: Data):
    """
    t5: Computes the Gini index of a node
    """
    total = len(part)
    if total == 0:
        return 0
    results = unique_counts(part)
    imp = 1
    for value in results.values():
        p = (value/ total)
        imp -= p**2
    return imp


def _log2(value: float):
    return log(value) / log(2)


def entropy(rows: Data):
    """
    t6: Entropy is the sum of p(x)log(p(x))
    across all the different possible results
    """
    results = unique_counts(rows)
    total = len(rows)
    imp = 0
    for value in results.values():
        p = (value/total)
        imp -= (p*_log2(p))
    return imp


def _split_numeric(prototype: List, column: int, value):
    return prototype[column] >= value


def _split_categorical(prototype: List, column: int, value) -> bool:
    return prototype[column] == value



def divideset(part: Data, column: int, value: int) -> Tuple[Data, Data]:
    """
    t7: Divide a set on a specific column. Can handle
    numeric or categorical values
    """
    if isinstance(value, (int, float)):
        split_function = _split_numeric
    else:
        split_function = _split_categorical

    set1, set2 = list(), list()
    for row in part:
        if split_function(row,column,value):
            set1.append(row)
        else:
            set2.append(row)

    return (set1, set2)


class DecisionNode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None,impurity = None):
        """
        t8: We have 5 member variables:
        - col is the column index which represents the
          attribute we use to split the node
        - value corresponds to the answer that satisfies
          the question
        - tb and fb are internal nodes representing the
          positive and negative answers, respectively
        - results is a dictionary that stores the result
          for this branch. Is None except for the leaves
        """
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb
        self.impurity = impurity

def _gain(part: Data, set1: Data, set2: Data, scoref):
    p1 = len(set1) / len(part)
    p2 = len(set2) / len(part)
    return scoref(part) - p1 * scoref(set1) - p2 * scoref(set2)

def buildtree(part: Data, scoref=entropy, beta=0):
    """
    t9: Define a new function buildtree. This is a recursive function
    that builds a decision tree using any of the impurity measures we
    have seen. The stop criterion is max_s\Delta i(s,t) < \beta
    """

    """
    1-Calcular impureza del node ; Nos la proporciona el codigo 
    2-Decidir mejor pregunta
    3-recursividad
    """
    if len(part) == 0:
        return DecisionNode()

    current_score = scoref(part)

    if current_score == 0:
        # The partition is pure
        return DecisionNode(results=unique_counts(part),impurity=current_score)

    # Set up some variables to track the best criteria
    best_gain = 0
    best_criteria = None
    best_sets = None

    n_cols = len(part[0]) - 1  # Skip the label

    for i in range(n_cols):
        possibles_cut_values = set()
        for row in part:
            possibles_cut_values.add(row[i])

        for value in possibles_cut_values:
            set1, set2 = divideset(part, i, value)
            gain = _gain(part, set1, set2, scoref)
            if gain > best_gain:
                best_gain = gain
                best_criteria = (i, value)
                best_sets = set1, set2

    if best_gain < beta:
        return DecisionNode(results=unique_counts(part),impurity=current_score)

    return DecisionNode(col=best_criteria[0], value=best_criteria[1],
        tb=buildtree(best_sets[0]), fb=buildtree(best_sets[1]),impurity=current_score)
    """
    if len(part) == 0:
        return DecisionNode()
    #1-Calcular impureza
    current_score = scoref(part)
    #No usamos Beta
    if current_score == 0:
        return DecisionNode(
            results=unique_counts(part),impurity=current_score
        )
    

    # Set up some variables to track the best criteria
    best_gain = 0
    best_criteria = None
    best_sets = None

    n_cols = len(part[0]) -1


    #2-Buscar mejor pregunta
    for col_idx in range(n_cols):
        for value in _get_values(part,col_idx):
            set1, set2 = divideset(part,col_idx,value)
            dism_true = len(set1) / len(part) * scoref(set1)
            dism_false = len(set2) / len(part) *scoref(set2)
            gain = current_score - dism_true - dism_false
            if gain > best_gain:
                best_gain = gain
                best_criteria = (col_idx,value)
                best_sets = set1,set2
    #Usamos beta, Logica de beta
    if best_gain < beta:
        return DecisionNode(results = unique_counts(part))

    return DecisionNode(col=best_criteria[0], value=best_criteria[1],
        tb=buildtree(best_sets[0]), fb=buildtree(best_sets[1]),impurity=current_score)
    """
    """
    Scoref representa el indice de impureza

    Cada particion disminuye la impureza por ello a mas ganancia, la impureza estara mas reduida.
    Valor de goodnes mas grande *Apuntes*
    """

def _get_values(rows,col_idx):
    value = set()
    for row in rows:
        value.add(row[col_idx])
    return value

def iterative_buildtree(part: Data, scoref=entropy, beta=0):
    #LIMPIAR CODIGO
    """
    t10: Define the iterative version of the function buildtree
    """

    """
    t9: Define a new function buildtree. This is a recursive function
    that builds a decision tree using any of the impurity measures we
    have seen. The stop criterion is max_s\Delta i(s,t) < \beta
    """

    """
    1-Calcular impureza del node ; Nos la proporciona el codigo 
    2-Decidir mejor pregunta
    3-recursividad
    """
    if len(part) == 0:
        return DecisionNode()
    nodes = []
    nodes.append(DecisionNode())
    data = []
    data.append(part)
    i = 0 
    while len(nodes) != 0:
        data_part  = data.pop() 
        # Set up some variables to track the best criteria
        curr_node :DecisionNode = nodes.pop()
        i+=1
        best_gain = 0
        best_criteria = None
        best_sets = None

        n_cols = len(part[0]) -1

        #1-Calcular impureza
        current_score = scoref(data_part)

        #No usamos Beta
        if current_score == 0:
            curr_node.results = unique_counts(data_part)
            curr_node.impurity = current_score
            continue
        for col_idx in range(n_cols):
            for value in _get_values(data_part,col_idx):
                set1, set2 = divideset(data_part,col_idx,value)
                dism_true = len(set1) / len(data_part) * scoref(set1)
                dism_false = len(set2) / len(data_part) *scoref(set2)
                gain = current_score - dism_true - dism_false
                if gain > best_gain:
                    best_gain = gain
                    best_criteria = (col_idx,value)
                    best_sets = set1,set2
        if best_gain < beta:
            curr_node.results = unique_counts(data_part)
        curr_node.col = best_criteria[0]
        curr_node.value = best_criteria[1]
        curr_node.impurity = current_score
        curr_node.tb = DecisionNode()
        nodes.append(curr_node.tb)
        data.append(best_sets[0])
        curr_node.fb = DecisionNode()
        nodes.append(curr_node.fb)
        data.append(best_sets[1])
        if i == 1:
            node_return = curr_node
    
    """
    Scoref representa el indice de impureza

    Cada particion disminuye la impureza por ello a mas ganancia, la impureza estara mas reduida.
    Valor de goodnes mas grande *Apuntes*
    """
    return node_return

    
def classify(tree: DecisionNode, values):
    #Values it must be a row
    #Te pasan un arbol y unos valores de una fila, recorrer el arbol encontrando los valores de esa columna y llegar a la ultima hoja que son los resultados. Devuelves hoja y te olvidas
    if tree.results is not None:
        biggest_value = 0
        valor = None
        for value in tree.results.values():
            if value > biggest_value:
                biggest_value = value
                #Guardar la etiqueta y devolverla
        return biggest_value #!cambiar por la etiqueta
    row = values[0]
    if tree.value == row:
        classify(tree.tb,values[1:])
    else:
        classify(tree.fb,values[1:])

def prune(tree: DecisionNode, threshold: float):
    if tree.tb is not None and tree.fb is not None:
        if tree.tb.results is not None and tree.fb.results is not None:
            tree.results = Counter()
            imp = tree.impurity
            tree.impurity = 0
            #Results es un diccionari el cual conte com a key l'etiqueta de les dades y com a valor, el nombre de cops que apareix l'etiqueta
            for res in tree.tb.results:
                tree.results[res] = tree.tb.results[res]
            for res in tree.fb.results:
                tree.results[res] = tree.fb.results[res]
            total = len(tree.results)
            for value in tree.results.values():
                p = (value/total)
                tree.impurity -= (p*_log2(p))
            if tree.impurity > threshold:
                tree.fb = None
                tree.tb = None
            else:
                tree.impurity = imp
                tree.results = None
        else:
            return DecisionNode(col=tree.col,value=tree.value,results=tree.results,
            tb=prune(tree.tb,threshold),fb=prune(tree.fb,threshold),impurity=tree.impurity)
    else:
        return tree
        



def print_tree(tree, headers=None, indent=""):
    """
    t11: Include the following function
    """
    # Is this a leaf node?
    if tree.results is not None:
        print(tree.results)
    else:
        # Print the criteria
        criteria = tree.col
        if headers:
            criteria = headers[criteria]
        print(f"{indent}{criteria}: {tree.value}?")

        # Print the branches
        print(f"{indent}T->")
        print_tree(tree.tb, headers, indent + "  ")
        print(f"{indent}F->")
        print_tree(tree.fb, headers, indent + "  ")

def print_data(headers, data):
    colsize = 15
    print('-' * ((colsize + 1) * len(headers) + 1))
    print("|", end="")
    for header in headers:
        print(header.center(colsize), end="|")
    print("")
    print('-' * ((colsize + 1) * len(headers) + 1))
    for row in data:
        print("|", end="")
        for value in row:
            if isinstance(value, (int, float)):
                print(str(value).rjust(colsize), end="|")
            else:
                print(value.ljust(colsize), end="|")
        print("")
    print('-' * ((colsize + 1) * len(headers) + 1))

def main():
    try:   
        filename = sys.argv[1]
    except IndexError:
        filename = "decision_tree_example.txt"
    headres,data = read(filename)

    """
    for value in ["USA","France","UK","NewZealand"]:
        set1, set2 = divideset(data,1,value)
        print("Split by:",value)
        print("gini set1:",gini_impurity(set1))
        print("gini set2:",gini_impurity(set2))
        print("entropy set1:",entropy(set1))
        print("entropy set2:",entropy(set2))
    """
    """
    #tree = buildtree(data)
    #print_tree(tree,headres)
    tree = iterative_buildtree(data)
    tree2 = buildtree(data)
    print_tree(tree,headres)
    print("----------")
    print_tree(tree2,headres)
    """
    """
    tree = buildtree(data)
    print_tree(tree)
    newtree = prune(tree,0.09) #Arrglar //un node es fique a null no se perque
    print_tree(newtree)
    
    """
    tree = buildtree(data)
    things = classify(tree,data[0])
    print(things)

    
    """
    print(headres)
        for row in data:
            print(row)

        print("-----")
        print(gini_impurity(data))
        print(entropy(data))
    """
if __name__=="__main__":
    main()


"""
1: caluclar impureza de nodo (i)
2: escoger pregunta
|   for query in possible_query():
|       set1,set2 = divide(query,data)
Calcular metrica para saber si la pregunta es buena o mala
|       function(data,set1,set2)
3: Generar hijos
|   buildTree(set1)
|   buildTree(set2)
"""
