import random
from typing import Union, List
import sys
import numpy as np


import treepredict

def train_test_split(dataset, test_size: Union[float, int], seed=None):
    if seed:
        random.seed(seed)

    # If test size is a float, use it as a percentage of the total rows
    # Otherwise, use it directly as the number of rows in the test dataset
    n_rows = len(dataset)
    if float(test_size) != int(test_size):
        test_size = int(n_rows * test_size)  # We need an integer number of rows

    # From all the rows index, we get a sample which will be the test dataset
    choices = list(range(n_rows))
    test_rows = random.choices(choices, k=test_size)

    test = [row for (i, row) in enumerate(dataset) if i in test_rows]
    train = [row for (i, row) in enumerate(dataset) if i not in test_rows]

    return train, test


def get_accuracy(classifier: treepredict.DecisionNode, dataset):
    #Contar las veces que se classifican correctamente, es decir passan por TB los datos de dataset
    accuracy = 0
    for row_data in dataset:
        curr_node = classifier
        for data in row_data:
            if curr_node.results is None:
                if data == curr_node.value:
                    curr_node = curr_node.tb
                else:
                    curr_node = curr_node.fb
            else:
                predict = row_data[-1]
                if predict in curr_node.results:
                    accuracy+=1
                else:
                    break
    return accuracy

                
def split_dataset(data,k):
    return np.array_split(data,k)       



def mean(values: List[float]):
    return sum(values) / len(values)


def cross_validation(dataset, k, agg, seed, scoref, beta, threshold):
    """
    Input: Dataset S, number of folds k, model arguments args
Output: Score sc
1: procedure Cross Validation(S, k)
2: F0, F1, ..., Fk ← split dataset(S, k) ▷ Separate in k folds
3: scores ← ∅
4: for i ∈ [0, k − 1] do
5: Strain ← S\Fi ▷ Use all the data but the fold selected
6: Seval ← Fi
7: model ← train model(Strain, args)
8: scorei ← score model(model, Seval)
9: scores ← scores ∪ scorei
10: end for
11: sc ← aggregate(scores) ▷ For example, apply the mean
12: end procedure

    """
    data_split = split_dataset(dataset,k)
    agg = mean(dataset)
    scoref = treepredict.entropy()
    beta = 0
    raise NotImplementedError

def main():
    try:   
        filename = sys.argv[1]
    except IndexError:
        filename = "decision_tree_example.txt"
    headres,data = treepredict.read(filename)

    acc = get_accuracy(treepredict.buildtree(data),data)
    print(acc)
if __name__=="__main__":
    main()
