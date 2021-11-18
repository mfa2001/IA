# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from node import Node
import util



class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    fringe = util.Stack()

    f_node=Node(problem.getStartState())
    if problem.isGoalState(f_node.state):
        return f_node.total_path()
    fringe.push(f_node)
    generated = set()
    while not fringe.isEmpty():
        n = fringe.pop()
        generated.add(n.state)
        for s,a,c in problem.getSuccessors(n.state):
            ns = Node(s,n,a,n.cost + c)
            if ns.state not in generated:
                if problem.isGoalState(ns.state):
                    return ns.total_path()
                fringe.push(ns)
                generated.add(ns.state)
    raise Exception




def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    """
    Import node + implement bfs / estudiar como funciona el bfs + fringe

    Fringe conjunto de nodos los cuales se van expandiendo, el primero sera el estado inicial
    """
    fringe=util.Queue()
    n_start=Node(problem.getStartState())
    if problem.isGoalState(n_start.state):
        return n_start.total_path
    fringe.push(n_start)
    generated=set()
    while not fringe.isEmpty():
        n = fringe.pop()
        generated.add(n.state) #Expandes
        for s,a,c in problem.getSuccessors(n.state):
                curr_node=Node(s,n,a,n.cost + c)
                if curr_node.state not in generated: #Not in exapnded not int fringe
                    if problem.isGoalState(curr_node.state):
                        return curr_node.total_path()
                    fringe.push(curr_node)
                    generated.add(curr_node.state) #Frigne
    raise Exception


  

    """
    Implementacion en grafo: state in fringe or state in expanded

    Pablo ha modificado la class queue, but lo mejor es generar un estado y expandir
    """    


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    n_start=Node(problem.getStartState())

    generated = {}
    fringe=util.PriorityQueue()
    fringe.push(n_start,0)

    generated[n_start.state] = ("F",0)
    while not fringe.isEmpty():
        n = fringe.pop()
        if problem.isGoalState(n.state):
            return n.total_path()
        if generated[n.state][0] == "E":
            #Node has been expanded, continue
            continue
        generated[n.state] = ("E",n.cost)
        for s,a,c in problem.getSuccessors(n.state):
            ns = Node(s,n,a,n.cost + c)
            if ns.state not in generated:
                fringe.push(ns,ns.cost)
                generated[ns.state] = ("F",ns.cost)
            elif generated[ns.state][0] == "F" and generated[ns.state][1] > ns.cost:
                fringe.update(ns,ns.cost)
                generated[ns.state] = ("F",ns.cost)
    raise Exception

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    n_start=Node(problem.getStartState())

    generated = {}
    fringe=util.PriorityQueue()
    h_cost = heuristic(n_start.state,problem)
    fringe.push(n_start,h_cost)

    generated[n_start.state] = ("F",h_cost)
    while not fringe.isEmpty():
        n = fringe.pop()
        if problem.isGoalState(n.state):
            return n.total_path()
        if generated[n.state][0] == "E":
            #Node has been expanded, continue
            continue
        generated[n.state] = ("E",n.cost+heuristic(n.state,problem))
        for s,a,c in problem.getSuccessors(n.state):
            ns = Node(s,n,a,n.cost + c + heuristic(s,problem))
            if ns.state not in generated:
                fringe.push(ns,ns.cost)
                generated[ns.state] = ("F",ns.cost)
            elif generated[ns.state][0] == "F" and generated[ns.state][1] > ns.cost:
                fringe.update(ns,ns.cost)
                generated[ns.state] = ("F",ns.cost)
    raise Exception


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
