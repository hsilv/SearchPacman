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
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    from game import Directions
    from util import Queue, Stack
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    u = Directions.NORTH
    d = Directions.SOUTH
    l = Directions.WEST
    r = Directions.EAST
    
    # Inicialización del nodo inicial
    start = problem.getStartState()

    # Si el nodo inicial es la meta, se retorna una lista vacía de movimientos
    if problem.isGoalState(start):
        return []

    # Inicialización de la frontera como una pila (LIFO), esto nos permitirá explorar el último nodo insertado
    # Esto nos ayudará a explorar siempre el nodo más profundo, es decir el último insertado
    frontier = Stack()
    
    # Se inserta el nodo inicial a la frontera con su camino vacío
    frontier.push((start, []))

    # Conjunto para almacenar los nodos alcanzados, set debido a que no se permitirán nodos repetidos
    reached = set()

    # Mientras la frontera no esté vacía, es decir, mientras queden nodos por explorar
    while not frontier.isEmpty():
        
        # Se obtiene el nodo y el camino que se ha recorrido hasta el momento
        node, path = frontier.pop()

        # Si el nodo es la meta, se retorna el camino recorrido hasta el momento
        if problem.isGoalState(node):
            return path
        
        # Si el nodo no ha sido alcanzado, se agrega al conjunto de nodos alcanzados
        if node not in reached:
            reached.add(node)

            # Por cada nodo adyacente al nodo actual, obteniendo los datos de su tupla...
            for child, direction, _ in problem.getSuccessors(node):
                
                # Si este no ha sido alcanzado aún, entonces se evaluará haciéndole push a la frontera e incluyendo el camino recorrido hasta el momento hacia ese nodo
                if child not in reached:
                    newPath = path + [direction]
                    frontier.push((child, newPath))

    return []

def breadthFirstSearch(problem):
    from game import Directions
    from util import Queue
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    u = Directions.NORTH
    d = Directions.SOUTH
    l = Directions.WEST
    r = Directions.EAST

    # Inicialización del nodo inicial
    start = problem.getStartState()

    # Si el nodo inicial es la meta, se retorna una lista vacía de movimientos
    if problem.isGoalState(start):
        return []

    # Inicialización de la frontera como una cola (FIFO), esto nos permitirá explorar el primer nodo insertado de todos
    # Esto nos ayudará a explorar siempre el nodo más superficial, es decir se recorrerán los nodos por nivel de profundidad
    frontier = Queue()
    
    # Se inserta el nodo inicial a la frontera con su camino vacío
    frontier.push((start, []))

    # Conjunto para almacenar los nodos alcanzados, set debido a que no se permitirán nodos repetidos
    reached = set()

    # Mientras la frontera no esté vacía, es decir, mientras queden nodos por explorar
    while not frontier.isEmpty():
        
        # Se obtiene el nodo y el camino que se ha recorrido hasta el momento
        node, path = frontier.pop()

        # Si el nodo es la meta, se retorna el camino recorrido hasta el momento
        if problem.isGoalState(node):
            return path

        # Si el nodo no ha sido alcanzado, se agrega al conjunto de nodos alcanzados
        if node not in reached:
            reached.add(node)

            # Por cada nodo adyacente al nodo actual, obteniendo los datos de su tupla...
            for child, direction, _ in problem.getSuccessors(node):
                
                # Si este no ha sido alcanzado aún, entonces se evaluará haciéndole push a la frontera e incluyendo el camino recorrido hasta el momento hacia ese nodo
                if child not in reached:
                    newPath = path + [direction]
                    frontier.push((child, newPath))

    return []


def uniformCostSearch(problem):
    from util import PriorityQueue
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()

    if problem.isGoalState(start):
        return []

    frontier = PriorityQueue()
    frontier.push((start, [], 0), 0)

    reached = {}

    while not frontier.isEmpty():
        node, path, pathCost = frontier.pop()

        if problem.isGoalState(node):
            return path

        if node not in reached or pathCost < reached[node]:
            reached[node] = pathCost

            for child, direction, child_cost in problem.getSuccessors(node):
                newCost = pathCost + child_cost
                if child not in reached or newCost < reached[child]:
                    newPath = path + [direction]
                    frontier.push((child, newPath, newCost), newCost)

    
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    from game import Directions
    from util import PriorityQueue
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    u = Directions.NORTH
    d = Directions.SOUTH
    l = Directions.WEST
    r = Directions.EAST
    
    # Inicialización del nodo inicial
    start = problem.getStartState()

    # Si el nodo inicial es la meta, se retorna una lista vacía de movimientos
    if problem.isGoalState(start):
        return []

    # Inicialización de la frontera como una cola de prioridad, en donde se hace pop al de menor prioridad, es decir, al de menor costo
    # Esto nos permite una aproximación muy similar a la combinación entre BFS y DFS, ya que se recorrerán los nodos por nivel de profundidad
    # pero se dará prioridad a los nodos con menor costo
    frontier = PriorityQueue()
    
    # Se inserta el nodo inicial a la frontera con su camino vacío y un costo de 0
    frontier.push((start, [], 0), 0)

    # Diccionario para almacenar los nodos alcanzados y poder obtener su costo
    reached = {}

    # Mientras la frontera no esté vacía, es decir, mientras queden nodos por explorar
    while not frontier.isEmpty():
        
        # Se obtiene el nodo y el camino que se ha recorrido hasta el momento
        node, path, pathCost = frontier.pop()

        # Si el nodo es la meta, se retorna el camino recorrido hasta el momento
        if problem.isGoalState(node):
            return path

        # Si el nodo no ha sido alcanzado, se agrega al conjunto de nodos alcanzados
        if node not in reached or pathCost < reached[node]:
            
            # Se almacena el costo del nodo
            reached[node] = pathCost

            # Por cada nodo adyacente al nodo actual, obteniendo los datos de su tupla...
            for child, direction, child_cost in problem.getSuccessors(node):
                
                # Se calcula el nuevo costo del camino
                newCost = pathCost + child_cost
                
                # Si este no ha sido alcanzado aún, entonces se evaluará haciéndole push a la frontera e incluyendo el camino recorrido hasta el momento hacia ese nodo
                if child not in reached or newCost < reached[child]:
                    
                    # Se calcula la prioridad del nodo, que es el costo del camino más la heurística del nodo, y el nuevo camino
                    newPath = path + [direction]
                    priority = newCost + heuristic(child, problem)
                    frontier.push((child, newPath, newCost), priority)

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
