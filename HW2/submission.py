import random, util
from collections import deque

from game import Agent
import math

#     ********* Reflex agent- sections a and b *********
from ghostAgents import RandomGhost, DirectionalGhost

class OriginalReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
  """

    def __init__(self):
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState):
        """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
    The evaluation function takes in the current GameState (pacman.py) and the proposed action
    and returns a number, where higher numbers are better.
    """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        return scoreEvaluationFunction(successorGameState)



class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
  """

    def __init__(self):
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState):
        """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
    The evaluation function takes in the current GameState (pacman.py) and the proposed action
    and returns a number, where higher numbers are better.
    """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        return betterEvaluationFunction(successorGameState)


#     ********* Evaluation functions *********

def scoreEvaluationFunction(gameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
  """
    return gameState.getScore()


######################################################################################
# b: implementing a better heuristic function
def betterEvaluationFunction(gameState):
    """

  The betterEvaluationFunction takes in a GameState (pacman.py) and should return a number, where higher numbers are better.

  A GameState specifies the full game state, including the food, capsules, agent configurations and more.
  Following are a few of the helper methods that you can use to query a GameState object to gather information about
  the present state of Pac-Man, the ghosts and the maze:

  gameState.getLegalActions():
  gameState.getPacmanState():
  gameState.getGhostStates():
  gameState.getNumAgents():
  gameState.getScore():
  The GameState class is defined in pacman.py and you might want to look into that for other helper methods.
  """

    fear_factor = 3
    brave_fator = 15
    ghost_proximity_penalty = 1000

    score = 0

    # Sub-Section a in Part B, q1
    pacman_position = gameState.getPacmanPosition()
    ghost_positions = gameState.getGhostPositions()
    distance_from_closest_ghost = math.inf
    distance_from_far_ghost = -math.inf

    for ghost_position in ghost_positions:
        current_distance = util.manhattanDistance(pacman_position, ghost_position)
        if (current_distance < distance_from_closest_ghost):
            distance_from_closest_ghost = current_distance
        if (current_distance > distance_from_far_ghost):
            distance_from_far_ghost = current_distance

    if (distance_from_closest_ghost < fear_factor):
        score += (distance_from_closest_ghost - ghost_proximity_penalty)
    if (distance_from_far_ghost > brave_fator):
        score += (distance_from_closest_ghost + ghost_proximity_penalty//2)

    # Sub-Section b in Part B, q1
    score += gameState.getScore()

    # Sub-Section c in Part B, q1
    distance_to_closest_point_with_food = find_distance_to_closest_point_with_food(gameState, 0)
    # distance = distance_with_board_constraints(gameState, closest_point_with_food)
    score += -(distance_to_closest_point_with_food)

    return score


def find_distance_to_closest_point_with_food(gameState, dist):
    """
  An implementation of a BFS to return the closest point that has food in it
  """
    max_dist = 4;

    visited = set()
    nodes_queue = deque()
    nodes_queue.appendleft((gameState, dist))

    while nodes_queue: # Not empty
        next_node_state, current_dist = nodes_queue.pop()

        # Limiting the bfs run to max_dist
        if current_dist > max_dist:
            return current_dist

        next_node_position = next_node_state.getPacmanPosition()
        x = next_node_position[0]
        y = next_node_position[1]
        game_food_state = gameState.getFood()
        if game_food_state[x][y]:
            return current_dist
        else:
            visited.add(next_node_position)

        actions = next_node_state.getLegalPacmanActions()
        for action in actions:
            new_state = next_node_state.generatePacmanSuccessor(action)
            if new_state.getPacmanPosition() not in visited:
                nodes_queue.appendleft((new_state, current_dist + 1))
            # Food will certainly be found because else the game is over with a win

    return 0 # At this point game is over with a win

#     ********* MultiAgent Search Agents- sections c,d,e,f*********

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & both ExpectimaxAgents.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

    def __init__(self, evalFn='betterEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


######################################################################################
# c: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent
  """

    def getAction(self, gameState):
        """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """

        # BEGIN_YOUR_CODE

        legal_pacman_actions = gameState.getLegalPacmanActions()
        minimax_values = [self.minimaxValue(gameState.generateSuccessor(0, action), 1, 0) for action in legal_pacman_actions]
        max_value = max(minimax_values)
        bestIndices = [index for index in range(len(minimax_values)) if minimax_values[index] == max_value]
        chosenIndex = random.choice(bestIndices)
        return legal_pacman_actions[chosenIndex] # Reutrns the minimax action

        # END_YOUR_CODE

    def minimaxValue(self, gameState, agentIndex, searchDepth):

        # The base cases
        # if reached self.depth - stop and return value of heuristic function of state
        if (searchDepth == self.depth):
            return self.evaluationFunction(gameState)
        # else if it's a final node that leads to win or lost - return the score
        if gameState.isWin() or gameState.isLose():
            return gameState.getScore()
        # The recursion
        current_agent_index = agentIndex
        if gameState.getNumAgents() == current_agent_index:
            current_agent_index = 0
        legal_agent_actions = gameState.getLegalActions(current_agent_index)
        children_states = [gameState.generateSuccessor(current_agent_index, action) for action in legal_agent_actions]

        if current_agent_index == 0: # It is pacman's turn - we want to maximize the choice
            cur_max = float('-inf')
            for c in children_states:
                v = self.minimaxValue(c, current_agent_index + 1, searchDepth + 1)
                cur_max = max(v, cur_max)
            return cur_max
        else:  # It is a ghost's turn - we want to minimize the choice
            cur_min = float('inf')
            for c in children_states:
                # Only pacman increases depth because it symbolizes a new iteration on all the agents
                v = self.minimaxValue(c, current_agent_index + 1, searchDepth)
                cur_min = min(v, cur_min)
            return cur_min





######################################################################################
# d: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning
  """

    def getAction(self, gameState):
        """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

        # BEGIN_YOUR_CODE

        legal_pacman_actions = gameState.getLegalPacmanActions()
        alpha_beta_values = [self.alphaBetaValue(gameState.generateSuccessor(0, action), 1, 0, float('-inf'), float('inf')) for action in legal_pacman_actions]
        max_value = max(alpha_beta_values)
        bestIndices = [index for index in range(len(alpha_beta_values)) if alpha_beta_values[index] == max_value]
        chosenIndex = random.choice(bestIndices)
        return legal_pacman_actions[chosenIndex] # Reutrns the alphabeta action

        #  END_YOUR_CODE

    def alphaBetaValue(self, gameState, agentIndex, searchDepth, alpha, beta):

        # The base cases
        # if reached self.depth - stop and return value of heuristic function of state
        if (searchDepth == self.depth):
            return self.evaluationFunction(gameState)
        # else if it's a final node that leads to win or lost - return the score
        if gameState.isWin() or gameState.isLose():
            return gameState.getScore()
        # The recursion
        current_agent_index = agentIndex
        if gameState.getNumAgents() == current_agent_index:
            current_agent_index = 0
        legal_agent_actions = gameState.getLegalActions(current_agent_index)
        children_states = [gameState.generateSuccessor(current_agent_index, action) for action in legal_agent_actions]

        if current_agent_index == 0:  # It is pacman's turn - we want to maximize the choice
            cur_max = float('-inf')
            for c in children_states:
                v = self.alphaBetaValue(c, current_agent_index + 1, searchDepth + 1, alpha, beta)
                cur_max = max(v, cur_max)
                # Added compared to minimax:
                alpha = max(cur_max, alpha)
                if cur_max >= beta:
                    return float('inf')
            return cur_max
        else:  # It is a ghost's turn - we want to minimize the choice
            cur_min = float('inf')
            for c in children_states:
                # Only pacman increases depth because it symbolizes a new iteration on all the agents
                v = self.alphaBetaValue(c, current_agent_index + 1, searchDepth, alpha, beta)
                cur_min = min(v, cur_min)
                # Added compared to minimax:
                beta = min(cur_min, beta)
                if cur_min <= alpha:
                    return float('-inf')
            return cur_min


######################################################################################
# e: implementing random expectimax

class RandomExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent
  """

    def getAction(self, gameState):
        """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as choosing uniformly at random from their legal moves.
    """

        # BEGIN_YOUR_CODE
        legal_pacman_actions = gameState.getLegalPacmanActions()
        expectimax_values = [self.randomExpectimaxValue(gameState.generateSuccessor(0, action), 1, 0) for action in legal_pacman_actions]
        max_value = max(expectimax_values)
        bestIndices = [index for index in range(len(expectimax_values)) if expectimax_values[index] == max_value]
        chosenIndex = random.choice(bestIndices)
        return legal_pacman_actions[chosenIndex] # Reutrns the expectimax action
        # END_YOUR_CODE


    def randomExpectimaxValue(self, gameState, agentIndex, searchDepth):

        # The base cases
        # if reached self.depth or reached a leaf - stop and return value of heuristic function of state
        if searchDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        # The recursion
        current_agent_index = agentIndex
        if gameState.getNumAgents() == current_agent_index:
            current_agent_index = 0

        # the randomGhost instance to call getDistribution on. It holds the correct ghost index
        random_ghost = RandomGhost(current_agent_index)

        legal_agent_actions = gameState.getLegalActions(current_agent_index)
        children_states = [gameState.generateSuccessor(current_agent_index, action) for action in legal_agent_actions]

        if current_agent_index == 0:  # It is pacman's turn - we want to maximize the choice
            cur_max = float('-inf')
            for c in children_states:
                v = self.randomExpectimaxValue(c, current_agent_index + 1, searchDepth + 1)
                cur_max = max(v, cur_max)
            return cur_max
        else:  # It is a ghost's turn - a probabilistic state
            sum = 0
            # get the Counter of probabilities
            probabilities = random_ghost.getDistribution(gameState)
            probabilities_keys = list(probabilities.keys())
            # Computer and return the sum on all the probabilities multiplied by the corresponding randomExpectimaxValue
            for i in range(len(children_states)):
                next_key = probabilities_keys[i]
                sum += probabilities.get(next_key)*self.randomExpectimaxValue(children_states[i], current_agent_index + 1, searchDepth)
            return sum


######################################################################################
# f: implementing directional expectimax

class DirectionalExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent
  """

    def getAction(self, gameState):
        """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as using the DirectionalGhost distribution to choose from their legal moves.
    """

        # BEGIN_YOUR_CODE
        legal_pacman_actions = gameState.getLegalPacmanActions()
        expectimax_values = [self.directionalExpectimaxValue(gameState.generateSuccessor(0, action), 1, 0) for action in
                             legal_pacman_actions]
        max_value = max(expectimax_values)
        bestIndices = [index for index in range(len(expectimax_values)) if expectimax_values[index] == max_value]
        chosenIndex = random.choice(bestIndices)
        return legal_pacman_actions[chosenIndex]  # Reutrns the expectimax action
        # END_YOUR_CODE

    def directionalExpectimaxValue(self, gameState, agentIndex, searchDepth):

        # The base cases
        # if reached self.depth or reached a leaf - stop and return value of heuristic function of state
        if searchDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(
                gameState)
        # The recursion
        current_agent_index = agentIndex
        if gameState.getNumAgents() == current_agent_index:
            current_agent_index = 0

        # the randomGhost instance to call getDistribution on. It holds the correct ghost index
        directional_ghost = DirectionalGhost(current_agent_index)

        legal_agent_actions = gameState.getLegalActions(current_agent_index)
        children_states = [gameState.generateSuccessor(current_agent_index, action) for action in legal_agent_actions]

        if current_agent_index == 0:  # It is pacman's turn - we want to maximize the choice
            cur_max = float('-inf')
            for c in children_states:
                v = self.directionalExpectimaxValue(c, current_agent_index + 1, searchDepth + 1)
                cur_max = max(v, cur_max)
            return cur_max
        else:  # It is a ghost's turn - a probabilistic state
            sum = 0
            # get the Counter of probabilities
            probabilities = directional_ghost.getDistribution(gameState)
            probabilities_keys = list(probabilities.keys())
            # Computer and return the sum on all the probabilities multiplied by the corresponding randomExpectimaxValue
            for i in range(len(children_states)):
                next_key = probabilities_keys[i]
                sum += probabilities.get(next_key) * self.directionalExpectimaxValue(children_states[i],
                                                                                current_agent_index + 1, searchDepth)
            return sum


######################################################################################
# I: implementing competition agent

class CompetitionAgent(MultiAgentSearchAgent):
    """
    Your competition agent
  """

    def getAction(self, gameState):
        """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as using the DirectionalGhost distribution to choose from their legal moves.
    """

        # BEGIN_YOUR_CODE
        legal_pacman_actions = gameState.getLegalPacmanActions()
        expectimax_values = [self.directionalExpectimaxValue(gameState.generateSuccessor(0, action), 1, 0) for action in
                             legal_pacman_actions]
        max_value = max(expectimax_values)
        bestIndices = [index for index in range(len(expectimax_values)) if expectimax_values[index] == max_value]
        chosenIndex = random.choice(bestIndices)
        return legal_pacman_actions[chosenIndex]  # Reutrns the expectimax action
        # END_YOUR_CODE

    def directionalExpectimaxValue(self, gameState, agentIndex, searchDepth):

        # The base cases
        # if reached self.depth or reached a leaf - stop and return value of heuristic function of state
        if searchDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(
                gameState)
        # The recursion
        current_agent_index = agentIndex
        if gameState.getNumAgents() == current_agent_index:
            current_agent_index = 0

        # the randomGhost instance to call getDistribution on. It holds the correct ghost index
        directional_ghost = DirectionalGhost(current_agent_index)

        legal_agent_actions = gameState.getLegalActions(current_agent_index)
        children_states = [gameState.generateSuccessor(current_agent_index, action) for action in legal_agent_actions]

        if current_agent_index == 0:  # It is pacman's turn - we want to maximize the choice
            cur_max = float('-inf')
            for c in children_states:
                v = self.directionalExpectimaxValue(c, current_agent_index + 1, searchDepth + 1)
                cur_max = max(v, cur_max)
            return cur_max
        else:  # It is a ghost's turn - a probabilistic state
            sum = 0
            # get the Counter of probabilities
            probabilities = directional_ghost.getDistribution(gameState)
            probabilities_keys = list(probabilities.keys())
            # Computer and return the sum on all the probabilities multiplied by the corresponding randomExpectimaxValue
            for i in range(len(children_states)):
                next_key = probabilities_keys[i]
                sum += probabilities.get(next_key) * self.directionalExpectimaxValue(children_states[i],
                                                                                current_agent_index + 1, searchDepth)
            return sum
