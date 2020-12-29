# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPosition).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        nextState = currentGameState.generatePacmanSuccessor(action)
        newPosition, foodList = nextState.getPacmanPosition(), nextState.getFood().asList()
        closestFood = min([manhattanDistance(newPosition, food) for food in foodList]+[float("inf")])
        minGDist = min([manhattanDistance(newPosition, ghost) for ghost in nextState.getGhostPositions()])
        if minGDist <= 1: return -float('inf')
        return nextState.getScore() + 1/closestFood - 1/minGDist

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(Index):
        Returns a list of legal actions for an agent
        Index=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(Index, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.MinMax(gameState, 0, 0)[1]

    def MinMax(self, gameState, Index, depth):
        """
            MinMax Function
        """
        if depth == self.depth * gameState.getNumAgents() \
                or gameState.isLose() or gameState.isWin():         #Base Case
            return (self.evaluationFunction(gameState), None)
        Actions = [(self.MinMax(gameState.generateSuccessor(Index, action), (depth + 1)
                                % gameState.getNumAgents(),depth + 1)[0], action)
                   for action in gameState.getLegalActions(Index)]
        if Index == 0: return max(Actions)  #Max Agent
        else: return min(Actions)           #Min Agent

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.MinMaxAB(gameState, 0, 0, -float("inf"), float("inf"))[1]

    def MinMaxAB(self, gameState, Index, depth, alpha, beta):
        if depth == self.depth * gameState.getNumAgents() \
                or gameState.isLose() or gameState.isWin():         #Base Case
            return (self.evaluationFunction(gameState), None)
        if Index == 0:                                      #Max Agent
            maxAction = (-float("inf"), None)
            for action in gameState.getLegalActions(Index):
                nextAction = (self.MinMaxAB(gameState.generateSuccessor(Index, action),(depth + 1) %
                                            gameState.getNumAgents(), depth + 1, alpha, beta)[0], action)
                if nextAction[0] > maxAction[0]: maxAction = nextAction
                if maxAction[0] >= beta: break
                alpha = max(alpha, maxAction[0])
            return maxAction
        else:                                               #Min Agent
            minAction = (float("inf"), None)
            for action in gameState.getLegalActions(Index):
                nextAction = (self.MinMaxAB(gameState.generateSuccessor(Index, action),(depth + 1) %
                                            gameState.getNumAgents(), depth + 1, alpha, beta)[0], action)
                if nextAction[0] < minAction[0]: minAction = nextAction
                if minAction[0] <= alpha: break
                beta = min(beta, minAction[0])
            return minAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        tDepth = self.depth * gameState.getNumAgents()
        return self.expectimax(gameState, None, tDepth, 0)[0]

    def expectimax(self, gameState, action, depth, Index):
        if depth == 0 or gameState.isLose() or gameState.isWin():   #Base Case
            return (action, self.evaluationFunction(gameState))
        if Index == 0:                                              #Max Agent
            maxAction = (None, -float('inf'))
            for lAction in gameState.getLegalActions(Index):
                nextAgent, nextAction = (Index + 1) % gameState.getNumAgents(), None
                if depth != self.depth * gameState.getNumAgents(): nextAction = action
                else: nextAction = lAction
                nextValue = self.expectimax(gameState.generateSuccessor(Index, lAction),
                                            nextAction, depth - 1, nextAgent)
                if nextValue[1] > maxAction[1]: maxAction = nextValue
            return maxAction
        else:                                                       #Expectation Agent
            Scores = [self.expectimax(gameState.generateSuccessor(Index, lAction),
                                      action, depth - 1, (Index + 1) % gameState.getNumAgents())[1]
                      for lAction in gameState.getLegalActions(Index)]
            return action, sum(Scores)/len(Scores)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    Evaluate by closest food, food left, capsules left, dist to ghost
    """
    "*** YOUR CODE HERE ***"
    position, foodList = currentGameState.getPacmanPosition(), currentGameState.getFood().asList()
    closestFood = min([manhattanDistance(position, food) for food in foodList]+[float('inf')])
    gDists = [manhattanDistance(position, ghost) for ghost in currentGameState.getGhostPositions()]
    if min(gDists) <= 1: return -float('inf')
    return 100000/ (currentGameState.getNumFood() + 1) + 1000/ (len(currentGameState.getCapsules()) + 1)\
           + sum(gDists)/len(gDists) + 100/ (closestFood + 1)


# Abbreviation
better = betterEvaluationFunction
