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

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
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
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        foodDistance = [manhattanDistance(newPos, i) for i in newFood.asList()]
        if foodDistance:
            foodScore = 10 / min(foodDistance)
        else:
            foodScore = 10 / 1

        ghostScore = -10 / (
            min(
                (
                    manhattanDistance(newPos, i)
                    for i in successorGameState.getGhostPositions()
                )
            )
            + 1
        )
        scaredGhostDistance = [
            manhattanDistance(newPos, i.getPosition())
            for i in newGhostStates
            if i.scaredTimer > 0
        ]
        if scaredGhostDistance:
            scaredScore = 20 / min(scaredGhostDistance)
        else:
            scaredScore = 20 / 1
        return successorGameState.getScore() + ghostScore + scaredScore + foodScore


def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        maxValue = float("-inf")
        maxAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            nextValue = self.getValue(nextState, 0, 1)
            if nextValue > maxValue:
                maxValue = nextValue
                maxAction = action
        return maxAction

    def getValue(self, gameState, currentDepth, agentIndex):
        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        elif agentIndex == 0:
            return self.maxValue(gameState, currentDepth)
        else:
            return self.minValue(gameState, currentDepth, agentIndex)

    def maxValue(self, gameState, currentDepth):
        maxValue = float("-inf")
        for action in gameState.getLegalActions(0):
            maxValue = max(
                maxValue,
                self.getValue(gameState.generateSuccessor(0, action), currentDepth, 1),
            )
        return maxValue

    def minValue(self, gameState, currentDepth, agentIndex):
        """
        If agentIndex==getNumAgents()-1,it means all the ghosts have been searched,
        or it will loop by adding 1
        """
        minValue = float("inf")
        for action in gameState.getLegalActions(agentIndex):
            if agentIndex == gameState.getNumAgents() - 1:
                minValue = min(
                    minValue,
                    self.getValue(
                        gameState.generateSuccessor(agentIndex, action),
                        currentDepth + 1,
                        0,
                    ),
                )
            else:
                minValue = min(
                    minValue,
                    self.getValue(
                        gameState.generateSuccessor(agentIndex, action),
                        currentDepth,
                        agentIndex + 1,
                    ),
                )
        return minValue


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        maxValue = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        maxAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            nextValue = self.getValue(nextState, 0, 1, alpha, beta)
            if nextValue > maxValue:
                maxValue = nextValue
                maxAction = action
            alpha = max(alpha, maxValue)
        return maxAction

    def getValue(self, gameState, currentDepth, agentIndex, alpha, beta):
        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        elif agentIndex == 0:
            return self.maxValue(gameState, currentDepth, alpha, beta)
        else:
            return self.minValue(gameState, currentDepth, agentIndex, alpha, beta)

    def maxValue(self, gameState, currentDepth, alpha, beta):
        maxValue = float("-inf")
        for action in gameState.getLegalActions(0):
            maxValue = max(
                maxValue,
                self.getValue(
                    gameState.generateSuccessor(0, action), currentDepth, 1, alpha, beta
                ),
            )
            if maxValue > beta:
                return maxValue
            alpha = max(alpha, maxValue)
        return maxValue

    def minValue(self, gameState, currentDepth, agentIndex, alpha, beta):
        """
        If agentIndex==getNumAgents()-1,it means all the ghosts have been searched,
        or it will loop by adding 1
        """
        minValue = float("inf")
        for action in gameState.getLegalActions(agentIndex):
            if agentIndex == gameState.getNumAgents() - 1:
                minValue = min(
                    minValue,
                    self.getValue(
                        gameState.generateSuccessor(agentIndex, action),
                        currentDepth + 1,
                        0,
                        alpha,
                        beta,
                    ),
                )
            else:
                minValue = min(
                    minValue,
                    self.getValue(
                        gameState.generateSuccessor(agentIndex, action),
                        currentDepth,
                        agentIndex + 1,
                        alpha,
                        beta,
                    ),
                )
            if minValue < alpha:  # not equality!
                return minValue
            beta = min(beta, minValue)
        return minValue

        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        maxValue = float("-inf")
        maxAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            nextValue = self.getValue(nextState, 0, 1)
            if nextValue > maxValue:
                maxValue = nextValue
                maxAction = action
        return maxAction

    def getValue(self, gameState, currentDepth, agentIndex):
        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        elif agentIndex == 0:
            return self.maxValue(gameState, currentDepth)
        else:
            return self.avgValue(gameState, currentDepth, agentIndex)

    def maxValue(self, gameState, currentDepth):
        maxValue = float("-inf")
        for action in gameState.getLegalActions(0):
            maxValue = max(
                maxValue,
                self.getValue(gameState.generateSuccessor(0, action), currentDepth, 1),
            )
        return maxValue

    def avgValue(self, gameState, currentDepth, agentIndex):
        avgValue = 0
        for action in gameState.getLegalActions(agentIndex):
            if agentIndex == gameState.getNumAgents() - 1:
                avgValue = avgValue + self.getValue(
                    gameState.generateSuccessor(agentIndex, action), currentDepth + 1, 0
                )
            else:
                avgValue = avgValue + self.getValue(
                    gameState.generateSuccessor(agentIndex, action),
                    currentDepth,
                    agentIndex + 1,
                )
        return avgValue

        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    Evaluate state，and Q1 is state-action pairs
    1.Score:5/6
    2.Score:
    """
    "*** YOUR CODE HERE ***"
    # # Useful information from the state
    # pacmanPos = currentGameState.getPacmanPosition()
    # food = currentGameState.getFood()
    # ghostStates = currentGameState.getGhostStates()
    # capsules = currentGameState.getCapsules()
    # currentScore = currentGameState.getScore()
    #
    # # Initialize evaluation score with current score
    # evaluationScore = currentScore
    #
    # # Calculate distance to the nearest food
    # foodDistances = [manhattanDistance(pacmanPos, foodPos) for foodPos in food.asList()]
    # if foodDistances:
    #     nearestFoodDistance = min(foodDistances)
    #     evaluationScore += 10.0 / nearestFoodDistance
    #
    # # Calculate distance to the ghosts
    # ghostDistances = [
    #     manhattanDistance(pacmanPos, ghost.getPosition()) for ghost in ghostStates
    # ]
    # scaredGhostDistances = [
    #     d for d, ghost in zip(ghostDistances, ghostStates) if ghost.scaredTimer > 0
    # ]
    # activeGhostDistances = [
    #     d for d, ghost in zip(ghostDistances, ghostStates) if ghost.scaredTimer == 0
    # ]
    #
    # # Penalize distances to active ghosts and reward distances to scared ghosts
    # if activeGhostDistances:
    #     nearestActiveGhostDistance = min(activeGhostDistances)
    #     if nearestActiveGhostDistance > 0:
    #         evaluationScore -= 2.0 / nearestActiveGhostDistance
    # if scaredGhostDistances:
    #     nearestScaredGhostDistance = min(scaredGhostDistances)
    #     if nearestScaredGhostDistance > 0:
    #         evaluationScore += 3.0 / nearestScaredGhostDistance
    #
    # # Calculate distance to the capsules
    # capsuleDistances = [manhattanDistance(pacmanPos, capsule) for capsule in capsules]
    # if capsuleDistances:
    #     nearestCapsuleDistance = min(capsuleDistances)
    #     evaluationScore += 5.0 / nearestCapsuleDistance
    #
    # # Consider remaining food count (fewer food items is better)
    # remainingFoodCount = len(food.asList())
    # evaluationScore -= 4 * remainingFoodCount
    #
    # return evaluationScore
    # util.raiseNotDefined()
    """
    ghostDist:distance of each ghost
    foodDist:distance of each food
    To evaluate current state,we need to deal with ghostDistacne,foodDistance,foodNum
    """
    score = scoreEvaluationFunction(currentGameState)
    newFood = currentGameState.getFood()
    newPos = currentGameState.getPacmanPosition()

    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")

    ghostDist = []
    for i in range(1, currentGameState.getNumAgents()):
        ghostDist.append(
            util.manhattanDistance(currentGameState.getGhostPosition(i), newPos)
        )
    if min(ghostDist) < 2:
        return float("-inf")

    foodDist = []
    for food in list(newFood.asList()):
        foodDist.append(util.manhattanDistance(food, newPos))

    return (
        score
        - 2 * min(foodDist)
        - max(foodDist)
        - 8 * currentGameState.getNumFood()
        + 2.5 * ghostDist[0]
    )


# Abbreviation
better = betterEvaluationFunction
