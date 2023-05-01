from __future__ import annotations
from enum import Enum
from random import random, randint

class Action(Enum):
    """
    Defines enumeration type specifying possible actions for an agent
        as well as methods for easily cycling through them.
    """
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def nextAction(self)->Action:
        return Action((self.value + 1) % 4)

    def prevAction(self)->Action:
        return Action((self.value + 3) % 4)

    def backAction(self)->Action:
        """
        Returns the opposite action, i.e., a move backwards.
        """
        return Action((self.value + 2) % 4)
    

    
class Field(Enum):
    """
    Defines enumeration type specifying possible single-cell world conditions. 
    """
    EMPTY = 0
    OBSTACLE = 1
    NEGREWARD = 2
    REWARD = 3

    
    
class Pos:
    """
    Defines (x,y) coordinate pair specifying a point in the
        world.
    """
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self)->str:
        return "(" + str(self.x) + ", " + str(self.y) + ")"

    def move(self, action: Action)->Pos:
        if action == Action.UP:
            return Pos(self.x, self.y-1)
        elif action == Action.RIGHT:
            return Pos(self.x+1, self.y)
        elif action == Action.DOWN:
            return Pos(self.x, self.y+1)
        else:
            return Pos(self.x-1, self.y)

    def __eq__(self, other)->bool:
        return isinstance(other, Pos) and self.x == other.x and self.y == other.y

    def __hash__(self)->hash:
        return hash((self.x, self.y))

    
class MDP:
    """
    Defines a Markov decision process (MDP) problem modeling a widthxheight-size rectangular world.
    
    The world is divided into square 'cells', each of which have some property: a field may either be
        'neutral' (empty), an obstacle (which cannot be traversed), a negative reward, or a positive
        reward. Algorithms which define an agent may then operate on this world.
    
    MDPs provide a framework for modeling decision making in discrete-time where outcomes are
        influenced by random occurences (namely, for each action there is a probability of
        an unintended action occuring instead), but are driven by deterministic planning. 
        They are 'Markov' decision processes, in the sense that they satisfy the Markov property:
        the probability of moving to a new state s' is influenced only by the chosen action a and
        the current state s; it is thus conditionally independent of all previous states and
        actions given s and a. 
    """
    def __init__(self, name:str, width:int=4, height:int=3, 
                 transition_probs:list=[0.7,0.2,0.1,0.0], rewards:list=[1,-1,-0.04],
                deterministic:bool=False):
        self.name = name

        self.width = width
        self.height = height

        self.field = [[Field.EMPTY for x in range(self.width)] for y in range(self.height)]

        self.deterministic = deterministic

        # probabilities (should add up to 1)
        self.pPerform = transition_probs[0]
        self.pSidestep = transition_probs[1]
        self.pBackstep = transition_probs[2]
        self.pNoStep = transition_probs[3]

        self.posReward = rewards[0] # reward for positive end states
        self.negReward = rewards[1] # reward for negative end states
        self.noReward = rewards[2] # reward for other states

    def startState(self)->Pos:
        return Pos(0, 0)

    def isValid(self, state:Pos)->bool:
        """
        Returns false if state is an obstacle or is outside of the
            defined world limits.
        """
        if state.x < 0 or state.x >= self.width or state.y < 0 or state.y >= self.height:
            return False
        return self.field[state.y][state.x] != Field.OBSTACLE

    def setField(self, x:int, y:int, value:Field)->None:
        """
        Sets the value of a field to an option defined in the
            Field enumeration.
        """
        self.field[y][x] = value

    def isTerminal(self, state)->bool:
        f = self.field[state.y][state.x]
        return f == Field.REWARD or f == Field.NEGREWARD

    def states(self)->list:
        """
        Returns a list of all positions (of type Pos) which can be moved to by an agent
        """
        w = self.width
        h = self.height
        return [Pos(x, y) for x in range(w) for y in range(h) if self.isValid(Pos(x, y))]

    def actions(self, state:Pos)->list:
        """
        Returns a list of all possible actions available to an agent at a
            given state.
        """
        if self.isTerminal(state) or not self.isValid(state):
            return []
        return [action for action in Action]

    def getReward(self, state:Pos)->int:
        """
        Returns the reward received for moving into a given state
        """
        f = self.field[state.y][state.x]
        if f == Field.REWARD:
            return self.posReward
        elif f == Field.NEGREWARD:
            return self.negReward
        else:
            return self.noReward

    def doAction(self, state:Pos, action:Action)->Pos:
        """
        Returns the new state which results from an action.
        """
        newState = state.move(action)
        if self.isValid(newState):
            return newState
        else:
            return state

    def succProbReward(self, state:Pos, action:Action)->list:
        """
        Accepts a current state and a proposed action from that state, generates all possible
            outcomes from attempting the action from the current state. If self.deterministic
            is True, this list will always be of length one triple (the new state resulting from
            the given action and its reward). Otherwise, the returned list will also include all
            possible unintended actions which have probability > 0.
        
        param state: the current state
        param action: the intended action; if self.deterministic = False, the intended action may not
            be executed, and an unintended action may occur in its place.
        
        Returns list of (newState, prob, reward)-triples (sorted in descending order on
            prob) described below:
        
        newState: (Pos) a state which may result (with probability prob) from a given action
        prob: (float) the probability of arriving in newState given the current state and intended action
        reward: (int) the reward which will result from successful execution of the action.
        """
        if self.deterministic:
            newState = self.doAction(state, action)
            return [(newState, 1.0, self.getReward(newState))]
        else:
            result = []
            if self.pPerform > 0:
                performState = self.doAction(state, action)
                result.append((performState, self.pPerform, self.getReward(performState)))
            if self.pSidestep > 0:
                leftState = self.doAction(state, action.prevAction())
                rightState = self.doAction(state, action.nextAction())
                result.append((leftState, self.pSidestep / 2, self.getReward(leftState)))
                result.append((rightState, self.pSidestep / 2, self.getReward(rightState)))
            if self.pBackstep > 0:
                backState = self.doAction(state, action.backAction())
                result.append((backState, self.pBackstep, self.getReward(backState)))
            if self.pNoStep > 0:
                result.append((state, self.pNoStep, self.getReward(state)))
                
            return sorted(result, reverse=True, key=lambda t: t[1])

def valueIteration(mdp: MDP, gamma: float)->(dict, dict):
    """
    Implements the value iteration algorithm on a given MDP-satisfying
        world.
    
    Returns an (approximately) optimal policy for an agent to follow
        to maximize its reward, along with the expected reward for
        each state in the world.
        
        The policy (pi) is a dictionary specifying the necessary action
        to take at each state to maximize overall reward.
    """
    states = mdp.states()
    V = {}
    for s in states:
        V[s] = 0.

    def Q(state:Pos, action:Action)->float:
        """
        Returns the Q-value for a given state/action pair: the expectation
            of the resulting reward.
        """
        result = 0.
        for newState, prob, reward in mdp.succProbReward(state, action):
            result = result + prob * (reward + gamma * V[newState])
        return result

    k = 0
    while True:
        newV = {}
        for s in states:
            if mdp.isTerminal(s):
                newV[s] = 0.
            else:
                newV[s] = max(Q(s, a) for a in mdp.actions(s))

        # check for convergence
        if max(abs(V[s] - newV[s]) for s in states) < 1e-10:
            #print("converged after " + str(k) + " iterations")
            break
        V = newV

        # not converging?
        k = k + 1
        if k > 1000:
            #print("didn't converge")
            break

    pi = {}
    for s in states:
        if mdp.isTerminal(s):
            pi[s] = 'none'
        else:
            pi[s] = max((Q(s, a), i, a) for i, a in enumerate(mdp.actions(s)))[2]

    return pi, V

def pickRandomPossibility(possibilities: list)->(Pos, int):
    """
    Generates a random number r and chooses the first potential state
        which has probability less than r.
        
        This works because states are organized in descending order of
        the action probabilities that result in them.
    """
    r = random()
    for state, prob, reward in possibilities:
        if r < prob:
            return (state, reward)
        else:
            r = r - prob
    # pick the last possibility
    state, prob, reward = possibilities[-1]
    return (state, reward)

def qLearning(mdp:MDP, gamma:float, alpha:float, epsilon:float=0.2, iterations:int=1000)->(dict, dict):
    """
    Implements the Q-Learning reinforcement algorithm
    
    param epsilon: epsilon-greedy exploration value
    """
    # stores cumulative Q-values at each iteration.
    values_per_step = []
    
    states = mdp.states()

    Q = {}

    # initialize Q[S, A] arbitrarily
    for s in states:
        for a in mdp.actions(s):
            Q[(s, a)] = 0.

    def bestActionAndValue(state:Pos)->(Action, float):
        """
        Returns the action with maximal Q-value, along with that Q-value.
        """
        value, i, action = max((Q[(state, a)], i, a) for i, a in enumerate(mdp.actions(state)))
        return (action, value)

    cumulative_reward = 0
    
    for i in range(0, iterations):
        s = mdp.startState()
        while True:
            actions = mdp.actions(s)

            if random() < epsilon:
                # pick a random action
                a = actions[randint(0, len(actions) - 1)]
            else:
                # pick best action according to learned values so far
                a, _ = bestActionAndValue(s)

            possibilities = mdp.succProbReward(s, a)
            newState, r = pickRandomPossibility(possibilities)

            if mdp.isTerminal(newState):
                Q_ = 0
            else:
                _, Q_ = bestActionAndValue(newState)

            Q[(s, a)] = Q[(s, a)] + alpha * (r + gamma * Q_ - Q[(s, a)])

            s = newState
            
            cumulative_reward += r
            values_per_step.append(cumulative_reward)

            if mdp.isTerminal(s):
                break
               

    pi = {}
    V = {}
    for s in states:
        if mdp.isTerminal(s):
            pi[s] = 'none'
            V[s] = 0
        else:
            action, value = bestActionAndValue(s)
            pi[s] = action
            V[s] = value

    return (pi, V, values_per_step)

def standardMdp()->MDP:
    mdp = MDP("standard")
    mdp.setField(1, 1, Field.OBSTACLE)
    mdp.setField(3, 1, Field.NEGREWARD)
    mdp.setField(3, 2, Field.REWARD)
    return mdp


# Example usage:

# mdp = standardMdp()

# print(mdp.actions(Pos(0,0)))
# print(mdp.succProbReward(Pos(2,1), Action.DOWN))

# print("Result with value iteration:")
# pi, V = valueIteration(mdp, 0.9)
# for s in mdp.states():
#     print("State " + str(s) + ", action: " + str(pi[s]) + ", value: " + str(V[s]))

# print("Result with q learning:")
# pi, V = qLearning(mdp, 0.9, 0.1)
# for s in mdp.states():
#     print("State " + str(s) + ", action: " + str(pi[s]) + ", value: " + str(V[s]))
