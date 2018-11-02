from solvers import Solver
from util.helper import rand_choice, randint, round
from util.helper import elem_distribution, ucb
from util.belief_tree import BeliefTree
from logger import Logger as log
import numpy as np
import time

MAX = np.inf

class UtilityFunction():
    @staticmethod
    def ucb1(c):
        def algorithm(action):
            return action.V + c * ucb(action.parent.N, action.N)
        return algorithm
    
    @staticmethod
    def mab_bv1(min_cost, c=1.0):
        def algorithm(action):
            if action.mean_cost == 0.0:
                return MAX
            ucb_value = ucb(action.parent.N, action.N)
            return action.mean_reward / action.mean_cost + c * ((1. + 1. / min_cost) * ucb_value) / (min_cost - ucb_value)
        return algorithm

    @staticmethod
    def sa_ucb(c0):
        def algorithm(action):
            if action.mean_cost == 0.0:
                return MAX
            return action.V + c0 * action.parent.budget * ucb(action.parent.N, action.N)
        return algorithm


class BSS(Solver):
    def __init__(self, model):
        Solver.__init__(self, model)
        self.tree = None

        self.gamma = None  # discount
        self.cur_state = None # current state for which action is produced
        self.horizon = None
        self.width = None
        self.max_reward = None # upperbound on possible reward for a state
        self.max_diff = None # max expected difference between optimal and computed
        self.utility_fn = None


    def _horizon(self):
        return None

    def _width(self):
        return None

    def _q_estimate_sample(self, state, action, horizon):
        return None

    def _estimate_qs(self, state, horizon):
        return none


