import math
import numpy as np
from env import *

def ucb_score(parent, child):
    """
    The score for an action that would transition between the parent and child.
    """
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        # The value of the child is from the perspective of the opposing player
        value_score = -child.value()
    else:
        value_score = 0

    return value_score + prior_score


class Node:
    def __init__(self, prior, to_play):
        self.visit_count = 0
        self.to_play = to_play
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_action(self, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        """
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

    def select_child(self):
        """
        Select the child with the highest UCB score.
        """
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, state, to_play, action_probs):
        """
        We expand a node and keep track of the prior policy probability given by neural network
        """
        self.to_play = to_play
        self.state = state
        for a, prob in action_probs:
            if prob != 0:
                self.children[a] = Node(prior=prob, to_play=self.to_play * -1)

    def __repr__(self):
        """
        Debugger pretty print node info
        """
        prior = "{0:.2f}".format(self.prior)
        return "{} Prior: {} Count: {} Value: {}".format(self.state.__str__(), prior, self.visit_count, self.value())


class MCTS:

    def __init__(self, deepq, env, args):
        self.game = Chess_env(env.board.fen())
        self.deepq = deepq
        self.args = args

    def run(self, state, to_play):

        root = Node(0, to_play)

        # EXPAND root
        outs_m, outs_p, value = self.deepq.predict_move_prob([self.game], white= bool(to_play == 1))
        outs_p[0] = [u.numpy() for u in outs_p[0]]
        action_probs = zip(outs_m[0], outs_p[0])
        root.expand(state, to_play, list(action_probs))

        for _ in range(self.args['num_simulations']):
            game = Chess_env(self.game.board.fen())
            node = root
            search_path = [node]

            # SELECT
            action_list = []
            while node.expanded():
                action, node = node.select_child()
                search_path.append(node)
                action_list.append(action)
            for ac in action_list[:-1]:
                _ = game.step(ac)

            parent = search_path[-2]
            state = parent.state
            # Now we're at a leaf node and we would like to expand
            # Players always play from their own perspective
            state_next, reward, done, _ = game.step(action)  
            # The value of the new state from the perspective of the other player
            if reward == 0:
                # If the game has not ended:
                # EXPAND
                outs_m, outs_p, value = self.deepq.predict_move_prob([game], white= bool(parent.to_play == -1))
                action_probs = zip(outs_m[0], outs_p[0])
                value = value[0]
                node.expand(value , parent.to_play * -1, action_probs)

            self.backpropagate(search_path, value, parent.to_play * -1)

        return root

    def backpropagate(self, search_path, value, to_play):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1