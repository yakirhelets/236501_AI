from .graph_problem_interface import *
from .best_first_search import BestFirstSearch
from typing import Optional
import numpy as np
import experiments.temperature as temp



class GreedyStochastic(BestFirstSearch):
    def __init__(self, heuristic_function_type: HeuristicFunctionType,
                 T_init: float = 1.0, N: int = 5, T_scale_factor: float = 0.95):
        # GreedyStochastic is a graph search algorithm. Hence, we use close set.
        super(GreedyStochastic, self).__init__(use_close=True)
        self.heuristic_function_type = heuristic_function_type
        self.T = T_init
        self.N = N
        self.T_scale_factor = T_scale_factor
        self.solver_name = 'GreedyStochastic (h={heuristic_name})'.format(
            heuristic_name=heuristic_function_type.heuristic_name)

    def _init_solver(self, problem: GraphProblem):
        super(GreedyStochastic, self)._init_solver(problem)
        self.heuristic_function = self.heuristic_function_type(problem)

    def _open_successor_node(self, problem: GraphProblem, successor_node: SearchNode):

        if self.open.has_state(successor_node.state):
            already_found_node_with_same_state = self.open.get_node_by_state(successor_node.state)
            if already_found_node_with_same_state.expanding_priority > successor_node.expanding_priority:
                self.open.extract_node(already_found_node_with_same_state)
                self.open.push_node(successor_node)

        elif self.close.has_state(successor_node.state):
            already_found_node_with_same_state = self.close.get_node_by_state(successor_node.state)
            if already_found_node_with_same_state.expanding_priority > successor_node.expanding_priority:
                self.close.remove_node(already_found_node_with_same_state)
                self.open.push_node(successor_node)

        else:
            self.open.push_node(successor_node)

    def _calc_node_expanding_priority(self, search_node: SearchNode) -> float:
        """
        Remember: `GreedyStochastic` is greedy.
        """
        h = self.heuristic_function
        return h.estimate(search_node.state)


    def _extract_next_search_node_to_expand(self) -> Optional[SearchNode]:
        """
        Extracts the next node to expand from the open queue,
         using the stochastic method to choose out of the N
         best items from open.
        Use `np.random.choice(...)` whenever you need to randomly choose
         an item from an array of items given a probabilities array `p`.
        You can read the documentation of `np.random.choice(...)` and
         see usage examples by searching it in Google.
        Notice: You might want to pop min(N, len(open) items from the
                `open` priority queue, and then choose an item out
                of these popped items. The other items have to be
                pushed again into that queue.
        """

        min_N_and_open_size = min(self.N, len(open()))

        best_N = np.array(1, min_N_and_open_size)

        for i in range(min_N_and_open_size):
            best_N[i] = self.open.pop_next_node()

        prob_array = np.array(1, min_N_and_open_size)

        for i in range(min_N_and_open_size):
            prob_array[i] = temp.calc_probability(self.T, best_N[i], min(best_N), best_N)

        chosen_element = np.random.choice(best_N, None, False, prob_array)

        np.delete(best_N, chosen_element, None)

        for i in best_N:
            self.open.push_node(i)