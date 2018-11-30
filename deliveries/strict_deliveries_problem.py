from framework.graph_search import *
from framework.ways import *
from .map_problem import MapProblem
from .deliveries_problem_input import DeliveriesProblemInput
from .relaxed_deliveries_problem import RelaxedDeliveriesState, RelaxedDeliveriesProblem

from typing import Set, FrozenSet, Optional, Iterator, Tuple, Union


class StrictDeliveriesState(RelaxedDeliveriesState):
    """
    An instance of this class represents a state of the strict
     deliveries problem.
    This state is basically similar to the state of the relaxed
     problem. Hence, this class inherits from `RelaxedDeliveriesState`.

    TODO:
        If you believe you need to modify the state for the strict
         problem in some sense, please go ahead and do so.
    """
    pass


class StrictDeliveriesProblem(RelaxedDeliveriesProblem):
    """
    An instance of this class represents a strict deliveries problem.
    """

    name = 'StrictDeliveries'

    def __init__(self, problem_input: DeliveriesProblemInput, roads: Roads,
                 inner_problem_solver: GraphProblemSolver, use_cache: bool = True):
        super(StrictDeliveriesProblem, self).__init__(problem_input)
        self.initial_state = StrictDeliveriesState(
            problem_input.start_point, frozenset(), problem_input.gas_tank_init_fuel)
        self.inner_problem_solver = inner_problem_solver
        self.roads = roads
        self.use_cache = use_cache
        self._init_cache()

    def _init_cache(self):
        self._cache = {}
        self.nr_cache_hits = 0
        self.nr_cache_misses = 0

    def _insert_to_cache(self, key, val):
        if self.use_cache:
            self._cache[key] = val

    def _get_from_cache(self, key):
        if not self.use_cache:
            return None
        if key in self._cache:
            self.nr_cache_hits += 1
        else:
            self.nr_cache_misses += 1
        return self._cache.get(key)

    def expand_state_with_costs(self, state_to_expand: GraphProblemState) -> Iterator[Tuple[GraphProblemState, float]]:
        """
        TODO: implement this method!
        This method represents the `Succ: S -> P(S)` function of the strict deliveries problem.
        The `Succ` function is defined by the problem operators as shown in class.
        The relaxed problem operators are defined in the assignment instructions.
        It receives a state and iterates over the successor states.
        Notice that this is an *Iterator*. Hence it should be implemented using the `yield` keyword.
        For each successor, a pair of the successor state and the operator cost is yielded.
        """
        assert isinstance(state_to_expand, StrictDeliveriesState)

        # Get the junction (in the map) that is represented by the state to expand.
        junction = state_to_expand.current_location

        remaining_drop_points_list = self.drop_points.difference(state_to_expand.dropped_so_far)
        gas_stations = self.gas_stations

        # TODO: need to do something with inner (with the heuristic) and the cache.
        # TODO: maybe the cache stores index num of a state and also h(s).

        # Iterate over the outgoing roads of the current junction.
        for link in junction.links:
            link_target = link.target
            target_junction = self.roads.get(link_target)
            distance = link.distance

            if target_junction is None or state_to_expand.fuel - distance < 0:
                continue

            # target_junction is a drop point that we haven't visited yet
            if target_junction in remaining_drop_points_list:
                # Adding the new drop point to the dropped_so_far list
                successor_dropped_so_far = {target_junction} | state_to_expand.dropped_so_far
                fuel_left = state_to_expand.fuel - distance

            # target_junction is a gas station
            elif target_junction in gas_stations:
                successor_dropped_so_far = state_to_expand.dropped_so_far
                fuel_left = self.gas_tank_capacity

            # target_junction is neither a drop point we haven't visited yet, nor a gas station
            else:
                successor_dropped_so_far = state_to_expand.dropped_so_far
                fuel_left = state_to_expand.fuel - distance

            successor_state = StrictDeliveriesState(target_junction, successor_dropped_so_far, fuel_left)

            yield successor_state, distance


    def is_goal(self, state: GraphProblemState) -> bool:
        """
        This method receives a state and returns whether this state is a goal.
        """
        assert isinstance(state, StrictDeliveriesState)

        state_is_a_drop_point = state.current_location in self.drop_points
        state_has_nonnegative_fuel_amount = state.fuel_as_int >= 0

        dropped = state.dropped_so_far
        to_drop_in_total = self.drop_points

        state_is_done_with_drops = dropped.issubset(to_drop_in_total) and to_drop_in_total.issubset(dropped)

        return state_is_a_drop_point and state_has_nonnegative_fuel_amount and state_is_done_with_drops

