from framework import *
from deliveries import *

from matplotlib import pyplot as plt
import numpy as np
from typing import List, Union

# Load the map
roads = load_map_from_csv(Consts.get_data_file_path("tlv.csv"))

# Make `np.random` behave deterministic.
Consts.set_seed()


# --------------------------------------------------------------------
# -------------------------- Map Problem -----------------------------
# --------------------------------------------------------------------

def plot_distance_and_expanded_wrt_weight_figure(
        weights: Union[np.ndarray, List[float]],
        total_distance: Union[np.ndarray, List[float]],
        total_expanded: Union[np.ndarray, List[int]]):
    """
    Use `matplotlib` to generate a figure of the distance & #expanded-nodes
     w.r.t. the weight.
    """
    assert len(weights) == len(total_distance) == len(total_expanded)

    fig, ax1 = plt.subplots()

    # See documentation here:
    # https://matplotlib.org/2.0.0/api/_as_gen/matplotlib.axes.Axes.plot.html
    # You can also search google for additional examples.

    ax1.plot(weights, total_distance, 'b-')

    # ax1: Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('distance traveled', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_xlabel('weight')

    # Create another axis for the #expanded curve.
    ax2 = ax1.twinx()

    ax2.plot(weights, total_expanded, 'r-')

    ax2.set_ylabel('states expanded', color='r')
    ax2.tick_params('y', colors='r')
    ax2.set_xlabel('weight')

    fig.tight_layout()
    plt.show()


def run_astar_for_weights_in_range(heuristic_type: HeuristicFunctionType, problem: GraphProblem):
    # 1. Create an array of 20 numbers equally spreaded in [0.5, 1]
    #    (including the edges). You can use `np.linspace()` for that.
    # 2. For each weight in that array run the A* algorithm, with the
    #    given `heuristic_type` over the map problem. For each such run,
    #    store the cost of the solution (res.final_search_node.cost)
    #    and the number of expanded states (res.nr_expanded_states).
    #    Store these in 2 lists (array for the costs and array for
    #    the #expanded.
    # Call the function `plot_distance_and_expanded_by_weight_figure()`
    #  with that data.

    num_of_experiments = 20
    start_point = 0.5
    end_point = 1.0

    weights_array = np.linspace(start_point, end_point, num_of_experiments, True, False, float)
    costs_array = list()
    expanded_states_array = list()

    for i in range(0, num_of_experiments, 1):
        a_star = AStar(heuristic_type, weights_array[i])
        res = a_star.solve_problem(problem)

        costs_array.append(res.final_search_node.cost)
        expanded_states_array.append(res.nr_expanded_states)

    plot_distance_and_expanded_wrt_weight_figure(weights_array, costs_array, expanded_states_array)


def map_problem():
    print()
    print('Solve the map problem.')

    # Ex.8
    map_prob = MapProblem(roads, 54, 549)
    uc = UniformCost()
    res = uc.solve_problem(map_prob)
    print(res)

    # Ex.10
    #       solve the same `map_prob` with it and print the results (as before).
    # Notice: AStar constructor receives the heuristic *type* (ex: `MyHeuristicClass`),
    #         and not an instance of the heuristic (eg: not `MyHeuristicClass()`).

    a_star_with_null = AStar(NullHeuristic)
    res = a_star_with_null.solve_problem(map_prob)
    print(res)


    # Ex.11
    #       solve the same `map_prob` with it and print the results (as before).

    a_star_with_air_dist = AStar(AirDistHeuristic)
    res = a_star_with_air_dist.solve_problem(map_prob)
    print(res)

    # Ex.12
    # 1. Complete the implementation of the function
    #    `run_astar_for_weights_in_range()` (upper in this file).
    # 2. Complete the implementation of the function
    #    `plot_distance_and_expanded_by_weight_figure()`
    #    (upper in this file).
    # 3. Call here the function `run_astar_for_weights_in_range()`
    #    with `AirDistHeuristic` and `map_prob`.

    run_astar_for_weights_in_range(AirDistHeuristic, map_prob)

# --------------------------------------------------------------------
# ----------------------- Deliveries Problem -------------------------
# --------------------------------------------------------------------

def relaxed_deliveries_problem():

    print()
    print('Solve the relaxed deliveries problem.')

    big_delivery = DeliveriesProblemInput.load_from_file('big_delivery.in', roads)
    big_deliveries_prob = RelaxedDeliveriesProblem(big_delivery)

    # Ex.16
    #       solve the `big_deliveries_prob` with it and print the results (as before).

    a_star_with_max_air_dist = AStar(MaxAirDistHeuristic)
    res = a_star_with_max_air_dist.solve_problem(big_deliveries_prob)
    print(res)

    # Ex.17
    #       solve the `big_deliveries_prob` with it and print the results (as before).
    a_star_with_mst_air_dist = AStar(MSTAirDistHeuristic)
    res = a_star_with_mst_air_dist.solve_problem(big_deliveries_prob)
    print(res)

    # Ex.18
    #       with `MSTAirDistHeuristic` and `big_deliveries_prob`.
    run_astar_for_weights_in_range(MSTAirDistHeuristic, big_deliveries_prob)


    # Ex.24
    # 1. Run the stochastic greedy algorithm for 100 times.
    #    For each run, store the cost of the found solution.
    #    Store these costs in a list.
    # 2. The "Anytime Greedy Stochastic Algorithm" runs the greedy
    #    greedy stochastic for N times, and after each iteration
    #    stores the best solution found so far. It means that after
    #    iteration #i, the cost of the solution found by the anytime
    #    algorithm is the MINIMUM among the costs of the solutions
    #    found in iterations {1,...,i}. Calculate the costs of the
    #    anytime algorithm wrt the #iteration and store them in a list.
    # TODO: switch back to 100
    run_times_num = 5
    # run_times_num = 100
    stochastic_greedy_result_list = list()
    anytime_result_list = list()

    for i in range(run_times_num):
        stochastic_greedy_ex_24 = GreedyStochastic(MSTAirDistHeuristic)
        result_node = stochastic_greedy_ex_24.solve_problem(big_deliveries_prob)
        stochastic_greedy_result_list.append(result_node.final_search_node.cost)
        anytime_result_list.append(min(stochastic_greedy_result_list))

    # 3. Calculate and store the cost of the solution received by
    #    the A* algorithm (with w=0.5).

    weight_for_a_star = 0.5

    a_star_ex_24 = AStar(MSTAirDistHeuristic, weight_for_a_star)
    a_star_result = a_star_ex_24.solve_problem(big_deliveries_prob)

    a_star_result_list = [a_star_result] * run_times_num

    # 4. Calculate and store the cost of the solution received by
    #    the deterministic greedy algorithm (A* with w=1).

    weight_for_deterministic_greedy = 1  # The weight for deterministic greedy is always 1

    deterministic_greedy_ex_24 = AStar(MSTAirDistHeuristic, weight_for_deterministic_greedy)
    deterministic_greedy_result = deterministic_greedy_ex_24.solve_problem(big_deliveries_prob)

    deterministic_greedy_result_list = [deterministic_greedy_result] * run_times_num

    # 5. Plot a figure with the costs (y-axis) wrt the #iteration
    #    (x-axis). Of course that the costs of A*, and deterministic
    #    greedy are not dependent with the iteration number, so
    #    these two should be represented by horizontal lines.

    iterations = (1, run_times_num + 1)

    results = list()
    results.append(stochastic_greedy_result_list)
    results.append(anytime_result_list)
    results.append(a_star_result_list)
    results.append(deterministic_greedy_result_list)

    plt.figure(2)
    for i in range(len(results)):
        plt.plot(iterations, results[i, :], label=str(iterations[i]))

    plt.xlabel("Iteration Number")
    plt.ylabel("Costs")
    plt.title("Quality of the algorithm solution as a function of the iteration number")
    plt.legend()
    plt.grid()
    plt.show()




def strict_deliveries_problem():
    print()
    print('Solve the strict deliveries problem.')

    small_delivery = DeliveriesProblemInput.load_from_file('small_delivery.in', roads)
    small_deliveries_strict_problem = StrictDeliveriesProblem(
        small_delivery, roads, inner_problem_solver=AStar(AirDistHeuristic))

    # Ex.26
    # TODO: Call here the function `run_astar_for_weights_in_range()`
    #       with `MSTAirDistHeuristic` and `big_deliveries_prob`.
    exit()  # TODO: remove!

    # Ex.28
    # TODO: create an instance of `AStar` with the `RelaxedDeliveriesHeuristic`,
    #       solve the `small_deliveries_strict_problem` with it and print the results (as before).
    exit()  # TODO: remove!


def main():
    map_problem()
    relaxed_deliveries_problem()
    strict_deliveries_problem()


if __name__ == '__main__':
    main()
