# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part A: Single Player Tetress

from .core import PlayerColor, Coord, PlaceAction
from .utils import *
from .data import *
from .generate_tetromino import *
import numpy as np
from collections import deque


def search(
    board: dict[Coord, PlayerColor], 
    target: Coord,
) -> list[PlaceAction] | None:
    """
    This is the entry point for your submission. You should modify this
    function to solve the search problem discussed in the Part A specification.
    See `core.py` for information on the types being used here.

    Parameters:
        `board`: a dictionary representing the initial board state, mapping
            coordinates to "player colours". The keys are `Coord` instances,
            and the values are `PlayerColor` instances.  
        `target`: the target BLUE coordinate to remove from the board.
    
    Returns:
        A list of "place actions" as PlaceAction instances, or `None` if no
        solution is possible.
    """
    import time

    start_time = time.time()
    
    # Declaration of pararmeters
    closest_red_heuristic = manhattan_distance
    heuristic_function = heuristic_version2
    K = 0.9
    M = 0.76

    print(f"K: {K}")
    print(f"M: {M}")


    # The render_board() function is handy for debugging. It will print out a
    # board state in a human-readable format. If your terminal supports ANSI
    # codes, set the `ansi` flag to True to print a colour-coded version!
    # print(render_board(board, target, ansi=False))


    ############################################################################
    red_cells = find_closest_red_cells(target = target,
                                      board =  board,
                                    closest_red_heuristic=closest_red_heuristic)
    # min_cell_dist = float("inf")
    # for red_cell in red_cells:
    #     for axis in ['vertical_fill', 'horizontal_fill']:
    #         cell_dist = heuristic_function(board, target, red_cell, axis)
    #         if cell_dist < min_cell_dist:
    #             min_cell_dist = cell_dist

    # min_path = None
    # for red in red_cells:
    #     print(f"SOURCE: {red}")
    #     path = bfs(board, target = target, source = red,
    #                        heuristic_function=heuristic_function)

    
    #     # if len(path) <= math.ceil(min_cell_dist/4):
    #     #     return path

    #     # failure handle
    #     if not path and len(red_cells) == 1:
    #         end_time = time.time()

    #         elapsed_time = end_time - start_time
    #         print(f"Program execution time: {elapsed_time:.2f} seconds")
    #         min_path = None
    #         return min_path
        
    #     # first update to min_path
    #     if not min_path and path:
    #         # end_time = time.time()

    #         # elapsed_time = end_time - start_time
    #         # print(f"Program execution time: {elapsed_time:.2f} seconds")
    #         min_path = path

    #     # # return straightaway
    #     # if len(min_path) <= math.ceil(min_cell_dist / 4):
    #     #     return min_path
        
    #     # afterwards
    #     if path and min_path and len(path) < len(min_path):
    #         min_path = path

    # end_time = time.time()

    # elapsed_time = end_time - start_time
    # print(f"Program execution time: {elapsed_time:.2f} seconds")

    # return min_path

################################################################################
    # path = multiple_a_star(board, target, red_cells, heuristic_function,
                        #    k = K, m = M)
    path = bfs(board, target, red_cells, None)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"Program execution time: {elapsed_time:.2f} seconds")
    return path



def a_star(board: dict[Coord, PlayerColor], 
    target: Coord,
    source: Coord,
    heuristic_function: Callable) -> list[PlaceAction]:

    queue = PriorityQueue()
    initial_state_vertical_fill = Node(parent_cell= source, parent_node= None,
                         path_cost= 0, heuristic_cost= 
                         heuristic_function(board=board, target=target, 
                                            source=source, 
                                            axis = 'vertical_fill'), 
                         place = PlaceAction(None, None, None, source),
                         actions = [PlaceAction(None, None, None, source)], 
                         board=board, axis = 'vertical_fill')
    
    initial_state_horizontal_fill = Node(parent_cell= source, parent_node= None,
                         path_cost= 0, heuristic_cost= 
                         heuristic_function(board=board, target=target, 
                                            source=source, 
                                            axis = 'horizontal_fill'), 
                         place = PlaceAction(None, None, None, source),
                         actions = [PlaceAction(None, None, None, source)], 
                         board=board, axis = 'horizontal_fill')
    # in the beginning, board 
    # mapping each placeaction to its total estimated cost to goal
    costs_dict_vertical_fill = {initial_state_vertical_fill.place: 0}
    costs_dict_horizontal_fill = {initial_state_horizontal_fill.place: 0} 
    queue.insert(initial_state_vertical_fill)
    queue.insert(initial_state_horizontal_fill)
    print(f"ORIGINAL QUEUE: {queue.heap}")
    print(f"ORIGINAL QUEUE LEN: {len(queue.heap)}")


    count = 0

    while queue.heap:
        print()
        print("POP")
        curr_node = queue.pop()
        print(f"COUNT: {count}")
        print(curr_node.place)
        print(f"PATHCOST: {curr_node.path_cost}")
        print(f"AXIS: {curr_node.axis}")
        print(f"Heuristic_cost: {curr_node.heuristic_cost}")
        print(f"List of actions: {curr_node.actions}")
        print(len(queue.heap))
        # print(f"QUEUE.HEAP: {queue.heap}")


        curr_node.board, removal = line_removal(curr_node.board)
        if removal:
            print("FINALLY BITCH")



        # if count >= 500:
        #     break
        
        if reach_goal_state(curr_node, target):
            print("SOLUTION FOUND") 
            print(len(curr_node.actions) - 1)
            print(render_board(curr_node.board, target))
            return curr_node.actions[1:]
        
        # could apply choose_best_k_cells here
        # here we apply the sort function based on the heuristic function to 
        # obtain the top score cells only, at the same time set the minimum 
        # requirements of nodes added for each length of actions
        flatten_lst = [item for place in curr_node.actions for item in place.coords if item and item not in removal]
        print(f"flatten_lst: {flatten_lst}")
        print(render_board(curr_node.board, target, ansi=False))
        best_k_cells = choose_best_k_cells(coord_lst=flatten_lst[::-1], 
                                        board=curr_node.board, target=target,
                                        heuristic=heuristic_function, k = None,
                                        axis = curr_node.axis)
        print(f"best_k_cells: {best_k_cells}")
        for cell in best_k_cells:
        # for cell in curr_node.place.coords:
            if not cell:
                continue
            # print(f"CELL expanded: {cell}")
            # could apply choose_best_tetromino here
            # sort the place in descending order of the tetromino
            # might be no need to choose best m tetrominos
            possible_tetrominos = generate_tetromino(cell = cell, board=curr_node.board)
            best_m_tetrominos = choose_best_m_tetrominos(possible_tetrominos, 
                                                         board=curr_node.board,
                                                         target=target, heuristic= heuristic_function,
                                                         m = 0.75, axis = curr_node.axis)
            # print(f"best_m_tetrominos: {best_m_tetrominos}")

            # set a lower bound for tetrominos and best_k_cells, so that it would
            # return the full size if required. Highly suggest change to np.quantile()
            for place in best_m_tetrominos:
                new_actions = curr_node.actions + [place]
                new_board = place_action(curr_node.board.copy(), place)

                # create a new node for each placeaction
                new_node = Node(parent_cell= cell, parent_node=curr_node, 
                                path_cost=curr_node.path_cost + 1,
                                heuristic_cost= 
                                heuristic_function(new_board, target, 
                                                   choose_best_k_cells(list(place.coords), new_board, target, heuristic_function, 1, curr_node.axis)[0], axis = curr_node.axis),
                                place = place, actions = new_actions, 
                                board = new_board, axis = curr_node.axis)
                
                if is_valid(new_node):
                    # if curr_node.axis == 'vertical_fill' and (place not in costs_dict_vertical_fill or \
                    #     new_node.path_cost < costs_dict_vertical_fill[place]):
                    #     costs_dict_vertical_fill[place] = new_node.path_cost
                    #     queue.insert(new_node)
                    # elif curr_node.axis == 'horizontal_fill' and (place not in costs_dict_horizontal_fill or \
                    #     new_node.path_cost < costs_dict_horizontal_fill[place]):
                    #     costs_dict_horizontal_fill[place] = new_node.path_cost
                    #     queue.insert(new_node)
                    queue.insert(new_node)
        # print(f"QUEUE.HEAP after being expanded: {len(queue.heap), queue.heap}")
        count += 1
                
    return None


def multiple_a_star(board: dict[Coord, PlayerColor], 
    target: Coord,
    sources: list[Coord],
    heuristic_function: Callable, 
    k: float | int | None,
    m: float | int | None) -> list[PlaceAction]:

    queue = PriorityQueue()
    for source in sources:
        initial_state_vertical_fill = Node(parent_cell= source, parent_node= None,
                            path_cost= 0, heuristic_cost= 
                            heuristic_function(board=board, target=target, 
                                                source=source, 
                                                axis = 'vertical_fill'), 
                            place = PlaceAction(None, None, None, source),
                            actions = [PlaceAction(None, None, None, source)], 
                            board=board, axis = 'vertical_fill')
        
        initial_state_horizontal_fill = Node(parent_cell= source, parent_node= None,
                            path_cost= 0, heuristic_cost= 
                            heuristic_function(board=board, target=target, 
                                                source=source, 
                                                axis = 'horizontal_fill'), 
                            place = PlaceAction(None, None, None, source),
                            actions = [PlaceAction(None, None, None, source)], 
                            board=board, axis = 'horizontal_fill')
        queue.insert(initial_state_vertical_fill)
        queue.insert(initial_state_horizontal_fill)
        
    # in the beginning, board 
    # mapping each placeaction to its total estimated cost to goal
    costs_dict_vertical_fill = {initial_state_vertical_fill.place: 0}
    costs_dict_horizontal_fill = {initial_state_horizontal_fill.place: 0} 

    print(f"ORIGINAL QUEUE: {queue.heap}")
    print(f"ORIGINAL QUEUE LEN: {len(queue.heap)}")


    count = 0

    while queue.heap:
        print()
        print("POP")
        curr_node = queue.pop()
        print(f"COUNT: {count}")
        print(curr_node.place)
        print(f"PATHCOST: {curr_node.path_cost}")
        print(f"AXIS: {curr_node.axis}")
        print(f"Heuristic_cost: {curr_node.heuristic_cost}")
        print(f"List of actions: {curr_node.actions}")
        print(len(queue.heap))
        # print(f"QUEUE.HEAP: {queue.heap}")

        prev_board = curr_node.board.copy()
        curr_node.board, removal = line_removal(curr_node.board)
        if removal:
            print(render_board(prev_board))
            print(removal)

            print("FINALLY BITCH")



        # if count >= 500:
        #     break
        
        if reach_goal_state(curr_node, target):
            print("SOLUTION FOUND") 
            print(len(curr_node.actions) - 1)
            print(render_board(curr_node.board, target))
            return curr_node.actions[1:]
        
        # could apply choose_best_k_cells here
        # here we apply the sort function based on the heuristic function to 
        # obtain the top score cells only, at the same time set the minimum 
        # requirements of nodes added for each length of actions
        # flatten_lst = [item for place in curr_node.actions for item in place.coords if item and item not in removal]
        flatten_lst = [item for item in curr_node.board if item and curr_node.board.get(item) == PlayerColor.RED]
        # print(f"{len(flatten_lst)} vs {len(extra_lst)}")
        print(f"flatten_lst: {flatten_lst}")
        print(render_board(curr_node.board, target, ansi=False))
        best_k_cells = choose_best_k_cells(coord_lst=flatten_lst[::-1], 
                                        board=curr_node.board, target=target,
                                        heuristic=heuristic_function, k = k,
                                        axis = curr_node.axis)
        print(f"best_k_cells: {best_k_cells}")
        for cell in best_k_cells:
        # for cell in curr_node.place.coords:
            if not cell:
                continue
            # print(f"CELL expanded: {cell}")
            # could apply choose_best_tetromino here
            # sort the place in descending order of the tetromino
            # might be no need to choose best m tetrominos
            possible_tetrominos = generate_tetromino(cell = cell, board=curr_node.board)
            best_m_tetrominos = choose_best_m_tetrominos(possible_tetrominos, 
                                                         board=curr_node.board,
                                                         target=target, heuristic= heuristic_function,
                                                         m = m, axis = curr_node.axis)
            # print(f"best_m_tetrominos: {best_m_tetrominos}")

            # set a lower bound for tetrominos and best_k_cells, so that it would
            # return the full size if required. Highly suggest change to np.quantile()
            for place in best_m_tetrominos:
                new_actions = curr_node.actions + [place]
                new_board = place_action(curr_node.board.copy(), place)

                # create a new node for each placeaction
                new_node = Node(parent_cell= cell, parent_node=curr_node, 
                                path_cost=curr_node.path_cost + 1,
                                heuristic_cost= 
                                heuristic_function(new_board, target, 
                                                   choose_best_k_cells(list(place.coords), new_board, target, heuristic_function, 1, curr_node.axis)[0], axis = curr_node.axis),
                                place = place, actions = new_actions, 
                                board = new_board, axis = curr_node.axis)
                
                if is_valid(new_node):
                    # if curr_node.axis == 'vertical_fill' and (place not in costs_dict_vertical_fill or \
                    #     new_node.path_cost < costs_dict_vertical_fill[place]):
                    #     costs_dict_vertical_fill[place] = new_node.path_cost
                    #     queue.insert(new_node)
                    # elif curr_node.axis == 'horizontal_fill' and (place not in costs_dict_horizontal_fill or \
                    #     new_node.path_cost < costs_dict_horizontal_fill[place]):
                    #     costs_dict_horizontal_fill[place] = new_node.path_cost
                    #     queue.insert(new_node)
                    queue.insert(new_node)
        # print(f"QUEUE.HEAP after being expanded: {len(queue.heap), queue.heap}")
        count += 1
                
    return None


def bfs(board: dict[Coord, PlayerColor], 
    target: Coord,
    sources: Coord,
    heuristic_function: Callable) -> list[PlaceAction]:

    queue = deque()
    for source in sources:
        initial_state_vertical_fill = Node(parent_cell= source, parent_node= None,
                            path_cost= 0, heuristic_cost= 0, 
                            place = PlaceAction(None, None, None, source),
                            actions = [PlaceAction(None, None, None, source)], 
                            board=board, axis = 'vertical_fill')
        
        initial_state_horizontal_fill = Node(parent_cell= source, parent_node= None,
                            path_cost= 0, heuristic_cost= 0,
                            place = PlaceAction(None, None, None, source),
                            actions = [PlaceAction(None, None, None, source)], 
                            board=board, axis = 'horizontal_fill')
        queue.append(initial_state_vertical_fill)
        queue.append(initial_state_horizontal_fill)
    # in the beginning, board 
    # mapping each placeaction to its total estimated cost to goal
    costs_dict_vertical_fill = {initial_state_vertical_fill.place: 0}
    costs_dict_horizontal_fill = {initial_state_horizontal_fill.place: 0} 

    # print(f"ORIGINAL QUEUE: {queue}")
    print(f"ORIGINAL QUEUE LEN: {len(queue)}")


    count = 0

    while queue:
        print()
        print("POP")
        curr_node = queue.popleft()
        print(f"COUNT: {count}")
        print(curr_node.place)
        print(f"PATHCOST: {curr_node.path_cost}")
        print(f"AXIS: {curr_node.axis}")
        print(f"Heuristic_cost: {curr_node.heuristic_cost}")
        print(f"List of actions: {curr_node.actions}")
        print(len(queue))
        # print(f"QUEUE.HEAP: {queue.heap}")


        curr_node.board, removal = line_removal(curr_node.board)
        if removal:
            print("FINALLY BITCH")



        # if count >= 500:
        #     break
        
        if reach_goal_state(curr_node, target):
            print("SOLUTION FOUND") 
            print(len(curr_node.actions) - 1)
            print(render_board(curr_node.board, target))
            return curr_node.actions[1:]
        
        # could apply choose_best_k_cells here
        # here we apply the sort function based on the heuristic function to 
        # obtain the top score cells only, at the same time set the minimum 
        # requirements of nodes added for each length of actions
        flatten_lst = [item for place in curr_node.actions for item in place.coords if item and item not in removal]
        print(f"flatten_lst: {flatten_lst}")
        print(render_board(curr_node.board, target, ansi=False))
        best_k_cells = choose_best_k_cells(coord_lst=flatten_lst[::-1], 
                                        board=curr_node.board, target=target,
                                        heuristic=None, k = None,
                                        axis = curr_node.axis)
        print(f"best_k_cells: {best_k_cells}")
        for cell in best_k_cells:
        # for cell in curr_node.place.coords:
            if not cell:
                continue
            # print(f"CELL expanded: {cell}")
            # could apply choose_best_tetromino here
            # sort the place in descending order of the tetromino
            # might be no need to choose best m tetrominos
            possible_tetrominos = generate_tetromino(cell = cell, board=curr_node.board)
            best_m_tetrominos = choose_best_m_tetrominos(possible_tetrominos, 
                                                         board=curr_node.board,
                                                         target=target, heuristic= None,
                                                         m = 0.75, axis = curr_node.axis)
            # print(f"best_m_tetrominos: {best_m_tetrominos}")

            # set a lower bound for tetrominos and best_k_cells, so that it would
            # return the full size if required. Highly suggest change to np.quantile()
            for place in best_m_tetrominos:
                new_actions = curr_node.actions + [place]
                new_board = place_action(curr_node.board.copy(), place)

                # create a new node for each placeaction
                new_node = Node(parent_cell= cell, parent_node=curr_node, 
                                path_cost=curr_node.path_cost + 1,
                                heuristic_cost= 0,
                                place = place, actions = new_actions, 
                                board = new_board, axis = curr_node.axis)
                
                if is_valid(new_node):
                    # if curr_node.axis == 'vertical_fill' and (place not in costs_dict_vertical_fill or \
                    #     new_node.path_cost < costs_dict_vertical_fill[place]):
                    #     costs_dict_vertical_fill[place] = new_node.path_cost
                    #     queue.insert(new_node)
                    # elif curr_node.axis == 'horizontal_fill' and (place not in costs_dict_horizontal_fill or \
                    #     new_node.path_cost < costs_dict_horizontal_fill[place]):
                    #     costs_dict_horizontal_fill[place] = new_node.path_cost
                    #     queue.insert(new_node)
                    queue.append(new_node)
        # print(f"QUEUE.HEAP after being expanded: {len(queue.heap), queue.heap}")
        count += 1
                
    return None



############################ GOAL STATE CHECKING ##############################
def reach_goal_state_no_clear(node: Node, target: Coord) -> bool:
    r,c = target.r, target.c
    horizontal_result = True
    for i in range(BOARD_N):
        if node.board.get(Coord(r, i)):
            continue
        else:
            horizontal_result = False
            break
    
    vertical_result = True
    for j in range(BOARD_N):
        if node.board.get(Coord(j, c)):
            continue
        else:
            vertical_result = False
            break

    return (vertical_result or horizontal_result) 
# """
# """
def reach_goal_state(node: Node, target: Coord) -> bool:
    if node.board.get(target, None):
        return False
    return True
################################################################################
# """

################################################################################
def choose_best_m_tetrominos(
    tetrominos: list[PlaceAction],
    board: dict[Coord, PlayerColor],
    target: Coord, 
    heuristic: Callable,
    m = None,
    axis = None
) -> list[PlaceAction]:
    if not heuristic:
        return tetrominos
    sorted_lst = sorted(tetrominos, key = lambda element: min(heuristic(board, target, src, axis) for src in element.coords))  
    # print(sorted_lst)  
    if not m: 
        return sorted_lst
    
    if 0 < m < 1:
        return sorted_lst[: int(len(sorted_lst) * m) + 1]
    
    if m < 10:
        return sorted_lst

    return sorted_lst[: m]
    
    
    return None
################################################################################


################################################################################
def choose_best_k_cells(coord_lst: list[Coord], board: dict[Coord, PlayerColor],
                        target: Coord, heuristic: Callable, k = 1, axis = None) -> list[Coord]: 
    if not heuristic:
        return coord_lst
    sorted_lst = sorted(coord_lst, key = lambda element: heuristic(board, target, element, axis ))  
    if not k: 
        return sorted_lst

    if 0 < k < 1:
        return sorted_lst[: int(len(sorted_lst) * k) + 1]
    return sorted_lst[: k]

################################################################################
def heuristic_version1(board: dict[Coord, PlayerColor],
                       target: Coord, source: Coord, axis = None) -> int:
    horizontal_cells = min(abs(target.c - source.c) - 1, 
                           BOARD_N - abs(target.c - source.c) - 1)
    if horizontal_cells < 0:
        horizontal_cells = 0 
    # print(horizontal_cells)
    vertical_gap_cells = 0
    vertical_gaps = obtain_gaps(board, target, source, axis = 'down')
    # print(f"vertical_gaps: {vertical_gaps}")
    for ind, gap in enumerate(vertical_gaps):
        # print(gap)
        res = None
        if ind != len(vertical_gaps) - 1:
            res =  abs(gap[0].r - gap[1].r) - 1
        else:
            res = BOARD_N - abs(gap[0].r - gap[1].r) - 1
        # print(res)
        vertical_gap_cells += res


        
    return horizontal_cells + vertical_gap_cells


#####################HANDLES BOTH VERTICAL+HORIZONTAL FILL######################
def heuristic_version2(board: dict[Coord, PlayerColor],
                       target: Coord, source: Coord, axis: str) -> int:
    if axis == 'vertical_fill':
        # horizontal steps to reach target's column
        horizontal_cells = min(abs(target.c - source.c) - 1, 
                            BOARD_N - abs(target.c - source.c) - 1)
        if horizontal_cells < 0:
            horizontal_cells = 0 

        # vertical steps to reach target's row, accounting for free cells
        # between gaps only
        vertical_gap_cells = 0
        vertical_gaps = obtain_gaps(board, target, source, axis = 'down')
        for ind, gap in enumerate(vertical_gaps):
            res = None
            if ind != len(vertical_gaps) - 1:
                res =  abs(gap[0].r - gap[1].r) - 1
            else:
                res = BOARD_N - abs(gap[0].r - gap[1].r) - 1
            vertical_gap_cells += res  

        return horizontal_cells + vertical_gap_cells

    elif axis == 'horizontal_fill':
        horizontal_cells = min(abs(target.r - source.r) - 1, 
                            BOARD_N - abs(target.r - source.r) - 1)
        if horizontal_cells < 0:
            horizontal_cells = 0 
        # print(horizontal_cells)
        vertical_gap_cells = 0
        vertical_gaps = obtain_gaps(board, target, source, axis = 'right')
        # print(f"vertical_gaps: {vertical_gaps}")
        for ind, gap in enumerate(vertical_gaps):
            # print(gap)
            res = None
            if ind != len(vertical_gaps) - 1:
                res =  abs(gap[0].c - gap[1].c) - 1
            else:
                res = BOARD_N - abs(gap[0].c - gap[1].c) - 1
            # print(res)
            vertical_gap_cells += res  
        return horizontal_cells + vertical_gap_cells


############################### GAPS GENERATION ###############################
def obtain_gaps(board: dict[Coord, PlayerColor],
                target: Coord, source: Coord, axis: str) -> list[list[Coord]]:
    r,c = target.r, target.c
    occupied_cells = []
    if axis == 'down':
        for i in range(BOARD_N):
            if is_occupied(Coord(i, c), board):
                occupied_cells.append(Coord(i,c))
    elif axis == 'right':
        for j in range(BOARD_N):
            if is_occupied(Coord(r, j), board):
                occupied_cells.append(Coord(r,j))
    
    return create_coords_pairs(occupied_cells)

################################################################################
def create_coords_pairs(lst_coords):
    pairs = []
    n = len(lst_coords)
    for i in range(n):
        pair = [lst_coords[i], lst_coords[(i + 1) % n]]
        pairs.append(pair)
    return pairs
################################################################################



def is_valid(node: Node) -> bool:
    return True

def manhattan_distance(
    coord1: Coord,
    coord2: Coord
):
    return abs(coord1.r - coord2.r) + abs(coord1.c - coord2.c)


def find_closest_red_cells(target, board, closest_red_heuristic):
    # change distance for other purposes
    red_cells_distance = dict()
    for coord in board:
        if board[coord] == PlayerColor.RED:
            red_cells_distance[coord] = closest_red_heuristic(coord, target)
    return dict(sorted(red_cells_distance.items(), 
                key=lambda item: item[1], reverse=False))

