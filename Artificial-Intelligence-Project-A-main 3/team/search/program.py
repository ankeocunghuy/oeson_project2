# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part A: Single Player Tetress

from .core import PlayerColor, Coord, PlaceAction
from .utils import *
from .data import *
from .generate_tetromino import *
import numpy as np


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


    # The render_board() function is handy for debugging. It will print out a
    # board state in a human-readable format. If your terminal supports ANSI
    # codes, set the `ansi` flag to True to print a colour-coded version!
    # print(render_board(board, target, ansi=False))

    """    # check occupied function
        print(f" is Coord(2, 5) occupied: {is_occupied(Coord(2, 5), board)}")
        print(f" is Coord(2, 4) occupied: {is_occupied(Coord(2, 4), board)}")
        print(f" is Coord(2, 9) occupied: {is_occupied(Coord(2, 9), board)}")
        print(f" is Coord(2, 12) occupied: {is_occupied(Coord(2, 9), board)}")
    """
    sample = PlaceAction(Coord(2, 5), Coord(2, 6), Coord(3, 6), Coord(3, 7))
    sample2 = PlaceAction(Coord(2, 10), Coord(3, 8), Coord(3, 6), Coord(3, 7))

    #################### TEST FOR GENERATING TETROMINO #########################
    # test for creating I tetrominoS and placing actions
    # place_lst = create_I_tetromino(Coord(9,8), board)
    # place_lst = create_O_tetromino(Coord(9,8), board)
    # place_lst = create_J_tetromino(Coord(8,1), board)
    # place_lst = generate_tetromino(Coord(8,1), board)
    # print(len(place_lst), place_lst)
    # ind = 6
    # # print(check_valid_place(place_lst[-1], board))
    # place_action(board, place_lst[ind])
    # # print(check_valid_place(place_lst[-1], board))
    # print(render_board(board, target, ansi=False))
    ############################################################################




    ################### PRIORITY QUEUE AND NODE CHECK ##########################
    # queue = PriorityQueue()
    # node1 = Node(parent_cell=Coord(2, 5), place=sample, parent_node=None,
    #              path_cost=1, heuristic_cost=100)
    # node2 = Node(parent_cell=Coord(2, 7), place=sample,parent_node=node1,
    #              path_cost=12, heuristic_cost=100)
    # node3 = Node(parent_cell=Coord(4, 5), place=sample,parent_node=node2,
    #              path_cost=1, heuristic_cost=80)
    # node4 = Node(parent_cell=Coord(8, 5), place=sample,parent_node=node1,
    #              path_cost=16, heuristic_cost=70)
    # node5 = Node(parent_cell=Coord(2, 10), place=sample2,parent_node=node3,
    #              path_cost=19, heuristic_cost=30)
    # node6 = Node(parent_cell=Coord(1, 5), place=sample,parent_node=node5,
    #              path_cost=1, heuristic_cost=0)    
    # queue.insert(node1)
    # queue.insert(node2)
    # queue.insert(node3)
    # queue.insert(node4)
    # queue.insert(node5)
    # queue.insert(node6)

    # print(len(queue.heap))
    # print(queue.pop().parent_node.place)
    # print(queue.pop())
    # print(queue.pop())
    # print(queue.pop())
    # print(queue.pop())
    # print(queue.pop())
    ############################################################################
    # print(board)
    # print(len(board))
    # print(render_board(board, target))

    # new_board = line_removal(board.copy())
    # print(new_board)
    # print(len(new_board))

    # print(render_board(new_board, target))
    ############################################################################


    red_cells = find_closest_red_cells(target = target,
                                      board =  board,
                                    closest_red_heuristic=closest_red_heuristic)
    min_cell_dist = float("inf")
    for red_cell in red_cells:
        for axis in ['vertical_fill', 'horizontal_fill']:
            cell_dist = heuristic_function(board, target, red_cell, axis)
            if cell_dist < min_cell_dist:
                min_cell_dist = cell_dist

    min_path = None
    for red in red_cells:
        print(f"SOURCE: {red}")
        path = a_star_search_tetromino_horizontal_vertical_nodes(board, target = target, source = red,
                           heuristic_function=heuristic_function)

        
        # if len(path) <= math.ceil(min_cell_dist/4):
        #     return path

        # failure handle
        if not path and len(red_cells) == 1:
            min_path = None
        
        # first update to min_path
        if not min_path and path:
            min_path = path
        
        # afterwards
        if path and min_path and len(path) < len(min_path):
            min_path = path

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Program execution time: {elapsed_time:.2f} seconds")

    return min_path




def a_star_search_tetromino_horizontal_vertical_nodes(board: dict[Coord, PlayerColor], 
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
        flatten_lst = [item for place in curr_node.actions for item in place.coords if item]
        print(f"flatten_lst: {flatten_lst}")
        print(render_board(curr_node.board, target, ansi=False))
        best_k_cells = choose_best_k_cells(coord_lst=flatten_lst[::-1], 
                                        board=curr_node.board, target=target,
                                        heuristic=heuristic_function, k = 0.7,
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
################################################################################
def choose_best_m_tetrominos(
    tetrominos: list[PlaceAction],
    board: dict[Coord, PlayerColor],
    target: Coord, 
    heuristic: Callable,
    m = None,
    axis = None
) -> list[PlaceAction]:
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

############################# GOAL STATE CHECKING ##############################
def reach_goal_state(node: Node, target: Coord) -> bool:
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
################################################################################

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


################################################################################

def find_box(source: Coord, board: dict[Coord, PlayerColor],
                      target: Coord) -> list[Coord]:
    """
    Given a board with a source (red_cell) and a target (B), this function
    returns a list of Coords if source is boxed. The list contains all 
    coordinates of the "box". If there is no such box, return None
    """
    return None

################################################################################
# def a_star_search_tetromino(board: dict[Coord, PlayerColor], 
#     target: Coord,
#     source: Coord,
#     heuristic_function: Callable) -> list[PlaceAction]:

#     queue = PriorityQueue()
#     initial_state = Node(parent_cell= source, parent_node= None,
#                          path_cost= 0, heuristic_cost= 
#                          heuristic_function(board, target, source), 
#                          place = PlaceAction(None, None, None, source),
#                          actions = [PlaceAction(None, None, None, source)], board=board)
#     # in the beginning, board 
#     # mapping each placeaction to its total estimated cost to goal
#     costs_dict = {initial_state.place: 0} # PlaceAction: int
#     queue.insert(initial_state)
#     count = 0

#     while queue:
#         curr_node = queue.pop()
#         print(f"COUNT: {count}")
#         print(curr_node.place)
#         print(f"PATHCOST: {curr_node.path_cost}")
#         print(f"Heuristic_cost: {curr_node.heuristic_cost}")
#         print(f"List of actions: {curr_node.actions}")


#         if count >= 250:
#             break
        
#         if reach_goal_state(curr_node, target):
#             print("SOLUTION FOUND") 
#             print(len(curr_node.actions) - 1)
#             print(render_board(curr_node.board, target))
#             return curr_node.actions[1:]
        
#         # could apply choose_best_k_cells here
#         # here we apply the sort function based on the heuristic function to 
#         # obtain the top score cells only, at the same time set the minimum 
#         # requirements of nodes added for each length of actions
#         flatten_lst = [item for place in curr_node.actions for item in place.coords if item]
#         print(f"flatten_lst: {flatten_lst}")
#         print(render_board(curr_node.board, target, ansi=False))
#         best_k_cells = choose_best_k_cells(coord_lst=flatten_lst[::-1], 
#                                         board=curr_node.board, target=target,
#                                         heuristic=heuristic_function, k = None)
#         print(best_k_cells)
#         for cell in best_k_cells:
#         # for cell in curr_node.place.coords:
#             if not cell:
#                 continue
#             # could apply choose_best_tetromino here
#             # sort the place in descending order of the tetromino
#             # might be no need to choose best m tetrominos
#             possible_tetrominos = generate_tetromino(cell = cell, board=curr_node.board)
#             best_m_tetrominos = choose_best_m_tetrominos(possible_tetrominos, 
#                                                          board=curr_node.board,
#                                                          target=target, heuristic= heuristic_function,
#                                                          m = None)
#             for place in best_m_tetrominos:
#                 new_actions = curr_node.actions + [place]
#                 new_board = place_action(curr_node.board.copy(), place)
                

#                 # create a new node for each placeaction
#                 new_node = Node(parent_cell= cell, parent_node=curr_node, 
#                                 path_cost=curr_node.path_cost + 1,
#                                 heuristic_cost= 
#                                 heuristic_function(new_board, target, 
#                                                    choose_best_k_cells(list(place.coords), new_board, target, heuristic_function, 1)[0]),
#                                 place = place, actions = new_actions, 
#                                 board = new_board)
                
#                 if is_valid(new_node):
#                     if place not in costs_dict or \
#                         new_node.path_cost < costs_dict[place]:
#                         costs_dict[place] = new_node.path_cost
#                         queue.insert(new_node)
#         count += 1
                
#     return None