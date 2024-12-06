# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part A: Single Player Tetress

from .core import Coord, PlayerColor, BOARD_N
from .core import PlayerColor, Coord, PlaceAction
from .data import *
from typing import Callable

def apply_ansi(
    text: str, 
    bold: bool = True, 
    color: str | None = None
):
    """
    Wraps some text with ANSI control codes to apply terminal-based formatting.
    Note: Not all terminals will be compatible!
    """
    bold_code = "\033[1m" if bold else ""
    color_code = ""
    if color == "r":
        color_code = "\033[31m"
    if color == "b":
        color_code = "\033[34m"
    return f"{bold_code}{color_code}{text}\033[0m"

def render_board(
    board: dict[Coord, PlayerColor], 
    target: Coord | None = None,
    ansi: bool = False
) -> str:
    """
    Visualise the Tetress board via a multiline ASCII string, including
    optional ANSI styling for terminals that support this.

    If a target coordinate is provided, the token at that location will be
    capitalised/highlighted.
    """
    output = ""
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            if board.get(Coord(r, c), None):
                is_target = target is not None and Coord(r, c) == target
                color = board[Coord(r, c)]
                color = "r" if color == PlayerColor.RED else "b"
                text = f"{color}" if not is_target else f"{color.upper()}"
                if ansi:
                    output += apply_ansi(text, color=color, bold=is_target)
                else:
                    output += text
            else:
                output += "."
            output += " "
        output += "\n"
    return output

def is_occupied(
    cell: Coord,
    board: dict[Coord, PlayerColor],
) -> bool:
    """
    Checks whether the current cell is free or not 
    Note this doesn't handle when row or column >= 11. This returns False
    This is resolved via adjust function declared below, or use default
    up and down method from data class


    Usage: is_occupied(Coord(2,5))
    E.g: cell (2,5) is unoccupied. is_occupied(Coord(2,5)) = False
    E.g: cell (2,12) is invalid. This returns False. 
    Solution: cell(2, adjust(12)) = cell(2, 1)
    """
    if board.get(cell, None):
        return True
    return False

def check_valid_place(
    place: PlaceAction,
    board: dict[Coord, PlayerColor]
) -> bool:
    """
    Returns whether a place is valid or not
    """
    for cell in place.coords:
        if is_occupied(cell, board):
            return False
    return True

def line_removal(board: dict[Coord, PlayerColor]):
    """
    This function checks if there exists any lines in the board (both 
    vertically and horizontally). If there is such a line, then return a new 
    board (of type dictionary) with all the cells on that line removed;
    otherwise, return the same board
    """
    lines_to_move = dict()
    for key in board:
        # print(f"KEY: {key}")
        # handle vertical line
        vertical_line = []
        if f"c{key.c}" not in lines_to_move:
            for offset in range(BOARD_N):
                temp = board.get(key.down(offset), None)
                # print(f"temp: {temp}")
                if temp:
                    vertical_line.append(key.down(offset))
                else:
                    vertical_line = []
                    break
            lines_to_move[f"c{key.c}"] = vertical_line
        
        # handle vertical line
        horizontal_line = []
        if f"r{key.r}" not in lines_to_move:
            # print("HI")
            for offset in range(BOARD_N):
                temp = board.get(key.right(offset), None)
                # print(f"horizontal temp: {temp}")
                if temp:
                    horizontal_line.append(key.right(offset))
                else:
                    horizontal_line = []
                    break
            lines_to_move[f"r{key.r}"] = horizontal_line
        # print(lines_to_move)
    finals = set()
    
    for position in lines_to_move:
        if lines_to_move[position]:
            for coord in lines_to_move[position]:
                board.pop(coord, None)
                finals.add(coord)

    return board, finals


def check_valid_coords(
    coord_lst: list[Coord],
    board: dict[Coord, PlayerColor]
) -> bool:
    """
    Returns whether a place is valid or not
    """
    for coord in coord_lst:
        if is_occupied(coord, board):
            return False
    return True


def adjust(
    number: int
) -> int:
    """
    Adjust the current column or row number
    Usage: adjust(11)
    """
    if number > (BOARD_N - 1):
        return number - BOARD_N
    return number


# Note that operations on coord such as coord.down() already adjusts for 11x11
# board


def move(
        cell: Coord, 
        offset: int, 
        direction: str) -> Coord:
    temp = None
    if direction == 'up':
        temp = cell.up(offset)
    elif direction == 'down':
        temp = cell.down(offset)
    elif direction == 'left':
        temp = cell.left(offset)
    elif direction == 'right':
        temp = cell.right(offset)
    
    return temp
    
def place_action(
board: dict[Coord, PlayerColor], 
action: PlaceAction
) -> None:
    """
    void method to update the board based on the action in the argument
    """
    for cell in action.coords:
        board[cell] = PlayerColor.RED
    return board
















######################## SINGLE-MOVE APPROACH ##################################
def single_move_expand(
board: dict[Coord, PlayerColor], 
node: Node,
target: Coord,
check_occupied: bool = True,
heuristic_function: Callable = None
) -> list[Node]:
    neighbor_nodes = []
    for direction in ['up', 'down', 'left', 'right']:
        nghbor_cell = move(node.cell, offset = 1, direction= direction)
        if check_occupied and single_move_is_occupied(nghbor_cell, board, target): 
            continue
        # total_cost = 1 +  
        temp_node = Node(cell = nghbor_cell, heuristic_cost= 
                         heuristic_function(nghbor_cell, target),
                         path_cost= node.path_cost + 1, parent = node) 
        neighbor_nodes.append(temp_node)
    return neighbor_nodes


def single_move_is_occupied(
    cell: Coord,
    board: dict[Coord, PlayerColor],
    target: Coord
) -> bool:
    if board.get(cell, None) and cell !=  target:
        return True
    return False



def single_move_search(
    board: dict[Coord, PlayerColor], 
    target: Coord,
    source: Coord,
    heuristic_function: Callable
) -> list[PlaceAction] | None:
    queue = PriorityQueue()
    closed_set = set()
    print("HI")
    initial_state = Node(cell = source, parent = None, path_cost= 0, 
                      heuristic_cost=heuristic_function(source, target))
    queue.insert(initial_state)
    iteration = 0

    while True:
        print("Start here")
        print(f"iteration: {iteration}")
        if not queue.heap:
            return None
        node = queue.pop()
        print(f"Node to be expanded: {node}")

        # Goal case
        if node.cell == target:
            path = []
            while node:
                path.append(node.cell)
                node = node.parent
            print(f"closed_set: {closed_set}")
            return path[::-1]
        
        closed_set.add(node.cell)
        expanded_nodes = single_move_expand(board, node, target, True,
                                             heuristic_function)
        # print(f"beginning queue.heap: {queue.heap}")
        
        # print(f"expanded_nodes: {expanded_nodes}")
        for expanded_node in expanded_nodes:
            if expanded_node.cell in closed_set:
                print("already found")
                continue
            queue.insert(expanded_node)
            # print(f"queue.heap: {queue.heap}")
        print(f"ending queue.heap: {len(queue.heap)}, {queue.heap}")
        print("End first expansion")
        iteration += 1
    
