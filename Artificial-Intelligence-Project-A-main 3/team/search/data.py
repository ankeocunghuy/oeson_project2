from .core import Coord, PlayerColor, BOARD_N
from .core import PlayerColor, Coord, PlaceAction
from dataclasses import dataclass
import heapq
import math
from .data import *


@dataclass(frozen=True, slots = True)
class Node():
    """
    Node is a data structure containing the cell (coordinate) and the distance
    from this node to the target

    parent_cell: the cell that generates this tetromino/node
    parent_node: The node where the parent_cell belongs to
    path_cost + heuristic_cost
    place: the placeaction that is about to be played
    actions: a list of actions taken so far to reach this state
    board: the current board reflecting changes done by actions parameter
    """
    axis: str                    # could
    path_cost: int
    heuristic_cost: int
    # heuristic_horizontal_fill: int 
    place: PlaceAction
    actions: list[PlaceAction]
    board: dict[Coord, PlayerColor]
    parent_cell: Coord                      # could be omitted
    parent_node: 'Node' 


    def __lt__(self, other):
        # Define custom comparison for sorting in the priority queue
        return self.path_cost + math.ceil(self.heuristic_cost /4) < \
            other.path_cost + math.ceil(other.heuristic_cost/4) 
        # return self.path_cost * 4 + self.heuristic_cost  < \
        #     other.path_cost * 4 + other.heuristic_cost 


    def __str__(self) -> str:
        return f"Node({self.parent_cell}, {self.path_cost + self.heuristic_cost})"

class PriorityQueue:
    def __init__(self):
        self.heap = []  # Initialize an empty heap


    def insert(self, node):
        # Insert the value into the heap
        heapq.heappush(self.heap, node)

    def pop(self):
        # Pop and return the smallest value from the heap
        node = heapq.heappop(self.heap)
        return node

    def __str__(self) -> str:
        return self.heap



# @dataclass(frozen=True, slots = True)
# class Node():
#     """
#     Node is a data structure containing the cell (coordinate) and the distance
#     from this node to the target

#     parent_cell: the cell that generates this tetromino/node
#     parent_node: The node where the parent_cell belongs to
#     path_cost + heuristic_cost
#     place: the placeaction that is about to be played
#     actions: a list of actions taken so far to reach this state
#     board: the current board reflecting changes done by actions parameter
#     """
#     parent_cell: Coord                      # could be omitted
#     parent_node: 'Node'                     # could
#     path_cost: int
#     heuristic_cost: int
#     place: PlaceAction
#     actions: list[PlaceAction]
#     board: dict[Coord, PlayerColor]

#     def __lt__(self, other):
#         # Define custom comparison for sorting in the priority queue
#         return self.path_cost * 4 + self.heuristic_cost  < \
#             other.path_cost * 4 + other.heuristic_cost


#     def __str__(self) -> str:
#         return f"Node({self.parent_cell}, {self.path_cost + self.heuristic_cost})"