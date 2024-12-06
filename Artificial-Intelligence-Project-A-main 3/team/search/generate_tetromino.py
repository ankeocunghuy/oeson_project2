from .core import Coord, PlayerColor, BOARD_N
from .core import PlayerColor, Coord, PlaceAction
from .data import *
from typing import Callable
from .utils import move, check_valid_coords


def create_I(
    cell: Coord,
    board: dict[Coord, PlayerColor],
    direction: str = None,
    check_occupied: bool = True
) -> list[PlaceAction]:
    """
    Generate new tetrominos vertically, which means the upper part and lower 
    part of the current cell. check_occupied further filters whether the 
    place action is a valid action or not
    Eg:   
     x                                      oooo       oooo 
    unu   , create_I_vertical(n, 'up') =    unu   ;     unu     ; etc
     x                                       x           x
    """
    temp = move(cell, 1, direction)

    lst = []
    # vertical I
    vertical = [temp, move(cell, 2, direction), \
        move(cell, 3, direction), \
        move(cell, 4, direction)]
    if check_valid_coords(vertical, board):
        lst = [PlaceAction(*vertical)]
    
    # horizontal
    horizontals = []
    for left_offset in range(0, 4, 1):
        if direction in ['up', 'down']:
            horizontal = [temp.left(left_offset), temp.left(left_offset).right(1), \
                temp.left(left_offset).right(2), temp.left(left_offset).right(3)]
        elif direction in ['left', 'right']:
            horizontal = [temp.down(left_offset), temp.down(left_offset).up(1), \
                temp.down(left_offset).up(2), temp.down(left_offset).up(3)]
        if check_valid_coords(horizontal, board):
            horizontals.append(PlaceAction(*horizontal))
    
    lst.extend(horizontals)
    return lst



def create_O(up_left_cell: Coord):
    up_right_cell = move(up_left_cell, 1, 'right')
    return [up_left_cell, up_right_cell, 
            move(up_left_cell, 1, 'down'), move(up_right_cell, 1, 'down')]

def create_T1(center_down_cell: Coord):
    center_up_cell = move(center_down_cell, 1, 'up')
    return [center_down_cell, center_up_cell,
            move(center_up_cell, 1, 'left'), move(center_up_cell, 1, 'right')]

def create_T2(outside_cell: Coord):
    temp_cell = move(outside_cell, 1, 'right')
    return [outside_cell, temp_cell,
            move(temp_cell, 1, 'up'), move(temp_cell, 1, 'down')]

def create_T3(outside_cell: Coord):
    center_down_cell = move(outside_cell, 1, 'down')
    return [outside_cell, center_down_cell,
            move(center_down_cell, 1, 'left'), move(center_down_cell, 1, 'right')]

def create_T4(outside_cell: Coord):
    temp_cell = move(outside_cell, 1, 'left')
    return [outside_cell, temp_cell,
            move(temp_cell, 1, 'up'), move(temp_cell, 1, 'down')]


def create_J1(outside_cell: Coord):
    temp_cell = move(outside_cell, 1, 'right')
    return [outside_cell,temp_cell,temp_cell.up(1), temp_cell.up(2)]

def create_J3(outside_cell: Coord):
    temp_cell = move(outside_cell, 1, 'left')
    return [outside_cell,temp_cell,temp_cell.down(1), temp_cell.down(2)]

def create_J2(outside_cell: Coord):
    temp_cell = move(outside_cell, 1, 'down')
    return [outside_cell,temp_cell,temp_cell.right(1), temp_cell.right(2)]

def create_J4(outside_cell: Coord):
    temp_cell = move(outside_cell, 1, 'up')
    return [outside_cell,temp_cell,temp_cell.left(1), temp_cell.left(2)]


def create_I_tetromino(    
    cell: Coord,
    board: dict[Coord, PlayerColor],
    check_occupied: bool = True):
    left_I = create_I(cell, board, 'left', True)
    right_I = create_I(cell, board, 'right', True)
    down_I = create_I(cell, board, 'down', True)
    up_I = create_I(cell, board, 'up', True)
    return left_I + right_I + down_I + up_I

def create_T_tetromino(cell: Coord,
                       board: dict[Coord, PlayerColor],
                       check_occupied = True) -> list:
    lst = []
    for spot in [move(cell, 1, 'up'), move(cell, 1, 'left'),
                 move(cell, 1, 'right'), cell.right(2).down(1),
                 cell.left(2).down(1), move(cell, 2, 'down'),
                 cell.down(2).left(1), cell.down(2).right(1)]:
        shape = create_T1(spot)
        if check_valid_coords(shape, board):
            lst.append(PlaceAction(*shape))

    for spot in [move(cell, 1, 'up'), move(cell, 1, 'right'), 
                 move(cell, 1, 'down'), cell.down(2).left(1),
                 cell.down(1).left(2), cell.left(2), cell.up(1).left(2),
                 cell.up(2).left(1)]:
        shape = create_T2(spot)
        if check_valid_coords(shape, board):
            lst.append(PlaceAction(*shape))
            
    for spot in [move(cell, 1, 'down'), move(cell, 1, 'left'),
                move(cell, 1, 'right'), cell.right(2).up(1),
                cell.left(2).up(1), move(cell, 2, 'up'),
                cell.up(2).left(1), cell.up(2).right(1)]:
        shape = create_T3(spot)
        if check_valid_coords(shape, board):
            lst.append(PlaceAction(*shape))

    for spot in [move(cell, 1, 'up'), move(cell, 1, 'left'), 
                 move(cell, 1, 'down'), cell.down(2).right(1),
                 cell.down(1).right(2), cell.right(2), cell.up(1).right(2),
                 cell.up(2).right(1)]:
        shape = create_T4(spot)
        if check_valid_coords(shape, board):
            lst.append(PlaceAction(*shape))
    return lst


def create_J_tetromino(cell: Coord, board: dict[Coord, PlayerColor],
                       check_occupied = True) -> list[PlaceAction]:
    lst = []
    for spot in [cell.up(1), cell.right(1), cell.down(1), cell.down(2),
                 cell.down(3).left(1), cell.down(2).left(2), 
                 cell.down(1).left(2), cell.left(2), cell.left(1).up(1)]:
        shape = create_J1(spot)
        if check_valid_coords(shape, board):
            lst.append(PlaceAction(*shape))

    for spot in [cell.up(2), cell.right(1).up(1), cell.right(1), cell.down(1),
                 cell.left(1), cell.left(2), 
                 cell.left(3).up(1), cell.left(2).up(2), cell.left(1).up(2)]:
        shape = create_J2(spot)
        if check_valid_coords(shape, board):
            lst.append(PlaceAction(*shape))

    for spot in [cell.up(3).right(1), cell.up(2).right(2), cell.up(1).right(2), 
                 cell.right(2), cell.right(1).down(1), cell.down(1), 
                cell.left(1), cell.up(1), cell.up(2)]:
        shape = create_J3(spot)
        if check_valid_coords(shape, board):
            lst.append(PlaceAction(*shape))

    for spot in [cell.down(2), cell.left(1).down(1), cell.left(1), cell.up(1),
                 cell.right(1), cell.right(2),
                 cell.right(3).down(1), cell.right(2).down(2), cell.right(1).down(2)]:
        shape = create_J4(spot)
        if check_valid_coords(shape, board):
            lst.append(PlaceAction(*shape))
    
    return lst
    
def create_O_tetromino(cell: Coord,
                       board: dict[Coord, PlayerColor],
                       check_occupied = True) -> list:
    lst = []
    spot_1 = move(cell, offset=2, direction='up')
    left_spot_1 = move(spot_1, 1, 'left')
    spot_2 = move(cell, offset = 1, direction='right')
    up_spot_2 = move(spot_2, 1, 'up')
    spot_3 = move(cell, offset = 1, direction='down')
    left_spot_3 = move(spot_3, 1, 'left')
    spot_4 = move(cell, offset=2, direction='left')
    up_spot_4 = move(spot_4, 1, 'up')

    for spot in [spot_1, left_spot_1, spot_2, up_spot_2, spot_3, left_spot_3,
                 spot_4, up_spot_4]:
        shape = create_O(up_left_cell=spot)
        if check_valid_coords(shape, board):
            lst.append(PlaceAction(*shape))

    return lst

def create_L_tetromino(cell: Coord, board: dict[Coord, PlayerColor],
                       check_occupied = True) -> list[PlaceAction]:
    lst = []
    for spot in [cell.up(1), cell.left(1), cell.down(1), cell.down(2),
                 cell.down(3).right(1), cell.down(2).right(2),
                 cell.down(1).right(2), cell.right(2), cell.right(1).up(1)]:
        shape = create_L1(spot)
        if check_valid_coords(shape, board):
            lst.append(PlaceAction(*shape))

    for spot in [cell.down(2), cell.right(1).down(1), cell.right(1), cell.up(1),
                 cell.left(1), cell.left(2),
                 cell.left(3).down(1), cell.left(2).down(2), cell.left(1).down(2)]:
        shape = create_L2(spot)
        if check_valid_coords(shape, board):
            lst.append(PlaceAction(*shape))

    for spot in [cell.up(3).left(1), cell.up(2).left(2), cell.up(1).left(2),
                 cell.left(2), cell.left(1).down(1), cell.down(1),
                cell.right(1), cell.up(1), cell.up(2)]:
        shape = create_L3(spot)
        if check_valid_coords(shape, board):
            lst.append(PlaceAction(*shape))

    for spot in [cell.up(2), cell.left(1).up(1), cell.left(1), cell.down(1),
                 cell.right(1), cell.right(2),
                 cell.right(3).up(1), cell.right(2).up(2), cell.right(1).up(2)]:
        shape = create_L4(spot)
        if check_valid_coords(shape, board):
            lst.append(PlaceAction(*shape))

    return lst

def create_L1(outside_cell: Coord):
    temp_cell = move(outside_cell, 1, 'left')
    return [outside_cell,temp_cell,temp_cell.up(1), temp_cell.up(2)]

def create_L3(outside_cell: Coord):
    temp_cell = move(outside_cell, 1, 'right')
    return [outside_cell,temp_cell,temp_cell.down(1), temp_cell.down(2)]

def create_L2(outside_cell: Coord):
    temp_cell = move(outside_cell, 1, 'up')
    return [outside_cell,temp_cell,temp_cell.right(1), temp_cell.right(2)]

def create_L4(outside_cell: Coord):
    temp_cell = move(outside_cell, 1, 'down')
    return [outside_cell,temp_cell,temp_cell.left(1), temp_cell.left(2)]


def create_Z_tetromino(cell: Coord, board: dict[Coord, PlayerColor],
                       check_occupied = True) -> list[PlaceAction]:
    lst = []
    for spot in [cell.up(1), cell.right(1), cell.down(1), cell.down(1).left(1),
                 cell.left(2), cell.up(1).left(3),
                 cell.up(2).left(2), cell.up(2).left(1)]:
        shape = create_Z1(spot)
        if check_valid_coords(shape, board):
            lst.append(PlaceAction(*shape))

    for spot in [cell.down(1), cell.right(1), cell.left(1), cell.up(1).left(1),
                 cell.up(2), cell.up(3).right(1),
                 cell.right(2).up(2), cell.right(2).up(1)]:
        shape = create_Z2(spot)
        if check_valid_coords(shape, board):
            lst.append(PlaceAction(*shape))

    return lst

def create_Z1(outside_cell: Coord):
    temp_cell = move(outside_cell, 1, 'right')
    return [outside_cell,temp_cell,temp_cell.down(1), temp_cell.down(1).right(1)]

def create_Z2(outside_cell: Coord):
    temp_cell = move(outside_cell, 1, 'down')
    return [outside_cell,temp_cell,temp_cell.left(1), temp_cell.left(1).down(1)]

def create_S_tetromino(cell: Coord, board: dict[Coord, PlayerColor],
                       check_occupied = True) -> list[PlaceAction]:
    lst = []
    for spot in [cell.up(1), cell.left(1), cell.down(1), cell.down(1).right(1),
                 cell.right(2), cell.up(1).right(3),
                 cell.up(2).right(2), cell.up(2).right(1)]:
        shape = create_S1(spot)
        if check_valid_coords(shape, board):
            lst.append(PlaceAction(*shape))

    for spot in [cell.down(1), cell.right(1), cell.left(1), cell.up(1).right(1),
                 cell.up(2), cell.up(3).left(1),
                 cell.left(2).up(2), cell.left(2).up(1)]:
        shape = create_S2(spot)
        if check_valid_coords(shape, board):
            lst.append(PlaceAction(*shape))

    return lst

def create_S1(outside_cell: Coord):
    temp_cell = move(outside_cell, 1, 'left')
    return [outside_cell,temp_cell,temp_cell.down(1), temp_cell.down(1).left(1)]

def create_S2(outside_cell: Coord):
    temp_cell = move(outside_cell, 1, 'down')
    return [outside_cell,temp_cell,temp_cell.right(1), temp_cell.right(1).down(1)]


def generate_tetromino(cell: Coord, 
                       board: dict[Coord, PlayerColor],
                       check_occupied = True) -> list[PlaceAction]:
    lst = []
    I_tetromino = create_I_tetromino(cell, board, check_occupied)
    O_tetromino = create_O_tetromino(cell, board, check_occupied)
    T_tetromino = create_T_tetromino(cell, board, check_occupied)
    J_tetromino = create_J_tetromino(cell, board, check_occupied)
    L_tetromino = create_L_tetromino(cell, board, check_occupied)
    Z_tetromino = create_Z_tetromino(cell, board, check_occupied)
    S_tetromino = create_S_tetromino(cell, board, check_occupied)


    lst = I_tetromino + O_tetromino + T_tetromino + J_tetromino + \
        L_tetromino + Z_tetromino + S_tetromino
    return lst
