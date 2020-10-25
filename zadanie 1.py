#Przemysław Scharmach 25.10.2020
#https://www.codingame.com/ide/puzzle/power-of-thor-episode-1

import sys
import math

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.
# ---
# Hint: You can use the debug stream to print initialTX and initialTY, if Thor seems not follow your orders.

# light_x: the X position of the light of power
# light_y: the Y position of the light of power
# initial_tx: Thor's starting X position
# initial_ty: Thor's starting Y position

light_x, light_y, initial_tx, initial_ty = [int(i) for i in input().split()]

x_of_thor = initial_tx
y_of_thor = initial_ty

# game loop
while True:
    remaining_turns = int(input())  # The remaining amount of turns Thor can move. Do not remove this line.
    direction_X = ""
    direction_Y = ""

    if y_of_thor > light_y:
        direction_Y = "N"
        y_of_thor = y_of_thor -1
    elif y_of_thor < light_y:
        direction_Y = "S"
        y_of_thor =  y_of_thor +1

    if x_of_thor > light_x:
        direction_X = "W"
        x_of_thor =  x_of_thor -1
    elif x_of_thor < light_x:
        x_of_thor = x_of_thor +1
        direction_X = "E"

    print(direction_Y + direction_X)
# koniec pętli
