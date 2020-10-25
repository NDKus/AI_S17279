#Przemysław Scharmach 25.10.2020
#https://www.codingame.com/ide/puzzle/horse-racing-duals

import sys
import math

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

s = set()
n = int(input())

for i in range(n):
    p = int(input())
    s.add(p)

sorted_s = sorted(s)
if len(sorted_s) > 1:
    min = sorted_s[1] - sorted_s[0]
    for i in range(2, len(sorted_s)):
        if sorted_s[i] - sorted_s[i - 1] < min:
            min = sorted_s[i] - sorted_s[i - 1]
    print(min)
else:
    print(0)

# Write an answer using print
# To debug: print("Debug messages...", file=sys.stderr, flush=True)

