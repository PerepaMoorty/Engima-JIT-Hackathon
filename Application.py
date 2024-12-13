import os
import time
import pygame

# Console Clear and Execution Time
os.system("cls" if os.name == "nt" else "clear")
_start_time = time.time()

pygame.init()

# ...

# Displaying Execution Time
_end_time = time.time()
print(f"Total Execution Time: {_end_time - _start_time:.2f}")
