import os
import time

# Console Clear and Execution Time
os.system("cls" if os.name == "nt" else "clear")
start_time = time.time()

# ...

# Displaying Execution Time
end_time = time.time()
print(f"Total Execution Time: {end_time - start_time:.2f}")
