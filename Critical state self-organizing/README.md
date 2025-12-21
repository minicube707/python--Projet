    # Critical State Self-Organizing – Sandpile Model

This repository contains several implementations of the **Self-Organized Criticality Sandpile model**, developed step by step to explore performance and behavior.

## Files description

### `Sandpile1.py`
Simple implementation of the sandpile model using a **deque**.

### `Sandpile1b.py`
Same as `Sandpile1.py`, but sand grains **cannot move into blocked cells** on the grid and remain in their current position.

### `Sandpile2.py`
Implementation using **NumPy vectorization** for better performance.

### `Sandpile2b.py`
Same as `Sandpile2.py`, with an added **graph showing the probability distribution** (power law).

### `Sandpile2bb.py`
Same as `Sandpile2.py`, but with **periodic boundary conditions**:  
if a grain leaves the grid, it **reappears on the opposite side**.

### `Sandpile3.py`
Optimized implementation using **NumPy and Numba** for maximum performance.
