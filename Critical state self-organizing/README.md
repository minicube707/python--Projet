# Critical State Self-Organizing – Sandpile Model

This repository contains several implementations of the **Self-Organized Criticality Sandpile model**, developed step by step to explore performance and behavior.
If you are interested in the topic, I highly recommend the following videos:
- (En) Veritasium: https://youtu.be/HBluLfX2F_k?t=1168  
- (Fr) ScienceEtonnante: https://youtu.be/MrsjMiL9W9o?t=816  

---

## 🧠 About the Model

The sandpile model is a classic example of **self-organized criticality**, a property of dynamical systems that naturally evolve toward a critical state where small events can trigger large-scale cascades (avalanches).

This project explores:
- Different implementation strategies
- Performance trade-offs
- Boundary conditions
- Statistical behavior (power laws)

---

## 🛠️ Technologies Used

- **Python**
- **NumPy**
- **Numba**
- **Matplotlib** (for statistical visualization)
- `collections.deque`

---

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
