# Traveling Salesman Problem (TSP) – Brute Force Variants

This repository contains several **brute-force implementations of the Traveling Salesman Problem (TSP)**, with progressive improvements and visual enhancements.

Each version builds upon the previous one, adding features such as progress tracking and graphical visualization using **Pygame**.

⚠️ These implementations are intended for **educational purposes only**, as brute-force solutions become extremely slow when the number of points increases.

---

## 📁 Files Overview

### 1️⃣ Travelling salesman problem.py
- Basic brute-force implementation of the TSP.
- Asks the user for the number of points.
- Outputs:
  - Number of combinations tested
  - Time taken to solve
  - Shortest distance found
  - Coordinates of the path

---

### 2️⃣ Travelling salesman problem +.py
- Same as the basic version.
- Displays the **percentage of progress** during computation.

---

### 3️⃣ Travelling salesman problem 2.py
- Same algorithm as the basic version.
- Adds **graphical visualization using Pygame**:
  - White lines: paths currently being tested
  - Purple lines: current shortest path
  - Bottom-left corner: current distance
- At the end:
  - Prints the total number of combinations
  - Displays the shortest distance and path
  - Draws the best path in **green**

---

### 4️⃣ Travelling salesman problem 2+.py
- Same as `Travelling salesman problem 2.py`
- Adds **progress percentage display** above the distance (bottom-left corner)

---

### 5️⃣ Travelling salesman problem worst.py
- Same as `Travelling salesman problem 2.py`
- Searches for the **longest possible path** instead of the shortest

---

### 6️⃣ Travelling salesman problem worst +.py
- Same as `Travelling salesman problem 2+.py`
- Searches for the **longest path**
- Includes graphical visualization and progress tracking

---

## 🛠 Requirements
- Python 3.x
- `pygame` (for graphical versions)

```bash
pip install pygame
```

🎯 Purpose

This project demonstrates:
- The exponential complexity of brute-force TSP
- Progressive code enhancement
- Real-time visualization of algorithm behavior

