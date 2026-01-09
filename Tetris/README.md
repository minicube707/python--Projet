# Tetris Variants Repository

This repository contains several variations of the classic Tetris game, each with different levels of complexity and improvements. Starting from the basic version, I've progressively added features like graphical enhancements and AI bots that can play the game.

## 📌 Projects

### 🎮 Tetris.py
The original version of Tetris, implemented in Python. This is the core game, where players can play the classic Tetris experience.

### 🎨 Tetris Deluxe.py
An enhanced version of the original Tetris game.  
In addition to the base gameplay, I’ve added shaders and improved the visuals to make the game more aesthetically appealing.

### 🤖 Tetris Bot.py
An Bot that plays Tetris.  
This bot analyzes the game state and searches for the best place to position the falling piece, optimizing the gameplay automatically.

### 🤖 Tetris Bot+.py
A more advanced version of the Tetris Bot.  
This bot can anticipate the placement of the next piece and evaluates the current piece placement with the next one. Additionally, a penalty is applied if an empty cell is surrounded by filled cells (to avoid creating inefficient gaps).

### 🤖 Tetris Bot++.py
An even more sophisticated version of the previous bot.  
The malus (penalty) calculation has been refined to better handle the positioning of pieces. It considers the number of neighbors around empty cells near full cells for a more compact and optimized placement.

### 🤖 Tetris Bot+++.py
An improvement on Tetris Bot++.py.  
In this version, the bot’s horizontal movement is optimized. The bot calculates its horizontal position only once at the beginning of the piece’s descent (instead of recalculating every frame), which reduces unnecessary calculations.

### 🤖 Tetris Bot++++.py
The final iteration of the Tetris Bot.  
This version optimizes both horizontal and vertical movements. The bot calculates its vertical position only once at the beginning, and the piece is placed directly in the optimal location without falling further.

## 🚀 Getting Started
To run any of the Tetris variants, simply execute the corresponding `.py` file in a Python environment. Each version has its own directory, and you can find additional instructions in the individual files.

## 🛠️ Technologies
- Python
- Pygame (for the graphical elements)
- AI algorithms

## 📄 License
This repository is intended for educational and personal use.
