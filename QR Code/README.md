
# 📱 QR Code Generator – Level 1

This repository contains a **from-scratch implementation of a QR Code generator**, limited to **Version 1 QR codes**.  
The project was developed step by step in order to understand the **internal structure and encoding process** of QR codes, without relying on external QR libraries.

The implementation follows official specifications and well-documented educational resources.

---

## 🧠 About the Project

A QR code is much more than a simple black-and-white image. It relies on a precise pipeline involving:

- Data encoding
- Bitstream construction
- Error correction
- Matrix placement and masking

This project focuses on:
- Understanding the **numeric encoding mode**
- Implementing the **QR Code structure manually**
- Visualizing and debugging each step of the generation process

Only **QR Code Version 1** is supported, making the project compact and educational.

---

## 🛠️ Technologies Used

- **Python**
- **NumPy** (for matrix manipulation)
- Standard Python libraries only (no QR generation packages)

---

## 📂 Files Description

### `QR_Code.py`
Main implementation of the QR code generator.

This program:
- Encodes numeric data
- Builds the QR Code bitstream
- Applies error correction
- Generates a **Version 1 QR Code matrix**
- Outputs the final QR code

Designed to be **clean, concise, and functional**.

---

### `QR_Code_Demo.py`
Educational and debug-oriented version of `QR_Code.py`.

This program:
- Performs the same operations as `QR_Code.py`
- Prints **each step of the QR Code generation process** in the terminal
- Displays intermediate representations (encoding, bitstream, matrix construction)

Useful for:
- Understanding the QR Code algorithm
- Debugging
- Learning purposes

---

## 📁 Subfolders

### `utils/`
Contains supporting data and configuration files required by the QR code generator:

- `log_table.npy` – Lookup table for **logarithm conversion** and **anti-log conversion**   
- `qr_capacity.json` – Maximum number of bits allowed for each **encoding mode**  
- `qr_rs_structure.json` – Defines the **size of error correction codewords** for different QR code configurations

These files allow the generator to hand

### `export/`
This folder is where **all generated QR codes** are saved.  
- Each QR code is exported as an image file (PNG or SVG)  
- Allows easy access and use of the generated codes in other applications
---

## 📚 References & Sources

This project is based on the following high-quality resources:

- Thonky – QR Code Tutorial  
  https://www.thonky.com/qr-code-tutorial/

- Nayuki – Creating a QR Code Step by Step  
  https://www.nayuki.io/page/creating-a-qr-code-step-by-step

- YouTube – I built a QR code with my bare hands to see how it works 
  https://www.youtube.com/watch?v=w5ebcowAJD8

---

## 🚀 Notes

- This project is **educational** and not intended to replace production-ready QR libraries
- Only **Version 1 QR Codes** are supported
- The focus is on **clarity, correctness, and understanding**, not performance
