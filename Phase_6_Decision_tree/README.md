# 🌳 Decision Tree From Scratch

This project demonstrates a **Decision Tree Classifier implemented entirely from scratch** using Python and NumPy.

## Features

- Calculates **entropy** to measure uncertainty
- Calculates **information gain** to choose the best split
- Handles categorical features with binary-style splitting
- Builds the tree recursively
- Makes predictions using the learned tree rules
- Includes detailed print statements to trace every step

## Project Structure

- `decision_tree.py` : implementation of the tree and prediction logic
- `README.md` : this file
- Training data is included in simple encoded form (like 0/1/2 for categories)

## Example

The code uses a simplified **Play Tennis** dataset:
- Outlook (0=Sunny, 1=Overcast, 2=Rain)
- Humidity (0=High, 1=Normal)
- Windy (0=False, 1=True)

The labels are:
- 0 = No
- 1 = Yes

## How to Run

```bash
python decision_tree.py
