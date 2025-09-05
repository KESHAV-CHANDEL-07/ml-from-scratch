# 🌳 Phase 6: Decision Tree — Machine Learning From Scratch

This project demonstrates a **Decision Tree Classifier implemented entirely from scratch** using Python and NumPy. In this phase, I explored both **Gini impurity** and **Entropy** as splitting criteria to deeply understand how a tree selects the best feature and threshold at each step.

## Features

- Calculates **entropy** to measure uncertainty and information gain
- Calculates **Gini impurity** to measure class impurity
- Supports choosing the best split based on either entropy or gini
- Manually selects the best split by minimizing weighted impurity or maximizing information gain
- Handles categorical features with binary-style splitting
- Builds the tree recursively with max depth control to avoid overfitting
- Makes predictions using the learned tree rules
- Includes detailed print statements to trace every calculation and step

## What I Learned

- How to manually calculate Gini impurity and entropy to measure data purity
- How to partition data into left and right branches
- How to build trees recursively from scratch
- How overfitting occurs in fully grown trees, and the importance of pruning or limiting max depth
- How decision trees form the base of more advanced ensemble models like Random Forests

## Example

A simplified **Play Tennis** dataset:
- Outlook (0=Sunny, 1=Overcast, 2=Rain)
- Humidity (0=High, 1=Normal)
- Windy (0=False, 1=True)

Target labels:
- 0 = No
- 1 = Yes

## How to Run

```bash
python decision_tree.py

