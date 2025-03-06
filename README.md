# Bio-Inspired Optimization Algorithms

This project was developed as part of our coursework in 2023, focusing on bio-inspired optimization algorithms. The goal was to implement and compare several evolutionary and swarm intelligence-based methods for optimizing various benchmark functions.

## Implemented Algorithms
- **Differential Evolution (DE)** – A population-based optimization method that iteratively improves candidate solutions based on mutation and crossover operations.
- **Particle Swarm Optimization (PSO)** – A nature-inspired algorithm that mimics the social behavior of birds and fish to find optimal solutions.
- **Self-Organizing Migrating Algorithm (SOMA)** – A swarm-based algorithm that moves solutions towards better regions in the search space.
- **Firefly Algorithm** – A metaheuristic inspired by the flashing behavior of fireflies to attract better solutions.
- **Teaching-Learning-Based Optimization (TLBO)** – An optimization method based on the learning process in a classroom, where a teacher influences learners to improve their knowledge.

## Benchmark Functions
The algorithms were tested on several well-known mathematical functions used in optimization research:
- **Ackley**
- **Griewank**
- **Levy**
- **Michalewicz**
- **Rastrigin**
- **Rosenbrock**
- **Schwefel**
- **Sphere**
- **Zakharov**

## Structure
- `Functions.py` – Contains implementations of the benchmark functions.
- `main.py` – The main script for running optimization algorithms.
- `functions.csv` and related CSV files – Datasets for function evaluations.

## How to Use
To run an algorithm on a specific benchmark function, execute `main.py` and configure the parameters accordingly. The results are stored in CSV files for further analysis.

## Purpose
This project serves as an educational tool for understanding and experimenting with bio-inspired optimization techniques, helping to analyze their performance on complex mathematical landscapes.
