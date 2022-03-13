# Adaptive Sampling on the go for bathymetry mapping

This repository contains implementation and results of adaptive sampling algorithms for bathymetry mapping of lakes an ponds. 

We propose few modifications of the existing GP-UCB algorithm to optimize the sensing task. Visit Project Page - https://sites.google.com/umass.edu/adaptive-sampling/home

## Project Directory Tree
- **ExplorationVsExploitation** : Contains implementation of different exploration versus exploitation trade-off algorithms and gaussian process regression.
- **GP-UCB** : Contains implementation of 2D scalar field mapping using baseline GP-UCB algorithm.
  - **GP_UCB_Modified** : Contains the implementation of the proposed _Adaptive-radius based GP-UCB algorithm_ for mapping a 2D scalar field.
  - **Sample_on_the_Way_GPUCB** : Contains the implementation of the proposed _On-the-way sampling GP-UCB algorithm_ for mapping a 2D scalar field.
- **NewApproach** : Contains the implementation and results of the proposed _dynamic programming GP-UCB algorithm_ for mapping a 2D scalar field.
  - **Evaluation of new approach**: Contains results of this approach
- **Results** : Contains results of all these algorithms, specifically trajectory length, error in estimation, and computation time of the algorithm.
- **Testing_with_RealLakeData**:  Contains results of trajectory length, error in estimation, and computation time for real lake data.
  - **lake**:  Contains real lake dataset
