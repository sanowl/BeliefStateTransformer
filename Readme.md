# Transformer Belief State Geometry Analysis

This project implements and extends the experiments described in the paper "Transformers Represent Belief State Geometry in their Residual Stream" by Adam S. Shai et al. The code provides a comprehensive framework for analyzing how transformer models encode belief states in their internal representations when trained on data generated by hidden Markov models (HMMs).

## Overview

The research paper demonstrates that transformers trained on next-token prediction tasks develop internal representations that capture the geometry of belief states over hidden states of the data-generating process. This implementation allows researchers and practitioners to replicate these findings and extend the analysis to new scenarios.

## Key Features

1. **Data Generation**: Implements the Mess3 and RRXOR processes as described in the paper, allowing for the generation of sequences with known underlying structure.

2. **Transformer Model**: Provides a simple transformer architecture that closely follows the model described in the paper.

3. **Belief State Analysis**: Includes functions to analyze the representation of belief states in the transformer's residual stream, replicating the core findings of the paper.

4. **Visualization Tools**: Offers various visualization methods to illustrate the belief state geometry, including PCA-based 3D plots and comparison graphs.

5. **Layer-wise Analysis**: Implements layer-wise analysis of belief state representations, extending the paper's investigations into how these representations evolve through the network.

6. **Training Dynamics**: Includes functionality to track how belief state representations change during the training process.

7. **Future Prediction Capability**:

8. **Mixed State Presentation (MSP) Structure**: Provides tools to examine and visualize the MSP structure of the underlying processes.

## Relation to the Paper

This implementation closely follows the methodology outlined in "Transformers Represent Belief State Geometry in their Residual Stream". It replicates key experiments, including:

- Training transformers on data from the Mess3 and RRXOR processes
- Analyzing the linear representation of belief states in the residual stream
- Comparing belief state distances with next-token prediction distances
- Visualizing the fractal-like structure of belief state geometry

Additionally, this code extends the paper's analysis by:

- Providing more detailed layer-wise analysis
- Implementing tools to track belief state representation across training epochs
- Offering a comprehensive examination of the MSP structure

## Usage

Researchers can use this code to:
1. Replicate the paper's findings
2. Extend the analysis to new types of hidden Markov models
3. Investigate how different transformer architectures represent belief states
4. Explore the relationship between belief state representation and model performance

By modifying the data generation processes or transformer architecture, users can investigate how these changes affect the model's ability to capture belief state geometry.

## Conclusion

This implementation serves as a valuable tool for researchers interested in the intersection of transformers, hidden Markov models, and belief state representation. It provides a solid foundation for further investigations into how neural networks capture and utilize the underlying structure of sequential data.