# neuroimage
# User Manual

## Overview
This application is designed to analyze and visualize functional networks in fMRI data using various decomposition techniques. The analysis is currently focused on a single subject's data from the ADHD200 dataset.

## How to use this app:
1. **Select Parameters**: Adjust the clustering parameters and decomposition settings in the sidebar according to your requirements.
2. **Run Analysis**: After adjusting the settings, click the **Run** button. 
3. **View Results**: The results will be displayed on the main panel, where you'll see visualizations and other outputs based on your selected parameters.

## Behind the Scenes
The application uses the `ComponentCorrelation` and `ComponentVisualization` classes from the `clustering.py` module to perform the analysis and visualization. 

`ComponentCorrelation` is responsible for fetching the fMRI data, performing the decomposition (either Dictionary Learning or ICA), computing the correlation matrix, and extracting clusters from the correlation matrix using hierarchical clustering.

`ComponentVisualization` is used to visualize the components of the fMRI data. It applies the decomposition to the data, and then generates a visualization for each component.

The results of the analysis are saved to a JSON file, and a download link is provided for the user to download the results.

## Neuroscience and Neuroimaging Background

The `clustering.py` module is based on principles from neuroscience and neuroimaging. It uses functional Magnetic Resonance Imaging (fMRI) data, a type of neuroimaging that measures brain activity by detecting changes associated with blood flow. This technique relies on the fact that cerebral blood flow and neuronal activation are coupled. When an area of the brain is in use, blood flow to that region also increases.

The module uses two main classes, `ComponentCorrelation` and `ComponentVisualization`, to analyze and visualize the fMRI data. These classes implement two key techniques in neuroimaging data analysis: decomposition and clustering.

Decomposition is a technique used to break down the fMRI data into separate components. The module supports two types of decomposition: Dictionary Learning and Independent Component Analysis (ICA). Dictionary Learning is a representation learning method which aims to find a sparse representation of the input data in the form of a linear combination of basic elements. These elements are called atoms and they compose a dictionary. ICA, on the other hand, is a computational method for separating a multivariate signal into additive subcomponents supposing the mutual statistical independence of the non-Gaussian source signals.

After decomposition, the module computes a correlation matrix to understand the relationship between different components. It then uses hierarchical clustering to group similar components together. Hierarchical clustering is a method of cluster analysis which seeks to build a hierarchy of clusters. It starts by treating each component as a separate cluster and then repeatedly executes the following two steps: (1) identify the two clusters that are closest together, and (2) merge the two most similar clusters. This iterative process continues until all the clusters are merged together.

The `ComponentVisualization` class is then used to visualize the components of the fMRI data. It applies the decomposition to the data, and then generates a visualization for each component. This allows users to visually inspect the results of the analysis.

These techniques provide a powerful tool for understanding the complex patterns of activity within the brain, and can be used to study a wide range of neurological and psychiatric conditions.
