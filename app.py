import base64
import streamlit as st
import time  # Add the time module
import pandas as pd  # Add the pandas library

from clustering import *
from encoder import *
from progress import *

from nilearn import datasets
import nilearn.datasets as datasets
import psutil
import json
import os

import base64

st.set_page_config(layout="wide")

order_components = 20
correlation_tool = ComponentCorrelation(n_order=order_components)

# Fetch the ADHD200 resting-state fMRI dataset
n_subjects = 1
adhd_dataset = datasets.fetch_adhd(n_subjects=n_subjects)
func_filenames = adhd_dataset.func
fwhm = 6

decomposition_key = {
    'Dictionary Learning':'dict_learning',
    'ICA':'ica'
}

def get_file_content_as_string(file_path):
    """Generate a download link for the file."""
    with open(file_path, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def measure_resources(func):
    def wrapper(*args, **kwargs):
        # Measure memory before function
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / 1024 ** 2  # Convert bytes to MB
        # Measure CPU usage before function
        start_cpu = time.process_time()
        result = func(*args, **kwargs)
        # Measure memory after function
        end_mem = process.memory_info().rss / 1024 ** 2
        # Measure CPU usage after function
        end_cpu = time.process_time()
        # Display in Streamlit
        st.toast(f"Memory used by function `{func.__name__}`: {end_mem - start_mem:.2f} MB")
        st.toast(f"CPU time used by function `{func.__name__}`: {end_cpu - start_cpu:.2f} seconds")
        
        return result
    return wrapper

# Add the @measure_resources decorator to functions you want to measure

def main():

    order_components = 20
    correlation_tool = ComponentCorrelation(n_order=order_components)

    # Introduction and Background
    st.info(
        """
        Welcome to the Subject-Level Functional Network Analysis App! This tool is designed to 
        analyze and visualize functional networks in fMRI data using various decomposition techniques.

        **Background**: Functional Magnetic Resonance Imaging (fMRI) provides insights into brain 
        activity by detecting changes in blood flow. Through fMRI data decomposition, one can isolate 
        individual networks or components of brain activity, leading to better understandings of cognitive 
        processes. This analysis is currently focused on a single subject's data from the [ADHD200 dataset](https://nilearn.github.io/dev/modules/generated/nilearn.datasets.fetch_adhd.html).

        """
    )
    # Tutorial Steps
    st.info(
        """
        **How to use this app:**
        1. **Select Parameters**: Adjust the clustering parameters and decomposition settings in the sidebar 
           according to your requirements.
        2. **Run Analysis**: After adjusting the settings, click the **Run** button. 
        3. **View Results**: The results will be displayed on this main panel, where you'll see visualizations 
           and other outputs based on your selected parameters.

        Start by adjusting the parameters in the sidebar to the left!
        """
    )

    st.sidebar.title("Subject-Level Functional Network Analysis")
       
    # Grouping & Spacing: Organize controls in expandable sections
    with st.sidebar.expander("Clustering Parameters",expanded=True):
        t = st.slider(
            "Hierarchical clustering distance threshold (t)", 
            min_value=0.05, max_value=5.0, value=1.5, step=0.1, 
            help="Adjust the distance threshold used during hierarchical clustering. Lower values yield more clusters, capturing finer details of the functional networks, while higher values result in fewer clusters, possibly representing larger network structures."
        )

        p_threshold = st.slider(
            "Pearson correlation p-value threshold", 
            min_value=0.01, max_value=1.0, value=0.01, step=0.01, 
            help="Set the significance level for Pearson correlation between time courses of regions/nodes in the functional network. Lower thresholds make correlations more stringent, potentially reducing false positives but may increase false negatives."
        )
    
    with st.sidebar.expander("Decomposition",expanded=True):
        order_components = st.slider(
            "Number of functional components", 
            min_value=5, max_value=50, value=10, step=1, 
            help="Specify the number of functional components (or networks) to extract from the fMRI data. This determines the dimensionality of the data after decomposition. Increasing the number of components can capture more nuanced functional activity but risks overfitting."
        )
        fwhm = st.slider(
            "FWHM of Gaussian smoothing kernel", 
            min_value=0, max_value=20, value=6, step=1, 
            help="Specify the full-width at half maximum (FWHM) of the Gaussian smoothing kernel applied to the fMRI data. This parameter controls the amount of spatial smoothing. Increasing the FWHM can improve signal-to-noise ratio but may blur distinct functional regions."
        )

        decomposition_type = st.radio(
            "Choose decomposition type", 
            {'Dictionary Learning': 'dict_learning', 'ICA': 'ica'}, 
            help="Select the decomposition technique. Both methods extract functional components from the fMRI data, but their underlying assumptions and processes differ."
        )

        if decomposition_type == 'Dictionary Learning':
            if st.checkbox("Show more information for Dictionary Learning"):
                st.info(
                    "- **How it works**: This technique learns a dictionary of basis functions (or 'atoms') which best represents the input data in a sparse manner.\n"
                    "- **Pros**: Good for extracting temporally independent networks. Often leads to more interpretable results.\n"
                    "- **Cons vs. ICA**: Dictionary Learning doesn't guarantee spatial or temporal independence and might be computationally intensive for large datasets."
                )
        
        elif decomposition_type == 'ICA':
            if st.checkbox("Show more information for ICA"):
                st.info(
                    "- **How it works**: Assumes fMRI signals are mixtures of independent non-Gaussian source signals. Decomposes data into such source signals maximizing their statistical independence.\n"
                    "- **Pros**: Widely used for its ability to extract spatially independent brain networks.\n"
                    "- **Cons vs. Dictionary Learning**: Some ICA components can be hard to interpret or might represent noise."
                )

    # Descriptions
    st.sidebar.markdown("After selecting parameters, click on **Run**. This will initiate the analysis based on your settings. The results will be visualized in the main panel.")

    # Add "Run" button
    run_button = st.sidebar.button("Run")
    
    def initialize_correlation_tool(order_components):
        return ComponentCorrelation(n_order=order_components)

    @measure_resources
    def visualize_correlation(correlation_tool, p_threshold, decomposition_type, decomposition_key):
        correlation_tool.visualize_component_correlation(streamlit=True, p_threshold=p_threshold, decomposition_type=decomposition_key[decomposition_type])
        return correlation_tool.extract_clusters(t=t)

    def create_clusters_dataframe(clusters):
        clusters_df = pd.DataFrame([(cluster_id, component_indices) for cluster_id, component_indices in clusters.items()], columns=['Cluster', 'Component Indices'])
        clusters_df['Component Indices'] = clusters_df['Component Indices'].apply(lambda x: ', '.join(map(str, x)))
        return clusters_df

    def display_clusters(clusters):
        st.write("Clusters:")
        for cluster_id, component_indices in clusters.items():
            with st.expander(f"Cluster {cluster_id}"):
                st.write("**Component Indices:**", ', '.join(map(str, component_indices)))

    @measure_resources
    def process_and_display_images(func_filenames, clusters, order_components, fwhm, decomposition_type, decomposition_key):
        # Define the list to hold our results
        all_clusters_coordinates = []
    
        progress_updater = ProgressUpdater(len(clusters))
    
        for i, func_file in enumerate(func_filenames):
            for cluster_id, component_indices in clusters.items():
                st.info(f"Visualizing components for cluster {cluster_id}")
    
                cluster_coordinates = {'cluster_id': cluster_id, 'components': {}}
                    
                visualizer = ComponentVisualization(func_file, order_components, component_indices, fwhm, i)
                    
                visualization_results = visualizer.process_and_visualize(streamlit=True, decomposition_type=decomposition_key[decomposition_type])
                    
                for component, coords in zip(component_indices, visualization_results):
                    coordinates_dict = {
                        'x': coords[0],
                        'y': coords[1],
                        'z': coords[2]
                    }
                    cluster_coordinates['components'][component] = coordinates_dict
    
                all_clusters_coordinates.append(cluster_coordinates)
    
                st.warning(f"Done with cluster {cluster_id}. Moving to the next cluster.")
                
                progress_updater.update()  # Update the progress
    
            st.info("Done! Getting Coordinates...")
    
        # Saving the results as a JSON file locally
        with open('clusters_coordinates.json', 'w') as json_file:
            json.dump(all_clusters_coordinates, json_file, cls=NumpyEncoder)
    
        # Generate a link for the user to download the file
        b64_file_data = get_file_content_as_string('clusters_coordinates.json')
        href = f'<a href="data:file/json;base64,{b64_file_data}" download="clusters_coordinates.json">Download JSON File</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        st.write("Results saved to `clusters_coordinates.json`")
        st.json(all_clusters_coordinates)
    
    if run_button:
        st.header("Starting analysis...")
        st.write(f"Visualizing component correlation with t = {t}")
        
        correlation_tool = initialize_correlation_tool(order_components)
        clusters = visualize_correlation(correlation_tool, p_threshold, decomposition_type, decomposition_key)
        clusters_df = create_clusters_dataframe(clusters)
        display_clusters(clusters)
        process_and_display_images(func_filenames, clusters, order_components, fwhm, decomposition_type, decomposition_key)

if __name__ == "__main__":
    main()
    

# import json
# import streamlit as st

# def generate_cluster_annotations(clusters, cluster_labels):
#     annotations = {}
#     for cluster_id, component_indices in clusters.items():
#         selected_label = st.selectbox(f'Select label for cluster {cluster_id}', cluster_labels)
#         annotations[cluster_id] = f"{selected_label} is determined by components {component_indices}"
#     with open('cluster_annotations.json', 'w') as f:
#         json.dump(annotations, f)

# # Call the function after clusters are generated
# # User defined cluster labels
# cluster_labels = ['','Left Hemisphere','Right Hemisphere','Background', 'Frontal Pole', 'Insular Cortex', 'Superior Frontal Gyrus', 'Middle Frontal Gyrus', 'Inferior Frontal Gyrus, pars triangularis', 'Inferior Frontal Gyrus, pars opercularis', 'Precentral Gyrus', 'Temporal Pole', 'Superior Temporal Gyrus, anterior division', 'Superior Temporal Gyrus, posterior division', 'Middle Temporal Gyrus, anterior division', 'Middle Temporal Gyrus, posterior division', 'Middle Temporal Gyrus, temporooccipital part', 'Inferior Temporal Gyrus, anterior division', 'Inferior Temporal Gyrus, posterior division', 'Inferior Temporal Gyrus, temporooccipital part', 'Postcentral Gyrus', 'Superior Parietal Lobule', 'Supramarginal Gyrus, anterior division', 'Supramarginal Gyrus, posterior division', 'Angular Gyrus', 'Lateral Occipital Cortex, superior division', 'Lateral Occipital Cortex, inferior division', 'Intracalcarine Cortex', 'Frontal Medial Cortex', 'Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)', 'Subcallosal Cortex', 'Paracingulate Gyrus', 'Cingulate Gyrus, anterior division', 'Cingulate Gyrus, posterior division', 'Precuneous Cortex', 'Cuneal Cortex', 'Frontal Orbital Cortex', 'Parahippocampal Gyrus, anterior division', 'Parahippocampal Gyrus, posterior division', 'Lingual Gyrus', 'Temporal Fusiform Cortex, anterior division', 'Temporal Fusiform Cortex, posterior division', 'Temporal Occipital Fusiform Cortex', 'Occipital Fusiform Gyrus', 'Frontal Operculum Cortex', 'Central Opercular Cortex', 'Parietal Operculum Cortex', 'Planum Polare', "Heschl's Gyrus (includes H1 and H2)", 'Planum Temporale', 'Supracalcarine Cortex', 'Occipital Pole']
# generate_cluster_annotations(clusters, cluster_labels)
