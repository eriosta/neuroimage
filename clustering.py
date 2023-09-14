import numpy as np
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from nilearn import datasets
from nilearn.decomposition import DictLearning, CanICA
from nilearn.plotting import plot_stat_map, find_xyz_cut_coords
from nilearn.image import index_img
from nilearn.input_data import NiftiMasker
from scipy.cluster.hierarchy import leaves_list, linkage, fcluster
import streamlit as st
from joblib import Parallel, delayed
from nilearn import image
from nilearn.masking import compute_epi_mask
from nilearn import plotting
import altair as alt

class ComponentCorrelation:
    def __init__(self, n_order, memory_level=2, cache_dir="nilearn_cache"):
        self.n_order = n_order
        self.cache_dir = cache_dir
        self.memory_level = memory_level

    def _fetch_data(self):
        """Fetch sample functional data for testing."""
        dataset = datasets.fetch_adhd(n_subjects=1)
        self.func_filename = [image.concat_imgs(dataset.func)]
        self.affine = self.func_filename[0].affine

    def _perform_decomposition(self, decomposition_type='dict_learning'):
        options = {
            "random_state": 0,
            "memory": self.cache_dir,
            "memory_level": self.memory_level
        }
        if decomposition_type == 'dict_learning':
            decomposition_model = DictLearning(n_components=self.n_order, **options, n_jobs=-1)
        elif decomposition_type == 'ica':
            decomposition_model = CanICA(n_components=self.n_order, **options, n_jobs=-1)
        else:
            raise ValueError("Invalid decomposition_type. Choose 'dict_learning' or 'ica'.")
            
        results = decomposition_model.fit_transform(self.func_filename)
        self.components_img = results[0]

    def _compute_correlation_matrix(self, p_threshold=0.01, corr_coefficient=0.5):
        self.correlation_matrix = np.zeros((self.n_order, self.n_order))
        self.results = []
        for i in range(self.n_order):
            for j in range(self.n_order):
                data_i = self.components_img[..., i]
                data_j = self.components_img[..., j]
                if data_i.size > 1 and data_j.size > 1:
                    correlation, p_value = pearsonr(data_i.ravel(), data_j.ravel())
                    
                    # Check if p-value is significant and correlation is above the threshold
                    if p_value < p_threshold and abs(correlation) > corr_coefficient:  
                        self.results.append({
                            'Component_1': i,
                            'Component_2': j,
                            'Pearson_r': correlation,
                            'p_value': p_value
                        })
                        self.correlation_matrix[i, j] = correlation
        self.correlation_matrix = np.nan_to_num(self.correlation_matrix)


    def _plot_heatmap(self, streamlit=None):
        diverging_cmap = plt.cm.RdBu_r
        figsize = (10, 5)
        cluster_grid = sns.clustermap(self.correlation_matrix, method="average", cmap=diverging_cmap, vmin=-1, vmax=1, annot=False, fmt=".2f", figsize=figsize)
        plt.close()  # Close the figure to prevent duplicate plots
        
        if streamlit is not None:
            st.pyplot(cluster_grid.fig)  # Display the clustermap figure in Streamlit
            
        # Get the order of the components after hierarchical clustering
        self.ordered_components = leaves_list(linkage(self.correlation_matrix, method='average'))

    def get_ordered_components(self):
        """Return the ordered list of component indices."""
        if hasattr(self, 'ordered_components'):
            return self.ordered_components
        else:
            raise AttributeError("Please run 'visualize_component_correlation' first to generate the ordered components.")

    def export_results_to_csv(self, filename="correlation_results.csv"):
        df = pd.DataFrame(self.results)
        df = df.sort_values(by='p_value')
        df.to_csv(filename, index=False)

    def visualize_component_correlation(self,streamlit,p_threshold,corr_coefficient,decomposition_type):
        self._fetch_data()
        self._perform_decomposition(decomposition_type)
        self._compute_correlation_matrix(p_threshold,corr_coefficient)
        self._plot_heatmap(streamlit)
        self.export_results_to_csv()
        
    def extract_clusters(self, t=1.5):
        """
        Extract clusters from the correlation matrix using hierarchical clustering.
        
        Parameters:
            t: float, optional (default=1.5)
                The threshold to form clusters. Components that are closer 
                than this threshold in the dendrogram will belong to the same cluster.
        
        Returns:
            dict
                A dictionary with cluster identifiers as keys and lists of 
                component indices as values.
        """
        # 1. Identify components that have significant correlations based on the cut-offs
        significant_components = set()
        for result in self.results:
            significant_components.add(result['Component_1'])
            significant_components.add(result['Component_2'])
    
        Z = linkage(self.correlation_matrix, method='average')
        cluster_assignments = fcluster(Z, t, criterion='distance')
        clusters = {}
        for idx, cluster_id in enumerate(cluster_assignments):
            if idx in significant_components:  # 2. Ensure only significant components are considered
                clusters.setdefault(cluster_id, []).append(idx)
        return clusters



class ComponentVisualization:
    def __init__(self, func_file,n_components,component_indices, fwhm, subject_index, ordered_components=None):
        self.func_file = func_file
        self.component_indices = component_indices
        self.fwhm = fwhm
        self.subject_index = subject_index
        self.n_components = n_components
        self.bg_img = datasets.load_mni152_template()
        if ordered_components is not None:
            self.ordered_components = ordered_components
        else:
            self.ordered_components = component_indices

    def apply_decomposition(self, decomposition_type='dict_learning'):
        fmri_subject = image.smooth_img(self.func_file, self.fwhm)
        if decomposition_type == 'dict_learning':
            decomposition_model = DictLearning(n_components=self.n_components, random_state=0, n_jobs=-1)
        elif decomposition_type == 'ica':
            decomposition_model = CanICA(n_components=self.n_components, random_state=0, n_jobs=-1)
        else:
            raise ValueError("Invalid decomposition_type. Choose 'dict_learning' or 'ica'.")
            
        decomposition_model.fit(fmri_subject)
        self.components_img_subject = decomposition_model.components_img_

    def visualize_components(self, streamlit=None):
        
        coordinates_list = []  # Initialize an empty list to store the coordinates. 
        
        # Get the mask image once outside the loop, assuming the first functional filename is representative for all
        mask_img = compute_epi_mask(self.func_file)
        masker = NiftiMasker(mask_img=mask_img, standardize=True)
        time_series_all = masker.fit_transform(self.func_file)
        
        for idx, component in enumerate(self.component_indices):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
            
            # Set the background color for the figure
            fig.patch.set_facecolor('white')
            
            # Set the background color for the individual subplots
            ax1.set_facecolor('white')
            ax2.set_facecolor('white')

            # Brain component visualization on ax1
            component_img = image.index_img(self.components_img_subject, component)
            x_coord, y_coord, z_coord = plotting.find_xyz_cut_coords(component_img)
            title_component = f'S{self.subject_index}C{component}'
            plotting.plot_stat_map(component_img, bg_img=self.bg_img, cut_coords=(x_coord, y_coord, z_coord), display_mode='ortho', title=title_component, colorbar=False, axes=ax1)
            
            coordinates_list.append((x_coord, y_coord, z_coord))  # Store the coordinates
            
            # Time series visualization on ax2
            time_series = time_series_all[:, component]
            max_int_timepoint = np.argmax(time_series)
            ax2.plot(time_series)
            ax2.scatter(max_int_timepoint, time_series[max_int_timepoint], color='red')
            ax2.set(title=f'Time Series of Component {component}', xlabel='Timepoints', ylabel='Intensity')
    
            plt.tight_layout()
            
            if streamlit is not None:
                st.pyplot(plt)  # Plot the figure in Streamlit
            
            plt.show()

        return coordinates_list  # Return the list of coordinates


    def process_and_visualize(self,streamlit,decomposition_type):
        self.apply_decomposition(decomposition_type)
        coordinates_list = self.visualize_components(streamlit)
        return coordinates_list
