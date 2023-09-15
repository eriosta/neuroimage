from nilearn.input_data import NiftiMasker
from nilearn.datasets import fetch_adhd
from nilearn.decomposition import DictLearning
from nilearn.plotting import plot_stat_map, find_xyz_cut_coords
from nilearn.image import index_img
from nilearn.masking import compute_epi_mask
import matplotlib.pyplot as plt
import os
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage, fcluster
from scipy.stats import pearsonr
import streamlit as st
from nilearn import image

from nilearn.input_data import NiftiMasker
from nilearn.datasets import fetch_adhd
from nilearn.decomposition import DictLearning
from nilearn.plotting import plot_stat_map, find_xyz_cut_coords
from nilearn.image import index_img
from nilearn.masking import compute_epi_mask
import matplotlib.pyplot as plt
import os
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage, fcluster
from scipy.stats import pearsonr
# import streamlit as st
from nilearn import image
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff


st.set_option('deprecation.showPyplotGlobalUse', False)

class ComponentVisualization:
    def __init__(self, n_subjects=1, n_components=5, fwhm=10, output_dir='component_images'):
        self.n_subjects = n_subjects
        self.n_components = n_components
        self.fwhm = fwhm
        self.output_dir = output_dir
        self.adhd_dataset = fetch_adhd(n_subjects=n_subjects)
        self.mask_img = compute_epi_mask(self.adhd_dataset.func[0])

    def preprocess_data(self):
        masker = NiftiMasker(mask_img=self.mask_img, standardize=True)
        self.func_data_cleaned = masker.fit_transform(self.adhd_dataset.func[0], 
                                                      confounds=self.adhd_dataset.confounds[0])

    def apply_decomposition(self):
        self.dict_learn = DictLearning(
            n_components=self.n_components,
            n_epochs=1,
            alpha=10,
            reduction_ratio='cd',
            random_state=0,
            batch_size=5,
            method='cd',
            mask=self.mask_img,
            smoothing_fwhm=self.fwhm,
            standardize='zscore_sample',
            detrend=True,
            mask_strategy='epi',
            n_jobs=1,
            verbose=1
        )

        # Fit the model to the data
        self.dict_learn.fit(self.adhd_dataset.func[0], confounds=self.adhd_dataset.confounds[0])

        # Transform the data
        self.func_img = [image.concat_imgs(self.adhd_dataset.func)]
        self.func_transformed = self.dict_learn.transform(self.func_img)

        self.components_img = self.dict_learn.components_img_


    def visualize_components(self, streamlit=True):
        os.makedirs(self.output_dir, exist_ok=True)

        for comp in range(self.n_components):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

            # Brain component visualization
            comp_img = index_img(self.components_img, comp)
            title = f'Subject 1, Component {comp}'
            x_coord, y_coord, z_coord = find_xyz_cut_coords(comp_img)
            plot_stat_map(comp_img, title=title, display_mode='ortho', colorbar=False, 
                          cut_coords=(x_coord, y_coord, z_coord), axes=ax1)

            # Time series visualization
            time_series = self.func_data_cleaned[:, comp]
            max_int_timepoint = np.argmax(time_series)
            ax2.plot(time_series)
            ax2.scatter(max_int_timepoint, time_series[max_int_timepoint], color='red')
            ax2.set(title=f'Time Series of Component {comp}', xlabel='Timepoints', ylabel='Intensity')

            plt.tight_layout()
            if streamlit:
                st.pyplot(fig)
            else:
                plt.show()

    def visualize_timeseries_interactive(self, streamlit=True):
        traces = []

        for comp in range(self.n_components):
            time_series = self.func_data_cleaned[:, comp]
            trace = go.Scatter(
                x=list(range(len(time_series))),
                y=time_series,
                mode='lines',
                name=f'Component {comp}',
                visible=True if comp == 0 else 'legendonly'  # display only the first component by default
            )
            traces.append(trace)

        layout = go.Layout(
            title='Time Series of Components',
            xaxis=dict(title='Timepoints'),
            yaxis=dict(title='Intensity'),
            showlegend=True,
            autosize=False,
            width=600,  # Adjust the width of the plot
            height=600,   # Adjust the height of the plot
            font=dict(size=12)  # Increase font size
        )

        fig = go.Figure(data=traces, layout=layout)
        
        if streamlit:
            st.plotly_chart(fig)
        else:
            fig.show()

    def _compute_correlation_matrix(self, p_threshold=0.01, corr_coefficient=0.5):
        self.correlation_matrix = np.zeros((self.n_components, self.n_components))
        self.results = []
        for i in range(self.n_components):
            for j in range(self.n_components):
                data_i = self.func_transformed[0][..., i]
                data_j = self.func_transformed[0][..., j]
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
        self.correlation_matrix = pd.DataFrame(self.correlation_matrix)

    def _plot_dendrogram(self, streamlit=None):
        linked = linkage(self.correlation_matrix, 'average')

        # Create a dendrogram using plotly
        fig = ff.create_dendrogram(
            self.correlation_matrix, 
            orientation='left', 
            labels=self.correlation_matrix.columns.tolist()
        )
        fig.update_layout(width=600, height=600, font=dict(size=12))  # Increase font size
        
        if streamlit is not None:
            st.plotly_chart(fig)  # Display the dendrogram figure in Streamlit

        # Get the order of the components after hierarchical clustering
        self.ordered_components = leaves_list(linkage(self.correlation_matrix, method='average'))
        
    def process_and_visualize(self,streamlit=True):
        self.preprocess_data()
        if streamlit:
            st.toast("Data denoised of nuissance confounds.")
        self.apply_decomposition()
        if streamlit:
            st.toast("Decomposition completed.")
        # self.visualize_components(streamlit=streamlit)
        # self.visualize_timeseries_interactive(streamlit=streamlit)
        self._compute_correlation_matrix()
        if streamlit:
            st.toast("Correlation matrix generated.")
        # self._plot_dendrogram(streamlit=streamlit)

# analyzer = ComponentVisualization(n_subjects=1, n_components=50, fwhm=10)
# analyzer.process_and_visualize()

