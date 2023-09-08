import streamlit as st

# Define the progress bar manager
class ProgressUpdater:
    def __init__(self, total_clusters):
        self.progress_bar = st.sidebar.progress(0)
        self.total_clusters = total_clusters
        self.current_cluster_num = 0

    def update(self):
        self.current_cluster_num += 1
        self.progress_bar.progress(self.current_cluster_num / self.total_clusters)
        st.sidebar.text(f"Processing cluster {self.current_cluster_num}/{self.total_clusters}")

    def reset(self):
        self.current_cluster_num = 0
        self.progress_bar.progress(0)
