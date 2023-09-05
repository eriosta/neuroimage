import streamlit as st
from settings import Settings
from data_manager import DataManager
from opengl_renderer import OpenGLRenderer
from volume_renderer import VolumeRenderer
from overlay_and_slicing import OverlayAndSlicing
from gui import GUI

class MainApp:
    def __init__(self):
        self.settings = Settings()
        self.data_manager = DataManager()
        self.opengl_renderer = OpenGLRenderer()
        self.volume_renderer = VolumeRenderer()
        self.overlay_and_slicing = OverlayAndSlicing()
        self.gui = GUI()

    def run(self):
        # Load data
        from nilearn import datasets
        adhd_dataset = datasets.fetch_adhd(n_subjects=1)
        nifti_file = adhd_dataset.func[0]
        self.data_manager.load_data(nifti_file)

        # Render main image
        self.opengl_renderer.render(self.data_manager.data)

        # Volume rendering
        self.volume_renderer.render(self.data_manager.data)

        # Overlay and slicing
        self.overlay_and_slicing.render(self.data_manager.data)

        # GUI
        self.gui.render(self.settings)

        # Update settings
        self.settings.update(self.gui.settings)

        # Update rendering based on settings
        self.opengl_renderer.update_settings(self.settings)
        self.volume_renderer.update_settings(self.settings)
        self.overlay_and_slicing.update_settings(self.settings)

if __name__ == "__main__":
    st.set_page_config(page_title="Streamlit FSLeyes-like App", layout="wide")
    app = MainApp()
    app.run()
