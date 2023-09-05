import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from data_manager import DataManager

class OverlayAndSlicing:
    def __init__(self):
        self.data_manager = DataManager()
        self.overlay_data = None
        self.slice_index = None

    def load_overlay(self, nifti_file):
        # Load the overlay data
        self.data_manager.load_data(nifti_file)
        self.overlay_data = self.data_manager.data

    def set_slice_index(self, index):
        # Set the slice index
        self.slice_index = index

    def get_slice(self):
        # Get the slice from the main data
        main_slice = self.data_manager.data[:, :, self.slice_index]

        # Get the slice from the overlay data
        overlay_slice = self.overlay_data[:, :, self.slice_index]

        # Combine the slices
        combined_slice = np.stack((main_slice, overlay_slice), axis=-1)

        return combined_slice

    def draw_slice(self):
        # Get the combined slice
        combined_slice = self.get_slice()

        # Draw the slice as a 2D texture
        glEnable(GL_TEXTURE_2D)

        # Generate a texture ID
        texture_id = glGenTextures(1)

        # Bind the texture
        glBindTexture(GL_TEXTURE_2D, texture_id)

        # Set the texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)

        # Upload the texture data
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, combined_slice.shape[0], combined_slice.shape[1], 0, GL_RGBA, GL_FLOAT, combined_slice)

        # Draw a quad with the texture
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(-1, -1)
        glTexCoord2f(1, 0); glVertex2f( 1, -1)
        glTexCoord2f(1, 1); glVertex2f( 1,  1)
        glTexCoord2f(0, 1); glVertex2f(-1,  1)
        glEnd()

        # Disable the 2D texture
        glDisable(GL_TEXTURE_2D)
