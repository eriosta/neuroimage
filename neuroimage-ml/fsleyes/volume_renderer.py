from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from shader_programs import ShaderPrograms
from opengl_renderer import OpenGLRenderer

class VolumeRenderer(OpenGLRenderer):
    def __init__(self):
        super().__init__()
        self.shader_programs = ShaderPrograms()

    def render(self, data):
        self.data = data

        # Initialize GLUT
        glutInit()

        # Set the display mode
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)

        # Create a window
        glutCreateWindow(b'Volume Renderer')

        # Set the idle function
        glutIdleFunc(self.draw_scene)

        # Set the display function
        glutDisplayFunc(self.draw_scene)

        # Start the main loop
        glutMainLoop()

    def draw_data(self):
        # Draw the data as a 3D texture
        glEnable(GL_TEXTURE_3D)

        # Generate a texture ID
        texture_id = glGenTextures(1)

        # Bind the texture
        glBindTexture(GL_TEXTURE_3D, texture_id)

        # Set the texture parameters
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER)

        # Upload the texture data
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RED, self.data.shape[0], self.data.shape[1], self.data.shape[2], 0, GL_RED, GL_FLOAT, self.data)

        # Use the volume rendering shader program
        self.shader_programs.use_shader_program()

        # Set the uniform values
        self.shader_programs.set_uniform_1f("u_opacity", self.settings.opacity)
        self.shader_programs.set_uniform_3f("u_color", self.settings.color)

        # Draw a cube with the texture
        glBegin(GL_QUADS)
        glTexCoord3f(0, 0, 0); glVertex3f(-1, -1, -1)
        glTexCoord3f(1, 0, 0); glVertex3f( 1, -1, -1)
        glTexCoord3f(1, 1, 0); glVertex3f( 1,  1, -1)
        glTexCoord3f(0, 1, 0); glVertex3f(-1,  1, -1)

        # Stop using the shader program
        self.shader_programs.stop_using_shader_program()

        # Disable the 3D texture
        glDisable(GL_TEXTURE_3D)
