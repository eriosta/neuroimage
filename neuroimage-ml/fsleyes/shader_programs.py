
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

class ShaderPrograms:
    def __init__(self):
        self.shader_program = None

    def create_shader_program(self, vertex_shader_source, fragment_shader_source):
        # Compile the vertex and fragment shaders
        vertex_shader = compileShader(vertex_shader_source, GL_VERTEX_SHADER)
        fragment_shader = compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)

        # Create a shader program
        self.shader_program = compileProgram(vertex_shader, fragment_shader)

    def use_shader_program(self):
        # Use the shader program
        glUseProgram(self.shader_program)

    def stop_using_shader_program(self):
        # Stop using the shader program
        glUseProgram(0)

    def set_uniform_1f(self, name, value):
        # Set a uniform float value
        location = glGetUniformLocation(self.shader_program, name)
        glUniform1f(location, value)

    def set_uniform_3f(self, name, value):
        # Set a uniform vec3 value
        location = glGetUniformLocation(self.shader_program, name)
        glUniform3f(location, value[0], value[1], value[2])

    def set_uniform_1i(self, name, value):
        # Set a uniform int value
        location = glGetUniformLocation(self.shader_program, name)
        glUniform1i(location, value)

    def set_uniform_matrix4fv(self, name, value):
        # Set a uniform mat4 value
        location = glGetUniformLocation(self.shader_program, name)
        glUniformMatrix4fv(location, 1, GL_FALSE, value)
