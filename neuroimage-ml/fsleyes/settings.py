class Settings:
    def __init__(self):
        # Initialize default settings
        self.opacity = 1.0
        self.colormap = 'gray'
        self.isosurface_level = 0.5
        self.rotation = [0, 0, 0]
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        self.slice_index = [0, 0, 0]
        self.overlay_opacity = 0.5
        self.overlay_colormap = 'jet'

    def update(self, new_settings):
        # Update settings based on user input
        self.opacity = new_settings.get('opacity', self.opacity)
        self.colormap = new_settings.get('colormap', self.colormap)
        self.isosurface_level = new_settings.get('isosurface_level', self.isosurface_level)
        self.rotation = new_settings.get('rotation', self.rotation)
        self.zoom_level = new_settings.get('zoom_level', self.zoom_level)
        self.pan_offset = new_settings.get('pan_offset', self.pan_offset)
        self.slice_index = new_settings.get('slice_index', self.slice_index)
        self.overlay_opacity = new_settings.get('overlay_opacity', self.overlay_opacity)
        self.overlay_colormap = new_settings.get('overlay_colormap', self.overlay_colormap)
