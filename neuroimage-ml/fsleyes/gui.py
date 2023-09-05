import streamlit as st

class GUI:
    def __init__(self):
        self.settings = {
            'opacity': 1.0,
            'colormap': 'gray',
            'isosurface_level': 0.5,
            'rotation': [0, 0, 0],
            'zoom_level': 1.0,
            'pan_offset': [0, 0],
            'slice_index': [0, 0, 0],
            'overlay_opacity': 0.5,
            'overlay_colormap': 'jet'
        }

    def render(self, settings):
        st.sidebar.header("Settings")

        # Opacity
        self.settings['opacity'] = st.sidebar.slider("Opacity", 0.0, 1.0, settings.opacity)

        # Colormap
        colormap_options = ['gray', 'jet', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter', 'bone', 'copper', 'greens', 'blues', 'reds', 'purples', 'inferno', 'magma', 'plasma', 'viridis']
        self.settings['colormap'] = st.sidebar.selectbox("Colormap", colormap_options, index=colormap_options.index(settings.colormap))

        # Isosurface level
        self.settings['isosurface_level'] = st.sidebar.slider("Isosurface Level", 0.0, 1.0, settings.isosurface_level)

        # Rotation
        self.settings['rotation'] = [
            st.sidebar.slider("Rotation X", -180, 180, settings.rotation[0]),
            st.sidebar.slider("Rotation Y", -180, 180, settings.rotation[1]),
            st.sidebar.slider("Rotation Z", -180, 180, settings.rotation[2])
        ]

        # Zoom level
        self.settings['zoom_level'] = st.sidebar.slider("Zoom Level", 0.1, 3.0, settings.zoom_level)

        # Pan offset
        self.settings['pan_offset'] = [
            st.sidebar.slider("Pan Offset X", -1.0, 1.0, settings.pan_offset[0]),
            st.sidebar.slider("Pan Offset Y", -1.0, 1.0, settings.pan_offset[1])
        ]

        # Slice index
        self.settings['slice_index'] = [
            st.sidebar.slider("Slice Index X", 0, 100, settings.slice_index[0]),
            st.sidebar.slider("Slice Index Y", 0, 100, settings.slice_index[1]),
            st.sidebar.slider("Slice Index Z", 0, 100, settings.slice_index[2])
        ]

        # Overlay opacity
        self.settings['overlay_opacity'] = st.sidebar.slider("Overlay Opacity", 0.0, 1.0, settings.overlay_opacity)

        # Overlay colormap
        self.settings['overlay_colormap'] = st.sidebar.selectbox("Overlay Colormap", colormap_options, index=colormap_options.index(settings.overlay_colormap))
