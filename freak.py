import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import srtm
from io import BytesIO

elevation_data = srtm.get_data()

def get_elevation(lat, lon):
    """Get elevation for the given latitude and longitude."""
    try:
        elevation = elevation_data.get_elevation(lat, lon)
        return elevation if elevation is not None else "Elevation data not available."
    except Exception as e:
        return f"Error: {e}"

def lonlat_to_tilexy(lat, lon, zoom):
    """Convert latitude and longitude to tile x, y coordinates."""
    lat_rad = np.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    y = int((1 - np.log(np.tan(lat_rad) + 1 / np.cos(lat_rad)) / np.pi) / 2 * n)
    return x, y

def fetch_image_from_arcgis(lat, lon, zoom):
    """Fetch an image tile from the ArcGIS server."""
    x, y = lonlat_to_tilexy(lat, lon, zoom)
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except requests.RequestException as e:
        print(f"Error fetching image: {e}")
        return None

def fetch_elevation_data(lat, lon, zoom, tile_size=256):
    """Fetch elevation data for the tile area."""
    x, y = lonlat_to_tilexy(lat, lon, zoom)
    bounds = tile_to_bounds(x, y, zoom)
    lat_step = (bounds['n'] - bounds['s']) / tile_size
    lon_step = (bounds['e'] - bounds['w']) / tile_size

    elevation_grid = np.zeros((tile_size, tile_size))
    for i in range(tile_size):
        for j in range(tile_size):
            lat_point = bounds['s'] + i * lat_step
            lon_point = bounds['w'] + j * lon_step
            elevation = get_elevation(lat_point, lon_point)
            if isinstance(elevation, (int, float)):
                elevation_grid[i, j] = elevation

    return elevation_grid

def tile_to_bounds(x, y, zoom):
    """Convert tile x, y and zoom to geographic bounds."""
    n = 2 ** zoom
    lon_deg = x / n * 360 - 180
    lat_deg = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / n))))
    return {
        'w': lon_deg,
        'e': lon_deg + 360 / n,
        's': lat_deg,
        'n': np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + 1) / n))))
    }

def generate_heightmap_overlay(data):
    """Create a heatmap overlay from elevation data."""
    fig, ax = plt.subplots(figsize=(data.shape[1] / 100, data.shape[0] / 100), dpi=100)
    cmap = plt.get_cmap('jet')
    norm = mcolors.Normalize(vmin=np.min(data), vmax=np.max(data))
    ax.imshow(data, cmap=cmap, norm=norm)
    plt.axis('off')

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def overlay_heightmap_on_image(image, heightmap_data):
    """Overlay heightmap on the provided image."""
    heightmap_overlay = generate_heightmap_overlay(heightmap_data)
    heightmap_overlay = heightmap_overlay.resize(image.size, Image.Resampling.BOX)
    combined = Image.blend(image.convert('RGBA'), heightmap_overlay.convert('RGBA'), alpha=0.5)
    return combined

def is_ocean(image):
    """Check if the image is mostly ocean based on common ocean colors."""
    ocean_colors = [(0, 0, 128), (0, 105, 148), (0, 149, 182)]
    image = image.convert("RGB")
    for count, color in image.getcolors(image.size[0] * image.size[1]):
        if color in ocean_colors:
            return True
    return False

def fetch_and_combine_images(lat, lon, zoom):
    """Fetch map image, get elevation data, and combine them."""
    image = fetch_image_from_arcgis(lat, lon, zoom)
    if image:
        heightmap_data = fetch_elevation_data(lat, lon, zoom)
        combined_image = overlay_heightmap_on_image(image, heightmap_data)
        if is_ocean(image):
            print("The image is mostly ocean.")
        else:
            combined_image.show()
    else:
        print("Failed to fetch the map image.")

if __name__ == "__main__":
    lat, lon, zoom = 37.7749, -122.4194, 6
    fetch_and_combine_images(lat, lon, zoom)