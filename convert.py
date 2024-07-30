import os
import random
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import srtm

NUM_IMAGES = 1000
ZOOM_LEVELS = range(7, 11)
LAT_RANGES = [(30, 60), (-30, 30)]
LON_RANGES = [(-130, -60), (30, 150)]
IMAGE_DIR = "new_data"

os.makedirs(IMAGE_DIR, exist_ok=True)

elevation_data = srtm.get_data()

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

def preprocess_image(image):
    """Preprocess image for model input."""
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    return image

def is_ocean_or_one_color(image):
    """Check if the image is mostly ocean or mostly one color."""
    try:
        img_array = np.array(image)

        if img_array.shape[2] != 3:
            raise ValueError("Image is not in RGB format")

        blue_channel = img_array[:, :, 2]
        green_channel = img_array[:, :, 1]
        red_channel = img_array[:, :, 0]
        
        is_blue = (blue_channel > 50) & (blue_channel > green_channel + 20) & (blue_channel > red_channel + 20)
        blue_pixel_count = np.sum(is_blue)
        total_pixel_count = img_array.shape[0] * img_array.shape[1]
        blue_pixel_ratio = blue_pixel_count / total_pixel_count
        
        is_mostly_ocean = blue_pixel_ratio > 0.5  

        std_dev = np.std(img_array, axis=(0, 1))
        is_one_color = np.all(std_dev < 30) 


        color_histograms = [np.histogram(img_array[:, :, i], bins=256, range=(0, 256))[0] for i in range(3)]
        max_counts = [np.max(histogram) for histogram in color_histograms]
        max_histogram_index = np.argmax(max_counts)
        dominant_color_count = max_counts[max_histogram_index]
        is_single_color = dominant_color_count > 0.75 * total_pixel_count  

        return is_mostly_ocean or is_one_color or is_single_color
    except Exception as e:
        print(f"Error in is_ocean_or_one_color function: {e}")
        return False

def generate_random_lat_lon():
    """Generate random latitude and longitude within specific land ranges."""
    lat_range = random.choice(LAT_RANGES)
    lon_range = random.choice(LON_RANGES)
    
    lat = random.uniform(lat_range[0], lat_range[1])
    lon = random.uniform(lon_range[0], lon_range[1])
    
    return lat, lon

def fetch_elevation_data(lat, lon, zoom, tile_size=256):
    """Fetch elevation data for the tile."""
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
            elevation_grid[i, j] = elevation

    return elevation_grid

def get_elevation(lat, lon):
    try:
        elevation = elevation_data.get_elevation(lat, lon)
        return elevation if elevation is not None else 0
    except Exception as e:
        return 0

def tile_to_bounds(x, y, zoom):
    n = 2 ** zoom
    lon_deg = x / n * 360 - 180
    lat_deg = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / n))))
    return {
        'w': lon_deg,
        'e': lon_deg + 360 / n,
        's': lat_deg,
        'n': np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + 1) / n))))
    }

class ImageDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, (lat, lon, zoom) = self.images[idx]
        heightmap = fetch_elevation_data(lat, lon, zoom)

        if self.transform:
            image = self.transform(image)

        heightmap = torch.tensor(heightmap, dtype=torch.float32)
        heightmap = heightmap.unsqueeze(0)  # Add channel dimension

        return image, heightmap, (lat, lon, int(zoom))

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, heightmaps, _ in dataloader:
            images = images.to(device)
            heightmaps = heightmaps.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, heightmaps)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    return model

def overlay_heightmap_on_image(image, heightmap_data):
    """Overlay heightmap on image."""
    fig, ax = plt.subplots(figsize=(heightmap_data.shape[1] / 100, heightmap_data.shape[0] / 100), dpi=100)
    cmap = plt.get_cmap('jet')
    norm = mcolors.Normalize(vmin=np.min(heightmap_data), vmax=np.max(heightmap_data))
    ax.imshow(heightmap_data, cmap=cmap, norm=norm)
    plt.axis('off')

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    heightmap_overlay = Image.open(buf)
    heightmap_overlay = heightmap_overlay.resize((image.shape[1], image.shape[0]), Image.NEAREST)
    combined = Image.blend(Image.fromarray((image * 255).astype(np.uint8)).convert('RGBA'), heightmap_overlay.convert('RGBA'), alpha=0.5)
    return combined

def save_image_with_heightmap(image, heightmap, filename):
    """Save the image with the heightmap overlay."""
    combined_image = overlay_heightmap_on_image(image, heightmap)
    combined_image.save(filename)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 8
learning_rate = 0.001
num_epochs = 10

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

images = []
for _ in range(NUM_IMAGES):
    lat, lon = generate_random_lat_lon()
    zoom = random.randint(7, 10)
    image = fetch_image_from_arcgis(lat, lon, zoom)
    if image and not is_ocean_or_one_color(image):
        images.append((image, (lat, lon, zoom)))

dataset = ImageDataset(images=images, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = CNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

trained_model = train_model(model, dataloader, criterion, optimizer, num_epochs)

if __name__ == "__main__":
    for i, (image, _, coords) in enumerate(dataset):
        image = image.unsqueeze(0).to(device)
        with torch.no_grad():
            model.eval()
            predicted_heightmap = model(image).squeeze().cpu().numpy()

        lat, lon, zoom = coords
        filename = os.path.join(IMAGE_DIR, f"{lat}_{lon}_{zoom}_combined.png")
        save_image_with_heightmap(image.squeeze().cpu().numpy().transpose(1, 2, 0), predicted_heightmap, filename)
        print(f"Saved combined image: {filename}")
