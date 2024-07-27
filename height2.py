import random
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from monodepth2.networks import ResnetEncoder, DepthDecoder
from tensorflow.keras.preprocessing.image import img_to_array

# Constants
ZOOM = 6
EPOCHS = 10
BATCH_SIZE = 8
NUM_SAMPLES = 5

class AerialImageDataset(Dataset):
    def __init__(self, lat_lon_pairs, zoom, transform=None):
        self.lat_lon_pairs = lat_lon_pairs
        self.zoom = zoom
        self.transform = transform
        self.images, self.depth_maps = self.load_images()

    def load_images(self):
        images, depth_maps = [], []
        for lat, lon in self.lat_lon_pairs:
            image, depth_map = fetch_and_process_images(lat, lon, self.zoom)
            if image and depth_map is not None:
                images.append(image)
                depth_maps.append(depth_map)
        return images, depth_maps

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        depth_map = self.depth_maps[idx]
        if self.transform:
            image = self.transform(image)
        depth_map = torch.tensor(depth_map, dtype=torch.float32).unsqueeze(0)
        return image, depth_map

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
    image = img_to_array(image) / 255.0
    return image

def load_monodepth2_model():
    """Load pre-trained Monodepth2 model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ResnetEncoder(18, False)
    depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc)

    encoder_path = 'monodepth2/models/mono+stereo_640x192/encoder.pth'
    depth_decoder_path = 'monodepth2/models/mono+stereo_640x192/depth.pth'

    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    depth_decoder.load_state_dict(torch.load(depth_decoder_path, map_location=device))

    encoder.to(device).eval()
    depth_decoder.to(device).eval()

    return encoder, depth_decoder, device

def predict_depth(image, encoder, depth_decoder, device):
    """Predict depth from image using Monodepth2."""
    transform = transforms.ToTensor()
    input_image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = encoder(input_image)
        outputs = depth_decoder(features)
        disp = outputs[("disp", 0)]
        depth_map = disp.squeeze().cpu().numpy()
    return depth_map

def visualize_depth_maps(images, depth_maps):
    """Visualize aerial images and their predicted depth maps."""
    num_images = len(images)
    plt.figure(figsize=(15, num_images * 5))
    for i in range(num_images):
        plt.subplot(num_images, 2, i * 2 + 1)
        plt.imshow(images[i])
        plt.title("Aerial Image")
        plt.axis('off')

        plt.subplot(num_images, 2, i * 2 + 2)
        plt.imshow(depth_maps[i], cmap='plasma')
        plt.title("Predicted Height Map")
        plt.axis('off')
    plt.show()

def is_ocean_or_one_color(image):
    """Check if the image is mostly ocean or mostly one color."""
    try:
        img_array = np.array(image)
        
        blue_threshold = 20
        blue_ratio_threshold = 0.4
        color_tolerance = 5

        blue_channel = img_array[:, :, 2]
        green_channel = img_array[:, :, 1]
        red_channel = img_array[:, :, 0]
        is_blue = (blue_channel > blue_threshold) & (blue_channel > green_channel + blue_threshold / 2) & (blue_channel > red_channel + blue_threshold / 2)
        blue_ratio = np.mean(is_blue)
        is_mostly_ocean = blue_ratio > blue_ratio_threshold
        
        std_dev = np.std(img_array, axis=(0, 1))
        is_one_color = np.all(std_dev < color_tolerance)
        
        return is_mostly_ocean or is_one_color
    except Exception as e:
        print(f"Error in is_ocean_or_one_color function: {e}")
        return False

def fetch_and_process_images(lat, lon, zoom):
    """Fetch map image, preprocess, and predict depth map."""
    image = fetch_image_from_arcgis(lat, lon, zoom)
    if image:
        if is_ocean_or_one_color(image):
            print("The image is mostly ocean or one color.")
            return None, None
        preprocessed_image = preprocess_image(image)
        encoder, depth_decoder, device = load_monodepth2_model()
        depth_map = predict_depth(preprocessed_image, encoder, depth_decoder, device)
        return image, depth_map
    else:
        print("Failed to fetch the map image.")
        return None, None

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_cnn(cnn, dataloader, epochs, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(cnn.parameters(), lr=0.001)

    cnn.to(device)
    cnn.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for images, depth_maps in dataloader:
            images, depth_maps = images.to(device), depth_maps.to(device)

            optimizer.zero_grad()
            outputs = cnn(images)
            loss = criterion(outputs, depth_maps.view(-1))  # Ensure depth_maps is correctly shaped
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader)}")

def generate_random_lat_lon():
    """Generate random latitude and longitude."""
    lat = random.uniform(-90, 90)
    lon = random.uniform(-180, 180)
    return lat, lon

# Example usage
if __name__ == "__main__":
    lat_lon_pairs = []
    while len(lat_lon_pairs) < NUM_SAMPLES:
        lat, lon = generate_random_lat_lon()
        image, depth_map = fetch_and_process_images(lat, lon, ZOOM)
        if image and depth_map is not None:
            lat_lon_pairs.append((lat, lon))

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = AerialImageDataset(lat_lon_pairs, ZOOM, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    cnn = SimpleCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cnn(cnn, dataloader, EPOCHS, device)
