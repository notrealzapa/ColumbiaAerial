import requests
from PIL import Image
from io import BytesIO
import numpy as np
import torch
from torchvision import transforms
from monodepth2.networks import ResnetEncoder, DepthDecoder
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array

# Fetch images from OpenAerialMap
def fetch_aerial_image(bbox):
    url = f"https://api.openaerialmap.org/tiles/{bbox}/256.png"
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        print(f"Failed to fetch image for bbox: {bbox}")
        return None

# Example bounding boxes (replace with real ones)
bboxes = [
    "12.4924,41.8902,12.4927,41.8905",
    "12.4928,41.8903,12.4931,41.8906",
    "12.4932,41.8904,12.4935,41.8907"
]

images = []
for bbox in bboxes:
    img = fetch_aerial_image(bbox)
    if img is not None:
        images.append(img)

print(f"Fetched {len(images)} images.")

# Preprocess images
def preprocess_image(image):
    image = image.resize((256, 256))
    image = img_to_array(image)
    image = image / 255.0
    return image

preprocessed_images = np.array([preprocess_image(img) for img in images])

# Load pre-trained Monodepth2 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = ResnetEncoder(18, False)
depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc)

encoder_path = 'monodepth2/models/mono+stereo_640x192/encoder.pth'
depth_decoder_path = 'monodepth2/models/mono+stereo_640x192/depth.pth'

encoder.load_state_dict(torch.load(encoder_path, map_location=device))
depth_decoder.load_state_dict(torch.load(depth_decoder_path, map_location=device))

encoder.to(device)
encoder.eval()
depth_decoder.to(device)
depth_decoder.eval()

# Predict depth
def predict_depth(image):
    transform = transforms.ToTensor()
    input_image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = encoder(input_image)
        outputs = depth_decoder(features)
        disp = outputs[("disp", 0)]
        depth_map = disp.squeeze().cpu().numpy()
    return depth_map

depth_maps = [predict_depth(img) for img in preprocessed_images]

# Visualize depth maps
def visualize_depth_maps(images, depth_maps):
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

visualize_depth_maps(images, depth_maps)
