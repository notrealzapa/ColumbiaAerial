# ColumbiaAerial

## Project Overview

ColumbiaAerial is a project aimed at fetching satellite images and their corresponding elevation data to predict heightmaps using a Convolutional Neural Network (CNN). The heightmaps are then visualized by overlaying them on the original satellite images.

## Features

- Fetches satellite images from the ArcGIS server.
- Retrieves elevation data using the SRTM data fetcher.
- Preprocesses images for model input.
- Trains a CNN to predict elevation heightmaps.
- Overlays the predicted heightmaps on the original images.
- Saves the combined images to a specified directory.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Git
- CodeSandbox or any local development environment

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/notrealzapa/ColumbiaAerial.git
    cd ColumbiaAerial
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. Initialize the Git repository (if not already initialized):

    ```bash
    git init
    ```

2. Add the remote repository:

    ```bash
    git remote add origin https://github.com/notrealzapa/ColumbiaAerial.git
    ```

3. Run the script to fetch images, train the model, and save the combined images:

    ```bash
    python convert.py
    ```

## Code Explanation

### Main Script (`convert.py`)

- **Constants and Initialization**: Defines constants for the number of images, zoom levels, latitude and longitude ranges. Ensures the `new_data` directory exists for saving output images. Initializes the SRTM data fetcher for elevation data.

- **Functions**: Includes various utility functions such as `lonlat_to_tilexy`, `fetch_image_from_arcgis`, `preprocess_image`, `is_ocean_or_one_color`, `generate_random_lat_lon`, `fetch_elevation_data`, `get_elevation`, `tile_to_bounds`, `overlay_heightmap_on_image`, and `save_image_with_heightmap`.

- **Dataset and Model**: Defines the `ImageDataset` class for handling images and their coordinates, and the `CNN` class for the Convolutional Neural Network model.

- **Training**: The `train_model` function is used to train the CNN model.

- **Main Execution**: Fetches a set of satellite images and their corresponding coordinates, creates a dataset and DataLoader, initializes and trains the CNN model, iterates through the dataset, predicts heightmaps, overlays the heightmaps on the original images, and saves the combined images to the `new_data` directory.

### Example Usage

The script iterates through the dataset, generates predicted heightmaps for each image, overlays the heightmaps on the original images, and saves the combined images to the `new_data` directory.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries, please contact the project maintainer at zyb2003@columbia.edu or zhumazhanbalapanov@gmail.com. 

## Thanks to...

Solal, Arthur and Anay contributed immensely to the project, thanks you very much guys! ajb2371@columbia.edu, sdp2170@columbia.edu, ac5683@columbia.edu
