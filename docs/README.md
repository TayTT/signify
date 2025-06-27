# SIGNIFY: Sign Language Recognition System

A Python-based tool for processing images and videos for sign language recognition using MediaPipe for hand, face, and pose detection.

## Features

- Hand tracking with MediaPipe Hands
- Face landmark detection with MediaPipe FaceMesh
- Pose estimation with MediaPipe Pose
- Support for processing both static images and videos
- Support for processing image sequences as videos
- Image enhancement for better hand detection
- Output visualization with annotated landmarks
- Structured JSON output for further analysis

## Installation

### Prerequisites

- Python 3.7+
- OpenCV
- MediaPipe
- Other dependencies (install via pip)

```bash
pip install opencv-python mediapipe numpy
```

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/your-username/sign-language-recognition.git
cd sign-language-recognition
```

2. Create directories for your data:
```bash
mkdir -p data/sequence1 data/sequence2
```

3. Add your image sequences or videos to the data directories.

## Usage

### Basic Command

```bash
python main.py --image-dirs data --detect-faces --detect-pose
```

This will process all subdirectories in the `data` folder, each as a separate image sequence.

### Command-line Arguments

| Argument | Description |
|----------|-------------|
| `--output-dir PATH` | Directory to save output files (default: `./output`) |
| `--images FILE [FILE ...]` | List of individual image files to process |
| `--videos FILE [FILE ...]` | List of video files to process |
| `--image-dirs DIR [DIR ...]` | Directories containing image sequences to process as videos |
| `--image-extension EXT` | File extension for image sequences (default: `png`) |
| `--skip-frames N` | Process every Nth frame in videos (default: 2) |
| `--detect-faces` | Enable face landmark detection |
| `--detect-pose` | Enable pose landmark detection |
| `--skip-image-processing` | Skip processing of individual images |
| `--skip-video-processing` | Skip processing of videos |
| `--visualize-enhancements` | Add visual enhancements to output images |

### Examples

Process specific image files:
```bash
python main.py --images data/image1.png data/image2.png
```

Process specific image file and show image enhancements:
```bash
python main.py --images data/stockimg.png --detect-faces --detect-pose --visualize-enhancements
```

Process both videos and image sequences:
```bash
python main.py --videos data/video1.mp4 --image-dirs data/sequence1
```

Process image sequences with different file extension:
```bash
python main.py --image-dirs data/jpg_sequence --image-extension jpg
```

Enable face and pose detection, with visual enhancements:
```bash
python main.py --image-dirs data --detect-faces --detect-pose --visualize-enhancements
```

## Output

The tool generates the following outputs:

1. **Annotated Images**: Visual representations of the detected landmarks
2. **JSON Data**: Structured data of all detected landmarks for further processing
3. **Directory Structure**: Organized by input type (images, videos, image sequences)

Outputs are saved in the specified output directory (default: `./output`).

## Implementation Details

### Image Processing Pipeline

1. Image loading and enhancement
2. Hand landmark detection
3. Optional face landmark detection
4. Optional pose landmark detection
5. Visualization and saving

### Video Processing Pipeline

1. Frame extraction
2. Processing of individual frames
3. Landmark tracking across frames
4. Output generation


## Acknowledgments

- [MediaPipe](https://developers.google.com/mediapipe) for the vision processing libraries
- [OpenCV](https://opencv.org/) for image and video processing capabilities