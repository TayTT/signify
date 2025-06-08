# SIGNIFY: Sign Language Recognition System

A Python-based tool for processing images and videos for sign language recognition using MediaPipe for hand, face, and pose detection.

## Features

- Hand tracking with MediaPipe Hands
- Face landmark detection with MediaPipe FaceMesh (full mesh or simplified key landmarks)
- Pose estimation with MediaPipe Pose
- Support for processing individual images, videos, and image sequences
- Flexible input handling: single files, directories with mixed content, or batch processing
- Image enhancement for better hand detection
- Output visualization with annotated landmarks and videos
- Structured JSON output for further analysis
- Frame skipping for performance optimization

## Installation

### Prerequisites

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy
- Other dependencies (install via pip)

```bash
pip install opencv-python mediapipe numpy pathlib argparse
```

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/your-username/sign-language-recognition.git
cd sign-language-recognition
```

2. Create directories for your data:
```bash
mkdir -p data/videos data/images data/sequences
```

3. Add your videos, images, or image sequences to the data directories.

## Usage

## Favourite commands
```bash
python src/main.py --save-all-frames --process-single data/01April_frames --skip-frames 1
python src/landmarksVisualizer3D.py output/images_01Dec/video_landmarks.json --track-hands --mode animated

```

### Basic Commands

Process all data in the default `./data` directory:
```bash
python main.py
```

Process data in a custom directory:
```bash
python main.py --input-directory ./my_data
```

### Single Item Processing

The `--process-single` option allows you to process individual items:

```bash
# Process a single video file
python main.py --process-single ./video.mp4

# Process a single image file
python main.py --process-single ./image.jpg

# Process a directory containing image frames/sequence
python main.py --process-single ./frames_directory

# Process a directory containing multiple videos
python main.py --process-single ./videos_directory

# Process a directory with mixed content (videos and images)
python main.py --process-single ./mixed_directory
```

### Command-line Arguments

#### Core Options
| Argument | Description |
|----------|-------------|
| `--input-directory PATH` | Input directory to search recursively for videos or frame folders (default: `./data`) |
| `--output-directory PATH` | Directory to save output files (default: `./output`) |
| `--process-single PATH` | Process a single item: video file, image file, directory with frames, directory with videos, or mixed directory |

#### Processing Options
| Argument | Description |
|----------|-------------|
| `--full-mesh` | Display full face mesh instead of simplified key landmarks |
| `--enhance` | Apply image enhancement for better hand detection |
| `--skip-frames N` | Process every Nth frame in videos (default: 1 = process all frames) |
| `--save-all-frames` | Save all annotated frames to disk (not just sample frames) |
| `--image-extension EXT` | File extension for image sequences (default: `png`) |

### Advanced Examples

Process with full face mesh and image enhancement:
```bash
python main.py --full-mesh --enhance
```

Process every 5th frame and save all processed frames:
```bash
python main.py --skip-frames 5 --save-all-frames
```

Process a mixed directory with custom output location:
```bash
python main.py --process-single ./my_data --output-directory ./results --enhance
```

High-performance processing (skip frames, no enhancement):
```bash
python main.py --skip-frames 10 ./large_video.mp4
```

## Input Data Organization

The tool supports flexible input organization:

### Supported File Types
- **Videos**: `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`, `.flv`, `.webm`
- **Images**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`

### Directory Structures
```
data/
├── video1.mp4                    # Individual video files
├── image1.jpg                    # Individual image files
├── frames_sequence/              # Directory with image sequence
│   ├── frame_001.png
│   ├── frame_002.png
│   └── ...
├── video_collection/             # Directory with multiple videos
│   ├── sign1.mp4
│   ├── sign2.mp4
│   └── ...
└── mixed_content/                # Directory with mixed content
    ├── video.mp4
    ├── image.jpg
    └── sequence_folder/
        ├── frame1.png
        └── frame2.png
```

## Output

The tool generates comprehensive outputs organized by input type:

### Output Structure
```
output/
├── video_filename/               # For video files
│   ├── annotated_video.mp4      # Annotated video with landmarks
│   ├── video_landmarks.json     # Complete landmark data
│   └── frames/                  # Individual annotated frames
│       ├── frame_0000.png
│       ├── frame_0010.png       # (every 10th if not --save-all-frames)
│       └── ...
├── images_dirname/              # For image sequences
│   ├── annotated_video.mp4      # Video created from sequence
│   ├── video_landmarks.json     # Landmark data for sequence
│   └── frames/                  # Processed frames
└── image_filename/              # For individual images
    ├── annotated_image.jpg      # Annotated image
    └── landmarks_image.json     # Landmark data
```

### Output Data

1. **Annotated Videos/Images**: Visual representations with detected landmarks overlaid
2. **JSON Data**: Structured landmark data including:
   - Hand landmarks (left/right with confidence scores)
   - Face landmarks (full mesh or simplified key points)
   - Pose landmarks (with visibility scores)
   - Frame timestamps and metadata
3. **Individual Frames**: PNG files of processed frames (configurable sampling)

### JSON Data Structure
```json
{
  "metadata": {
    "input_source": "video.mp4",
    "input_type": "video",
    "total_frames_in_source": 300,
    "processed_frames": 30,
    "frame_skip": 10,
    "fps": 30,
    "resolution": "1920x1080",
    "processing_options": {
      "enhancement_applied": true,
      "full_face_mesh": false,
      "save_all_frames": false
    }
  },
  "frames": {
    "0": {
      "frame": 0,
      "timestamp": 0.0,
      "hands": {
        "left_hand": {"landmarks": [...], "confidence": 0.95},
        "right_hand": {"landmarks": [...], "confidence": 0.87}
      },
      "face": {
        "all_landmarks": [...],
        "mouth_landmarks": [...]
      },
      "pose": {...}
    }
  }
}
```

## Performance Optimization

### Frame Skipping
- Use `--skip-frames N` to process every Nth frame
- Reduces processing time significantly for long videos
- Frame numbers in output correspond to actual source frames

### Storage Management
- Default: Saves every 10th processed frame as samples
- Use `--save-all-frames` to save every processed frame
- Balance between data completeness and storage requirements

### Enhancement Options
- `--enhance`: Improves hand detection but increases processing time
- `--full-mesh`: More detailed face landmarks but higher computational cost

## Implementation Details

### Processing Pipeline

1. **Input Detection**: Automatically identifies file types and directory structures
2. **Frame Processing**: 
   - Hand landmark detection (always enabled)
   - Optional face landmark detection (simplified or full mesh)
   - Optional pose landmark detection
3. **Enhancement**: Optional image preprocessing for improved detection
4. **Output Generation**: Annotated visuals and structured data export

### MediaPipe Configuration
- **Hands**: Model complexity 1, max 2 hands, confidence thresholds 0.5
- **Face**: Refined landmarks, max 1 face, confidence threshold 0.5
- **Pose**: Model complexity 1, confidence thresholds 0.5

## Troubleshooting

### Common Issues
- **Video codec errors**: Tool automatically falls back to XVID codec if MP4V fails
- **Large file processing**: Use `--skip-frames` to reduce memory usage
- **Mixed directory processing**: Tool automatically detects and handles different content types

### Performance Tips
- Start with `--skip-frames 5` for initial testing
- Use `--enhance` only when hand detection is poor
- Save all frames only when needed for detailed analysis

## Acknowledgments

- [MediaPipe](https://developers.google.com/mediapipe) for the vision processing libraries
- [OpenCV](https://opencv.org/) for image and video processing capabilities