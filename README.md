<div align="center">
<p>
   <a align="center" target="_blank">
   <img width="900" src="./images/MiVOLO.jpg"></a>
</p>
<br>
</div>

# MiVOLO: Multi-input Transformer for Age and Gender Estimation

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mivolo-multi-input-transformer-for-age-and/age-estimation-on-utkface)](https://paperswithcode.com/sota/age-estimation-on-utkface?p=mivolo-multi-input-transformer-for-age-and) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/beyond-specialization-assessing-the-1/age-estimation-on-imdb-clean)](https://paperswithcode.com/sota/age-estimation-on-imdb-clean?p=beyond-specialization-assessing-the-1) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/beyond-specialization-assessing-the-1/facial-attribute-classification-on-fairface)](https://paperswithcode.com/sota/facial-attribute-classification-on-fairface?p=beyond-specialization-assessing-the-1)

## 🙏 Acknowledgments

This project is built upon the outstanding work of the original MiVOLO authors. We extend our deepest gratitude to:

**Original Authors:**
- [Maksim Kuprashevich](https://github.com/wildchlamydia) 
- [Irina Tolstykh](https://github.com/iitolstykh)
- Grigorii Alekseenko

**Original Research Papers:**
- [MiVOLO: Multi-input Transformer for Age and Gender Estimation (2023)](https://arxiv.org/abs/2307.04616)
- [Beyond Specialization: Assessing the Capabilities of MLLMs in Age and Gender Estimation (2024)](https://arxiv.org/abs/2403.02302)

This enhanced version builds upon their groundbreaking work by adding advanced tracking capabilities, JSON export functionality, and improved Re-ID models for better performance in video analysis scenarios.

## 🔗 Quick Links

| Resource | Link | Description |
|----------|------|-------------|
| **Original Repository** | [WildChlamydia/MiVOLO](https://github.com/WildChlamydia/MiVOLO) | Original MiVOLO implementation |
| **Papers** | [2023 Paper](https://arxiv.org/abs/2307.04616) • [2024 Paper](https://arxiv.org/abs/2403.02302) | Original research papers |
| **Live Demo** | [HuggingFace Demo](https://huggingface.co/spaces/iitolstykh/age_gender_estimation_demo) | Try online |
| **Telegram Bot** | [@AnyAgeBot](https://t.me/AnyAgeBot) | Mobile testing |
| **Dataset** | [Lagenda Dataset](https://wildchlamydia.github.io/lagenda/) | Training data |
| **Models** | [Google Drive](https://drive.google.com/drive/folders/1F84L0j0T8XIsOLnBArZjDPxH0XP2NwlU) | Pre-trained models |
| **Documentation** | [Advanced Re-ID Guide](REID_MODELS.md) | Enhanced tracking models |

## ✨ What's New in This Version

### 🚀 **Enhanced Tracking System**
- **Advanced Re-ID Models**: OSNet, ResNeSt, CLIP-based, ConvNeXt, and MobileNet backbones
- **BoT-SORT Integration**: Superior tracking with reduced ID switches
- **Confidence-Based Selection**: Keeps only highest-confidence predictions per track
- **Global Temporal Smoothing**: Single final prediction per track across entire video

### 📊 **JSON Export & Analytics**
- **Comprehensive Demographics**: Frame-by-frame age group and gender statistics
- **Spatial Tracking**: Bounding box coordinates and trajectories
- **Cropped Images**: Automatic person crop extraction with unique filenames
- **Timestamp Precision**: Accurate timing for each detection

### ⚡ **Performance Optimizations**
- **Runtime Monitoring**: Total processing time and FPS tracking
- **Frame Skipping**: Configurable frame processing for speed
- **Batch Processing**: Optimized inference pipeline
- **Half-Precision Support**: 2x speed improvement on compatible hardware

## 🛠️ Installation

### Prerequisites
- Python 3.8+ 
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Quick Install
```bash
# Clone this enhanced version
git clone https://github.com/your-username/MiVOLO.git
cd MiVOLO

# Install dependencies
pip install -r requirements.txt
pip install .
```

### Download Required Models
```bash
# Create models directory
mkdir -p models

# Download detector (required)
wget -O models/yolov8x_person_face.pt \
  "https://drive.google.com/uc?export=download&id=1CGNCkZQNj5WkP3rLpENWAOgrBQkUWRdw"

# Download age/gender model (required)
wget -O models/mivolo_imbd.pth.tar \
  "https://drive.google.com/uc?export=download&id=11i8pKctxz3wVkDBlWKvhYIh7kpVFXSZ4"
```

## 🎮 Quick Start Demo

### Basic Image Processing
```bash
# Download test image
wget https://variety.com/wp-content/uploads/2023/04/MCDNOHA_SP001.jpg -O test_image.jpg

# Run basic demo
python demo.py \
  --input test_image.jpg \
  --output output/ \
  --detector-weights models/yolov8x_person_face.pt \
  --checkpoint models/mivolo_imbd.pth.tar \
  --with-persons \
  --draw
```

### Video Processing with Enhanced Features
```bash
# Enhanced video processing with JSON export
python demo.py \
  --input video.mp4 \
  --output output/ \
  --detector-weights models/yolov8x_person_face.pt \
  --checkpoint models/mivolo_imbd.pth.tar \
  --with-persons \
  --draw \
  --tracker botsort.yaml \
  --confidence-based \
  --save-json \
  --json-output-path output/demographics.json \
  --save-crops \
  --crops-dir crops
```

### YouTube Video Processing
```bash
python demo.py \
  --input "https://www.youtube.com/watch?v=VIDEO_ID" \
  --output output/ \
  --detector-weights models/yolov8x_person_face.pt \
  --checkpoint models/mivolo_imbd.pth.tar \
  --with-persons \
  --draw \
  --tracker botsort.yaml
```

## 🔧 Advanced Configuration

### Speed Optimization
```bash
# Fast processing (skip frames)
python demo.py \
  --input video.mp4 \
  --output output/ \
  --detector-weights models/yolov8x_person_face.pt \
  --checkpoint models/mivolo_imbd.pth.tar \
  --with-persons \
  --draw \
  --skip-frames 4 \
  --half
```

### Maximum Accuracy
```bash
# Setup advanced Re-ID model
python setup_reid_models.py --download clip_vit_l14_msmt17.pt --use clip_vit_l14_msmt17.pt

# Run with best accuracy settings
python demo.py \
  --input video.mp4 \
  --output output/ \
  --detector-weights models/yolov8x_person_face.pt \
  --checkpoint models/mivolo_imbd.pth.tar \
  --with-persons \
  --draw \
  --tracker botsort.yaml \
  --confidence-based \
  --min-detection-conf 0.7 \
  --min-gender-conf 0.8
```

### Compare Re-ID Models
```bash
# Interactive comparison of different Re-ID models
python demo_advanced_reid.py --input video.mp4 --compare
```

## 📊 Output Examples

### Real-time Console Output
```
Video processing completed!
Total frames processed: 3638
Total processing time: 125.43 seconds
Average processing FPS: 29.01
Output video saved to: output/out_video.avi
```

### JSON Export Sample
```json
[
  {
    "time": "0.5 sec",
    "total_people": 3,
    "man": 2,
    "woman": 1,
    "age_twenty": 1,
    "age_thirty": 2,
    "people_data": [
      {
        "id": 1,
        "gender": "man",
        "age": 32,
        "cropped_img": "/crops/1_15.jpg",
        "location": [425, 280]
      }
    ]
  }
]
```

## 🏆 Performance Benchmarks

### Model Performance (Original Research)
| Model | Type | Dataset | Age MAE | Gender Acc | Download |
|-------|------|---------|---------|------------|----------|
| mivolo_d1 | face+body | IMDB | 4.24 | 99.46% | [Link](https://drive.google.com/file/d/11i8pKctxz3wVkDBlWKvhYIh7kpVFXSZ4/view) |
| mivolov2_d1 | face+body | Lagenda | 3.65 | 97.99% | [Telegram Bot](https://t.me/AnyAgeBot) |

### Re-ID Model Comparison (New)
| Model | Speed | Accuracy | Use Case |
|-------|--------|----------|----------|
| OSNet x0.5 | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Real-time |
| CLIP ViT-L | ⚡⚡ | ⭐⭐⭐⭐⭐ | Best accuracy |
| MobileNet | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | Edge devices |

## 🚀 Command Line Options

### Core Options
```bash
--input PATH              # Input image/video/YouTube URL
--output PATH             # Output directory
--detector-weights PATH   # YOLO detector model
--checkpoint PATH         # MiVOLO model checkpoint
--device DEVICE          # cuda:0, cpu
```

### Quality & Performance
```bash
--with-persons            # Enable person detection
--draw                    # Save visualized outputs
--half                    # Use half-precision (2x faster)
--skip-frames N          # Process every N+1 frames
```

### Tracking & Smoothing
```bash
--tracker CONFIG         # botsort.yaml, bytetrack.yaml
--confidence-based       # Use best predictions only
--temporal-smoothing     # Frame-by-frame smoothing
--global-smoothing       # Video-wide smoothing
```

### Export & Analysis
```bash
--save-json              # Export demographics JSON
--json-output-path PATH  # JSON file location
--save-crops             # Save person crops
--crops-dir DIR          # Crops directory name
```

## 📚 Advanced Features

### 🎯 Confidence-Based Selection
Superior to traditional smoothing - keeps only the highest-confidence prediction per track:
- Immune to outliers and noisy predictions
- Provides detailed confidence metrics
- Shows which frame had the best prediction
- Takes priority over global smoothing

### 🔄 Advanced Re-ID Models
State-of-the-art person re-identification for better tracking:
- **OSNet**: Best balance of speed and accuracy
- **ResNeSt**: Maximum accuracy for challenging scenarios
- **CLIP**: Cross-modal understanding for diverse conditions
- **ConvNeXt**: Modern CNN architecture
- **MobileNet**: Optimized for edge deployment

### 📊 JSON Analytics
Comprehensive demographic analysis:
- Frame-by-frame statistics
- Age group distributions (teen, twenty, thirty, etc.)
- Gender ratios and counts
- Spatial coordinates and trajectories
- Cropped image references

## 🔬 Validation & Testing

```bash
# Validate on standard datasets
python eval_pretrained.py \
  --dataset_images /path/to/dataset \
  --dataset_annotations /path/to/annotations \
  --dataset_name utk \
  --checkpoint models/mivolo_imbd.pth.tar \
  --batch-size 512 \
  --half \
  --with-persons
```

Supported datasets: `utk`, `imdb`, `lagenda`, `fairface`, `adience`, `agedb`

## 📄 License

This enhanced version maintains the same license as the original work. Please see [license](./license) for details.

## 📖 Citation

If you use this enhanced version or the original MiVOLO work, please cite:

```bibtex
@article{mivolo2023,
   Author = {Maksim Kuprashevich and Irina Tolstykh},
   Title = {MiVOLO: Multi-input Transformer for Age and Gender Estimation},
   Year = {2023},
   Eprint = {arXiv:2307.04616},
}

@article{mivolo2024,
   Author = {Maksim Kuprashevich and Grigorii Alekseenko and Irina Tolstykh},
   Title = {Beyond Specialization: Assessing the Capabilities of MLLMs in Age and Gender Estimation},
   Year = {2024},
   Eprint = {arXiv:2403.02302},
}
```

## 🤝 Contributing

We welcome contributions! This enhanced version builds upon the original authors' excellent foundation. Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/MiVOLO/issues)
- **Original Authors**: [WildChlamydia/MiVOLO](https://github.com/WildChlamydia/MiVOLO)
- **Documentation**: [Advanced Re-ID Guide](REID_MODELS.md)

---

⭐ **If this enhanced version helps your research or project, please give it a star!** ⭐
