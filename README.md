<div align="center">
<p>
   <a align="center" target="_blank">
   <img width="900" src="./images/MiVOLO.jpg"></a>
</p>
<br>
</div>



## MiVOLO: Multi-input Transformer for Age and Gender Estimation

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mivolo-multi-input-transformer-for-age-and/age-estimation-on-utkface)](https://paperswithcode.com/sota/age-estimation-on-utkface?p=mivolo-multi-input-transformer-for-age-and) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/beyond-specialization-assessing-the-1/age-estimation-on-imdb-clean)](https://paperswithcode.com/sota/age-estimation-on-imdb-clean?p=beyond-specialization-assessing-the-1) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/beyond-specialization-assessing-the-1/facial-attribute-classification-on-fairface)](https://paperswithcode.com/sota/facial-attribute-classification-on-fairface?p=beyond-specialization-assessing-the-1) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/beyond-specialization-assessing-the-1/age-and-gender-classification-on-adience)](https://paperswithcode.com/sota/age-and-gender-classification-on-adience?p=beyond-specialization-assessing-the-1) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/beyond-specialization-assessing-the-1/age-and-gender-classification-on-adience-age)](https://paperswithcode.com/sota/age-and-gender-classification-on-adience-age?p=beyond-specialization-assessing-the-1) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/beyond-specialization-assessing-the-1/age-and-gender-estimation-on-lagenda-age)](https://paperswithcode.com/sota/age-and-gender-estimation-on-lagenda-age?p=beyond-specialization-assessing-the-1) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/beyond-specialization-assessing-the-1/gender-prediction-on-lagenda)](https://paperswithcode.com/sota/gender-prediction-on-lagenda?p=beyond-specialization-assessing-the-1) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mivolo-multi-input-transformer-for-age-and/age-estimation-on-agedb)](https://paperswithcode.com/sota/age-estimation-on-agedb?p=mivolo-multi-input-transformer-for-age-and) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mivolo-multi-input-transformer-for-age-and/gender-prediction-on-agedb)](https://paperswithcode.com/sota/gender-prediction-on-agedb?p=mivolo-multi-input-transformer-for-age-and) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/beyond-specialization-assessing-the-1/age-estimation-on-cacd)](https://paperswithcode.com/sota/age-estimation-on-cacd?p=beyond-specialization-assessing-the-1)

> [**MiVOLO: Multi-input Transformer for Age and Gender Estimation**](https://arxiv.org/abs/2307.04616),
> Maksim Kuprashevich, Irina Tolstykh,
> *2023 [arXiv 2307.04616](https://arxiv.org/abs/2307.04616)*

> [**Beyond Specialization: Assessing the Capabilities of MLLMs in Age and Gender Estimation**](https://arxiv.org/abs/2403.02302),
> Maksim Kuprashevich, Grigorii Alekseenko, Irina Tolstykh
> *2024 [arXiv 2403.02302](https://arxiv.org/abs/2403.02302)*

[[`Paper 2023`](https://arxiv.org/abs/2307.04616)] [[`Paper 2024`](https://arxiv.org/abs/2403.02302)] [[`Demo`](https://huggingface.co/spaces/iitolstykh/age_gender_estimation_demo)] [[`Telegram Bot`](https://t.me/AnyAgeBot)] [[`BibTex`](#citing)] [[`Data`](https://wildchlamydia.github.io/lagenda/)]


## MiVOLO pretrained models

Gender & Age recognition performance.

<table style="margin: auto">
  <tr>
    <th align="left">Model</th>
    <th align="left" style="color:LightBlue">Type</th>
    <th align="left">Dataset (train and test)</th>
    <th align="left">Age MAE</th>
    <th align="left">Age CS@5</th>
    <th align="left">Gender Accuracy</th>
    <th align="left">download</th>
  </tr>
  <tr>
    <td>volo_d1</td>
    <td align="left">face_only, age</td>
    <td align="left">IMDB-cleaned</td>
    <td align="left">4.29</td>
    <td align="left">67.71</td>
    <td align="left">-</td>
    <td><a href="https://drive.google.com/file/d/17ysOqgG3FUyEuxrV3Uh49EpmuOiGDxrq/view?usp=drive_link">checkpoint</a></td>
  </tr>
    <tr>
    <td>volo_d1</td>
    <td align="left">face_only, age, gender</td>
    <td align="left">IMDB-cleaned</td>
    <td align="left">4.22</td>
    <td align="left">68.68</td>
    <td align="left">99.38</td>
    <td><a href="https://drive.google.com/file/d/1NlsNEVijX2tjMe8LBb1rI56WB_ADVHeP/view?usp=drive_link">checkpoint</a></td>
  </tr>
    <tr>
    <td>mivolo_d1</td>
    <td align="left">face_body, age, gender</td>
    <td align="left">IMDB-cleaned</td>
    <td align="left">4.24 [face+body]<br>6.87 [body]</td>
    <td align="left">68.32 [face+body]<br>46.32 [body]</td>
    <td align="left">99.46 [face+body]<br>96.48 [body]</td>
    <td><a href="https://drive.google.com/file/d/11i8pKctxz3wVkDBlWKvhYIh7kpVFXSZ4/view?usp=drive_link">model_imdb_cross_person_4.24_99.46.pth.tar</a></td>
  </tr>
  <tr>
    <td>volo_d1</td>
    <td align="left">face_only, age</td>
    <td align="left">UTKFace</td>
    <td align="left">4.23</td>
    <td align="left">69.72</td>
    <td align="left">-</td>
    <td><a href="https://drive.google.com/file/d/1LtDfAJrWrw-QA9U5IuC3_JImbvAQhrJE/view?usp=drive_link">checkpoint</a></td>
  </tr>
    <tr>
    <td>volo_d1</td>
    <td align="left">face_only, age, gender</td>
    <td align="left">UTKFace</td>
    <td align="left">4.23</td>
    <td align="left">69.78</td>
    <td align="left">97.69</td>
    <td><a href="https://drive.google.com/file/d/1hKFmIR6fjHMevm-a9uPEAkDLrTAh-W4D/view?usp=drive_link">checkpoint</a></td>
  </tr>
  <tr>
    <td>mivolo_d1</td>
    <td align="left">face_body, age, gender</td>
    <td align="left">Lagenda</td>
    <td align="left">3.99 [face+body]</td>
    <td align="left">71.27 [face+body]</td>
    <td align="left">97.36 [face+body]</td>
    <td><a href="https://huggingface.co/spaces/iitolstykh/demo">demo</a></td>
  </tr>
  <tr>
    <td>mivolov2_d1_384x384</td>
    <td align="left">face_body, age, gender</td>
    <td align="left">Lagenda</td>
    <td align="left">3.65 [face+body]</td>
    <td align="left">74.48 [face+body]</td>
    <td align="left">97.99 [face+body]</td>
    <td><a href="https://t.me/AnyAgeBot">telegram bot</a></td>
  </tr>

</table>

## MiVOLO regression benchmarks

Gender & Age recognition performance.

Use [valid_age_gender.sh](scripts/valid_age_gender.sh) to reproduce results with our checkpoints.

<table style="margin: auto">
  <tr>
    <th align="left">Model</th>
    <th align="left" style="color:LightBlue">Type</th>
    <th align="left">Train Dataset</th>
    <th align="left">Test Dataset</th>
    <th align="left">Age MAE</th>
    <th align="left">Age CS@5</th>
    <th align="left">Gender Accuracy</th>
    <th align="left">download</th>
  </tr>

  <tr>
    <td>mivolo_d1</td>
    <td align="left">face_body, age, gender</td>
    <td align="left">Lagenda</td>
    <td align="left">AgeDB</td>
    <td align="left">5.55 [face]</td>
    <td align="left">55.08 [face]</td>
    <td align="left">98.3 [face]</td>
    <td><a href="https://huggingface.co/spaces/iitolstykh/demo">demo</a></td>
  </tr>
  <tr>
    <td>mivolo_d1</td>
    <td align="left">face_body, age, gender</td>
    <td align="left">IMDB-cleaned</td>
    <td align="left">AgeDB</td>
    <td align="left">5.58 [face]</td>
    <td align="left">55.54 [face]</td>
    <td align="left">97.93 [face]</td>
    <td><a href="https://drive.google.com/file/d/11i8pKctxz3wVkDBlWKvhYIh7kpVFXSZ4/view?usp=drive_link">model_imdb_cross_person_4.24_99.46.pth.tar</a></td>
  </tr>

</table>

## MiVOLO classification benchmarks

Gender & Age recognition performance.

<table style="margin: auto">
  <tr>
    <th align="left">Model</th>
    <th align="left" style="color:LightBlue">Type</th>
    <th align="left">Train Dataset</th>
    <th align="left">Test Dataset</th>
    <th align="left">Age Accuracy</th>
    <th align="left">Gender Accuracy</th>
  </tr>

  <tr>
    <td>mivolo_d1</td>
    <td align="left">face_body, age, gender</td>
    <td align="left">Lagenda</td>
    <td align="left">FairFace</td>
    <td align="left">61.07 [face+body]</td>
    <td align="left">95.73 [face+body]</td>
  </tr>
  <tr>
    <td>mivolo_d1</td>
    <td align="left">face_body, age, gender</td>
    <td align="left">Lagenda</td>
    <td align="left">Adience</td>
    <td align="left">68.69 [face]</td>
    <td align="left">96.51[face]</td>
  </tr>
  <tr>
    <td>mivolov2_d1_384</td>
    <td align="left">face_body, age, gender</td>
    <td align="left">Lagenda</td>
    <td align="left">Adience</td>
    <td align="left">69.43 [face]</td>
    <td align="left">97.39[face]</td>
  </tr>

</table>

## Dataset

**Please, [cite our papers](#citing) if you use any this data!**

- Lagenda dataset: [images](https://drive.google.com/file/d/1QXO0NlkABPZT6x1_0Uc2i6KAtdcrpTbG/view?usp=sharing) and [annotation](https://drive.google.com/file/d/1mNYjYFb3MuKg-OL1UISoYsKObMUllbJx/view?usp=sharing).
- IMDB-clean: follow [these instructions](https://github.com/yiminglin-ai/imdb-clean) to get images and [download](https://drive.google.com/file/d/17uEqyU3uQ5trWZ5vRJKzh41yeuDe5hyL/view?usp=sharing) our annotations.
- UTK dataset: [origin full images](https://susanqq.github.io/UTKFace/) and our annotation: [split from the article](https://drive.google.com/file/d/1Fo1vPWrKtC5bPtnnVWNTdD4ZTKRXL9kv/view?usp=sharing), [random full split](https://drive.google.com/file/d/177AV631C3SIfi5nrmZA8CEihIt29cznJ/view?usp=sharing).
- Adience dataset: follow [these instructions](https://talhassner.github.io/home/projects/Adience/Adience-data.html) to get images and [download](https://drive.google.com/file/d/1wS1Q4FpksxnCR88A1tGLsLIr91xHwcVv/view?usp=sharing) our annotations.
   <details>
      <summary>Click to expand!</summary>

   After downloading them, your `data` directory should look something like this:

   ```console
   data
   └── Adience
       ├── annotations  (folder with our annotations)
       ├── aligned      (will not be used)
       ├── faces
       ├── fold_0_data.txt
       ├── fold_1_data.txt
       ├── fold_2_data.txt
       ├── fold_3_data.txt
       └── fold_4_data.txt
   ```

   We use coarse aligned images from `faces/` dir.

   Using our detector we found a face bbox for each image (see [tools/prepare_adience.py](tools/prepare_adience.py)).

   This dataset has five folds. The performance metric is accuracy on five-fold cross validation.

   | images before removal | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
   | --------------------- | ------ | ------ | ------ | ------ | ------ |
   | 19,370                | 4,484  | 3,730  | 3,894  | 3,446  | 3,816  |

   Not complete data

   | only age not found | only gender not found | SUM           |
   | ------------------ | --------------------- | ------------- |
   | 40                 | 1170                  | 1,210 (6.2 %) |

   Removed data

   | failed to process image | age and gender not found | SUM         |
   | ----------------------- | ------------------------ | ----------- |
   | 0                       | 708                      | 708 (3.6 %) |

   Genders

   | female | male  |
   | ------ | ----- |
   | 9,372  | 8,120 |

   Ages (8 classes) after mapping to not intersected ages intervals

   | 0-2   | 4-6   | 8-12  | 15-20 | 25-32 | 38-43 | 48-53 | 60-100 |
   | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ------ |
   | 2,509 | 2,140 | 2,293 | 1,791 | 5,589 | 2,490 | 909   | 901    |

   </details>

- FairFace dataset: follow [these instructions](https://github.com/joojs/fairface) to get images and [download](https://drive.google.com/file/d/1EdY30A1SQmox96Y39VhBxdgALYhbkzdm/view?usp=drive_link) our annotations.
    <details>
      <summary>Click to expand!</summary>

    After downloading them, your `data` directory should look something like this:

    ```console
    data
    └── FairFace
       ├── annotations  (folder with our annotations)
       ├── fairface-img-margin025-trainval   (will not be used)
           ├── train
           ├── val
       ├── fairface-img-margin125-trainval
           ├── train
           ├── val
       ├── fairface_label_train.csv
       ├── fairface_label_val.csv

    ```

    We use aligned images from `fairface-img-margin125-trainval/` dir.

    Using our detector we found a face bbox for each image and added a person bbox if it was possible (see [tools/prepare_fairface.py](tools/prepare_fairface.py)).

    This dataset has 2 splits: train and val. The performance metric is accuracy on validation.

    | images train | images val |
    | ------------ | ---------- |
    | 86,744       | 10,954     |

    Genders for **validation**

    | female | male  |
    | ------ | ----- |
    | 5,162  | 5,792 |

    Ages for **validation** (9 classes):

    | 0-2 | 3-9   | 10-19 | 20-29 | 30-39 | 40-49 | 50-59 | 60-69 | 70+ |
    | --- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | --- |
    | 199 | 1,356 | 1,181 | 3,300 | 2,330 | 1,353 | 796   | 321   | 118 |

    </details>
- AgeDB dataset: follow [these instructions](https://ibug.doc.ic.ac.uk/resources/agedb/) to get images and [download](https://drive.google.com/file/d/1Dp72BUlAsyUKeSoyE_DOsFRS1x6ZBJen/view) our annotations.
    <details>
      <summary>Click to expand!</summary>

  **Ages**: 1 - 101

  **Genders**: 9788 faces of `M`, 6700 faces of `F`

  | images 0 | images 1 | images 2 | images 3 | images 4 | images 5 | images 6 | images 7 | images 8 | images 9 |
  |----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
  | 1701     | 1721     | 1615     | 1619     | 1626     | 1643     | 1634     | 1596     | 1676     | 1657     |

    Data splits were taken from [here](https://github.com/paplhjak/Facial-Age-Estimation-Benchmark-Databases)

    !! **All splits(all dataset) were used for models evaluation.**
    </details>

## Install

Install pytorch 1.13+ and other requirements.

```
pip install -r requirements.txt
pip install .
```


## Demo

1. [Download](https://drive.google.com/file/d/1CGNCkZQNj5WkP3rLpENWAOgrBQkUWRdw/view) body + face detector model to `models/yolov8x_person_face.pt`
2. [Download](https://drive.google.com/file/d/11i8pKctxz3wVkDBlWKvhYIh7kpVFXSZ4/view) mivolo checkpoint to `models/mivolo_imbd.pth.tar`

```bash
wget https://variety.com/wp-content/uploads/2023/04/MCDNOHA_SP001.jpg -O jennifer_lawrence.jpg

python3 demo.py \
--input "jennifer_lawrence.jpg" \
--output "output" \
--detector-weights "models/yolov8x_person_face.pt " \
--checkpoint "models/mivolo_imbd.pth.tar" \
--device "cuda:0" \
--with-persons \
--draw
```

To run demo for a youtube video:
```bash
python3 demo.py \
--input "https://www.youtube.com/shorts/pVh32k0hGEI" \
--output "output" \
--detector-weights "models/yolov8x_person_face.pt" \
--checkpoint "models/mivolo_imbd.pth.tar" \
--device "cuda:0" \
--draw \
--with-persons
```

## Advanced Features

### Skip Frames for Faster Processing

You can skip frames to speed up processing:

```bash
python3 demo.py \
--input "video.mp4" \
--output "output" \
--detector-weights "models/yolov8x_person_face.pt" \
--checkpoint "models/mivolo_imbd.pth.tar" \
--device "cuda:0" \
--draw \
--with-persons \
--skip-frames 4  # Process every 5th frame (skip 4 frames)
```

### BOT-SORT Tracking and Temporal Smoothing

For better tracking and more stable predictions across frames:

```bash
python3 demo.py \
--input "video.mp4" \
--output "output" \
--detector-weights "models/yolov8x_person_face.pt" \
--checkpoint "models/mivolo_imbd.pth.tar" \
--device "cuda:0" \
--draw \
--with-persons \
--tracker "botsort.yaml" \
--temporal-smoothing \
--smoothing-window 10  # Use 10 frames for temporal smoothing
```

### Global Temporal Smoothing

For single final prediction per track across the entire video:

```bash
python3 demo.py \
--input "video.mp4" \
--output "output" \
--detector-weights "models/yolov8x_person_face.pt" \
--checkpoint "models/mivolo_imbd.pth.tar" \
--device "cuda:0" \
--draw \
--with-persons \
--tracker "botsort.yaml" \
--temporal-smoothing \
--global-smoothing  # Aggregate across ALL frames
```

**Tracker Options:**
- `bytetrack.yaml` (default): Fast and efficient
- `botsort.yaml`: Better re-identification capabilities with advanced ReID models

**Advanced Re-ID Models (NEW!):**
MiVOLO now supports state-of-the-art Re-ID backbone models for enhanced tracking accuracy:

- **OSNet Models** (Recommended): Best balance of speed and accuracy
- **ResNeSt Models**: Maximum accuracy for challenging scenarios  
- **CLIP-based Models**: State-of-the-art cross-modal understanding
- **ConvNeXt Models**: Modern CNN architecture with excellent performance
- **MobileNet Models**: Optimized for edge deployment

**Setup Advanced Re-ID Models:**
```bash
# List available models
python setup_reid_models.py --list

# Quick setup with recommended model
python setup_reid_models.py --download osnet_x1_0_msmt17.pt --use osnet_x1_0_msmt17.pt

# Or run the interactive demo
python demo_advanced_reid.py --quick-demo
```

**Re-ID Model Benefits:**
- 🎯 **Better Identity Consistency**: Reduced ID switches during occlusions
- 🔄 **Cross-Camera Tracking**: Enhanced matching across different viewpoints
- 📈 **Improved Accuracy**: More discriminative features for person matching
- ⚡ **Multiple Speed Options**: From ultra-fast mobile models to high-accuracy research models

**Temporal Smoothing Benefits:**
- Reduces flickering between age/gender predictions
- More stable tracking results
- Better handling of temporary occlusions

**Important Note:** When temporal smoothing is enabled, the system uses its own smoothing algorithm instead of the default tracking-based aggregation to prevent conflicts that could cause continued flickering.

### **Confidence-Based Selection (Recommended)**

A superior alternative to smoothing that keeps only the highest-confidence prediction per track:

```bash
python demo.py \
  --input video.mp4 \
  --output output/ \
  --detector-weights yolov8x_person_face.pt \
  --checkpoint models/mivolo_imbd.pth.tar \
  --with-persons \
  --draw \
  --confidence-based \
  --min-detection-conf 0.5 \
  --min-gender-conf 0.6 \
  --tracker botsort.yaml
```

### **JSON Export with Demographics (NEW!)**

Export detailed tracking results with demographics and cropped images in JSON format:

```bash
python demo.py \
  --input video.mp4 \
  --output output/ \
  --detector-weights yolov8x_person_face.pt \
  --checkpoint models/mivolo_imbd.pth.tar \
  --with-persons \
  --draw \
  --tracker botsort.yaml \
  --save-json \
  --json-output-path output/demographics.json \
  --save-crops \
  --crops-dir crops
```

**JSON Export Features:**
- 📊 **Frame-by-frame demographics**: Age group and gender counts per frame
- 📍 **Person locations**: Bounding box center coordinates for each detection
- 🖼️ **Cropped images**: Automatically saves person crops with unique filenames
- ⏱️ **Timestamp tracking**: Precise timing information for each frame
- 🆔 **ID consistency**: Maintains track IDs across frames for trajectory analysis

**Sample JSON Output:**
```json
[
  {
    "time": "0.5 sec",
    "total_people": 3,
    "man": 2,
    "woman": 1,
    "age_teen": 0,
    "age_twenty": 1,
    "age_thirty": 2,
    "age_fourty": 0,
    "age_fifty": 0,
    "age_sisxty": 0,
    "age_seventy": 0,
    "people_data": [
      {
        "id": 1,
        "gender": "man",
        "age": 32,
        "cropped_img": "/crops/1_15.jpg",
        "location": [425, 280]
      },
      {
        "id": 2,
        "gender": "woman",
        "age": 28,
        "cropped_img": "/crops/2_15.jpg",
        "location": [650, 320]
      }
    ]
  }
]
```

**Confidence-Based Selection Benefits:**
- Uses best quality predictions (no averaging of noisy data)
- Immune to outliers (unlike smoothing which can be affected)
- More stable than smoothing for consistent lighting/angles
- Provides detailed confidence metrics per track
- Shows which frame had the best prediction

**⚠️ Important:** Confidence-based selection takes priority over global smoothing. If both `--confidence-based` and `--global-smoothing` flags are specified, the system will use confidence-based selection and ignore global smoothing.

**Confidence Scoring:**
- **Detection confidence**: YOLO's confidence in detecting the person/face
- **Gender confidence**: Model's confidence in gender prediction  
- **Composite confidence**: Weighted combination (70% detection + 30% gender)
- **Thresholds**: Only predictions above minimum confidence are considered

**Global Smoothing Features:**
- Progressive smoothing: Video shows running averages that converge to final predictions
- Single final age/gender prediction per track ID
- Aggregates all predictions across the entire video
- Saves results to `global_smoothing_results.json`
- Includes statistics (mean, std, min, max, confidence) for each track

**Analyzing Results:**

For global smoothing:
```bash
# After running with --global-smoothing, analyze the results
python analyze_global_results.py global_smoothing_results.json
```

For confidence-based selection:
```bash
# Results are automatically saved to confidence_based_results.json
# Analyze the confidence-based results
python analyze_confidence_results.py confidence_based_results.json
```

## Validation

To reproduce validation metrics:

1. Download prepared annotations for imbd-clean / utk / adience / lagenda  / fairface.
2. Download checkpoint
3. Run validation:

```bash
python3 eval_pretrained.py \
  --dataset_images /path/to/dataset/utk/images \
  --dataset_annotations /path/to/dataset/utk/annotation \
  --dataset_name utk \
  --split valid \
  --batch-size 512 \
  --checkpoint models/mivolo_imbd.pth.tar \
  --half \
  --with-persons \
  --device "cuda:0"
````

Supported dataset names: "utk", "imdb", "lagenda", "fairface", "adience".


## Changelog

[CHANGELOG.md](CHANGELOG.md)

## ONNX and TensorRT export

As of now (11.08.2023), while ONNX export is technically feasible, it is not advisable due to the poor performance of the resulting model with batch processing.
**TensorRT** and **OpenVINO** export is impossible due to its lack of support for col2im.

If you remain absolutely committed to utilizing ONNX export, you can refer to [these instructions](https://github.com/WildChlamydia/MiVOLO/issues/14#issuecomment-1675245889).

The most highly recommended export method at present **is using TorchScript**. You can achieve this with a single line of code:
```python
torch.jit.trace(model)
```
This approach provides you with a model that maintains its original speed and only requires a single file for usage, eliminating the need for additional code.

## License

Please, see [here](./license)


## Citing

If you use our models, code or dataset, we kindly request you to cite the following paper and give repository a :star:

```bibtex
@article{mivolo2023,
   Author = {Maksim Kuprashevich and Irina Tolstykh},
   Title = {MiVOLO: Multi-input Transformer for Age and Gender Estimation},
   Year = {2023},
   Eprint = {arXiv:2307.04616},
}
```
```bibtex
@article{mivolo2024,
   Author = {Maksim Kuprashevich and Grigorii Alekseenko and Irina Tolstykh},
   Title = {Beyond Specialization: Assessing the Capabilities of MLLMs in Age and Gender Estimation},
   Year = {2024},
   Eprint = {arXiv:2403.02302},
}
```
