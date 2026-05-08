# 🧊 Multi-Object 3D Reconstruction from Single Images

> Reconstructing 3D voxel representations of real-world objects from a single RGB image using deep learning.

🚀 **[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/CharukeshPanjala/image-3d-reconstruction)** | 🤗 **[Model Weights](https://huggingface.co/CharukeshPanjala/image-3d-reconstruction)**

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-orange?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-Pix3D-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![HF Spaces](https://img.shields.io/badge/🤗-Live%20Demo-blue?style=flat-square)

---

## 📌 Overview

This project implements an end-to-end deep learning pipeline that takes a **single 2D RGB image** as input and outputs a **3D voxel occupancy grid** representing the object's geometry.

Single-view 3D reconstruction is a fundamentally hard problem — depth information is lost when projecting 3D to 2D, and multiple plausible shapes can correspond to the same image. This project tackles that challenge using convolutional neural networks trained on the [Pix3D](http://pix3d.csail.mit.edu/) dataset.

---

## 🎬 Demo

Visit the live demo and upload any image of a chair, sofa, table, bed, bookcase, desk, wardrobe, or tool to get a predicted 3D reconstruction:

👉 **[huggingface.co/spaces/CharukeshPanjala/image-3d-reconstruction](https://huggingface.co/spaces/CharukeshPanjala/image-3d-reconstruction)**

---

## 🏗️ Architecture Comparison

Four model architectures were designed, trained, and evaluated:

| Model  | Architecture        | Key Idea                                                  |
| ------ | ------------------- | --------------------------------------------------------- |
| **M1** | Baseline CNN        | Image encoder → flat FC → voxel reshape                   |
| **M2** | Classification Head | Voxel prediction as binary classification per cell        |
| **M3** | 3D CNN Decoder      | Explicit 3D spatial decoding with transposed convolutions |
| **M4** | ResNet Encoder ✅   | Transfer learning with pretrained ResNet-18 + 3D decoder  |

**Best Model: M4 (ResNet-based encoder)** — produces the most coherent and structured 3D reconstructions.

---

## 📦 Dataset

**[Pix3D](http://pix3d.csail.mit.edu/)** — a large-scale benchmark for single-view 3D object reconstruction.

- 10,001 real-world RGB images with aligned CAD models
- 8 object categories: `bed`, `bookcase`, `chair`, `desk`, `sofa`, `table`, `wardrobe`, `tool`
- Challenges: background clutter, occlusions, viewpoint variation, lighting changes
- Trained on **full dataset** for **20 epochs**

---

## 🔧 Tech Stack

- **Deep Learning:** PyTorch 2.11, torchvision
- **Transfer Learning:** ResNet-18 pretrained on ImageNet
- **3D Processing:** trimesh, scipy
- **Data & Viz:** NumPy, matplotlib, PIL
- **Training Environment:** Apple Silicon M4 (MPS)
- **Demo:** Gradio
- **Model Hosting:** Hugging Face Hub
- **Deployment:** Hugging Face Spaces

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/CharukeshPanjala/image-3d-reconstruction.git
cd image-3d-reconstruction
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install torch torchvision trimesh scipy matplotlib pillow tqdm huggingface_hub gradio
```

### 4. Download Pix3D dataset

The dataset will be downloaded automatically when you run the notebook, or manually:

```bash
wget http://pix3d.csail.mit.edu/data/pix3d.zip
unzip pix3d.zip -d ./pix3d
```

### 5. Run the notebook

```bash
jupyter notebook Image_3D_Reconstruction.ipynb
```

### 6. Run inference with pretrained model

```python
from huggingface_hub import hf_hub_download
import torch
from torchvision import transforms, models
from PIL import Image

# Load model
class ImageToVoxelCNN_ResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-2])
        self.fc = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(512, 32 * 32 * 32)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x.view(-1, 32, 32, 32)

# Download and load weights
weights_path = hf_hub_download(
    repo_id="CharukeshPanjala/image-3d-reconstruction",
    filename="M4_ResNet_final.pth"
)
model = ImageToVoxelCNN_ResNet()
model.load_state_dict(torch.load(weights_path, map_location="cpu"))
model.eval()

# Run inference
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
image = Image.open("your_image.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    voxel_output = model(input_tensor)

print("Output shape:", voxel_output.shape)  # [1, 32, 32, 32]
```

---

## 📁 Project Structure

```
image-3d-reconstruction/
├── Image_3D_Reconstruction.ipynb   # Full training notebook
├── app.py                          # Gradio demo app (HF Spaces)
├── requirements.txt                # Dependencies
└── README.md
```

---

## 📊 Training Details

| Parameter        | Value                       |
| ---------------- | --------------------------- |
| Dataset          | Pix3D (full, 10,001 images) |
| Epochs           | 20                          |
| Batch size       | 10                          |
| Optimizer        | Adam (lr=1e-4)              |
| Loss             | MSE                         |
| Device           | Apple M4 MPS                |
| Training time    | ~15 hours                   |
| Final model size | 107MB                       |

---

## ⚠️ Limitations

- **Voxel resolution is coarse (32³)** — fine geometric details like chair legs or thin surfaces are hard to recover
- **MSE loss produces smooth predictions** — the model tends to predict average occupancy values rather than sharp 0/1 boundaries, resulting in dense cube-like outputs at low thresholds
- **Single viewpoint** — depth ambiguity means multiple 3D shapes can correspond to the same 2D image
- **Category-specific** — model works best on the 8 Pix3D categories it was trained on

---

## 🔮 Future Work

- Replace MSE loss with **Binary Cross-Entropy (BCE)** for sharper voxel occupancy predictions
- Increase voxel resolution to **64³ or 128³** for finer detail
- Explore **implicit representations** (NeRF, signed distance functions) for smoother surfaces
- Use **point cloud or mesh decoders** (PointNet) instead of voxels
- Incorporate **multi-view supervision** when available

---

## 👨‍💻 Authors

- **Charukesh Panjala** — [LinkedIn](https://linkedin.com/in/charukeshpanjala) | [GitHub](https://github.com/CharukeshPanjala)

_MSc Software Engineering — University of Europe for Applied Sciences, Potsdam_

---

## 📄 License

MIT License — feel free to use and build on this work.
