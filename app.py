import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms, models
from PIL import Image
from huggingface_hub import hf_hub_download
import io

class M4_ResNet(nn.Module):
    def __init__(self, voxel_size=32):
        super().__init__()
        resnet = models.resnet18(pretrained=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.decoder = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, voxel_size ** 3),
            nn.Sigmoid(),
        )
        self.voxel_size = voxel_size

    def forward(self, x):
        features = self.encoder(x).view(x.size(0), -1)
        voxels = self.decoder(features)
        return voxels.view(-1, 1, self.voxel_size, self.voxel_size, self.voxel_size)

REPO_ID = "CharukeshPanjala/image-3d-reconstruction"
device = torch.device("cpu")
model = M4_ResNet()

try:
    weights_path = hf_hub_download(repo_id=REPO_ID, filename="M4_ResNet_final.pth")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print("✅ Model loaded!")
except Exception as e:
    print(f"⚠️ Could not load weights: {e}")

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict_3d(image, threshold=0.2):
    if image is None:
        return None, "Please upload an image."
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_voxel = model(input_tensor).squeeze().cpu().numpy()
    coords = np.argwhere(pred_voxel > threshold)
    fig = plt.figure(figsize=(6, 6), facecolor="#0f0f0f")
    ax = fig.add_subplot(111, projection="3d", facecolor="#0f0f0f")
    if len(coords) > 0:
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                   c=coords[:, 2], cmap="cool", s=8, alpha=0.7)
        stats = f"Occupied voxels: {len(coords)} / {pred_voxel.size} ({100*len(coords)/pred_voxel.size:.1f}%)"
    else:
        stats = "No voxels above threshold — try lowering the slider."
    ax.set_title("Predicted 3D Reconstruction", color="white", fontsize=13)
    ax.tick_params(colors="white")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0f0f0f")
    buf.seek(0)
    plt.close()
    return Image.open(buf), stats

with gr.Blocks(title="3D Reconstruction") as demo:
    gr.Markdown("# 🧊 3D Reconstruction from a Single Image")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Input Image")
            threshold = gr.Slider(0.05, 0.5, value=0.2, step=0.05, label="Voxel Threshold")
            run_btn = gr.Button("🔮 Reconstruct 3D", variant="primary")
        with gr.Column():
            voxel_output = gr.Image(label="3D Voxel Output")
            stats_output = gr.Textbox(label="Stats", interactive=False)
    run_btn.click(fn=predict_3d, inputs=[image_input, threshold], outputs=[voxel_output, stats_output])

if __name__ == "__main__":
    demo.launch()
