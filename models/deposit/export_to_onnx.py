import torch
import torch.onnx
from .deposit import DePOSit

# Configuration for your model (adjust as necessary)
args = {
    'config': {
        'layers': 12,
        'channels': 64,
        'nheads': 8,
        'beta_start': 0.0001,
        'beta_end': 0.5,
        'num_steps': 1,
        'schedule': "cosine",
        'model': {
            'is_unconditional': 0,
            'timeemb': 128,
            'featureemb': 16
        },
        'diffusion': {
            'layers': 12,
            'channels': 64,
            'nheads': 8,
            'diffusion_embedding_dim': 128,
            'beta_start': 0.0001,
            'beta_end': 0.5,
            'num_steps': 1,
            'schedule': "cosine",
            'type': 'ddim'
        },
    },
    'target_dim': 26, # joints * dim
    'device': torch.device('cpu'), # (use CPU for ONNX export to avoid issues)
}

IN_N = 50
OUT_N = 5
INPUT_PATH = "example.pth"
OUTPUT_PATH = "out.onnx"

TARGET_DIM = args['target_dim']
device = args['device']

# Create and load the model
model = DePOSit(type('Args', (object,), args)())
model.load_state_dict(torch.load(INPUT_PATH, map_location=device))

model.eval()

dummy_pose = torch.randn((1, IN_N + OUT_N, TARGET_DIM))
dummy_mask = torch.randn((1, IN_N + OUT_N, TARGET_DIM))
dummy_timepoints = torch.randn((1, IN_N + OUT_N))

torch.onnx.export(
    model,
    (dummy_pose, dummy_mask, dummy_timepoints),
    OUTPUT_PATH,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['pose', 'mask', 'timepoints'],
    output_names=['output'],
)
