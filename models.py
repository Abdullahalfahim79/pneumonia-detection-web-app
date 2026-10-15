"""Loads the 3 trained models (CNN, ViT-B/16, Hybrid) and produces
predictions + Grad-CAM / Score-CAM heatmaps. Mirrors the notebook's preprocessing."""
import base64, io, os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import torch
import torch.nn as nn
from torchvision.models import vit_b_16

tf.config.set_visible_devices([], "GPU")          # keep TF on CPU, torch may use GPU
WEIGHTS = os.environ.get("WEIGHTS_DIR", os.path.join(os.path.dirname(__file__), "weights"))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
LABELS = ["NORMAL", "PNEUMONIA"]                  # 0 = NORMAL, 1 = PNEUMONIA (same as notebook)


class HybridCNNViTClassifier(nn.Module):          # exact layout from the notebook checkpoint
    def __init__(self, input_dim=896):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2))

    def forward(self, x):
        return self.classifier(x)


# ---------- load models ----------
cnn = tf.keras.models.load_model(os.path.join(WEIGHTS, "best_cnn_model.keras"), compile=False)
cnn(np.zeros((1, 224, 224, 3), np.float32))
_layers = cnn.layers
CONV_IDX = max(i for i, l in enumerate(_layers) if isinstance(l, tf.keras.layers.Conv2D))
DENSE_IDX = [i for i, l in enumerate(_layers) if isinstance(l, tf.keras.layers.Dense)][-2]  # Dense(128)

vit = vit_b_16(weights=None)
vit.heads.head = nn.Linear(vit.heads.head.in_features, 2)
vit.load_state_dict(torch.load(os.path.join(WEIGHTS, "best_vit_pneumonia.pth"), map_location=dev))
vit.to(dev).eval()

hyb = HybridCNNViTClassifier()
hyb.load_state_dict(torch.load(os.path.join(WEIGHTS, "best_hybrid_cnn_vit.pth"), map_location=dev))
hyb.to(dev).eval()


# ---------- forward helpers ----------
def cnn_run(x):
    """x: float32 NHWC in [0,1]. Returns (last conv maps, 128-d features, P(pneumonia))."""
    conv = feats = None
    for i, layer in enumerate(_layers):
        x = layer(x)
        if i == CONV_IDX: conv = x
        if i == DENSE_IDX: feats = x
    return conv, feats, x


def to_vit(b01):
    t = torch.tensor(((b01 - MEAN) / STD).transpose(0, 3, 1, 2).copy(), dtype=torch.float32)
    return t.to(dev)


def vit_feats(x):
    t = vit._process_input(x)
    t = torch.cat([vit.class_token.expand(t.shape[0], -1, -1), t], dim=1)
    return vit.encoder(t)[:, 0]


def _chunks(fn, b, n=32):
    return np.concatenate([fn(b[i:i + n]) for i in range(0, len(b), n)])


def p_cnn(b):
    return _chunks(lambda x: cnn(tf.constant(x)).numpy().ravel(), b)


def p_vit(b):
    def f(x):
        with torch.no_grad():
            return torch.softmax(vit.heads(vit_feats(to_vit(x))), 1)[:, 1].cpu().numpy()
    return _chunks(f, b)


def p_hyb(b):
    def f(x):
        feats = cnn_run(tf.constant(x))[1].numpy()
        with torch.no_grad():
            v = vit_feats(to_vit(x)).cpu().numpy()
            z = hyb(torch.tensor(np.concatenate([feats, v], 1), dtype=torch.float32).to(dev))
            return torch.softmax(z, 1)[:, 1].cpu().numpy()
    return _chunks(f, b)


# ---------- Grad-CAM ----------
def _relu_cam(acts, grads):
    w = grads.mean(axis=tuple(range(grads.ndim - 1)))
    return np.maximum((acts * w).sum(-1), 0)


def _cnn_gradcam(x01, cls):
    with tf.GradientTape() as tape:
        conv, _, out = cnn_run(tf.constant(x01[None]))
        score = out[:, 0] if cls == 1 else 1.0 - out[:, 0]
    g = tape.gradient(score, conv)
    return _relu_cam(conv[0].numpy(), g[0].numpy())


def _vit_tokens_and_grad(x01, cls, cnn_feat_grad_fn=None):
    """Runs ViT (and optionally the hybrid head) with a hook on the last block's ln_1."""
    store = {}
    layer = vit.encoder.layers[-1].ln_1

    def hook(_, __, out):
        out.retain_grad(); store["a"] = out
    h = layer.register_forward_hook(hook)
    try:
        vit.zero_grad(); hyb.zero_grad()
        feats = vit_feats(to_vit(x01[None]))
        if cnn_feat_grad_fn is None:
            logits = vit.heads(feats)
        else:
            logits = cnn_feat_grad_fn(feats)
        logits[0, cls].backward()
        a, g = store["a"][0, 1:], store["a"].grad[0, 1:]
        return a.detach().cpu().numpy().reshape(14, 14, -1), g.cpu().numpy().reshape(14, 14, -1)
    finally:
        h.remove()


def _vit_gradcam(x01, cls):
    a, g = _vit_tokens_and_grad(x01, cls)
    return _relu_cam(a, g)


def _hybrid_gradcam(x01, cls):
    with tf.GradientTape() as tape:
        conv, f, _ = cnn_run(tf.constant(x01[None]))
    ft = torch.tensor(f.numpy(), dtype=torch.float32, device=dev, requires_grad=True)
    a, g = _vit_tokens_and_grad(x01, cls, lambda v: hyb(torch.cat([ft, v], 1)))
    g_conv = tape.gradient(f, conv, output_gradients=tf.constant(ft.grad.cpu().numpy()))
    cam_cnn = _relu_cam(conv[0].numpy(), g_conv[0].numpy())
    return _fuse(cam_cnn, _relu_cam(a, g))


# ---------- Score-CAM ----------
def _norm(m):
    m = cv2.resize(m.astype(np.float32), (224, 224))
    r = m.max() - m.min()
    return (m - m.min()) / r if r > 1e-8 else np.zeros_like(m)


def _fuse(a, b):
    return (_norm(a) + _norm(b)) / 2


def _scorecam(acts, x01, score_fn, k=24):
    C = acts.shape[-1]
    idx = np.argsort(acts.reshape(-1, C).var(0))[-k:]
    maps = np.stack([_norm(acts[:, :, c]) for c in idx])
    s = score_fn(x01[None] * maps[..., None]) - score_fn(x01[None])[0]
    w = np.exp(s - s.max()); w /= w.sum()
    return np.maximum((w[:, None, None] * maps).sum(0), 0)


def _cnn_acts(x01):
    return cnn_run(tf.constant(x01[None]))[0][0].numpy()


def _vit_acts(x01):
    a, _ = _vit_tokens_and_grad(x01, 1)
    return a


# ---------- output ----------
def _overlay(base_img, cam):
    cam = _norm(cam)
    cam = cv2.resize(cam, base_img.size)
    heat = cv2.cvtColor(cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    out = (0.55 * np.asarray(base_img) + 0.45 * heat).astype(np.uint8)
    return _png(Image.fromarray(out))


def _png(img):
    buf = io.BytesIO(); img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def analyze(data: bytes, method: str = "gradcam"):
    img = Image.open(io.BytesIO(data)).convert("RGB")
    small = img.resize((224, 224), Image.BILINEAR)
    x01 = np.asarray(small, np.float32) / 255.0
    show = img.resize((448, 448), Image.BILINEAR)
    batch = x01[None]
    probs = {"CNN": float(p_cnn(batch)[0]), "ViT-B/16": float(p_vit(batch)[0]), "Hybrid CNN-ViT": float(p_hyb(batch)[0])}
    results = []
    for name, p in probs.items():
        cls = int(p >= 0.5)
        if method == "gradcam":
            cam = {"CNN": _cnn_gradcam, "ViT-B/16": _vit_gradcam, "Hybrid CNN-ViT": _hybrid_gradcam}[name](x01, cls)
        else:
            sc = (lambda f: (lambda b: f(b) if cls == 1 else 1 - f(b)))
            if name == "CNN":
                cam = _scorecam(_cnn_acts(x01), x01, sc(p_cnn))
            elif name == "ViT-B/16":
                cam = _scorecam(_vit_acts(x01), x01, sc(p_vit))
            else:
                cam = _fuse(_scorecam(_cnn_acts(x01), x01, sc(p_hyb)), _scorecam(_vit_acts(x01), x01, sc(p_hyb)))
        results.append({"name": name, "label": LABELS[cls], "p_pneumonia": p,
                        "confidence": p if cls == 1 else 1 - p, "heatmap": _overlay(show, cam)})
    return {"method": method, "original": _png(show), "models": results,
            "agree": len({r["label"] for r in results}) == 1}
