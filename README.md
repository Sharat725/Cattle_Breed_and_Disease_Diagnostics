# 🐄 Cattle Breed & Disease Recognition System

*AI-powered livestock diagnostics using Hybrid CNN-ViT deep learning and modern web technologies.*

---

## ⚡ Quick Start

### 1. Installation (2 minutes)
```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/Cattle-Disease-Diagnostics.git
cd Cattle-Disease-Diagnostics

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Application
```bash
# Windows
run.bat

# Or manually
python app.py
```

### 3. Access Web Interface
Open browser → `http://localhost:8000`

---

## ✨ Key Features

- **🎯 15 Cattle Breeds**: Holstein, Jersey, Gir, Sahiwal, and 11 others
- **🏥 3 Disease Categories**: Healthy, Foot-and-Mouth, Lumpy Skin
- **🚀 Real-time Inference**: <1 second on GPU with TTA
- **🧠 Hybrid CNN-ViT**: EfficientNet-B3 + Vision Transformer
- **📱 Modern Web UI**: Responsive, drag-and-drop interface
- **📊 Confidence Metrics**: Probability distributions for all predictions

---

## 🏗️ Technology Stack

### Backend
| Technology | Purpose |
|---|---|
| **PyTorch 2.0+** | Deep learning inference |
| **TorchVision** | Pre-trained models (EfficientNet-B3, ViT-B/16) |
| **FastAPI** | RESTful API with auto-documentation |
| **Uvicorn** | Production ASGI server |
| **Pillow** | Image processing |

### Frontend
| Technology | Purpose |
|---|---|
| **HTML5** | Semantic markup |
| **CSS3** | Glassmorphism animations |
| **JavaScript (ES6+)** | Async API calls, DOM manipulation |

### ML Pipeline
| Component | Details |
|---|---|
| **CNN** | EfficientNet-B3 (1536 features) |
| **Transformer** | ViT-B/16 (768 features) |
| **Fusion** | Concatenation + FC layers → 15 classes |
| **Disease Model** | ConvNeXt-Tiny (27.8M params) |
| **Inference** | TTA (5-crop ensemble) |
| **Augmentation** | Medical-grade (preserves disease features) |

---

## 📊 Model Architecture

### Breed Classification (Hybrid CNN-ViT)
```
Image → EfficientNet-B3 (1536) ──┐
        ↓                         ├→ Concatenate (2304) → FC layers → 15 Breeds
Image → ViT-B/16 (768) ──────────┘
```

### Disease Detection (ConvNeXt-Tiny + TTA)
```
Image → [5 Different Crops] → ConvNeXt-Tiny → Average Predictions → 3 Diseases
```

**Why This Approach?**
- CNN extracts local features (textures, patterns)
- ViT captures global context (structure, relationships)
- TTA (5-crop) ensemble for medical-grade reliability

---

## 📁 Project Structure

```
.
├── app.py                          # FastAPI backend (model inference)
├── cattle_classification.ipynb      # Breed model training
├── disease.ipynb                   # Disease model training
├── debug_load.py                   # Model debugging
├── requirements.txt                # Dependencies
├── run.bat                         # Windows startup
├── frontend/
│   ├── index.html                 # Web interface
│   ├── app.js                     # Frontend logic
│   └── style.css                  # Styling
└── README.md                       # Quick start (this file)
```

---

## 🚀 API Usage

### Health Check
```bash
curl http://localhost:8000/api/health
# Response: {"status": "ok", "model": "ready", "device": "cuda"}
```

### Predict
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -F "file=@cattle_image.jpg"
```

**Response**:
```json
{
  "breed": "Holstein_Friesian",
  "breed_probability": 0.9847,
  "disease": "healthy",
  "disease_probability": 0.9612,
  "all_breed_predictions": {...},
  "all_disease_predictions": {...}
}
```

---

## 🐄 Supported Breeds (15)

**Indian Dairy**: Gir, Red Sindhi, Sahiwal  
**Indian Draft**: Hallikar, Kankrej, Rathi  
**Buffaloes**: Murrah, Nagpuri  
**International**: Ayrshire, Brown Swiss, Holstein, Jersey, Red Dane, Tharparkar, Ongole

---

## 🏥 Disease Categories (3)

| Disease | Status | Description |
|---------|--------|-------------|
| Healthy | ✅ | Normal cattle |
| Foot-and-Mouth | ⚠️ | Viral, blisters |
| Lumpy Skin | ⚠️ | Viral, nodules |

---

## ⚙️ Training Details

### Dataset
- **Split**: 70% train, 15% validation, 15% test
- **Augmentation**: Medical-grade (preserves disease features)
- **Strategies**: MixUp, Random erasing, Rotation (±10°)

### Optimization
- **Optimizer**: AdamW with discriminative learning rates
- **Loss**: Focal Loss (γ=2.0) for class imbalance
- **Scheduler**: Cosine Annealing
- **Mixed Precision**: AMP for faster training

### Test Time Augmentation (TTA)
- **5-Crop Ensemble**: Center, flips, multiple sizes
- **Averaging**: Mean softmax across crops
- **Benefit**: Medical-grade reliability

### Expected Performance
- **Accuracy**: 95%+ (varies with dataset)
- **Per-class Recall**: ≥90% (critical for diseases)
- **Inference Time**: 50-100ms (GPU), 2-5 sec (CPU)

---

## 🔧 System Requirements

| Aspect | Minimum | Recommended |
|--------|---------|-------------|
| **RAM** | 8GB | 16GB |
| **GPU VRAM** | 2GB | 8GB+ |
| **Disk** | 2GB | 5GB |
| **Python** | 3.8 | 3.10+ |
| **OS** | Windows/Mac/Linux | Any |

---

## ❓ Troubleshooting

### Port 8000 Already in Use
```bash
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### CUDA Out of Memory
```bash
# Use CPU: device = torch.device("cpu")
# Or reduce batch size
```

### Module Not Found
```bash
venv\Scripts\activate
pip install -r requirements.txt --upgrade
```

### Model Loading Fails
```bash
python debug_load.py
```

---

## 📚 Documentation

For **comprehensive documentation** including:
- Detailed architecture diagrams
- Full API reference
- Training pipeline details
- Contributing guidelines
- Roadmap and future plans

👉 See [README_COMPREHENSIVE.md](README_COMPREHENSIVE.md)

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Single Image (GPU)** | ~100ms |
| **With TTA (GPU)** | ~1.5 sec |
| **Model Size** | ~240MB |
| **Accuracy** | 95%+ |
| **Parameters** | ~75M (both models) |

---

## 🔗 API Documentation

Interactive API docs available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork repository
2. Create feature branch
3. Submit pull request

Areas for improvement:
- Add more breeds/diseases
- Improve model accuracy
- Mobile app integration
- Docker deployment

---

## 📜 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **PyTorch & TorchVision**: Deep learning frameworks
- **FastAPI**: Modern web framework
- **ImageNet Community**: Pre-trained models
- **Open Source ML**: Tools & inspiration

---

## 📞 Support

- **Issues**: Open on GitHub
- **Discussions**: Start GitHub discussion
- **API Docs**: `/docs` endpoint

---

**Status**: Active Development  
**Last Updated**: 2026-07-03  
**Python**: 3.8+  
**License**: MIT
