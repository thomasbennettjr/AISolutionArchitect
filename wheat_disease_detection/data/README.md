# Wheat Disease Detection - Example Dataset Structure

This folder should contain your wheat disease images organized by class.

## Directory Structure

```
data/
└── wheat_diseases/
    ├── train/
    │   ├── Healthy/
    │   │   ├── healthy_001.jpg
    │   │   ├── healthy_002.jpg
    │   │   └── ...
    │   ├── Leaf_Rust/
    │   │   ├── leaf_rust_001.jpg
    │   │   ├── leaf_rust_002.jpg
    │   │   └── ...
    │   ├── Stem_Rust/
    │   │   ├── stem_rust_001.jpg
    │   │   └── ...
    │   ├── Yellow_Rust/
    │   │   ├── yellow_rust_001.jpg
    │   │   └── ...
    │   └── Septoria/
    │       ├── septoria_001.jpg
    │       └── ...
    └── test/ (optional)
        ├── Healthy/
        ├── Leaf_Rust/
        ├── Stem_Rust/
        ├── Yellow_Rust/
        └── Septoria/
```

## Dataset Requirements

- **Image Format**: JPG, PNG, or BMP
- **Minimum Images**: 100+ per class (500+ recommended)
- **Image Quality**: Clear, well-lit aerial photographs
- **Image Size**: Any size (will be resized to 224x224)
- **Balance**: Try to have similar numbers of images per class

## Where to Get Data

1. **Public Datasets**:
   - PlantVillage Dataset (Kaggle)
   - CGIAR Wheat Atlas
   - Agricultural Research Repositories

2. **Collect Your Own**:
   - Use drones for aerial photography
   - Ensure consistent lighting
   - Capture multiple angles
   - Label by agricultural experts

3. **Data Augmentation**:
   - The training script includes automatic augmentation
   - Rotation, flipping, zooming applied during training
   - Helps with small datasets

## Tips

- Start with at least 500 images per class
- Ensure proper lighting in images
- Validate images are correctly labeled
- Remove corrupted or unclear images
- Balance your dataset across classes
