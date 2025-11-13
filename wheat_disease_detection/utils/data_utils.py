"""
Utility functions for wheat disease detection
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess a single image

    Args:
        image_path: Path to image file
        target_size: Target size for resizing

    Returns:
        Preprocessed image array
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return img_array


def display_sample_images(data_dir, classes, samples_per_class=5):
    """
    Display sample images from each class

    Args:
        data_dir: Root directory containing class folders
        classes: List of class names
        samples_per_class: Number of samples to display per class
    """
    fig, axes = plt.subplots(len(classes), samples_per_class,
                             figsize=(15, 3*len(classes)))

    for i, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            continue

        images = [f for f in os.listdir(class_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for j in range(min(samples_per_class, len(images))):
            img_path = os.path.join(class_dir, images[j])
            img = Image.open(img_path)

            if len(classes) == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]

            ax.imshow(img)
            ax.axis('off')
            if j == 0:
                ax.set_title(class_name, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
    plt.show()


def count_images_per_class(data_dir):
    """
    Count number of images in each class folder

    Args:
        data_dir: Root directory containing class folders

    Returns:
        Dictionary with class names and image counts
    """
    counts = {}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            count = sum(1 for f in os.listdir(class_path)
                        if os.path.splitext(f)[1].lower() in image_extensions)
            counts[class_name] = count

    return counts


def visualize_dataset_distribution(data_dir):
    """
    Visualize the distribution of images across classes

    Args:
        data_dir: Root directory containing class folders
    """
    counts = count_images_per_class(data_dir)

    plt.figure(figsize=(10, 6))
    plt.bar(counts.keys(), counts.values(), color='skyblue', edgecolor='navy')
    plt.xlabel('Disease Class', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.title('Dataset Distribution', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')

    # Add count labels on bars
    for i, (class_name, count) in enumerate(counts.items()):
        plt.text(i, count + 10, str(count), ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('dataset_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print statistics
    total = sum(counts.values())
    print(f"\nDataset Statistics:")
    print(f"Total Images: {total}")
    print(f"Number of Classes: {len(counts)}")
    print(f"Average per Class: {total / len(counts):.1f}")
    print(f"\nPer-class counts:")
    for class_name, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count} ({count/total*100:.1f}%)")


def augment_image(image, augmentation_type='rotation'):
    """
    Apply augmentation to an image

    Args:
        image: PIL Image or numpy array
        augmentation_type: Type of augmentation to apply

    Returns:
        Augmented image
    """
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image

    if augmentation_type == 'rotation':
        angle = np.random.randint(-30, 30)
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        img = cv2.warpAffine(img, M, (cols, rows))

    elif augmentation_type == 'flip_horizontal':
        img = cv2.flip(img, 1)

    elif augmentation_type == 'flip_vertical':
        img = cv2.flip(img, 0)

    elif augmentation_type == 'brightness':
        factor = np.random.uniform(0.7, 1.3)
        img = np.clip(img * factor, 0, 255).astype(np.uint8)

    elif augmentation_type == 'zoom':
        scale = np.random.uniform(0.8, 1.2)
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 0, scale)
        img = cv2.warpAffine(img, M, (cols, rows))

    return Image.fromarray(img.astype(np.uint8))


def create_prediction_report(results, output_path='prediction_report.txt'):
    """
    Create a text report of prediction results

    Args:
        results: List of prediction results
        output_path: Path to save report
    """
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("WHEAT DISEASE DETECTION - PREDICTION REPORT\n")
        f.write("="*80 + "\n\n")

        # Summary statistics
        total = len(results)
        diseases_detected = sum(1 for r in results
                                if r.get('predicted_class', '').lower() != 'healthy')

        f.write(f"Total Images Analyzed: {total}\n")
        f.write(f"Healthy Samples: {total - diseases_detected}\n")
        f.write(f"Diseased Samples: {diseases_detected}\n")
        f.write(f"Disease Rate: {diseases_detected/total*100:.1f}%\n\n")

        # Disease breakdown
        disease_counts = {}
        for result in results:
            disease = result.get('predicted_class', 'Unknown')
            disease_counts[disease] = disease_counts.get(disease, 0) + 1

        f.write("Disease Breakdown:\n")
        f.write("-"*80 + "\n")
        for disease, count in sorted(disease_counts.items(),
                                     key=lambda x: x[1], reverse=True):
            f.write(f"{disease}: {count} ({count/total*100:.1f}%)\n")

        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("="*80 + "\n\n")

        # Individual results
        for i, result in enumerate(results, 1):
            f.write(f"\nImage {i}: {result.get('image_path', 'N/A')}\n")
            f.write(f"Prediction: {result.get('predicted_class', 'N/A')}\n")
            f.write(f"Confidence: {result.get('confidence', 0):.2%}\n")

            if 'top_predictions' in result:
                f.write("Top 3 Predictions:\n")
                for pred in result['top_predictions']:
                    f.write(f"  - {pred['class']}: {pred['confidence']:.2%}\n")
            f.write("-"*80 + "\n")

    print(f"Report saved to {output_path}")


def validate_dataset_structure(data_dir, required_classes=None):
    """
    Validate dataset directory structure

    Args:
        data_dir: Root directory to validate
        required_classes: List of required class names (optional)

    Returns:
        Boolean indicating if structure is valid
    """
    print(f"Validating dataset structure in: {data_dir}")

    if not os.path.exists(data_dir):
        print(f"❌ Directory does not exist: {data_dir}")
        return False

    # Check for class folders
    class_folders = [d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d))]

    if not class_folders:
        print(f"❌ No class folders found in {data_dir}")
        return False

    print(f"✓ Found {len(class_folders)} class folders: {class_folders}")

    # Check required classes
    if required_classes:
        missing = set(required_classes) - set(class_folders)
        if missing:
            print(f"⚠️  Missing required classes: {missing}")
        else:
            print(f"✓ All required classes present")

    # Check for images in each class
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    empty_classes = []

    for class_name in class_folders:
        class_path = os.path.join(data_dir, class_name)
        images = [f for f in os.listdir(class_path)
                  if os.path.splitext(f)[1].lower() in image_extensions]

        if not images:
            empty_classes.append(class_name)
            print(f"⚠️  No images in class: {class_name}")
        else:
            print(f"✓ {class_name}: {len(images)} images")

    if empty_classes:
        print(f"\n⚠️  Warning: {len(empty_classes)} class(es) are empty")
        return False

    print(f"\n✓ Dataset structure is valid!")
    return True


if __name__ == "__main__":
    # Example usage
    data_dir = "data/wheat_diseases/train"

    if os.path.exists(data_dir):
        # Validate structure
        validate_dataset_structure(
            data_dir,
            required_classes=['Healthy', 'Leaf Rust', 'Stem Rust',
                              'Yellow Rust', 'Septoria']
        )

        # Display distribution
        visualize_dataset_distribution(data_dir)

        # Display samples
        classes = ['Healthy', 'Leaf Rust',
                   'Stem Rust', 'Yellow Rust', 'Septoria']
        display_sample_images(data_dir, classes)
    else:
        print(f"Data directory not found: {data_dir}")
        print("Please create the directory and add your training images.")
