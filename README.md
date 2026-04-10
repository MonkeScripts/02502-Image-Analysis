# DTU 02502 - Image Analysis

Course repository for **02502 Image Analysis** at the Technical University of Denmark (DTU). Contains lecture slides, weekly exercises with solutions, utility functions, and exam preparation materials.

## Tech Stack

- **Language:** Python
- **Core Libraries:** NumPy, SciPy, Matplotlib, scikit-image, OpenCV, scikit-learn
- **Medical Imaging:** pydicom, SimpleITK
- **Format:** Jupyter Notebooks, Python scripts, PDF lectures

## Repository Structure

```
.
├── week1/          # PCA tutorial, intro to image analysis in Python
├── week2/          # Image analysis fundamentals
├── week3/          # Pixel-wise operations, cat landmark exercises
├── week4/          # Video image filtering, image morphology
├── week5/          # Blob analysis
├── week6/          # Geometric transformations
├── week8/          # Continuation of blob/transform topics
├── week9/          # Video image transformations
├── week10/         # Advanced topics, medical imaging
├── week11/         # Advanced topics (lecture only)
├── week12/         # Advanced topics (lectures only)
├── week13/         # Advanced topics (lecture only)
├── cats/           # Exercise 8: Cat face alignment & recognition (PCA + warping)
├── exam_2024/      # Fall 2024 exam solution (PCA, LDA, classification)
├── exam_2024_2/    # Spring 2024 exam (3D medical image registration)
├── test_exam/      # Practice exam notebook
└── utils/          # Utility functions and example DICOM images
```

Each week folder typically contains:
- A **PDF lecture** slide deck
- An **exercise notebook** (`.ipynb`)
- A **solution notebook** (`.ipynb`)
- A **data/** folder with images and datasets

## Course Progression

1. **Foundations (Weeks 1-2):** Python basics for image analysis, PCA, data visualization
2. **Core Image Processing (Weeks 3-6):** Pixel operations, filtering, morphology, blob analysis
3. **Advanced Applications (Weeks 8-10):** Video processing, geometric transformations, medical imaging
4. **Specialized Topics (Weeks 11-13):** Advanced image analysis techniques

## Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install numpy scipy matplotlib scikit-image opencv-python scikit-learn pydicom SimpleITK
   ```
3. Open the weekly notebooks in Jupyter:
   ```bash
   jupyter notebook
   ```

## Notes

- Image files (`.jpg`, `.png`), cat landmark files (`.cat`), and databases (`.db`) are excluded via `.gitignore`.
- The `week10/data/` folder contains large medical imaging datasets.
- Week 7 is not present in the repository (likely a course break or holiday).
