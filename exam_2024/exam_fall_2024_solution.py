import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
from skimage.morphology import binary_closing, binary_opening
from skimage.morphology import disk
from scipy.stats import norm
from skimage import color, io, measure
from scipy.spatial import distance
import glob
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
import LDA


def pca_on_breast_data_e_2024():
    breast = load_breast_cancer()

    x = breast.data
    n_feat = x.shape[1]
    n_obs = x.shape[0]
    print(f"Answer: number of features: {n_feat} and number of observations: {n_obs}")
    mn = np.mean(x, axis=0)
    data = x - mn
    std = np.std(data, axis=0)
    data = data / std
    c_x = np.cov(data.T)
    values, vectors = np.linalg.eig(c_x)

    # Plot the first two principal components and color it with the target
    pc_proj = vectors.T.dot(data.T)

    pc_1 = pc_proj[0, :]

    is_true_positive = breast.target == 1

    # Compute the mean of the first principal component for the two classes
    mean_pos = np.mean(pc_1[is_true_positive])
    mean_neg = np.mean(pc_1[~is_true_positive])
    print(f"Answer: Mean of true positive {mean_pos:.2f} and true negative {mean_neg:.2f}")

    # Classify based on the first principal component
    is_positive = pc_1 < 0

    # how many are classified as positive (healthy)
    n_pos = np.sum(is_positive)
    print(f"Answer: Number of positive classified {n_pos}")

    # Calculate the accuracy
    accuracy = np.sum(is_positive == is_true_positive) / len(is_true_positive)
    print(f"Answer: Accuracy {accuracy:.2f}")

    # 1 and 2
    pc_1 = pc_proj[0, :]
    pc_2 = pc_proj[1, :]
    plt.scatter(pc_1, pc_2, c=breast.target)
    plt.show()


def pca_on_screws_bolts_e_2024():
    """
    """
    in_dir = "final_data/screws/"
    all_images = glob.glob(in_dir + "*.jpg")
    n_samples = len(all_images)

    # Read first image to get image dimensions
    im_org = io.imread(all_images[0])
    im_shape = im_org.shape
    height = im_shape[0]
    width = im_shape[1]
    channels = im_shape[2]
    n_features = height * width * channels

    print(f"Found {n_samples} image files. Height {height} Width {width} Channels {channels} n_features {n_features}")

    data_matrix = np.zeros((n_samples, n_features))

    idx = 0
    for image_file in all_images:
        img = io.imread(image_file)
        flat_img = img.flatten()
        data_matrix[idx, :] = flat_img
        idx += 1

    print("Computing PCA")
    image_pca = PCA(n_components=7)
    image_pca.fit(data_matrix)

    plt.plot(image_pca.explained_variance_ratio_ * 100)
    plt.xlabel('Principal component')
    plt.ylabel('Percent explained variance')
    plt.show()

    # Variation explained by first three components
    sum_var = image_pca.explained_variance_ratio_[0] + image_pca.explained_variance_ratio_[1] + image_pca.explained_variance_ratio_[2]
    print(f"Answer: Total variation explained by 3 components {sum_var * 100}")

    components = image_pca.transform(data_matrix)

    idx_test = 7
    print(f"Finding position of {all_images[idx_test]}")
    im_test = io.imread(all_images[idx_test])

    # project on PCA space
    im_test_flat = im_test.flatten()
    im_test_flat = im_test_flat.reshape(1, -1)

    pca_coords = image_pca.transform(im_test_flat)
    pca_coords = pca_coords.flatten()

    pc_1 = components[:, 0]
    pc_2 = components[:, 1]

    plt.plot(pc_1, pc_2, '.')
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    plt.plot(pca_coords[0], pca_coords[1], "*", color="green", label=f"Weird screw")
    plt.show()

    extreme_pc_1_image_m = np.argmin(pc_1)
    extreme_pc_1_image_p = np.argmax(pc_1)
    print(f'Answer: PC 1 extreme minus image: {all_images[extreme_pc_1_image_m]}')
    print(f'Answer:  PC 1 extreme plus image: {all_images[extreme_pc_1_image_p]}')


    idx_1 = 7
    idx_2 = 8
    print(f"Computing PCA distance between {all_images[idx_1]} and {all_images[idx_2]} ")
    im_1 = io.imread(all_images[idx_1])
    im_2 = io.imread(all_images[idx_2])

    # project on PCA space
    im_1_flat = im_1.flatten()
    im_1_flat = im_1_flat.reshape(1, -1)
    im_2_flat = im_2.flatten()
    im_2_flat = im_2_flat.reshape(1, -1)

    pca_coords_1 = image_pca.transform(im_1_flat)
    pca_coords_1 = pca_coords_1.flatten()
    pca_coords_2 = image_pca.transform(im_2_flat)
    pca_coords_2 = pca_coords_2.flatten()

    # Distance between the two images in PCA space
    dist = np.linalg.norm(pca_coords_1 - pca_coords_2)
    print(f"Answer: Distance between {all_images[idx_1]} and {all_images[idx_2]} is {dist:.2f}")

    # find the two photos that are most similar in PCA space
    # This is not a very efficient way to do it
    min_dist = np.inf
    min_idx_1 = -1
    min_idx_2 = -1
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            im_1 = io.imread(all_images[i])
            im_2 = io.imread(all_images[j])

            # project on PCA space
            im_1_flat = im_1.flatten()
            im_1_flat = im_1_flat.reshape(1, -1)
            im_2_flat = im_2.flatten()
            im_2_flat = im_2_flat.reshape(1, -1)

            pca_coords_1 = image_pca.transform(im_1_flat)
            pca_coords_1 = pca_coords_1.flatten()
            pca_coords_2 = image_pca.transform(im_2_flat)
            pca_coords_2 = pca_coords_2.flatten()

            # Distance between the two images in PCA space
            dist = np.linalg.norm(pca_coords_1 - pca_coords_2)
            if dist < min_dist:
                min_dist = dist
                min_idx_1 = i
                min_idx_2 = j
    print(f"Answer: Most similar images are {all_images[min_idx_1]} and {all_images[min_idx_2]} with distance {min_dist:.2f}")



def kidney_analysis_e_2024():
    in_dir = "final_data/kidneys/"
    im_name = "1-189.dcm"

    ct = dicom.read_file(in_dir + im_name)
    img = ct.pixel_array
    io.imshow(img, vmin=50, vmax =300, cmap = 'gray')
    io.show()

    min_hu = 100
    max_hu = 250

    bin_img = (img > min_hu) & (img < max_hu)
    vb_label_colour = color.label2rgb(bin_img)
    io.imshow(vb_label_colour)
    plt.title("First kidney estimate")
    io.show()

    label_img = measure.label(bin_img)
    n_labels = label_img.max()
    print(f"Number of labels: {n_labels}")

    region_props = measure.regionprops(label_img)

    min_area = 2000
    max_area = 5000
    min_per = 400
    max_per = 600
    print(f"Answer: Minimum area {min_area}")

    min_found_area = np.inf
    max_found_area = -np.inf
    # Create a copy of the label_img
    label_img_filter = label_img.copy()
    for region in region_props:
        a = region.area
        p = region.perimeter
        if a < min_found_area:
            min_found_area = a
        if a > max_found_area:
            max_found_area = a

        if a < min_area or a > max_area or p < min_per or p > max_per:
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0
        else:
            print(f"Area {a} Perimeter {p}")

    # print(f"Answer: Min area {min_found_area:.0f} and max area {max_found_area:.0f}")

    # Create binary image from the filtered label image
    i_kidney = label_img_filter > 0
    io.imshow(i_kidney)
    io.show()

    footprint = disk(3)
    closing = binary_closing(i_kidney, footprint)
    io.imshow(closing)
    plt.title("Closed kidney estimate")
    io.show()
    kidney_estimate = closing

    hu_values = img[kidney_estimate]
    med_hu = np.median(hu_values)
    print(f"Answer: Found HU median {med_hu:.0f}")

    label_img = measure.label(kidney_estimate)
    n_labels = label_img.max()
    print(f"Number of labels: {n_labels}")
    region_props = measure.regionprops(label_img)
    sum_area = 0
    for region in region_props:
        a = region.area
        print(f"Area {a}")
        sum_area += a

    # Voxel side in cm
    voxel_side = 0.78 / 10
    sum_area = sum_area * voxel_side * voxel_side
    print(f"Answer: Total area {sum_area:.0f} cm^2")

    # Ground truth
    gt_kidney = io.imread(in_dir + 'kidneys_gt.png')
    gt_kidney = gt_kidney > 0

    dice_score = 1 - distance.dice(kidney_estimate.ravel(), gt_kidney.ravel())
    print(f"Answer: DICE score {dice_score:.3f}")


# f = 7 * x_1 * x_1 + x_1 * x_2 + 3 * x_2 * x_2
# d_x_1 = 14 * x_1 + x_2
# d_x_2 = x_1 + 6 * x_2
def gradient_descent_e_2024():
    x_1_start = -5
    x_2_start = 4
    step_length = 0.1

    n_steps = 13
    x_1 = x_1_start
    x_2 = x_2_start
    cs = []
    x_1_s = []
    x_2_s = []
    for i in range(n_steps):
        x_1_s.append(x_1)
        x_2_s.append(x_2)

        grad_x_1 = 14 * x_1 + x_2
        grad_x_2 = x_1 + 6 * x_2

        new_x_1 = x_1 - step_length * grad_x_1
        new_x_2 = x_2 - step_length * grad_x_2
        x_1 = new_x_1
        x_2 = new_x_2
        c = 7 * x_1 * x_1 + x_1 * x_2 + 3 * x_2 * x_2
        if c < 2.0:
            print(f"After {i+1} steps: x1 {x_1:.2f} x2 {x_2:.2f} c {c:.2f}")

        cs.append(c)
    # plt.plot(cs)
    # plt.show()

    plt.scatter(x_1_s, x_2_s, c = cs)
    plt.plot(x_1_s, x_2_s)
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.show()


def cell_blob_analysis_e_2024():
    moving_image_in = "final_data/cells/x_NisslStain_9-260.81.png"
    fixed_image_in = "final_data/cells/y_NisslStain_9-260.81.png"

    moving_image = io.imread(moving_image_in)
    m_bin_img = moving_image > 30

    footprint = disk(3)
    opening = binary_opening(m_bin_img, footprint)
    io.imshow(opening)
    plt.title("Moving image estimate")
    io.show()

    label_img = measure.label(opening)
    n_labels = label_img.max()
    print(f"Number of labels: {n_labels}")

    fixed_image = io.imread(fixed_image_in)
    m_bin_img = fixed_image > 30

    footprint = disk(3)
    opening = binary_opening(m_bin_img, footprint)
    io.imshow(opening)
    plt.title("fixed image estimate")
    io.show()

    label_img = measure.label(opening)
    n_labels = label_img.max()
    print(f"Number of labels: {n_labels}")


def cell_blob_registration_e_2024():
    moving_image_in = "final_data/cells/LabelsFixedImg.png"
    fixed_image_in = "final_data/cells/LabelsMovingImg.png"

    lm_moving = io.imread(moving_image_in)
    lm_fixed = io.imread(fixed_image_in)

    lms_moving = []
    lms_fixed = []

    img_shape = lm_moving.shape
    # print(f"Image shape: {img_shape}")

    # Really simple way to find all landmarks
    for lm_idx in range(1, 6):
        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                if lm_moving[i, j] == lm_idx:
                    lms_moving.append([i, j])
                if lm_fixed[i, j] == lm_idx:
                    lms_fixed.append([i, j])

    print(f"Found {len(lms_moving)} moving landmarks and {len(lms_fixed)} fixed landmarks")

    # Compute the mean of the landmarks
    mean_moving = np.mean(lms_moving, axis=0)
    mean_fixed = np.mean(lms_fixed, axis=0)

    # Compute distance between the means
    diff_means = mean_fixed - mean_moving
    distance = np.linalg.norm(diff_means)
    print(f"Answer: Distance between means of landmarks: {distance:.2f}")


def cell_sensitivity_e_2024():
    tp = 18
    fn = 4
    tn = 12
    fp = 6
    sensitivity = tp / (tp + fn)
    print(f"Answer: Sensitivity: {sensitivity:.2f}")

    sensitivity = 0.82
    tp = 18
    fn = tp / sensitivity - tp
    print(f"Answer: Number of false negatives: {fn:.2f}")



def lda_on_traffic_e_2024():
    train_file = "final_data/traffic/traffic_train.txt"
    test_file = "final_data/traffic/traffic_test.txt"
    data_train = np.loadtxt(train_file, comments="%", delimiter=",")
    data_test = np.loadtxt(test_file, comments="%", delimiter=",")

    # Training samples per morning and afternoon
    n_train = 140
    n_test = 60

    print(data_train.shape)
    print(data_test.shape)

    # Training data
    morning_traffic_train = data_train[:n_train]
    afternoon_traffic_train = data_train[n_train:]

    # Test data
    morning_traffic_test = data_test[:n_test]
    afternoon_traffic_test = data_test[n_test:]


    morning_traffic_train_with_rain = morning_traffic_train[morning_traffic_train[:, 2] == 1]
    print(f"Answer: Number of rainy days in the morning: {morning_traffic_train_with_rain.shape[0]}")


    plt.scatter(morning_traffic_train[:100, 0], morning_traffic_train[:100, 1])
    plt.scatter(afternoon_traffic_train[:100, 0], afternoon_traffic_train[:100, 1])
    plt.show()

    # Concatenate the two classes
    data_train = np.concatenate((morning_traffic_train, afternoon_traffic_train), axis=0)
    # Remove last column
    data_train = data_train[:, :-1]
    # generate binary outcome
    y_train = np.concatenate((np.zeros(n_train), np.ones(n_train)), axis=0)

    print(f"Computing LDA")
    w = LDA.LDA(data_train, y_train)
    print(w.shape)

    # Test the classifier
    data_test = np.concatenate((morning_traffic_test, afternoon_traffic_test), axis=0)
    data_test = data_test[:, :-1]

    y = np.c_[np.ones((len(data_test), 1)), data_test] @ w.T
    PosteriorProb = np.clip(np.exp(y) / np.sum(np.exp(y), 1)[:, np.newaxis], 0, 1)
    true_afternoon_prop = PosteriorProb[60:]

    # Number of faulty morning predictions.
    # If the probability of being morning (column 0) is greater than 0.5,
    # even though we know the true value is afternoon, it is then a false negative.
    faulty_pred_morning = np.sum(true_afternoon_prop[:, 0] > 0.5)

    print(f"The number of faulty morning predictions are {faulty_pred_morning}")


def dice_segmentation_e_2024():
    in_file = "final_data/dice/CubesG.png"
    val_file_1 = "final_data/dice/B_Cubes.txt"
    val_file_2 = "final_data/dice/C_Cubes.txt"
    val_file_3 = "final_data/dice/D_Cubes.txt"

    val_file_1_p = "final_data/dice/D_Cubes.txt"
    val_file_2_p = "final_data/dice/E_Cubes.txt"

    img = io.imread(in_file)

    vals_1 = np.loadtxt(val_file_1, comments="%")
    vals_2 = np.loadtxt(val_file_2, comments="%")
    vals_3 = np.loadtxt(val_file_3, comments="%")

    mean_1 = np.mean(vals_1, axis=0)
    mean_2 = np.mean(vals_2, axis=0)
    mean_3 = np.mean(vals_3, axis=0)

    thres_1 = 0.5 * (mean_1 + mean_2)
    thres_2 = 0.5 * (mean_2 + mean_3)

    print(f"Answer: Thresholds: {thres_1} {thres_2}")
    bin_img_1 = (img < thres_1) & (img > 0)
    bin_img_2 = (thres_1 <= img) & (img < thres_2)
    bin_img_3 = img >= thres_2

    io.imshow(bin_img_1)
    io.show()

    combined = bin_img_1 + 2 * bin_img_2 + 3 * bin_img_3
    io.imshow(combined)
    io.show()

    vals_1_p = np.loadtxt(val_file_1_p, comments="%")
    vals_2_p = np.loadtxt(val_file_2_p, comments="%")

    (mu_1, std_1) = norm.fit(vals_1_p)
    (mu_2, std_2) = norm.fit(vals_2_p)

    for v in range(256):
        # Find crossing of the two normal distributions
        test_pdf_1 = norm.pdf(v, mu_1, std_1)
        test_pdf_2 = norm.pdf(v, mu_2, std_2)
        if test_pdf_1 > test_pdf_2:
            print(f"val: {v} 1 bigger than 2")
        else:
            print(f"val: {v} 2 bigger than 1")

    avg_mu = 0.5 * (mu_1 + mu_2)
    print(f"Shortest dist threshold {avg_mu:.2f}")


if __name__ == '__main__':
    pca_on_breast_data_e_2024()
    pca_on_screws_bolts_e_2024()
    kidney_analysis_e_2024()
    gradient_descent_e_2024()
    cell_blob_analysis_e_2024()
    cell_blob_registration_e_2024()
    cell_sensitivity_e_2024()
    lda_on_traffic_e_2024()
    dice_segmentation_e_2024()
