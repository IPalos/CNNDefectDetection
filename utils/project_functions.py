import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, f1_score
import numpy as np
import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy.ndimage import label, center_of_mass
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.cluster import DBSCAN
import pandas as pd
import seaborn as sns
import torch.nn.functional as F
plt.rcParams['figure.figsize'] = [16, 9]

# Import classes
import utils.project_classes as pc

# VTK Functions
def load_and_pad_vtk(file_path, pad_width=0):
    """
    Loads a vtk file and pads it with the given width
    Returns:
        eigen_val: numpy array of shape (simulation_size, simulation_size)
        eigen_vec: numpy array of shape (simulation_size, simulation_size, 2)
    """
    # Read the structured grid
    reader = vtk.vtkStructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()
    data = reader.GetOutput()

    # Get eigen_val array and reshape
    eigen_val_array = data.GetPointData().GetArray("eigen_val")
    if eigen_val_array is None:
        raise ValueError("No 'eigen_val' array found in the dataset.")
    eigen_val = vtk_to_numpy(eigen_val_array)
    simulation_size = int(np.sqrt(eigen_val.shape[0]))  # Assuming square grid
    eigen_val = eigen_val.reshape(simulation_size, simulation_size)

    # Get eigen_vec array and reshape
    eigen_vec_array = data.GetPointData().GetArray("eigen_vec")
    if eigen_vec_array is None:
        raise ValueError("No 'eigen_vec' array found in the dataset.")
    eigen_vec = vtk_to_numpy(eigen_vec_array)
    eigen_vec = eigen_vec.reshape(simulation_size, simulation_size, 3)
    # Take only x,y components
    eigen_vec = eigen_vec[:,:,:2]

    # Add padding
    padded_eigen_val = np.pad(eigen_val, pad_width=pad_width, mode='edge')
    padded_eigen_vec = np.pad(eigen_vec, pad_width=((pad_width,pad_width), (pad_width,pad_width), (0,0)), mode='edge')

    return padded_eigen_val, padded_eigen_vec

def find_defects(eigen_vals, eigen_vecs, order_threshold):
    """
    Finds the defects in the eigen_vals array
    Returns:
        defect_points: numpy array of shape (n_defects, 2)
    """
    # Get dimensions from input arrays
    simulation_size = eigen_vals.shape[0]
    
    # Create coordinate grid
    x = np.arange(simulation_size)
    y = np.arange(simulation_size)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)  # Assuming 2D data with z=0
    points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)

    # Identify points below the threshold
    below_threshold_indices = np.where(eigen_vals < order_threshold)
    # Convert 2D indices to flattened indices for points array
    flat_indices = below_threshold_indices[0] * simulation_size + below_threshold_indices[1]
    defect_points = points[flat_indices]

    # Cluster adjacent points
    clustering = DBSCAN(eps=1.0, min_samples=1).fit(defect_points)
    cluster_labels = clustering.labels_

    # Calculate centroids for each cluster
    centroids = []
    for label in np.unique(cluster_labels):
        cluster = defect_points[cluster_labels == label]
        centroid = cluster.mean(axis=0)
        centroids.append(centroid[0:2])

    return np.round(centroids).astype(int)

def build_tensor(eigen_vals, eigen_vecs, center_x, center_y, window_size=7):
    """
    Extracts a 7x7 window around (center_x, center_y) and builds multi-channel tensor.
    """
    # Define the half window size
    half_window = window_size // 2
    
    # Initialize channels (7 channels for 7x7 window)
    channels = {
        "U": np.zeros((window_size, window_size)),
        "V": np.zeros((window_size, window_size)),
        "Magnitude": np.zeros((window_size, window_size)),
        "Orientation": np.zeros((window_size, window_size)),
        "Divergence": np.zeros((window_size, window_size)),
        "Curl": np.zeros((window_size, window_size)),
        "OrderFactor": np.zeros((window_size, window_size)),
    }

    # Extract a 7x7 window of eigenvectors
    u_window = eigen_vecs[center_y - half_window:center_y + half_window + 1, 
                          center_x - half_window:center_x + half_window + 1, 0]  # U-components
    v_window = eigen_vecs[center_y - half_window:center_y + half_window + 1, 
                          center_x - half_window:center_x + half_window + 1, 1]  # V-components

    # Compute divergence and curl for the 7x7 window
    du_dx, du_dy = np.gradient(u_window, axis=(1, 0))  # Gradients of U
    dv_dx, dv_dy = np.gradient(v_window, axis=(1, 0))  # Gradients of V

    divergence = du_dx + dv_dy  # Divergence
    curl = dv_dx - du_dy        # Curlfire

    # Fill the channels
    for i in range(window_size):
        for j in range(window_size):
            u = u_window[i, j]
            v = v_window[i, j]

            channels["U"][i, j] = u
            channels["V"][i, j] = v
            channels["Magnitude"][i, j] = np.sqrt(u**2 + v**2)
            channels["Orientation"][i, j] = np.arctan2(v, u)
            channels["Divergence"][i, j] = divergence[i, j]
            channels["Curl"][i, j] = curl[i, j]
            channels["OrderFactor"][i, j] = eigen_vals[center_y - half_window + i, center_x - half_window + j]

    # Stack the channels into a single tensor (7x7x7)
    tensor = np.stack(list(channels.values()), axis=-1)  # Shape: (7, 7, 7)
    return tensor




# Visualization functions
def plot_labeled_vector_field(U, V, order, labeled_defects):
    """
    Plots a 2D vector field using U and V components.
    Args:
        U (2D array): X-components of the vectors
        V (2D array): Y-components of the vectors
    """
    x, y = np.meshgrid(range(U.shape[1]), range(U.shape[0]))
    plt.figure(figsize=(10, 10))
    plt.quiver(x, y, U, V, order, scale=1,cmap='viridis',scale_units='xy', angles='xy')
    plt.title("Vector Field (U, V)")

    window_size = 7  # Size of window to draw around defects
    for defect in labeled_defects:
        x, y = defect[0]
        if defect[1] == 0:
            rect = plt.Rectangle((x - window_size//2, y - window_size//2), 
                                window_size, window_size,
                                fill=False, color='red', linewidth=2)
            plt.gca().add_patch(rect)
        elif defect[1] == 1:
            rect = plt.Rectangle((x - window_size//2, y - window_size//2), 
                                window_size, window_size,
                                fill=False, color='blue', linewidth=2)
            plt.gca().add_patch(rect)
    plt.gca().invert_yaxis()  # Match array coordinate system
    plt.show()

def plot_vector_field_with_confidence(U, V, order, df, confidence_threshold=0.8):
    """
    Plots a 2D vector field using U and V components.
    Args:
        U (2D array): X-components of the vectors
        V (2D array): Y-components of the vectors
    """
    x, y = np.meshgrid(range(U.shape[1]), range(U.shape[0]))

    plt.figure(figsize=(20, 10))
    plt.subplot(1,2,1)

    plt.quiver(x, y, U, V, order, scale=1,cmap='viridis',scale_units='xy', angles='xy')
    plt.title("Vector Field (U, V)")

    window_size = 7  # Size of window to draw around defects
    for defect in df.to_dict(orient="records"):
        x, y = defect["x"], defect["y"]
        if defect["prediction"] == 0:
            rect = plt.Rectangle((x - window_size//2, y - window_size//2), 
                                window_size, window_size,
                                fill=False, color='red', linewidth=2, alpha=defect["confidence"])
            plt.gca().add_patch(rect)
        elif defect["prediction"] == 1:
            rect = plt.Rectangle((x - window_size//2, y - window_size//2), 
                                window_size, window_size,
                                fill=False, color='blue', linewidth=2, alpha=defect["confidence"])
            plt.gca().add_patch(rect)
        elif defect["prediction"] == 2 and defect["confidence"] < confidence_threshold:
            rect = plt.Rectangle((x - window_size//2, y - window_size//2), 
                                window_size, window_size,
                                fill=False, color='black', linewidth=2, alpha=1)
            plt.gca().add_patch(rect)
    plt.gca().invert_yaxis()  # Match array coordinate system

    plt.subplot(1,2,2)
    sns.boxplot(data=df, x='prediction', y='confidence')
    plt.title('Confidence Distribution by Prediction Class')
    plt.xlabel('Prediction Class')
    plt.ylabel('Confidence Score')
    plt.show()


    plt.show()

def visualize_tensor_as_heatmaps(tensor, channel_names):
    """
    Visualizes each channel of a tensor as a heatmap.
    Args:
        tensor (numpy array): Tensor of shape (7, 7, 7)
        channel_names (list): List of channel names
    """
    num_channels = tensor.shape[-1]
    fig, axes = plt.subplots(1, num_channels, figsize=(15, 5))

    for i in range(num_channels):
        ax = axes[i]
        im = ax.imshow(tensor[:, :, i], cmap='coolwarm', origin='upper')
        ax.set_title(channel_names[i])
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()

def plot_experiments_evolution(prediction_df, y_scale=10):
    """
    Plots the evolution of the experiments
    df_cols : [x,y,prediction,tag,frame]
    Returns:
        fig: The matplotlib figure object
    """
    _ = prediction_df.copy()    
    _["+"] = _["prediction"]==1
    _["-"] = _["prediction"]==0
    experiments = _.groupby(["tag","frame"]).sum().reset_index()
    experiments = experiments["tag"].unique()
    fig, axes = plt.subplots(len(experiments), 1, figsize=(10, 3 * len(experiments)), sharex=True)

    # Grouped data
    grouped_data = _.groupby(["tag", "frame"]).sum().reset_index()
    grouped_data["-"] = -grouped_data["-"]


    # Handle case of single experiment (axes is not array)
    if len(experiments) == 1:
        axes = [axes]
        
    for i, exp in enumerate(experiments):
        # Filter data for the current experiment
        exp_data = grouped_data[grouped_data["tag"] == exp]
        
        # Create horizontal bar plots
        sns.barplot(data=exp_data, x="frame", y="+", color="#FDE725", label="+ (Defectos positivos)", ax=axes[i])
        sns.barplot(data=exp_data, x="frame", y="-", color="#440154", label="- (Defectos negativos)", ax=axes[i])
        
        # Add labels and title
        axes[i].set_title(f"Experimento: {exp}")
        axes[i].set_ylabel("Conteo de defectos")
        axes[i].legend(loc="upper right")

        # Fix x-ticks: set ticks in multiples of 10, rotate for readability
        frames = exp_data["frame"].unique()
        max_frame = frames.max()
        tick_values = np.arange(0, max_frame + 1, 10)
        axes[i].set_xticks(tick_values)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_ylim(-y_scale, y_scale)

    # Common X-label
    plt.xlabel("Tiempo")
    plt.tight_layout()
    return fig

def plot_vector_field(U, V, order):
    """
    Plots a 2D vector field using U and V components.
    Args:
        U (2D array): X-components of the vectors
        V (2D array): Y-components of the vectors
    """
    x, y = np.meshgrid(range(U.shape[1]), range(U.shape[0]))
    plt.figure(figsize=(10, 10))
    plt.quiver(x, y, U, V, order, scale=1,cmap='viridis',scale_units='xy', angles='xy')
    plt.title("Vector Field (U, V)")
    plt.plot(3,3, 'ro', markersize=10)

    plt.gca().invert_yaxis()  # Match array coordinate system
    plt.show()


def plot_defect_detail(eigen_vecs, hue_data, centroid):
    """
    Creates a detailed visualization for each defect showing:
    1. A zoomed 7x7 quiver plot of the vector field around the defect
    2. The full hue field with a red box indicating the defect location
    
    Args:
        eigen_vecs (numpy array): Vector field of shape (H, W, 2)
        hue_data (numpy array): Hue values of shape (H, W)
        centroid (list): List of (x,y) coordinates of defects
    """
    x, y = int(centroid[0]), int(centroid[1])
        
    # Create figure with 1x2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot 1: 7x7 quiver plot centered on defect
    window_size = 7
    half_window = window_size // 2
    
    # Extract vector components for the window
    U = eigen_vecs[y-half_window:y+half_window+1, 
                    x-half_window:x+half_window+1, 1]
    V = eigen_vecs[y-half_window:y+half_window+1, 
                    x-half_window:x+half_window+1, 0]
    
    # Create coordinate grid for quiver
    xx, yy = np.meshgrid(range(window_size), range(window_size))
    
    # Plot quiver with hue background
    window_hue = hue_data[y-half_window:y+half_window+1, 
                            x-half_window:x+half_window+1]
    ax1.quiver(xx, yy, U, V, window_hue, scale=1, 
                scale_units='xy', angles='xy', cmap='viridis')
    ax1.set_title(f'Vector Field Around Defect at ({x},{y})')
    
    # Plot 2: Full hue field with rectangle
    ax2.imshow(hue_data, cmap='viridis')
    rect = Rectangle((x-half_window, y-half_window), 
                    window_size, window_size, 
                    linewidth=2, edgecolor='r', facecolor='none')
    ax2.add_patch(rect)
    ax2.set_title('Full Hue Field')
    
    plt.tight_layout()
    plt.show()



# Prediction functions
def predict_field(model, eigen_vals, eigen_vecs, defects,device, *args, **kwargs):
    """
    Predicts the defects in the field
    Returns:
        df (pandas DataFrame): DataFrame with the predicted defects
        df shape: (n_defects, 4) (x, y, prediction, confidence)
    """
    samples = []
    model = model.to(device)
    for i in defects:
        tensor = build_tensor(eigen_vals, eigen_vecs, int(i[0]), int(i[1]))
        tensor = process_single_tensor(tensor)
        tensor = tensor.to(device)
        with torch.no_grad():
            outputs = model(tensor)  # Forward pass
            probabilities = torch.softmax(outputs, dim=1)  # For classification
            predictions = torch.argmax(probabilities, dim=1)  # Predicted class
        sample = pc.Sample(
            filename=kwargs["filename"], 
            position=i, 
            tensor=tensor, 
            label=None, 
            pseudo_label=predictions.item(), 
            pseudo_label_confidence=probabilities.max().item()
        )
        samples.append(sample)
    return samples

def generate_pseudo_labels(model, loader, confidence_threshold=0.8, device=None):
    """
    Generates pseudo labels for the data
    Returns:
        valid_data (list): List of tensors with confidence >= confidence_threshold
        pseudo_labels (list): List of predicted labels for the valid_data
    """ 
    model = model.to(device)
    model.eval()
    pseudo_labels = []
    valid_data = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            outputs = model(data)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)
            for i in range(len(data)):
                if confidence[i] >= confidence_threshold:
                    pseudo_labels.append(predicted[i].item())
                    valid_data.append(data[i].cpu())

    return valid_data, pseudo_labels


# tensor data processing
def process_single_tensor(tensor):
    """
    Process the tensor to be used in the model
    Args:
        tensor (numpy array): Tensor of shape (7, 7, 7) (H, W, C)
    Returns:
        tensor (torch tensor): Processed tensor of shape (5, 7, 7) (C, H, W)
    """
    tensor[:, :, [4, 6]] = tensor[:,:, [6, 4]]
    tensor = torch.tensor(tensor[:,:,:5], dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1)  # Convert to (C, H, W)
    tensor = tensor.reshape(1, 5, 7, 7)
    return tensor

def process_batch_tensor(tensor):
    """
    Process a batch of tensors to be used in the model
    Args:
        tensor (numpy array): Tensor of shape (B, 7, 7, 7) (B, H, W, C)
    Returns:
        tensor (torch tensor): Processed tensor of shape (B, 5, 7, 7) (B, C, H, W)
    """
    tensor[:, :, :, [4, 6]] = tensor[:,:,:, [6, 4]]
    tensor = torch.tensor(tensor[:,:,:,:5], dtype=torch.float32)
    tensor = tensor.permute(0, 3, 1, 2)  # Convert to (B, C, H, W)
    return tensor

def load_tensor_data(file_path, ignore_unknown=True, is_unlabeled=False):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    tensors = np.array(data["tensor"])
    labels = np.array(data["labels"])
    positions = np.array(data["positions"])


    label_mapping = {"+": 0, "-": 1, "0": 2, "?": 3}

    if ignore_unknown:
        mask = labels != "?"
        labels = labels[mask]
        positions = positions[mask]
        tensors = tensors[mask]
        label_mapping = {"+": 0, "-": 1, "0": 2}
    
    if not is_unlabeled:
        encoded_labels = [label_mapping[label] for label in labels]
    else:
        encoded_labels = None

    return tensors, labels, encoded_labels, positions

def create_dataset(directory, ignore_unknown=True, n_files=None, *args, **kwargs):
    """
    Creates a dataset from the directory
    Returns:
        tensors (numpy array): Tensor of shape (B, 5, 7, 7) (B, C, H, W)
        labels (numpy array): Label of shape (B,)
        encoded_labels (numpy array): Encoded label of shape (B,)
        positions (numpy array): Position of shape (B, 2)
    """
    # List all files in the directory that end with _tensors.pkl
    files = [f for f in os.listdir(directory) if f.endswith('_tensors.pkl')]
    is_unlabeled = kwargs["is_unlabeled"]

    if n_files is not None:
        files = files[:n_files]

    # Initialize variables
    tensors, labels, encoded_labels, positions = None, None, None, None

    # Load and concatenate data
    for file in files:
        try:
            _tensors, _labels, _encoded_labels, _positions = load_tensor_data(os.path.join(directory, file), ignore_unknown, is_unlabeled=is_unlabeled)
        except Exception as e:
            print(f"Error loading file {file}: {e}")
            continue
        
        # Initialize or concatenate
        if tensors is None:
            tensors = _tensors
            labels = _labels
            encoded_labels = _encoded_labels
            positions = _positions
        else:
            try:
                tensors = np.concatenate((tensors, _tensors), axis=0)
                if not is_unlabeled:
                    labels = np.concatenate((labels, _labels), axis=0)
                    encoded_labels = np.concatenate((encoded_labels, _encoded_labels), axis=0)
                positions = np.concatenate((positions, _positions), axis=0)
            except ValueError as e:
                print(f"Error concatenating data for file {file}: {e}")
                continue

    tensors = process_batch_tensor(tensors)

    # Return the concatenated dataset
    return tensors, labels, encoded_labels, positions

def create_label_tensor_data(eigen_vals, eigen_vecs, centroids, hue_data):
    labels = []
    tensors = []
    for i in range(len(centroids)):
        center_x, center_y = [ int(i) for i in centroids[i]]
        tensor = build_tensor(eigen_vals, eigen_vecs, center_x, center_y)
        tensors.append(tensor)
        plot_defect_detail(eigen_vecs, hue_data, centroids[i])
        label = input(f"Enter label (+, -, 0): ({i}/{len(centroids)})")
        if label == "":
            label = "0"
        elif label == "exit":
            break
        labels.append(label)
    return labels, tensors

def save_tensor_data(tensors, labels,positions,file_path):
    data = {
        "positions": positions,
        "tensor": tensors,
        "labels": labels
    }
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

# Model functions
def load_model(file_path):
    model = pc.CNNClassifier()
    model.load_state_dict(torch.load(file_path))
    model.eval()
    return model



# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50, device=None):
    model = model.to(device)
    criterion = criterion.to(device)  # Ensure the criterion is on the same device

    training_losses = []
    val_losses = []
    train_f1_scores = []
    val_f1_scores = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for tensors, labels in train_loader:
            tensors = tensors.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(tensors)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # Collect predictions for F1 calculation
            _, predicted = torch.max(outputs, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        # Calculate training F1 score
        train_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0)
        
        val_loss, val_accuracy, val_f1 = evaluate_model(model, val_loader, criterion)
        training_losses.append(train_loss)
        val_losses.append(val_loss)
        train_f1_scores.append(train_f1)
        val_f1_scores.append(val_f1)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
    
    return {
        'train_losses': training_losses,
        'val_losses': val_losses,
        'train_f1_scores': train_f1_scores,
        'val_f1_scores': val_f1_scores
    }

def evaluate_model(model, data_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for tensors, labels in data_loader:
            outputs = model(tensors)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(classification_report(all_labels, all_preds, target_names=["+", "-", "0"]))
    return val_loss, accuracy, f1


# Winding number functions

def get_closest_angle(theta1, theta2):
    """
    Returns the closest angle between two angles
    Returns:
        angle_diff (float): The closest angle between the two angles
        is_supplementary (bool): True if the angle is supplementary, False otherwise
    """
    angle_diff = (theta2 -theta1 + np.pi) % (2 * np.pi) - np.pi
    supplementary_angle = np.pi - np.abs(angle_diff)

    if np.abs(angle_diff) < supplementary_angle:
        return angle_diff, 0
    else:
        new_angle = -1 * np.sign(angle_diff) * supplementary_angle
        return new_angle, np.sign(new_angle)

def calculate_winding_number(eigen_vecs, pad_width, **kwargs):
    """
    Calculates the winding number of the vector field
    Args:
        eigen_vecs (numpy array): Eigenvectors of shape (H, W, 2)
        pad_width (int): Padding width
        method (int): Method to calculate the winding number
    Returns:
        winding_numbers (numpy array): Winding numbers of shape (H, W) ( an entire frame)
    """

    border_indices = [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 2),
        (2, 2),
        (2, 1),
        (2, 0),
        (1, 0),
        (0, 0)
    ]
    angles = np.arctan2(eigen_vecs[:,:,1], eigen_vecs[:,:,0])
    winding_numbers = np.zeros((eigen_vecs.shape[0], eigen_vecs.shape[1]))
    flips = np.zeros((eigen_vecs.shape[0], eigen_vecs.shape[1]))
    for i in range(pad_width, eigen_vecs.shape[0]-pad_width):
        for j in range(pad_width, eigen_vecs.shape[1]-pad_width):

            window = angles[i-1:i+2, j-1:j+2]
            angle_array = [window[border_indices[k]] for k in range(len(border_indices))    ]
            result = [get_closest_angle(angle_array[k + 1], angle_array[k]) for k in range(len(angle_array) - 1)]
            deltas = [r[0] for r in result]
            flip = sum([r[1] for r in result])
            delta_sum = sum(deltas)
            winding_numbers[i,j] = delta_sum / (2 * np.pi)
            flips[i,j] = flip
    return winding_numbers, flips

def find_candidates_by_winding_number(winding_numbers, threshold=0.05):
    positive_mask = np.abs(winding_numbers - 0.5) < threshold
    negative_mask = np.abs(winding_numbers + 0.5) < threshold

    labeled_pos, num_pos = label(positive_mask)
    pos_centroids = center_of_mass(positive_mask, labeled_pos, range(1, num_pos + 1))

    labeled_neg, num_neg = label(negative_mask)
    neg_centroids = center_of_mass(negative_mask, labeled_neg, range(1, num_neg + 1))

    centroids = list(pos_centroids) + list(neg_centroids)
    return np.rint(centroids)