import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import tkinter as tk
from tkinter import filedialog, simpledialog


def load_and_prepare_data(df, class_column, class_a, class_b, feature_columns=None):
    # Filter to two-class subset
    df_sub = df[df[class_column].isin([class_a, class_b])].copy()
    df_sub['Label'] = df_sub[class_column].apply(lambda x: 'ClassA' if x == class_a else 'ClassB')

    # Determine feature columns
    if feature_columns is None:
        feature_columns = df_sub.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in feature_columns if col != class_column]

    # Normalize features to [0,1]
    df_sub[feature_columns] = (df_sub[feature_columns] - df_sub[feature_columns].min()) / \
                               (df_sub[feature_columns].max() - df_sub[feature_columns].min())

    return df_sub, feature_columns


def create_split_data(df_sub, feature_columns, split_fraction=0.5):
    num_features = len(feature_columns)
    split_count = int(num_features * split_fraction)
    overlap_indices = sorted(np.random.choice(range(num_features), size=split_count, replace=False))
    non_overlap_indices = sorted(set(range(num_features)) - set(overlap_indices))

    df_classA = df_sub[df_sub['Label'] == 'ClassA'].copy()
    df_classB = df_sub[df_sub['Label'] == 'ClassB'].copy()

    for i in non_overlap_indices:
        col = feature_columns[i]
        margin = 0.02
        gap_size = np.random.uniform(0.05, 0.15)
        available = 1.0 - 2 * margin - gap_size
        classA_frac = np.random.uniform(0.45, 0.55)
        classB_frac = 1.0 - classA_frac

        b_max = margin + available * classB_frac
        b_min = margin
        a_min = b_max + gap_size
        a_max = a_min + available * classA_frac

        df_classB[col] = np.interp(df_classB[col], (df_classB[col].min(), df_classB[col].max()), (b_min, b_max))
        df_classA[col] = np.interp(df_classA[col], (df_classA[col].min(), df_classA[col].max()), (a_min, a_max))

    return df_classA, df_classB, overlap_indices


def shift_split_axes(df_classA, df_classB, feature_columns, overlap_indices):
    df_a_shifted = df_classA.copy()
    df_b_shifted = df_classB.copy()
    for i in overlap_indices:
        col = feature_columns[i]
        df_a_shifted[col] = np.interp(df_a_shifted[col], (0, 1), (0.5, 1.5))
        df_b_shifted[col] = np.interp(df_b_shifted[col], (0, 1), (-0.5, 0.5))
    return df_a_shifted, df_b_shifted


def manual_parallel_coordinates_split_with_axis_ranges(
    df_blue, df_red,
    top_blue, bottom_blue, top_red, bottom_red,
    feature_columns, xtick_labels, overlap_indices, title, blue_on_top=True
):
    plt.figure(figsize=(14, 6))
    x = np.arange(len(feature_columns))

    # Plot all data lines - only use feature columns, not Label
    if blue_on_top:
        for _, row in df_red[feature_columns].iterrows():
            plt.plot(x, row.values, color='red', alpha=0.2)
        for _, row in df_blue[feature_columns].iterrows():
            plt.plot(x, row.values, color='blue', alpha=0.2)
    else:
        for _, row in df_blue[feature_columns].iterrows():
            plt.plot(x, row.values, color='blue', alpha=0.2)
        for _, row in df_red[feature_columns].iterrows():
            plt.plot(x, row.values, color='red', alpha=0.2)

    # Outlines
    blue_outline = top_blue + bottom_blue + [top_blue[0]]
    red_outline = top_red + bottom_red + [top_red[0]]
    blue_x, blue_y = zip(*blue_outline)
    red_x, red_y = zip(*red_outline)

    # Outline drawing
    if blue_on_top:
        plt.plot(red_x, red_y, linestyle='--', linewidth=3, color='red')
        plt.plot(blue_x, blue_y, linestyle='--', linewidth=3, color='blue')
    else:
        plt.plot(blue_x, blue_y, linestyle='--', linewidth=3, color='blue')
        plt.plot(red_x, red_y, linestyle='--', linewidth=3, color='red')

    # Draw vertical bars for axis ranges
    for i in range(len(feature_columns)):
        # Default: full axis
        if i not in overlap_indices:
            plt.plot([i, i], [0, 1], color='black', linewidth=1.2)
        else:
            # Split axes: blue on top, red on bottom
            plt.plot([i, i], [0.5, 1.5], color='blue', linewidth=1.2)
            plt.plot([i, i], [-0.5, 0.5], color='red', linewidth=1.2)

    plt.xticks(x, xtick_labels, rotation=45)
    plt.xlim(-0.5, len(feature_columns) - 0.5)
    plt.ylim(-0.6, 1.6)
    plt.grid(True)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def manual_parallel_coordinates_no_split(
    df_blue, df_red, top_blue, bottom_blue, top_red, bottom_red,
    feature_columns, xtick_labels, title, blue_on_top=True
):
    plt.figure(figsize=(14, 6))
    x = np.arange(len(feature_columns))

    if blue_on_top:
        for _, row in df_red[feature_columns].iterrows():
            plt.plot(x, row.values, color='red', alpha=0.2)
        for _, row in df_blue[feature_columns].iterrows():
            plt.plot(x, row.values, color='blue', alpha=0.2)
    else:
        for _, row in df_blue[feature_columns].iterrows():
            plt.plot(x, row.values, color='blue', alpha=0.2)
        for _, row in df_red[feature_columns].iterrows():
            plt.plot(x, row.values, color='red', alpha=0.2)

    # Outlines
    blue_outline = top_blue + bottom_blue + [top_blue[0]]
    red_outline = top_red + bottom_red + [top_red[0]]
    blue_x, blue_y = zip(*blue_outline)
    red_x, red_y = zip(*red_outline)

    if blue_on_top:
        plt.plot(red_x, red_y, linestyle='--', linewidth=3, color='red')
        plt.plot(blue_x, blue_y, linestyle='--', linewidth=3, color='blue')
    else:
        plt.plot(blue_x, blue_y, linestyle='--', linewidth=3, color='blue')
        plt.plot(red_x, red_y, linestyle='--', linewidth=3, color='red')

    plt.xticks(x, xtick_labels, rotation=45)
    plt.xlim(-0.5, len(feature_columns) - 0.5)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def select_classes(df, class_column):
    """Prompt user to select two classes to compare."""
    root = tk.Tk()
    root.withdraw()
    
    # Get unique classes
    unique_classes = df[class_column].unique().tolist()
    
    # Create dialogs to select two classes
    class_a = simpledialog.askstring("Select First Class", 
                                    f"Enter the name of the first class from: {', '.join(unique_classes)}")
    
    # Validate first selection
    while class_a not in unique_classes:
        class_a = simpledialog.askstring("Invalid Selection", 
                                        f"Please enter a valid class name from: {', '.join(unique_classes)}")
    
    # Get remaining classes
    remaining_classes = [c for c in unique_classes if c != class_a]
    
    # Select second class
    class_b = simpledialog.askstring("Select Second Class", 
                                    f"Enter the name of the second class from: {', '.join(remaining_classes)}")
    
    # Validate second selection
    while class_b not in remaining_classes:
        class_b = simpledialog.askstring("Invalid Selection", 
                                        f"Please enter a valid class name from: {', '.join(remaining_classes)}")
    
    return class_a, class_b


if __name__ == "__main__":
    # Create root window and withdraw it
    root = tk.Tk()
    root.withdraw()
    
    # Load data with file dialog
    file_path = filedialog.askopenfilename(title="Select Dataset File", 
                                          filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    
    if not file_path:
        print("No file selected. Exiting.")
        exit()
    
    # Load the dataset
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {', '.join(df.columns)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()
    
    # Find the class column (case-insensitive)
    class_column = None
    for col in df.columns:
        if col.lower() == 'class':
            class_column = col
            break
    
    if class_column is None:
        print("Error: No column named 'class' (case-insensitive) found in the dataset.")
        exit()
    
    print(f"Using '{class_column}' as the class column")
    
    # Select classes to compare
    class_a, class_b = select_classes(df, class_column)
    print(f"Selected classes to compare: {class_a} and {class_b}")
    
    # Process data with proper normalization
    df_sub, feature_columns = load_and_prepare_data(df, class_column, class_a, class_b)
    print(f"Processed data with {len(feature_columns)} features")
    
    # Split dataframe by class
    df_blue = df_sub[df_sub['Label'] == 'ClassA']
    df_red = df_sub[df_sub['Label'] == 'ClassB']
    
    # Create split data
    df_blue_split, df_red_split, overlap_indices = create_split_data(df_sub, feature_columns)
    
    # Shift split axes
    df_blue_split_axes, df_red_split_axes = shift_split_axes(df_blue_split, df_red_split, feature_columns, overlap_indices)
    
    # Compute outlines for split axes plot
    blue_top_split = df_blue_split_axes[feature_columns].max()
    blue_bottom_split = df_blue_split_axes[feature_columns].min()
    red_top_split = df_red_split_axes[feature_columns].max()
    red_bottom_split = df_red_split_axes[feature_columns].min()
    
    x = np.arange(len(feature_columns))
    blue_top_line_split = list(zip(x, blue_top_split.values))
    blue_bottom_line_split = list(zip(x[::-1], blue_bottom_split.values[::-1]))
    red_top_line_split = list(zip(x, red_top_split.values))
    red_bottom_line_split = list(zip(x[::-1], red_bottom_split.values[::-1]))
    
    # Compute outlines for normal plot
    blue_top_full = df_blue[feature_columns].max()
    blue_bottom_full = df_blue[feature_columns].min()
    red_top_full = df_red[feature_columns].max()
    red_bottom_full = df_red[feature_columns].min()
    
    blue_top_line_full = list(zip(x, blue_top_full.values))
    blue_bottom_line_full = list(zip(x[::-1], blue_bottom_full.values[::-1]))
    red_top_line_full = list(zip(x, red_top_full.values))
    red_bottom_line_full = list(zip(x[::-1], red_bottom_full.values[::-1]))
    
    # Create xtick labels with asterisk for overlap features
    xtick_labels = [f'{col}*' if i in overlap_indices else col for i, col in enumerate(feature_columns)]
    
    # Plot with split axes
    manual_parallel_coordinates_split_with_axis_ranges(
        df_blue=df_blue_split_axes,
        df_red=df_red_split_axes,
        top_blue=blue_top_line_split,
        bottom_blue=blue_bottom_line_split,
        top_red=red_top_line_split,
        bottom_red=red_bottom_line_split,
        feature_columns=feature_columns,
        xtick_labels=xtick_labels,
        overlap_indices=overlap_indices,
        title=f"Split Axis Parallel Coordinates: {class_a} vs {class_b}",
        blue_on_top=True
    )
    
    # Plot without axis split
    manual_parallel_coordinates_no_split(
        df_blue=df_blue,
        df_red=df_red,
        top_blue=blue_top_line_full,
        bottom_blue=blue_bottom_line_full,
        top_red=red_top_line_full,
        bottom_red=red_bottom_line_full,
        feature_columns=feature_columns,
        xtick_labels=xtick_labels,
        title=f"Normal Parallel Coordinates: {class_a} vs {class_b}",
        blue_on_top=True
    )
