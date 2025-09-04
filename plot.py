
import pandas as pd
import matplotlib.pyplot as plt

# Load Excel with correct header row (adjust header row if needed)
file_path = "Server testing results (1).xlsx"
df = pd.read_excel(file_path)  # try header=2 if needed

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Print column names for verification (optional)
print("Detected columns:", df.columns.tolist())

# Rename columns if necessary
df.rename(columns={
    df.columns[0]: "Code",
    df.columns[1]: "Processors",
    df.columns[2]: "Run",
    df.columns[3]: "Read Time",
    df.columns[4]: "Main Program Time",
    df.columns[5]: "Total Time"
}, inplace=True)

# Define datasets and metrics
datasets = ['data_64_64_64_3.bin.txt', 'data_64_64_96_7.bin.txt']
metrics = ['Read Time', 'Main Program Time', 'Total Time']

# Loop over both datasets
for dataset in datasets:
    df_subset = df[df["Code"] == dataset]
    # Group by number of processors and average over runs
    avg_df = df_subset.groupby("Processors")[metrics].mean().reset_index()

    # Sort by processor count (ensure numeric order)
    avg_df['Processors'] = avg_df['Processors'].astype(int)

    avg_df = avg_df.sort_values(by="Processors")

    # Plot each metric
    for metric in metrics:
        plt.figure()
        plt.plot(avg_df["Processors"], avg_df[metric], marker='o', linestyle='-', color='blue')
        plt.xlabel("Number of Processors")
        plt.ylabel(metric)
        plt.title(f"{metric} vs Number of Processors\n({dataset})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{metric.replace(' ', '_')}_{dataset.replace('.bin.txt', '')}.png")
        plt.show()



import pandas as pd
import matplotlib.pyplot as plt

# Load Excel file
file_path = "comparative.xlsx"
df = pd.read_excel(file_path)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Rename columns if necessary
df.rename(columns={
    df.columns[0]: "File",
    df.columns[1]: "Processors",
    df.columns[2]: "Run",
    df.columns[3]: "Read Time",
    df.columns[4]: "Main Program Time",
    df.columns[5]: "Total Time"
}, inplace=True)

# Make sure the datasets are strings
df["File"] = df["File"].astype(str)

# Define datasets with their custom legend names
datasets = ['1', '2', '3', '4', '5']
custom_legends = ['Sequential Read', 'Parallel input method 3', 'Parallel input method 4',
                 'Parallel input method 2', 'Parallel input method 1']

# Create a dictionary to map dataset IDs to their custom legends
legend_map = dict(zip(datasets, custom_legends))

metrics = ['Read Time', 'Main Program Time', 'Total Time']

# Define distinct colors for the 5 datasets
colors = ['blue', 'red', 'green', 'purple', 'orange']

# Plot each metric with all datasets overlapped
for metric in metrics:
    plt.figure(figsize=(10, 6))

    for i, dataset in enumerate(datasets):
        # Check if this dataset exists in the dataframe
        df_subset = df[df["File"] == dataset]

        if len(df_subset) > 0:
            # Group by number of processors and average over runs
            avg_df = df_subset.groupby("Processors")[metrics].mean().reset_index()

            # Sort by processor count (ensure numeric order)
            avg_df['Processors'] = avg_df['Processors'].astype(int)
            avg_df = avg_df.sort_values(by="Processors")

            # Use the custom legend name for this dataset
            legend_name = legend_map[dataset]

            # Plot this dataset with its assigned color and custom legend
            plt.plot(avg_df["Processors"], avg_df[metric],
                    marker='o', linestyle='-', color=colors[i],
                    label=legend_name)
        else:
            print(f"Warning: Dataset '{dataset}' not found in the Excel file")

    plt.xlabel("Number of Processors")
    plt.ylabel(metric + " (seconds)")
    plt.title(f"{metric} vs Number of Processors")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{metric.replace(' ', '_')}_comparison.png")
    plt.show()

