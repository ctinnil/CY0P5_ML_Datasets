import os
import pandas as pd

encoding = 'ISO-8859-1'  # Specify the correct encoding

# Directory where the CSV files are located
# Get user input with a prompt
csv_folder = input("Provide folder path: ")

# Check if the input is empty
if not csv_folder:
    # If empty, use the default value
    csv_folder = os.getcwd()

# List to store individual DataFrames
dfs = []

# Iterate over the CSV files in the folder
for filename in os.listdir(csv_folder):
    if filename.endswith(".csv"):

	# Read the CSV file with the specified encoding
        try:
            df = pd.read_csv(os.path.join(csv_folder, filename), encoding=encoding)
        except UnicodeDecodeError:
            print(f'Error: Unable to read {file_path} with encoding {encoding}')
        # Read each CSV file into a DataFrame
        #df = pd.read_csv(os.path.join(csv_folder, filename))
        dfs.append(df)

# Concatenate all DataFrames into one
result_df = pd.concat(dfs, ignore_index=True)

# Output file name for the concatenated CSV
output_csv = os.path.join(csv_folder, "concatenated_result.csv")

# Save the concatenated DataFrame to a new CSV file
result_df.to_csv(output_csv, index=False)

print(f"Concatenated CSV saved to {output_csv}")
