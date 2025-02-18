import os

# Define the folder containing the text files
folder_path = r"D:\grad project\customed dataset\HOW object detection dataset\hands on wheel labels\kaggle\working\out_handsON_labels"  # Change this to your actual path

# Loop through all .txt files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):
        file_path = os.path.join(folder_path, file_name)

        # Read the file content
        with open(file_path, "r") as file:
            lines = file.readlines()

        # Process each line
        modified_lines = []
        for line in lines:
            parts = line.strip().split()  # Split by spaces
            if parts and parts[0] == '0':  # If the first element is '0', change it to '1'
                parts[0] = '1'
            modified_lines.append(" ".join(parts))

        # Write the modified content back to the file
        with open(file_path, "w") as file:
            file.write("\n".join(modified_lines))

print("All files have been updated successfully!")
