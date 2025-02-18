import os
import pandas as pd

def add_images_to_excel(excel_file, image_folder, output_file):
    # Load the existing Excel file
    df = pd.read_csv(excel_file)  # Change to pd.read_excel() if it's an .xlsx file

    # Get all image filenames in the folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Create new rows for each image
    new_data = [['p9999', 'c10', img] for img in image_files]

    # Append new data
    df_new = pd.DataFrame(new_data, columns=df.columns)
    df = pd.concat([df, df_new], ignore_index=True)

    # Save the updated file
    df.to_csv(output_file, index=False)  # Change to df.to_excel(output_file, index=False) for .xlsx files
    print(f"Updated file saved as: {output_file}")

# Example usage
excel_file = "D:\\grad project\\state_farm_dataset\\driver_imgs_list.csv"  # Change to .xlsx if needed
image_folder = r"D:\grad project\excel"
output_file = "D:\\grad project\\state_farm_dataset\\driver_imgs_list.csv"  # Change to .xlsx if needed

add_images_to_excel(excel_file, image_folder, output_file)
