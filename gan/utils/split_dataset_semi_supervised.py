import os
import shutil

def organize_wav_files(source_dir, destination_dir):
    # Ensure the source directory exists
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist.")
        return

    # Ensure the destination directory exists, create if not
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    
    # List all files in the source directory
    files = [f for f in os.listdir(source_dir) if f.endswith('.wav')]
    
    for file in files:
        # Extract the first four characters from the filename
        folder_name = file[:4]
        
        # Create a new folder path in the destination directory
        new_folder_path = os.path.join(destination_dir, folder_name)
        
        # Create the folder if it doesn't already exist
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
        
        # Build the full old and new file paths
        old_file_path = os.path.join(source_dir, file)
        new_file_path = os.path.join(new_folder_path, file)
        
        # Copy the file to the new folder
        shutil.copy(old_file_path, new_file_path)
        print(f'Copied "{file}" to "{new_folder_path}"')


def flatten_folder_structure(base_folder):
    # Iterate over all the items in the base folder
    for item in os.listdir(base_folder):
        item_path = os.path.join(base_folder, item)
        
        # Check if this item is a directory
        if os.path.isdir(item_path):
            # List all files in this subdirectory
            for file in os.listdir(item_path):
                file_path = os.path.join(item_path, file)
                
                # Construct the new path for this file directly in the base folder
                new_file_path = os.path.join(base_folder, file)
                
                # Move the file to the base folder
                shutil.move(file_path, new_file_path)
            
            # Optionally, remove the now empty subdirectory
            os.rmdir(item_path)

# Usage example:
folder_x = '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/clean_raw_speakers/unsuper50p'
flatten_folder_structure(folder_x)


# # Specify the directory containing the original .wav files
# source_directory = '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/noisy_raw'
# # Specify the destination directory for organized files
# destination_directory = '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/noisy_raw_speakers'
# organize_wav_files(source_directory, destination_directory)