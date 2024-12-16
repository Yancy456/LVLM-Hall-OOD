import os
import glob
import numpy as np


class StoreData:
    def __init__(self, path) -> None:
        create_folder(path)
        self.path = path
        self.idx = 0
        self.remove_old_files()

    def remove_old_files(self):
        # Get all files in the directory
        files = glob.glob(os.path.join(self.path, '*'))

        # Remove each file
        for file in files:
            os.remove(file)

        print(f"All files in the directory {self.path} have been removed.")

    def store(self, dict):
        name = os.path.join(self.path, f'{self.idx}')
        np.save(name, dict)
        self.idx += 1


class ReadData:
    def __init__(self, path) -> None:
        self.path = path
        self.max, self.min = self.get_max_min_file_names(self.path)

    def read_all(self):
        results = []
        for idx in range(self.min, self.max+1):
            name = os.path.join(self.path, f'{idx}.npy')
            if os.path.isfile(name):
                results.append(np.load(name, allow_pickle=True).item())
            else:
                continue
        return results

    def get_max_min_file_names(self, folder_path):
        # Get a list of files in the folder
        file_names = os.listdir(folder_path)

        # Filter out directories, only keep files
        file_names = [
            f[:-4] for f in file_names if os.path.isfile(os.path.join(folder_path, f))]

        # Convert file names to numbers
        file_numbers = [int(f) for f in file_names if f.isdigit()]

        if not file_numbers:
            return None, None

        # Find the maximum and minimum file numbers
        max_file_name = max(file_numbers)
        min_file_name = min(file_numbers)

        return max_file_name, min_file_name


def create_folder(path):
    if not os.path.exists(path):
        # If it does not exist, create it
        os.makedirs(path)
