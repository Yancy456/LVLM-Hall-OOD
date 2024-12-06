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
        self.idx = 0

    def read_all(self):
        results = []
        while True:
            name = os.path.join(self.path, f'{self.idx}.npy')
            if os.path.isfile(name):
                results.append(np.load(name, allow_pickle=True).item())
                self.idx += 1
            else:
                break
        return results


def create_folder(path):
    if not os.path.exists(path):
        # If it does not exist, create it
        os.makedirs(path)
