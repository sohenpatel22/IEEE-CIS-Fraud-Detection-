import kagglehub
import os
import shutil

def download_dataset():
    # Download latest version
    path = kagglehub.competition_download('ieee-fraud-detection')

    print("Path to competition files:", path)

    target_dir = "data/raw/"

    os.makedirs(target_dir, exist_ok = True)

    for file in os.listdir(path):
        if file.endswith(".csv"):

            shutil.copy2(os.path.join(path,file),target_dir)

    print("Data copied to:", target_dir)

if __name__ == "__main__":
    download_dataset()
