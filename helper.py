import os
import zipfile
import shutil
import gdown
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt


DATA_DIR = 'data/'
DOCS_DIR = 'docs/'
LABELS_FILE = 'All_labels.txt'
DATA_URL = 'https://drive.google.com/uc?id=1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf'
ZFILE = 'SCUT-FBP5500_v2.1.zip'

def extract_zipfile():
    with zipfile.ZipFile(ZFILE) as zip_file:
        for member in zip_file.namelist():
            filename = os.path.basename(member)
            # skip directories, all non-jpgs, except labels
            if filename.endswith(".jpg") or filename == LABELS_FILE:
                # copy file (taken from zipfile's extract)
                source = zip_file.open(member)
                target = open(os.path.join(DATA_DIR, filename), "wb")
                with source, target:
                    shutil.copyfileobj(source, target)

def download_data():
    # Download Dataset
    if os.path.isfile(ZFILE) or os.path.isfile(DATA_DIR+LABELS_FILE):
        print('data already downloaded')
    else:
        print ("data does not exist. downloading it.")
        gdown.download(DATA_URL, ZFILE, quiet=False)
    # Extract ZipFile
    if os.path.isfile(DATA_DIR+LABELS_FILE):
        print("data already extracted.")
    else:
        print("extracting data.")
        if not os.path.exists(DATA_DIR):
            os.mkdir(DATA_DIR)
        extract_zipfile()
        os.remove(ZFILE)
        
def preprocess_image(image,target_size):
    return cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),target_size) / .255

def create_dataset(target_size):
    labels_dict = get_labels_dict()
    img_files = glob.glob(DATA_DIR+'*.jpg')
    print(f'reading {len(img_files)} images into dataset')
    
    # Create a generator that yields batches of images and labels
    def data_generator(batch_size=32):
        while True:
            # Shuffle the files
            np.random.shuffle(img_files)
            
            for i in range(0, len(img_files), batch_size):
                batch_files = img_files[i:i + batch_size]
                batch_images = []
                batch_labels = []
                
                for f in batch_files:
                    img = preprocess_image(cv2.imread(f), target_size)
                    batch_images.append(img)
                    batch_labels.append(labels_dict[os.path.split(f)[-1]])
                
                yield np.array(batch_images), np.array(batch_labels)
    
    return data_generator, len(img_files)

def get_labels_dict():
    labels_dict = {}
    with open(DATA_DIR + LABELS_FILE) as fp:
        for line in fp:
            img,label = line.split(' ', 1)
            labels_dict[img] = float(label)
    return labels_dict

def plot_metrics(history, model_name, stage):
    f,(ax1) = plt.subplots(1, 1, figsize=(15,7))
    f.suptitle(f'Stage {stage} Model "{model_name}" training Metrics')
    ax1.plot(history.history["loss"], color='darkblue', label="Train")
    ax1.plot(history.history["val_loss"], color='darkred', label="Test")
    ax1.set_title('Loss (Mean Squared Error) over epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()
    plt.savefig(f'{DOCS_DIR}metrics_stage_{stage}_{model_name}')
    plt.show()


if __name__ == "__main__":
    download_data()