import urllib.request, os
import numpy as np
import torch

BASE_URL = "http://yann.lecun.com/exdb/mnist/"

FILE = {
    "train_images" : "train-images.idx3-ubyte",
    "train_labels" : "train-labels.idx1-ubyte",
    "test_images" : "t10k-images.idx3-ubyte",
    "test_labels" : "t10k-labels.idx1-ubyte"
}

def download(name):
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", FILE[name])
    if not os.path.exists(path):   #checks if file is already downloaded
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(BASE_URL + FILE[name], path)
    return path

def load_images(path):
    with open(path,"rb") as f:
        #skips header and useless files
        f.read(16)        
        #reads byte code
        data = np.frombuffer(f.read(), dtype = np.uint8)
        #reshapes into pytorch tensor, and normalizes into 0-1 instead of 0-255
    return torch.from_numpy(data.reshape(-1,28,28).copy()).float()/255.0

def load_labels(path):
    with open(path,"rb") as f:
        f.read(8)
        data = np.frombuffer(f.read(), dtype = np.uint8)
        #converts into 64 bit ints
    return torch.from_numpy(data.copy()).long()


def load_all():
    return (
        load_images(download("train_images")), 
        load_labels(download("train_labels")),
        load_images(download("test_images")),
        load_labels(download("test_labels"))
    )


def main():
    img1, lab1,img2,lab2 = load_all()

    print(img1)
    print(img2)
    print(lab1)
    print(lab2)


if __name__ == "__main__":
    main()