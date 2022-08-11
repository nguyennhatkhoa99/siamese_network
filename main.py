from train import Train
from train_triplet import Train_Triplet
from wok import ImageFolderWithPaths
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torch
if __name__ == '__main__':
    dataset = Train_Triplet()
    # dataset.train_data()
    dataset.train_triplet_data()
    # train.feature_extraction()
    # train.get_test_images("D:/work/siamese_networks/real_data/luan")
    # a = train.load_feature()
    # acc = train.test_data()
    # train.feature_extraction()
    # print("accuracy" , acc)
    # dataset.online_image_retrieval()