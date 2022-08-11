from util import SiameseNetworkDataset, TripletLossDataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys
class DataProcessing():

    def get_train_data(training_dir, train_batch_size):
        folder_dataset = dset.ImageFolder(root=training_dir)
        siamese_dataset_train = SiameseNetworkDataset(imageFolderDataset = folder_dataset,
                                                    transform= transforms.Compose([
                                                                            transforms.Resize((256,256)),
                                                                            transforms.ToTensor(),
                                                                            ]))

        train_dataloader = DataLoader(siamese_dataset_train,
                                shuffle=True,
                                num_workers=4,
                                batch_size=train_batch_size)
        return train_dataloader

    def triplet_train_data(training_dir, train_batch_size):
        folder_dataset = dset.ImageFolder(root=training_dir)
        
        siamese_dataset_train = TripletLossDataset(imageFolderDataset = folder_dataset,
                                                    transform= transforms.Compose([
                                                                            transforms.Resize((256,100)),
                                                                            transforms.ToTensor(),
                                                                            ]))

        train_dataloader = DataLoader(siamese_dataset_train,
                                shuffle=True,
                                num_workers=4,
                                batch_size=train_batch_size)
        return train_dataloader

    def get_test_data(testing_dir):
        folder_dataset_test = dset.ImageFolder(root=testing_dir)
        siamese_dataset_test = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                                transform=transforms.Compose([transforms.Resize((100,100)),
                                                                            transforms.ToTensor()
                                                                            ]))
        test_dataloader = DataLoader(siamese_dataset_test,num_workers=3,batch_size=32,shuffle=True)
        return test_dataloader
