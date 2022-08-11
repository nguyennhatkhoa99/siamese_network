from torch.utils import data
from siamese_network import SiameseNetwork
from data_processing import DataProcessing
import torch
import torch.nn as nn
from configuration import Config
from torch import optim
import torch.nn.functional as F
from pathlib import Path
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from torch.utils.data import DataLoader
from wok import ImageFolderWithPaths
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from img2vec_pytorch import Img2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from PIL import Image
import pickle
import pathlib
import time
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import traceback
import sys

 

def imshow(img,text=None,should_save=False):
        npimg = img.numpy()
        plt.axis("off")
        if text:
            plt.text(75, 8, text, style='italic',fontweight='bold',
                bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()    

def show_plot(iteration,loss):
        plt.plot(iteration,loss)
        plt.show()

class Train_Triplet():
    def __init__(self):
        # super(Train, self).__init__()
        self.training_dir = Config.training_dir
        self.testing_dir = Config.testing_dir
        self.train_batch_size = Config.train_batch_size
        self.train_number_epochs = Config.train_number_epochs
        self.model = SiameseNetwork()
        self.learning_rate = 0.0001
        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate )
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = 0.5)
        # self.loss = TripletLoss()
        # self.loss = torch.nn.MarginRankingLoss(margin= 1.0)
        self.loss = nn.TripletMarginLoss(margin = 0.1)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    def get_triplet_train_data(self):
        train_data = DataProcessing.triplet_train_data(self.training_dir, self.train_batch_size)
        return train_data
        
    def get_test_data(self):
        test_data = DataProcessing.get_test_data(self.testing_dir)
        return test_data
    
    def load_model_file(self):
        weight_model_saved_filename = 'model_weights.pth'
        weight_opt_saved_filename = 'model_optimizer.pth'
        weight_model_saved_file = Path(weight_model_saved_filename)
        weight_opt_saved_file = Path(weight_model_saved_filename)
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        if weight_model_saved_file.exists() and weight_opt_saved_file.exists():
            self.model.load_state_dict(torch.load(weight_model_saved_filename))
            self.optimizer.load_state_dict(torch.load(weight_opt_saved_filename))
            return (self.model, resnet)
        else: 
            return (False, resnet)
    
    def train_triplet_data(self):
        # try:
        self.model = self.model.cuda()
        train_dataloader = self.get_triplet_train_data()
        loaded_model , resnet = self.load_model_file()
        counter = []
        loss_history = [] 
        iteration_number= 0

        if loaded_model == False:
            for epoch in range(0,self.train_number_epochs):
                for i, (anchor, pos, neg) in enumerate(train_dataloader,0):
                    anchor, pos , neg = anchor.cuda(), pos.cuda() , neg.cuda()
                    anchor_feature = resnet(anchor).to(self.device)
                    pos_feature = resnet(pos).to(self.device)
                    neg_feature = resnet(neg).to(self.device)
                    anchor_feature, pos_feature, neg_feature = Variable(anchor_feature), Variable(pos_feature), Variable(neg_feature)
                    output1, output2, output3 = self.model(anchor_feature,pos_feature, neg_feature)
                    self.optimizer.zero_grad()
                    criterion = self.loss(output1, output2, output3)
                    criterion.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    if i % 10 == 0 :
                        iteration_number +=10
                        counter.append(iteration_number)
                        loss_history.append(criterion.item())
                print("Epoch number {}\n Current loss {}\n".format(epoch,criterion.item()))
            torch.save(self.model.state_dict(), 'model_weights.pth')
            torch.save(self.optimizer.state_dict(), 'model_optimizer.pth')
            show_plot(counter,loss_history)
        # except Exception:
        #     print(traceback.format_exc())
        
    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def get_test_images(self, image_link, model_file_name = "finalized_model.sav"):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        loaded_model = pickle.load(open(model_file_name, 'rb'))
        test_X, test_Y = self.file_to_features('val_embeding.npy', 'val_embedname.npy')
        score = loaded_model.score(test_X, test_Y)
        print(score)
        data_dir = pathlib.Path(image_link)
        transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        ])
        print(loaded_model.classes_)
        for filename in data_dir.glob('**/*.jpg'):
            
            img = Image.open(filename)
            img0 = transform(img)
            img0 = img0.unsqueeze(0)
            img0 = img0.cuda()
            output = resnet(img0).detach().cpu().squeeze().numpy()
            pred = loaded_model.predict(output.reshape(1,-1))
            print(pred, filename)
            
    def test_images(self):
        loaded_model , resnet = self.load_model_file()
        
