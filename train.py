from torch.utils import data
from siamese_network import SiameseNetwork, ContrastiveLoss
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

class Train():
    def __init__(self):
        # super(Train, self).__init__()
        self.training_dir = Config.training_dir
        self.testing_dir = Config.testing_dir
        self.train_batch_size = Config.train_batch_size
        self.train_number_epochs = Config.train_number_epochs
        self.model = SiameseNetwork()
        self.learning_rate = 0.0001
        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate )
        self.loss = TripletLoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def get_train_data(self):
        train_data = DataProcessing.get_train_data(self.training_dir, self.train_batch_size)
        return train_data

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

    def train_data(self):
        # try:
        sys.getrecursionlimit(100000)
        self.model = self.model.cuda()
        train_dataloader = self.get_train_data()
        loaded_model , resnet = self.load_model_file()
        counter = []
        loss_history = [] 
        iteration_number= 0
        if loaded_model == False:
            for epoch in range(0,self.train_number_epochs):
                for i, (img0, img1, label) in enumerate(train_dataloader,0):
                    img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
                    img0_feature = resnet(img0).to(self.device)
                    img1_feature = resnet(img1).to(self.device)
                    output1,output2 = self.model(img0_feature,img1_feature)
                    self.optimizer.zero_grad()
                    criterion = self.loss(output1,output2,label)
                    criterion.backward()
                    self.optimizer.step()
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

    
    def train_triplet_data(self):
        # try:
        self.model = self.model.cuda()
        train_dataloader = self.train_triplet_data()
        loaded_model , resnet = self.load_model_file()
        counter = []
        loss_history = [] 
        iteration_number= 0
        if loaded_model == False:
            for epoch in range(0,self.train_number_epochs):
                # print(train_dataloader)
                for i, (anchor, pos, neg) in enumerate(train_dataloader,0):
                    anchor, pos , neg = anchor.cuda(), pos.cuda() , neg.cuda()
                    anchor_feature = resnet(anchor).to(self.device)
                    pos_feature = resnet(pos).to(self.device)
                    neg_feature = resnet(neg).to(self.device)
                    output1, output2, output3 = self.model(anchor_feature,pos_feature, neg_feature)
                    self.optimizer.zero_grad()
                    criterion = self.loss(output1, output2, output3)
                    criterion.backward()
                    self.optimizer.step()
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

    def file_to_feautes_with_batch(self, features, label):
        features_array = []
        labels_array = []
        embeds = np.load(open(features,'rb'), allow_pickle=True)
        names = np.load(open(label,'rb'), allow_pickle=True)
        for batch_code, label in zip(embeds,names):
            for index, person in enumerate (batch_code):
                person = person.squeeze()
                features_array.append(person)
                labels_array.append(label)
        features_array = np.array(features_array)
        labels_array = np.array(labels_array)
        return features_array, labels_array
        
    def file_to_features(self, features, label):
        features_array = []
        labels_array = []
        embeds = np.load(open(features,'rb'), allow_pickle=True)
        names = np.load(open(label,'rb'), allow_pickle=True)
        for feature, label in zip(embeds, names):
            features_array.append(feature)
            labels_array.append(label)
        return features_array, labels_array

    def feature_extraction(self):
        self.load_model_file()
        feature_array = []
        labels_array = []
        test_feature_array = []
        test_labels_array = []
        
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(128),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            'val': transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(112),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            }
        image_datasets = {x: datasets.ImageFolder(os.path.join("D:/work/siamese_networks/", x), data_transforms[x]) for x in ['train', 'val']}
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size= 1, shuffle=True, num_workers=4) for x in ['train', 'val']}
        print(len(dataloaders_dict['train']))
        # train_time_start = time.process_time()
        # for i, data in enumerate(dataloaders_dict['train'],0):
        #     img, label = data
        #     img = img.cuda()
        #     image_time_start = time.process_time()
        #     output =  resnet(img).detach().cpu().squeeze().numpy()
        #     train_time_end = time.process_time() - image_time_start
        #     feature_array.append(output)
        #     labels_array.append(label)
        # # train_time_end = time.process_time() - train_time_start
        # print(train_time_end)
        # for i, data in enumerate(dataloaders_dict['val'],0):
        #     img, label = data
        #     img = img.cuda()
        #     output =  resnet(img).detach().cpu().squeeze().numpy()
        #     test_feature_array.append(output)
        #     test_labels_array.append(label)
        # np.array(feature_array).dump(open('embeding.npy', 'wb'))
        # np.array(labels_array).dump(open('embedname.npy', 'wb'))
        # np.array(test_feature_array).dump(open('val_embeding.npy', 'wb'))
        # np.array(test_labels_array).dump(open('val_embedname.npy', 'wb'))
        train_X, train_Y = self.file_to_features('embeding.npy', 'embedname.npy')
        test_X, test_Y = self.file_to_features('val_embeding.npy', 'val_embedname.npy')
        train_X = np.array(train_X)
        test_X = np.array(test_X)
        train_time_start = time.process_time()
        clf = SVC().fit(train_X, train_Y)
        train_time_end = time.process_time() - train_time_start
        print(train_time_end)
        filename = 'finalized_model.sav'
        pickle.dump(clf, open(filename, 'wb'))
        loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
        score = loaded_model.score(test_X, test_Y)
        print(score)
        # tsne = TSNE(n_components=2, learning_rate=200, perplexity=50, method = "barnes_hut", random_state=np.random.RandomState(1)).fit_transform(train_X)
        # clf.fit(tsne, train_Y)
        # plt.figure(figsize=(8, 5))
        # plot_decision_regions(X = tsne, y = train_Y, clf = clf, legend = 1)
        # plt.title("Nhân RBF")
        # plt.show()
    

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

    def online_image_retrieval(self):
        cv2.namedWindow("preview")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        loaded_model = pickle.load(open("finalized_model.sav", 'rb'))
        transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        ])
        video_capture = cv2.VideoCapture(0)
        frame_interval = 10  # Number of frames after which to run face detection
        fps_display_interval = 5  # seconds
        frame_rate = 0
        frame_count = 0
        if video_capture.isOpened(): # try to get the first frame
            rval, frame = video_capture.read()
        else:
            rval = False
        start_time = time.time()
        preds_arr = []
        while rval:
            cv2.imshow("preview", frame)
            rval, frame = video_capture.read()
            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                final_pred = max(preds_arr, key= preds_arr.count)
                print(final_pred, "\t", np.array(preds_arr) )
                break
            if (frame_count % frame_interval) == 0:
                img = transforms.ToPILImage()(frame.squeeze()).convert("RGB")
                img0 = transform(img)
                img0 = img0.unsqueeze(0)
                img0 = img0.cuda()
                embed = resnet(img0).detach().cpu().squeeze().numpy()
                pred = loaded_model.predict(embed.reshape(1,-1))
                preds_arr.append(pred)
                print(pred)
                # break
                end_time = time.time()
                if (end_time - start_time) > fps_display_interval:
                    frame_rate = int(frame_count / (end_time - start_time))
                    start_time = time.time()
                    frame_count = 0
            # add_overlays(frame, faces, frame_rate, colors)
            frame_count += 1
        cv2.destroyWindow("preview")
