# # def feature_extraction(self):
    #     torch.cuda.empty_cache()
    #     model = torchvision.models.resnet50(pretrained=True)
    #     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #     modules=list(model.children())[:-1]
    #     resnet50= nn.Sequential(*modules)
    #     for p in resnet50.parameters():
    #         p.requires_grad = False
    #     resnet50 =resnet50.eval().to(device)
    #     # model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    #     data_transforms = {
    #         'train': transforms.Compose([
    #             transforms.RandomResizedCrop(224),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #         ]),
    #         'val': transforms.Compose([
    #             transforms.Resize(224),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #         ]),
    #         }
    #     image_datasets = {x: datasets.ImageFolder(os.path.join("D:/work/siamese_networks/", x), data_transforms[x]) for x in ['train', 'val']}
    #     dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size= self.train_batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
    #     people_array = []
    #     label_array = []
    #     for i, data in enumerate(dataloaders_dict['train'],0):
    #         img1 , label = data
    #         label = label.data
    #         label = label.numpy()
    #         img1 = img1.to(device)
    #         features_var = resnet50(img1).detach().cpu().numpy()
    #         # features_var = model(img1).detach().cpu().numpy()
    #         people_array.append(features_var)
    #         label_array.append(label)
    #     np.array(people_array).dump(open('embeding.npy', 'wb'))
    #     np.array(label_array).dump(open('embedname.npy', 'wb'))
    #     embed = np.load(open('embeding.npy','rb'), allow_pickle=True)
    #     names = np.load(open('embedname.npy','rb'),allow_pickle=True)
    #     features_array = []
    #     labels = []
    #     for batch_code, label in zip(embed,names):
    #         for index, person in enumerate (batch_code):
    #             person = person.squeeze()
    #             features_array.append(person)
    #             labels.append(label[index])
    #     features_array = np.array(features_array)
    #     mean_euclidean_distance = 0
    #     neighbor = 0
    #     # for j in range(pca_features.shape[0]):
    #     j = 120
    #     for i in range (features_array.shape[0]):
    #         # if int(labels[i]) == int(labels[j]):           
    #         euclidean_distance = F.pairwise_distance(torch.Tensor([features_array[j]]), torch.Tensor([features_array[i]]))   
    #         if euclidean_distance < 9:
    #             neighbor += 1
    #             if int(labels[i]) != int(labels[j]): 
    #                 print(euclidean_distance, labels[i], labels[j])
    #             mean_euclidean_distance += euclidean_distance
    #     # print(len)
    #     # mean_euclidean_distance = mean_euclidean_distance.data / neighbor
    #     # print(mean_euclidean_distance.data)
    #     print(neighbor)



# temp_network = self.model
        # temp_network.load_state_dict(torch.load(Path("model_weights.pth")))
        # temp_network.eval()
        # print(len(dataloaders_dict['val']))
        # for test_index, test_value in enumerate(test_dataloader, 0):
        #     torch.cuda.empty_cache()
        #     # print(test_index)
        #     test_img, test_link, test_label = test_value
        #     neighhbor = []
        #     distance = []
        #     high_distance = []
        #     wrong_neighhbor = []
        #     true_index = []
        #     count = 0
        #     min_dist = 9999999
        #     min_index_label = 0
        #     print("Link test", test_link)
            # dataiter = iter(dataloader)
        #     for train_index, train_value in enumerate(dataloader, 0):
        #         count += 1
        #         train_img, link_img, train_label = train_value
        #         with torch.no_grad():      
        #             output1,output2 = self.model(Variable(torch.Tensor(train_img)).cuda(),Variable(torch.Tensor(test_img)).cuda()) 
        #             # output1 = temp_network.forward_once(Variable(torch.Tensor(train_img)).cuda()).detach().cpu()
        #             # output2 = temp_network.forward_once(Variable(torch.Tensor(test_img)).cuda()).detach().cpu()
        #         euclidean_distance = F.pairwise_distance(output1, output2)
        #         if(euclidean_distance.item() < 0.2):
        #             pred = train_label
        #             distance.append(euclidean_distance.item())
        #             neighhbor.append(pred.numpy()[0])
        #             true_index.append(train_index)
        #             print("link train ", link_img)
        #             if len(neighhbor) == 6:
        #                 break
        #         elif(euclidean_distance.item() > 0.2):
        #             wrong_pred = train_label
        #             high_distance.append(euclidean_distance.item())
        #             wrong_neighhbor.append(wrong_pred.numpy()[0])
        #         if (euclidean_distance < min_dist):
        #             min_index_label = train_label

        #         # if pred == actual:
        #         #         correct += 1
        #         # else: incorrect += 1
        #     if  len(neighhbor) != 0:
        #         final_pred = max(neighhbor, key= neighhbor.count)
        #     else: 
        #         final_pred = min_index_label
        #     print("True neighbor", neighhbor, "\n", distance)
        #     # print("Index", true_index)
        #     # print("len_wrong", len(wrong_neighhbor))
        #     # print("Wrong neighbor", wrong_neighhbor[:10], "\n", high_distance[:10])
        #     if final_pred == test_label:
        #         correct += 1
        #     print(test_index, " predict: ", final_pred, " actual: " ,test_label, "\n")
        # # print(correct)
        # # print(incorrect)

     #test
        # for i, data in enumerate(dataloaders_dict['val'],0):
        #     img, label = data
        #     img = img.cuda()
        #     temp_network = self.model
        #     label = label.data
        #     label = label.numpy()
        #     torch.cuda.empty_cache()
        #     output = temp_network.forward_once(img).detach().cpu().numpy()
        #     test_feature_array.append(output)
        #     test_labels_array.append(label)


            # for i, data in enumerate(dataloaders_dict['train'],0):
        #     img, label = data
        #     img = img.cuda()
        #     img = transforms.ToPILImage()(img.squeeze()).convert("RGB")
        #     output = img2vec.get_vec(img, tensor=True).squeeze().numpy()
        #     feature_array.append(output)
        #     labels_array.append(label)
        # for i, data in enumerate(dataloaders_dict['val'],0):
        #     img, label = data
        #     img = img.cuda()
        #     img = transforms.ToPILImage()(img.squeeze()).convert("RGB")
        #     output = img2vec.get_vec(img, tensor=True).squeeze().numpy()
        #     test_feature_array.append(output)
        #     test_labels_array.append(label)