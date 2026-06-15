import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchvision.models.resnet import Bottleneck, ResNet
from sklearn import metrics
import timm

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score



class ClipAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(ClipAdapter, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck, c_in, bias=False),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        x = self.fc1(x)
        y = self.fc2(x)
        return x, y
        
        
        
        
class CLIP_Inplanted(nn.Module):
    def __init__(self, model, features=[4,8,12,16,20,24]): 
        super().__init__()
        self.image_encoder = model
        self.features = features 
        self.det_adapters = nn.ModuleList( [ClipAdapter(1024, bottleneck=2048) for i in range(len(features))] )#768
        
        self.fc = nn.Sequential(
                          #nn.Linear(2048, 1024),
                          #nn.ReLU(),
                          #nn.Dropout(0.25),
                          nn.Linear(1024, 256),
                          nn.ReLU(),
                          nn.Dropout(0.25),
                          #nn.Linear(512, 256),
                          #nn.ReLU(),
                          #nn.Dropout(0.25),
                          nn.Linear(256, num_classes),
                          nn.LogSoftmax(dim=1))


        


    def forward(self, x):

        B = x.shape[0]
        x = self.image_encoder.patch_embed(x).reshape(B,-1,1024)
        cls_tokens = self.image_encoder.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens,x),dim=1)
        x = x + self.image_encoder.pos_embed[:,:x.size(1),:]
        x = self.image_encoder.pos_drop(x)




        #for i in range(24):
        
        
        for i,layer in enumerate(list(self.image_encoder.blocks)): #24

            x = layer(x)
                
            if (i + 1) in self.features:
                det_adapt_med, det_adapt_out = self.det_adapters[self.features.index(i+1)](x)
                x = 0.8 * x + 0.2 * det_adapt_out 


        #pooled, tokens = self.image_encoder._global_pool(x)
        x = self.image_encoder.norm(x)
        #x = x[:, 1:].mean(dim=1)# take tokens,ecept cls token

        x = x[:, 0]

        x = self.fc(x)
        
        return x

print('models_adapter_test')


image_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) # uni model
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
}






dataset = "PATH TO DATASET"
train_directory = os.path.join(dataset, 'train')
valid_directory = os.path.join(dataset, 'test')

batch_size = 16
num_classes = 3
print(train_directory)
data = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['test'])
}


train_data_size = len(data['train'])
valid_data_size = len(data['valid'])

train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True, num_workers=8)
valid_data = DataLoader(data['valid'], batch_size=batch_size, shuffle=True, num_workers=8)

print(train_data_size, valid_data_size)

#resnet50 = models.resnet50(pretrained=True)

#model = ResNetTrunk(Bottleneck, [3, 4, 6, 3])
#model.load_state_dict(torch.load("/data15/data15_5/zhengke/generation/quality_assess/resnet50/mocov2_rn50_ep200.torch"))
#model.eval()


# transformer

visual_model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
visual_model.load_state_dict(torch.load(os.path.join("PATH TO UNI", "pytorch_model.bin"), map_location="cuda"), strict=True)
visual_model.eval()

model = CLIP_Inplanted(model=visual_model).to("cuda")
model.eval()


for name, param in model.named_parameters():
    param.requires_grad = True 

model.to('cuda:0')


loss_func = nn.CrossEntropyLoss().to('cuda:0')

fc_optimizer = optim.Adam(list(model.fc.parameters()), lr=0.00001, betas=(0.5, 0.999)) 
det_optimizer = optim.Adam(list(model.det_adapters.parameters()), lr=0.00001, betas=(0.5, 0.999))




def train_and_valid(model, loss_function, fc_optimizer, det_optimizer, epochs=50):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    history = []
    best_acc = 0.0
    best_epoch = 0
    best_micro = 0.0
    best_macro = 0.0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))

        model.eval()

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(tqdm(train_data)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            #loss.requires_grad_(True)

            fc_optimizer.zero_grad()
            det_optimizer.zero_grad()
            

            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            loss.backward()
            fc_optimizer.step()
            det_optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)
            
            
        pred_list = []
        label_list = []

        with torch.no_grad():
            model.eval()

            for j, (inputs, labels) in enumerate(tqdm(valid_data)):
            
                save_labels = torch.eye(2)[labels,:]
                save_labels = save_labels.numpy().tolist()
                label_list.extend(save_labels)
            
            
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
             
                
                loss = loss_function(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                valid_acc += acc.item() * inputs.size(0)
                
                
                save_predictions = torch.eye(2)[predictions.cpu(),:]
                save_predictions = save_predictions.numpy().tolist()
                pred_list.extend(save_predictions)
                

        # compute micro & macro auroc
        
        macro_auc = metrics.roc_auc_score(np.array(label_list), np.array(pred_list), average='macro')
        micro_auc = metrics.roc_auc_score(np.array(label_list), np.array(pred_list), average='micro')
        auc = metrics.roc_auc_score(np.array(label_list), np.array(pred_list))

        avg_train_loss = train_loss/train_data_size
        avg_train_acc = train_acc/train_data_size

        avg_valid_loss = valid_loss/valid_data_size
        avg_valid_acc = valid_acc/valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc

            
        if best_micro < micro_auc:
            best_micro = micro_auc
        if best_macro < macro_auc:
            best_macro = macro_auc

            best_epoch = epoch + 1
            
        torch.save(model, "./SAVE"+'_model_'+str(epoch+1)+'.pt')
            # save total model, model = torch.load(PATH) to load, don't define the model, ans use torch.load_state_dict
        epoch_end = time.time()

        print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
            epoch+1, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start
        ))
        
        print("macro_aur:", macro_auc, "; micro_aur:", micro_auc, "best_macro_aur:", best_macro, "; best_micro_aur:", best_micro, )
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))



        #torch.save(model, 'models/'+'_model_'+str(epoch+1)+'.pt')
        
    return model, history

num_epochs = 20
trained_model, history = train_and_valid(model,loss_func, fc_optimizer, det_optimizer, num_epochs)
torch.save(history, "./SAVE"+'_history.pt')

plt.figure()
history = np.array(history)
plt.plot(history[:, 0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0, 2)
plt.savefig('adapter_loss_curve.png')
#plt.show()

plt.figure()
plt.plot(history[:, 2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig('adapter_accuracy_curve.png')
#plt.show()
