import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
torch.__version__

BATCH_SIZE=512
#大概需要2G的显存 
#il faut environ 2G du RAM de la carte graphique

EPOCHS=3
# 总共训练批次 
# l'epoch totale

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
# 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多 
# cette ligne décide si l'entrainement utilise GPU, si oui, la calculation va être beaucoup plus vite

#On va procéder le dataset. Ici, torch.utils.data.DataLoader() est le coeur de la pytorch, veuillez conseiller des introductions partout.
#On a negligé(par default) l'implémentation de la fonction _getitem_ qui est le coeur de dataset. je vous conseille de voir avec command + click sur datasets.MNIST.
#pour datasets.MNIST voyez https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html.
#La difference entre les deux dataset est train=True/False. 

#dataset of train
train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, 
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)

#dataset of test
test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)

#Ici on définie le module, avec l'input et l'output déstinée. 
#Ici chaque image a la resolution de 28x28, donc l'input est 28x28=784.
#On a 10 classes, donc l'output est 10.

class ConvNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        # batch*1*28*28（ "1" répresent l'image est monocoleur，28x28 est la résolution des images）
        # Pour nn.Conv2d(), voyez https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d 
        # Pour la convolution, il faut l'aprendre par coeur, voyez https://github.com/vdumoulin/conv_arithmetic
        self.conv1 = nn.Conv2d(1, 10, 5) 
        self.conv2 = nn.Conv2d(10, 20, 3) 
        # batch*20*10*10 
        self.fc1 = nn.Linear(20*10*10, 500) # 输入通道数是2000，输出通道数是500
        self.fc2 = nn.Linear(500, 10) # 输入通道数是500，输出通道数是10，即10分类
        
    def forward(self,x):
        in_size = x.size(0) # 在本例中in_size=512，也就是BATCH_SIZE的值。输入的x可以看成是512*1*28*28的张量。
        out = self.conv1(x) # batch*1*28*28 -> batch*10*24*24（28x28的图像经过一次核为5x5的卷积，输出变为24x24）
        out = F.relu(out) # batch*10*24*24（激活函数ReLU不改变形状））
        out = F.max_pool2d(out, 2, 2) # batch*10*24*24 -> batch*10*12*12（2*2的池化层会减半）
        out = self.conv2(out) # batch*10*12*12 -> batch*20*10*10（再卷积一次，核的大小是3）
        out = F.relu(out) # batch*20*10*10
        out = out.view(in_size, -1) # batch*20*10*10 -> batch*2000（out的第二维是-1，说明是自动推算，本例中第二维是20*10*10）
        out = self.fc1(out) # batch*2000 -> batch*500
        out = F.relu(out) # batch*500
        out = self.fc2(out) # batch*500 -> batch*10
        out = F.log_softmax(out, dim=1) # 计算log(softmax(x))
        return out

#On va définir le model, l'optimizer et la fonction de loss.    
model = ConvNet().to(DEVICE)
optimizer = optim.Adam(model.parameters())

def train(model, device, train_loader, optimizer, epoch):
    # On va mettre le model en mode d'entrainement
    model.train()
    # On va parcourir le dataset
    for batch_idx, (data, target) in enumerate(train_loader):
        # On va mettre les données sur le GPU
        data, target = data.to(device), target.to(device)
        # On va mettre le gradient à zéro
        optimizer.zero_grad()
        # On va faire la prédiction
        output = model(data)
        # On va calculer la loss
        loss = F.nll_loss(output, target)
        # On va faire la backpropagation
        loss.backward()
        # On va faire la mise à jour des paramètres
        optimizer.step()
        # On va afficher le résultat
        if(batch_idx+1)%30 == 0: 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
def test(model, device, test_loader):
    # On va mettre le model en mode d'évaluation
    model.eval()
    # On va initialiser les variables
    test_loss = 0
    correct = 0
    # On va désactiver le calcul du gradient
    with torch.no_grad():
        # On va parcourir le dataset
        for data, target in test_loader:
            # On va mettre les données sur le GPU
            data, target = data.to(device), target.to(device)
            # On va faire la prédiction
            output = model(data)
            # On va calculer la loss
            test_loss += F.nll_loss(output, target, reduction='sum').item() # 将一批的损失相加
            # On va trouver la prédiction
            pred = output.max(1, keepdim=True)[1] 
            # On va calculer le nombre de prédiction correct
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    # On va afficher le résultat   
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

#main    
for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)