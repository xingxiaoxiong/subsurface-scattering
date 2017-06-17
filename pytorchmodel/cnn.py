import torch
from torch import nn
from torch.autograd import Variable


class VGG(torch.nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        filter_num = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
        self.convs = nn.Sequential(
            nn.Conv2d(12, filter_num[0], kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(filter_num[0], filter_num[1], kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(filter_num[1], filter_num[2], kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(filter_num[2], filter_num[3], kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(filter_num[3], filter_num[4], kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(filter_num[4], filter_num[5], kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(filter_num[5], filter_num[6], kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(filter_num[6], filter_num[7], kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(filter_num[7], filter_num[8], kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(filter_num[8], filter_num[9], kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(filter_num[9], filter_num[10], kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(filter_num[10], filter_num[11], kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(filter_num[11], filter_num[12], kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc = nn.Sequential(
            nn.Linear(131072, 3),
            nn.Sigmoid()
        )

        # self.fc6 = nn.Sequential(
        #     nn.Linear(131072, 128),
        #     nn.ReLU()
        # )
        #
        # self.fc7 = nn.Sequential(
        #     nn.Linear(128, 128),
        #     nn.ReLU()
        # )
        #
        # self.fc8 = nn.Sequential(
        #     nn.Linear(128, 3),
        #     nn.Sigmoid()
        # )
        self.dropout = nn.Dropout()

    def forward(self, x):
        conv_out = self.convs(x)
        conv_out = conv_out.view(conv_out.size(0), -1)
        return self.fc(conv_out)
        # fc6_out = self.fc6(conv_out)
        # fc6_out = self.dropout(fc6_out)
        # fc7_out = self.fc7(fc6_out)
        # fc7_out = self.dropout(fc7_out)
        # return self.fc8(fc7_out)


class CNN:
    def __init__(self, opt):
        self.network = VGG()
        if opt.gpu:
            self.network.cuda()
        self.opt = opt

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=opt.lr)

    def set_input(self, X, y):
        self.tX = torch.from_numpy(X)
        self.ty = torch.from_numpy(y)

    def forward(self):
        self.X = Variable(self.tX)
        self.y = Variable(self.ty)
        if self.opt.gpu:
            self.X = self.X.cuda()
            self.y = self.y.cuda()
        self.y_pred = self.network.forward(self.X)

    def backward(self):
        self.loss = self.criterion(self.y_pred, self.y)
        self.loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def validate(self):
        self.network.train(False)
        self.forward()
        self.loss = self.criterion(self.y_pred, self.y)
        self.network.train(True)

    def predict(self, X):
        X = Variable(torch.from_numpy(X))
        if self.opt.gpu:
            X = X.cuda()
        return self.network.forward(X).data[0]

    def save(self, save_path):
        torch.save(self.network.cpu().state_dict(), save_path)
        if self.opt.gpu:
            self.network.cuda()


    def load(self, load_path):
        self.network.load_state_dict(torch.load(load_path))