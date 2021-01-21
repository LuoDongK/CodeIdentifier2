from abc import ABC

import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from PIL import Image
import json
from ResNet import resnet18, BasicBlock


class CRNN(nn.Module, ABC):
    def __init__(self, input_length=30, alphabet='0123456789', hidden_size=64, dropout=0.):
        super(CRNN, self).__init__()

        self.alphabet = alphabet
        self.num_classes = len(self.alphabet)

        self.feature_extractor = resnet18()
        num_layers = 1 if dropout == 0 else 2
        self.extra = BasicBlock(1, input_length, 1, 1)

        self.rnn = nn.LSTM(512, hidden_size, num_layers, dropout=dropout, bidirectional=True)
        self.out_layer = nn.Linear(hidden_size * 2, self.num_classes + 1)

    def forward(self, inputs, decode=False):
        features = self.feature_extractor(inputs)
        features = self.features_to_sequence(features)
        seq, _ = self.rnn(features)
        seq = self.out_layer(seq)
        seq = seq.log_softmax(2)
        if decode:
            seq = self.decode(seq)
        return seq

    def features_to_sequence(self, features):
        features = features.mean(2)
        b, c, w = features.size()
        features = features.reshape(b, c, 1, w)
        features = features.permute(0, 3, 2, 1)
        features = self.extra(features)
        features = features.permute(1, 0, 2, 3)
        features = features.squeeze(2)
        return features

    def pred_to_string(self, pred):
        seq = []
        for ii in range(pred.shape[0]):
            label = np.argmax(pred[ii])
            seq.append(label - 1)
        out = []
        for ii in range(len(seq)):
            if ii == 0:
                if seq[ii] != -1:
                    out.append(seq[ii])
            else:
                if seq[ii] != -1 and seq[ii] != seq[ii - 1]:
                    out.append(seq[ii])

        out = ''.join(self.alphabet[ii] for ii in out)
        return out

    def decode(self, pred):
        pred = pred.permute(1, 0, 2).cpu().data.numpy()
        seq = []
        for ii in range(pred.shape[0]):
            seq.append(self.pred_to_string(pred[ii]))
        return seq


class LRStep(object):
    def __init__(self, optimizer, step_size=1000, max_iter=10000):
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.step_size = step_size
        self.last_iter = -1
        self.base_lrs = list(map(lambda group: group['lr'], optimizer.param_groups))

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def step(self):
        if self.last_iter + 1 == self.max_iter:
            self.last_iter = -1
        self.last_iter = (self.last_iter + 1) % self.max_iter
        for ids, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.base_lrs[ids] * 0.8 ** (self.last_iter // self.step_size)


class TextLoader(Dataset):
    def __init__(self, img_path, img_label, alphabet='0123456789', transform=None):
        super(TextLoader, self).__init__()
        self.img_path = img_path
        self.img_label = img_label
        self.transform = transform
        self.alphabet = alphabet

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, item):
        if self.img_label is not None:
            text = self.img_label[item]
            text = self.text_to_seq(text)
            img = self.transform(self.img_path[item])
            sample = {'img': img, 'text': text, 'length': len(text)}
        else:
            img = self.transform(self.img_path[item])
            sample = {'img': img}
        return sample

    def text_to_seq(self, text):
        seq = []
        for c in text:
            seq.append(self.alphabet.find(str(c)) + 1)
        return seq


def text_collate(batch):
    img = []
    text = []
    text_length = []
    for sample in batch:
        img.append(sample['img'].float())
        if sample.get('text') is not None:
            text.extend(sample['text'])
            text_length.append(sample['length'])
    img = torch.stack(img)
    if text:
        text = torch.tensor(text).int()
        text_length = torch.tensor(text_length).int()
        batch = {'img': img, 'text': text, 'length': text_length}
    else:
        batch = {'img': img}
    return batch


def accuracy(pred, text, text_length, alphabet='0123456789'):
    pos = 0
    correct = 0
    for ii in range(len(pred)):
        target = ''.join(alphabet[c] for c in text[pos:pos + text_length[ii]])
        pos += text_length[ii]
        if target == pred[ii]:
            correct += 1
    return correct / len(pred)


class TextIdentifier(object):
    def __init__(self, model, input_length, device):
        torch.manual_seed(123)
        self.criterion = nn.CTCLoss(zero_infinity=True)
        self.model = model
        self.model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.lr_schedule = LRStep(self.optimizer, 1000, 10000)
        self.input_length = input_length
        self.best_acc, self.best_epoch = 0, 0
        self.train_loss = []
        self.validate_accuracy = []
        self.predictions = []

    def train(self, sample):
        self.model.train()
        img = sample['img'].to(self.device)
        text = sample['text']
        text_length = sample['length'].int()
        output = self.model(img)
        input_lengths = torch.full((len(img),), self.input_length, dtype=torch.int)
        loss = self.criterion(output.cpu(), text, input_lengths, text_length)
        self.train_loss.append(loss.item())
        with open('loss.txt', mode='a+') as f:
            f.write('\r\n' + str(float(loss)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self.lr_schedule.step()

    def validate(self, sample):
        self.model.eval()
        img = sample['img'].to(self.device)
        text = list(sample['text'].numpy() - 1)
        text_length = list(sample['length'].numpy())
        with torch.no_grad():
            output = self.model(img, True)
        acc = accuracy(output, text, text_length)
        self.validate_accuracy.append(acc)

    def predict(self, sample):
        self.model.eval()
        img = sample['img'].to(self.device)
        with torch.no_grad():
            out = self.model(img, True)
        return out


if __name__ == '__main__':
    train_json = json.load(open('train.json'))
    train_label = [train_json[x]['label'] for x in train_json.keys()]
    train_path = ['train/' + x for x in train_json.keys()]

    validate_json = json.load(open('val.json'))
    validate_label = [validate_json[x]['label'] for x in validate_json.keys()]
    validate_path = ['val/' + x for x in validate_json.keys()]

    train_path.extend(validate_path[-5000:])
    train_label.extend(validate_label[-5000:])
    validate_path = validate_path[:-5000]
    validate_label = validate_label[:-5000]
    tf_train = transforms.Compose([
        lambda x: Image.open(x).convert('RGB'),
        transforms.Resize((100, 200)),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tf_val = transforms.Compose([
        lambda x: Image.open(x).convert('RGB'),
        transforms.Resize((100, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_set = TextLoader(train_path, train_label, transform=tf_train)
    validate_set = TextLoader(validate_path, validate_label, transform=tf_val)

    device = torch.device('cpu')
    crnn = CRNN(dropout=0.5)
    identifier = TextIdentifier(crnn, 30, device)
    # identifier.model.load_state_dict(torch.load('best.mdl'))
    for epoch in range(30):
        identifier.train_loss = []
        identifier.validate_accuracy = []
        train_loader = DataLoader(train_set, batch_size=1, collate_fn=text_collate)
        train_iter = tqdm(train_loader)
        for index, samples in enumerate(train_iter):
            identifier.train(samples)
            status = 'epoch: {}; iter: {}; lr: {}; loss_mean: {}; loss: {}'.format(epoch, index,
                                                                                   identifier.lr_schedule.get_lr(),
                                                                                   np.mean(identifier.train_loss),
                                                                                   np.mean(identifier.train_loss[-32:]))
            train_iter.set_description(status)

        validate_loader = DataLoader(validate_set, batch_size=2, collate_fn=text_collate)
        validate_iter = tqdm(validate_loader)
        for index, samples in enumerate(validate_iter):
            identifier.validate(samples)
            validate_iter.set_description(
                'epoch: {}, accuracy: {}'.format(epoch, np.mean(identifier.validate_accuracy)))
        avg_acc = np.mean(identifier.validate_accuracy)
        with open('./accuracy.txt', mode='a+') as f:
            f.write('\r\n' + str(avg_acc))
        if avg_acc > identifier.best_acc:
            identifier.best_epoch = epoch
            identifier.best_acc = avg_acc
            torch.save(identifier.model.state_dict(), 'best.mdl')
