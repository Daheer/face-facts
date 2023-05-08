import torch.nn as nn
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
class CustomModelMain(nn.Module):
    def __init__(self, problem_type, n_classes):
        super().__init__()
        if problem_type == 'Classification' and n_classes == 1:
            output = nn.Sigmoid()
        elif problem_type == 'Regression' and n_classes == 1:
            output = nn.ReLU()
        elif problem_type == 'Classification' and n_classes > 1:
            output = nn.Softmax(dim = 1)
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 25 * 25 , 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.5)
        self.fc2 = nn.Linear(128, n_classes)
        self.output = output
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.output(x)
        return x
class age_lightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = CustomModelMain('Regression', 1)
    def forward(self, x):
        return self.model(x)
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y[:, 0]
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.unsqueeze(-1).float())
        acc = torch.eq((y_hat > 0.5).int().to(torch.int64), y.unsqueeze(-1).int()).all(dim=1).sum() / len(y)
        self.log('train loss', loss, prog_bar = True)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_val = y[:, 0]
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y_val.unsqueeze(-1).float())
        self.log('valid loss', loss, prog_bar = True)
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
class gender_lightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = CustomModelMain('Classification', 1)
    def forward(self, x):
        return self.model(x)
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y[:, 1]
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y.unsqueeze(-1).float())
        acc = torch.eq((y_hat > 0.5).int().to(torch.int64), y.unsqueeze(-1).int()).all(dim=1).sum() / len(y)
        self.log('train loss', loss, prog_bar = True)
        self.log('accuracy', acc, prog_bar = True)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_val = y[:, 1]
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y_val.unsqueeze(-1).float())
        acc =  torch.eq((y_hat > 0.5).int().to(torch.int64), y_val.unsqueeze(-1).int()).all(dim=1).sum() / len(y_val)
        self.log('valid loss', loss, prog_bar = True)
        self.log('val accuracy', acc, prog_bar = True)
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
class race_lightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = CustomModelMain('Classification', 5)
    def forward(self, x):
        return self.model(x)
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y[:, 2]
        y_hat = self(x)
        y_oh = F.one_hot(y, num_classes = 5)
        loss = F.cross_entropy(y_hat.log(), y_oh.float())
        preds = y_hat.argmax(dim = 1)
        acc = torch.eq(y, preds).float().mean()
        self.log('train loss', loss, prog_bar = True)
        self.log('accuracy', acc, prog_bar = True)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_val = y[:, 2]
        y_hat = self(x)
        y_oh = F.one_hot(y_val, num_classes = 5)
        loss = F.cross_entropy(y_hat, y_oh.float())
        preds = y_hat.argmax(dim = 1)
        acc = torch.eq(y_val, preds).float().mean()
        self.log('valid loss', loss, prog_bar = True)
        self.log('val accuracy', acc, prog_bar = True)
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
class Ultimate_Lightning(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.age_model = CustomModelMain('Regression', 1)
    self.gender_model = CustomModelMain('Classification', 1)
    self.race_model = CustomModelMain('Classification', 5)
  def forward(self, x):
    return self.age_model(x), self.gender_model(x), self.race_model(x)
  def training_step(self, batch, batch_idx):
    x, y = batch
    y_age, y_gender, y_race = y[:, 0], y[:, 1], y[:, 2]
    y_hat_age, y_hat_gender, y_hat_race = self(x)
    
    age_loss = F.mse_loss(y_hat_age, y_age.unsqueeze(-1).float())
    age_acc = torch.eq((y_hat_age > 0.5).int().to(torch.int64), y_age.unsqueeze(-1).int()).all(dim=1).sum() / len(y_age)
    
    gender_loss = F.binary_cross_entropy(y_hat_gender, y_gender.unsqueeze(-1).float())
    gender_acc = torch.eq((y_hat_gender > 0.5).int().to(torch.int64), y_gender.unsqueeze(-1).int()).all(dim=1).sum() / len(y_gender)

    y_race_oh = F.one_hot(y_race, num_classes = 5)
    race_loss = F.cross_entropy(y_hat_race.log(), y_race_oh.float())
    race_preds = y_hat_race.argmax(dim = 1)
    race_acc = torch.eq(y_race, race_preds).float().mean()
    
    total_loss = (0.001 * age_loss) + gender_loss + race_loss

    self.log('age loss', age_loss, prog_bar = True)
    self.log('gender loss', gender_loss, prog_bar = True)
    self.log('race loss', race_loss, prog_bar = True)
    self.log('gender acc', gender_acc, prog_bar = True)
    self.log('race acc', race_acc, prog_bar = True)
    self.log('total loss', total_loss, prog_bar = True)
    
    return total_loss
      
  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_age, y_gender, y_race = y[:, 0], y[:, 1], y[:, 2]
    y_hat_age, y_hat_gender, y_hat_race = self(x)
    
    age_loss = F.mse_loss(y_hat_age, y_age.unsqueeze(-1).float())
    age_acc = torch.eq((y_hat_age > 0.5).int().to(torch.int64), y_age.unsqueeze(-1).int()).all(dim=1).sum() / len(y_age)
    
    gender_loss = F.binary_cross_entropy(y_hat_gender, y_gender.unsqueeze(-1).float())
    gender_acc = torch.eq((y_hat_gender > 0.5).int().to(torch.int64), y_gender.unsqueeze(-1).int()).all(dim=1).sum() / len(y_gender)

    y_race_oh = F.one_hot(y_race, num_classes = 5)
    race_loss = F.cross_entropy(y_hat_race.log(), y_race_oh.float())
    race_preds = y_hat_race.argmax(dim = 1)
    race_acc = torch.eq(y_race, race_preds).float().mean()
    
    total_loss = (0.001 * age_loss) + gender_loss + race_loss

    self.log('val age loss', age_loss, prog_bar = True)
    
    self.log('val gender acc', gender_acc, prog_bar = True)
    
    self.log('val race acc', race_acc, prog_bar = True)
    
  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=1e-4)

