from tqdm import tqdm
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from IPython.display import display

from pretrained_models import BERTClassifier

class FakeNewsDetector():

  def __init__(self, nb_classes, device):
    self.model = BERTClassifier(nb_classes).to(device)
    self.device = device
    # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
    self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=3e-5)
    self.criterion = nn.CrossEntropyLoss(reduction="sum")


  def fit(self, train_dataloader, nb_epochs, val=None, val_dataloader=None):

    for epoch in range(nb_epochs):

      self.model.train()
      progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{nb_epochs}")
      accuracy, f1_macro = 0.0, 0.0
      for batch in progress_bar:

        x, y = batch
        x, y = [ input.to(self.device) for input in x], y.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(*x)
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()
        preds = torch.argmax(outputs, dim=1)
        accuracy += accuracy_score(y.cpu(), preds.cpu())
        f1_macro += f1_score(y.cpu(), preds.cpu(), average="macro")
        progress_bar.set_postfix({"accuracy": accuracy / (progress_bar.n + 1), "f1_macro": f1_macro / (progress_bar.n + 1)})
        del outputs, preds, loss, x, y
        torch.cuda.empty_cache()

      # Validation Time
      if val:
        self.model.eval()
        with torch.no_grad():
          accuracy, f1_macro = 0.0, 0.0
          for batch in val_dataloader:
            x, y = batch
            x, y = [ input.to(self.device) for input in x], y.to(self.device)
            outputs = self.model(*x)
            preds = torch.argmax(outputs, dim=1)
            accuracy += accuracy_score(y.cpu(), preds.cpu())
            f1_macro += f1_score(y.cpu(), preds.cpu(), average="macro")
          del x, y, outputs, preds
          torch.cuda.empty_cache()

        val_scores = {
            "Accuracy" : accuracy / len(val_dataloader),
            "F1_macro" : f1_macro / len(val_dataloader)
        }
        display(pd.DataFrame(val_scores, index=["Validation Scores"]))

      print("")

  def test(self, test_dataloader):

    self.model.eval()

    accuracy, f1_macro = 0.0, 0.0
    progress_bar = tqdm(test_dataloader, desc="Testing model...")
    for batch in progress_bar:
      x, y = batch
      x, y = [ input.to(self.device) for input in x], y.to(self.device)
      with torch.no_grad():
        outputs = self.model(*x)
      preds = torch.argmax(outputs, dim=1)
      accuracy += accuracy_score(y.cpu(), preds.cpu())
      f1_macro += f1_score(y.cpu(), preds.cpu(), average="macro")
    del x, y, outputs, preds
    torch.cuda.empty_cache()
    scores = {
        "Accuracy" : accuracy / len(test_dataloader),
        "F1_macro" : f1_macro / len(test_dataloader)
    }
    display(pd.DataFrame(scores, index=["Test Scores"]))