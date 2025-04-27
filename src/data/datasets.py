from torch.utils.data import Dataset

class FakeNewsDataset(Dataset):
  """
    Dataset class for the TRECIS Dataset.
    Warning : it represents only 1 dataset (1 task)
    Therefore, the full dataset will be a dictionary whose keys are the different tasks and values are TRECISDataset objects.
  """

  def __init__(self, inputs, y):
    """
    inputs : should be a list of torch.tensor that represents all the desirable input for the model
    y : should be a torch.tensor that represents the target
    """
    self.inputs = inputs
    self.y = y

  def __len__(self):
    return len(self.y)

  def __getitem__(self, idx):
    x = [self.inputs[i][idx] for i in range(len(self.inputs))]
    y = self.y[idx]
    return (x, y)