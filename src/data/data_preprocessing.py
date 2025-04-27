from datasets import FakeNewsDataset

from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import train_test_split

class TextPreprocessing():

  def __init__(self):

    def lower_text(text):
      return text.lower()

    self.preprocessing_functions = [
      lower_text,
    ]

  def preprocessing(self, text):
    for function in self.preprocessing_functions:
      text = function(text)
    return text

  def preprocess_text(self, texts):
    """
      Preprocessing of the text Series texts

    """
    return texts.apply(self.preprocessing)

class DataPreProcessing():

  def __init__(self, tokenizer, df, batch_size, test_size=0.2, val_size=None):
    """
      df has to contain the following columns ["title", "text", "subject", "date", "label"]
    """
    self.text_preprocessing = TextPreprocessing()
    self.df = df[df["text"].apply(lambda x: isinstance(x, str))]
    self.tokenizer = tokenizer
    self.test_size = test_size
    self.val_size = val_size
    self.batch_size = batch_size

  def data_preprocessing(self, feature_columns=["text"]):
    """
      This method computes the final data_train and data_test under the form of a PyTorch DataLoader
    """
    print("Text Preprocessing...")
    print("")
    for feature in feature_columns:
      self.df[feature] = self.text_preprocessing.preprocess_text(self.df[feature])

    print("Tokenization...")
    print("")
    list_features = [ list(self.df[feature]) for feature in feature_columns ]
    labels = torch.tensor(list(self.df["label"]))
    tokenized_inputs = self.tokenizer(
        *list_features,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    print("Splitting data...")
    print("")
    ids_train, ids_test, attention_mask_train, attention_mask_test, labels_train, labels_test = train_test_split(
        tokenized_inputs["input_ids"],
        tokenized_inputs["attention_mask"],
        labels,
        test_size=self.test_size,
        random_state=42
    )

    if self.val_size:
      ids_train, ids_val, attention_mask_train, attention_mask_val, labels_train, labels_val = train_test_split(
          ids_train,
          attention_mask_train,
          labels_train,
          test_size=self.val_size / (1 - self.test_size),
          random_state=42
      )
      train_dataset, test_dataset, val_dataset = FakeNewsDataset([ids_train, attention_mask_train], labels_train), FakeNewsDataset([ids_test, attention_mask_test], labels_test), FakeNewsDataset([ids_val, attention_mask_val], labels_val)
      return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True), DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False), DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

    else:
      train_dataset, test_dataset = FakeNewsDataset([ids_train, attention_mask_train], labels_train), FakeNewsDataset([ids_test, attention_mask_test], labels_test)
      return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True), DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)