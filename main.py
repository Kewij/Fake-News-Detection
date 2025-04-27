import os
import pandas as pd
import kagglehub
import torch
import gc
import argparse

from src.data.data_preprocessing import DataPreProcessing
from src.models.fake_news_detectors import FakeNewsDetector
from transformers import BertTokenizer

def main(dataset_name: str):

    if dataset_name == "emineyetm":

        path = kagglehub.dataset_download("emineyetm/fake-news-detection-datasets")
        print("Path to dataset files:", path)
        path_files = os.path.join(path, "News _dataset")
        true_df, fake_df = pd.read_csv(os.path.join(path_files, "True.csv")), pd.read_csv(os.path.join(path_files, "Fake.csv"))
        fake_df["label"] = 1
        true_df["label"] = 0
        df = pd.concat([fake_df, true_df], axis=0)

    elif dataset_name == "saurabhshahane":

        # Download latest version
        path = kagglehub.dataset_download("saurabhshahane/fake-news-classification")
        print("Path to dataset files:", path)
        df = pd.read_csv(os.path.join(path, "WELFake_Dataset.csv"), index_col=0)
    
    else:

        raise ValueError(f"Dataset {dataset_name} not supported.")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_size, val_size = 0.2, 0.1
    batch_size = 64
    data_processing = DataPreProcessing(tokenizer=tokenizer, df=df, test_size=test_size, val_size=val_size, batch_size=batch_size)

    train_dataloader, test_dataloader, val_dataloader = data_processing.data_preprocessing()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nb_classes = 2
    fake_news_detector = FakeNewsDetector(nb_classes=nb_classes, device=device)

    # Fit the model
    fake_news_detector.fit(train_dataloader, nb_epochs=1, val=True, val_dataloader=val_dataloader)
    del train_dataloader, val_dataloader
    gc.collect()
    torch.cuda.empty_cache()

    # Test the model
    fake_news_detector.test(test_dataloader)
    del test_dataloader
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Launch Fake News Detector")
    parser.add_argument(
        "dataset_name",
        type=str,
        choices=["emineyetm", "saurabhshahane"],
        help="Name of the dataset to use."
    )
    args = parser.parse_args()

    main(args.dataset_name)