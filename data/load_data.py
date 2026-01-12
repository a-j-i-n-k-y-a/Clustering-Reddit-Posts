from datasets import load_dataset
import pandas as pd

def load_reddit_data(subreddits=None):
    dataset = load_dataset("wenknow/reddit_dataset_44", split="train")
    df = pd.DataFrame(dataset)

    if subreddits:
        df = df[df["subreddit"].isin(subreddits)]

    return df
