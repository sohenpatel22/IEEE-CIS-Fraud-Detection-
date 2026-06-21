import pandas as pd

def load_data():

    identity = pd.read_csv("data/raw/train_identity.csv")
    transaction = pd.read_csv("data/raw/train_transaction.csv")

    return identity, transaction

def merge_data(identity, transaction):

    df = transaction.merge(identity, on="TransactionID", how='left')
    
    return df

def processed_dataset():

    identity, transaction = load_data()

    df = merge_data(identity, transaction)

    return df



    

