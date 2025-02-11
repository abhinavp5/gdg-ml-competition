import logging
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.impute import SimpleImputer
from collections import Counter
from textblob import TextBlob

# Load dataset and return cleaned and preprocessed features and target
# TODO: use more of the columns, preprocess them in a different way,
#       or even include new features (e.g. from other datasets)

# columns: 'Poster_Link', 'Series_Title', 'Released_Year', 'Certificate',
#          'Runtime', 'Genre', 'IMDB_Rating', 'Overview', 'Meta_score', 'Director',
#          'Star1', 'Star2', 'Star3', 'Star4', 'No_of_Votes'

# columns used in template: 'Released_Year', 'Certificate', 'Runtime', 'Genre' (kind of',
# 'IMDB_Rating', 'Meta_score', 'No_of_Votes'
#defining functino to encode the target variable
#Helper method
def smooth_encode(df, col, target, smoothing_param =.3):
    mean_target = df[target].mean()
    encoded =df.groupby(col)[target].agg(["count", "mean"])
    counts = encoded["count"]
    means = encoded["mean"]
    smooth_encodings = (means*counts + mean_target*smoothing_param)/(counts+smoothing_param)
    return df[col].map(smooth_encodings)
# def get_prepared_data(data_path="data"):

#     #Our Code
#     df = pd.read_csv('data/thesheetlessdata.csv')
#     logging.warning(f"DF SHAPE: {df.shape}")
#     df.dropna(inplace=True)
#     logging.warning(f"DF SHAPE: {df.shape}")

    
#     # Convert Data Types
#     df["Released_Year"] = pd.to_numeric(df["Released_Year"], errors='coerce')
#     df["Released_Day"] = pd.to_numeric(df["Released_Day"], errors='coerce')
    
    
#     #gpt o1 pro deeplearning operator r1
#     df["Released_Month"] = (df["Released_Day"] // 30) + 1  # approximate monthly grouping
#     df["Released_Quarter"] = (df["Released_Month"] - 1) // 3 + 1  # approximate quarterly grouping
#     df["Overview_Length"] = df["Overview"].apply(lambda x: len(x.split()))
#     df["Overview_Sentiment"] = df["Overview"].apply(lambda x: TextBlob(x).sentiment.polarity)
#     df["Log_No_of_Votes"] = np.log1p(df["No_of_Votes"])
#     df["RatingVotes_Interaction"] = df["IMDB_Rating"] * df["No_of_Votes"]

#     # bins = [0, 4, 6, 8, 10]  
#     # labels = ["low", "mid", "good", "excellent"]  
#     # df["Rating_Bin"] = pd.cut(df["IMDB_Rating"], bins=bins, labels=labels, include_lowest=True)
#     # Then smooth-encode or one-hot-encode the new Rating_Bin feature
    
#     # bins = [0, 90, 120, 180, np.inf]
#     # labels = ['short', 'average', 'long', 'very_long']
#     # df["Runtime_Bin"] = pd.cut(df["Runtime"], bins=bins, labels=labels)


#     ### end gpt oo1 1


#     # df["Runtime"] = df["Runtime"].str.replace(" min", "").astype(float)
#     # df["Gross"] = df["Gross"].str.replace(",", "").astype(float)



#     # df["Gross"] = np.log1p(df["Gross"])  # Apply log transformation



#     # Feature Engineering
#     # df["Log_No_of_Votes"] = np.log1p(df["No_of_Votes"])
#     df["Decade"] = (df["Released_Year"] // 10) * 10  # Convert Year to Decades


#     # Encoding Categorical Variables
#     df["Series_Title"] = smooth_encode(df, "Series_Title", "Gross")
#     # df["Certificate"] = smooth_encode(df, "Certificate", "Gross")
#     #df["Production_Company"] = smooth_encode(df, "Production_Company", "Gross")
#     df["Production_Country"] = smooth_encode(df, "Production_Country", "Gross")
#     df["Director"] = smooth_encode(df, "Director", "Gross")
#     df["Star1"] = smooth_encode(df, "Star1", "Gross")
#     df["Star2"] = smooth_encode(df, "Star2", "Gross")
#     df["Star3"] = smooth_encode(df, "Star3", "Gross")
#     df["Star4"] = smooth_encode(df, "Star4", "Gross")
    

#     # Multi-label binarization for Genre
#     genre_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
#     genre_matrix = genre_encoder.fit_transform(df[["Genre"]])
#     genre_df = pd.DataFrame(genre_matrix, columns=genre_encoder.get_feature_names_out(["Genre"]))
#     df = pd.concat([df, genre_df], axis=1)
#     df.drop(columns=["Genre"], inplace=True)
    
#     # Process Overview Text
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     df["Overview"] = df["Overview"].fillna("No overview available").astype(str)
#     overview_embeddings = df["Overview"].apply(lambda x: model.encode(x))
    
#     # Reduce Dimensionality with PCA
#     pca = PCA(n_components=50)
#     overview_reduced = pca.fit_transform(np.stack(overview_embeddings))
#     overview_df = pd.DataFrame(overview_reduced, columns=[f'Overview_PC{i}' for i in range(50)])
#     df = pd.concat([df.reset_index(drop=True), overview_df.reset_index(drop=True)], axis=1)
#     df.drop(columns=["Overview"], inplace=True)

#     df.dropna(inplace = True) #dropping null valued columns
#     logging.warning(f"DF SHAPE: {df.shape}")
    
#     # Define Features and Target
#     Y = df["Gross"].values
#     df.drop(columns=["Gross"], inplace=True)
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(df.values)
#     scaler_Y = StandardScaler()
#     Y_scaled = scaler_Y.fit_transform(Y.reshape(-1, 1))
    
#     # Convert to torch tensors
#     features = torch.tensor(X_scaled, dtype=torch.float32)
#     target = torch.tensor(Y_scaled, dtype=torch.float32).view(-1, 1)


        
#     return features, target

    

def get_prepared_data(data_path="data"):
    #FOr csv they gave us 

    #Our Code
    df = pd.read_csv('data/imdb_top_1000.csv')
    df.at[966, "Released_Year"] = 1995  # Fix incorrect value for Apollo 13
    df.drop(columns=["Poster_Link"], inplace=True)

    
    # Convert Data Types
    df["Released_Year"] = pd.to_numeric(df["Released_Year"], errors='coerce')

    df["Runtime"] = df["Runtime"].str.replace(" min", "").astype(float)
    df["Gross"] = df["Gross"].str.replace(",", "").astype(float)
    df["Gross"], df["Log_No_of_Votes"] = np.power(df["Gross"], 0.15),np.log1p(df["No_of_Votes"])
    df["Decade"] = (df["Released_Year"] // 10) * 10  # Convert Year to Decades
    df["Series_Title"] = smooth_encode(df, "Series_Title", "Gross")
    df["Certificate"] = smooth_encode(df, "Certificate", "Gross")
    df["Director"] = smooth_encode(df, "Director", "Gross")
    df["Star1"] = smooth_encode(df, "Star1", "Gross")
    df["Star2"] = smooth_encode(df, "Star2", "Gross")
    df["Star3"] = smooth_encode(df, "Star3", "Gross")
    df["Star4"] = smooth_encode(df, "Star4", "Gross")
    

    # Multi-label binarization for Genre
    genre_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    genre_matrix = genre_encoder.fit_transform(df[["Genre"]])
    genre_df = pd.DataFrame(genre_matrix, columns=genre_encoder.get_feature_names_out(["Genre"]))
    df = pd.concat([df, genre_df], axis=1)
    df.drop(columns=["Genre"], inplace=True)
    
    # Process Overview Text
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df["Overview"] = df["Overview"].fillna("No overview available").astype(str)
    overview_embeddings = df["Overview"].apply(lambda x: model.encode(x))
    
    # Reduce Dimensionality with PCA
    pca = PCA(n_components=50)
    overview_reduced = pca.fit_transform(np.stack(overview_embeddings))
    overview_df = pd.DataFrame(overview_reduced, columns=[f'Overview_PC{i}' for i in range(50)])
    df = pd.concat([df.reset_index(drop=True), overview_df.reset_index(drop=True)], axis=1)
    df.drop(columns=["Overview"], inplace=True)

    df.dropna(inplace = True) #dropping null valued columns
    print("NaN count per column after dropna:")
    print(df.isna().sum())
    
    # Define Features and Target
    Y = df["Gross"].values
    df.drop(columns=["Gross"], inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)
    scaler_Y = StandardScaler()
    Y_scaled = scaler_Y.fit_transform(Y.reshape(-1, 1))
    
    # Convert to torch tensors
    features = torch.tensor(X_scaled, dtype=torch.float32)
    target = torch.tensor(Y_scaled, dtype=torch.float32).view(-1, 1)

    print(f"Features contain NaN: {torch.isnan(features).any()}, Inf: {torch.isinf(features).any()}")
    print(f"Target contains NaN: {torch.isnan(target).any()}, Inf: {torch.isinf(target).any()}")
        
    return features, target

    


def get_all_titles(data_path="data"):
    data = get_raw_data(data_path)
    return data["Series_Title"]

def get_raw_data(path="data"):
    # read in every csv file, join on "Series_Title"
    # return the raw data
    import os
    files = os.listdir(path)
    data = pd.DataFrame()
    for file in files:
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(path, file))
            df = df.dropna()
            # join on "Series_Title"
            if data.empty:
                data = df
            else:
                data = data.merge(df, on="Series_Title")
    return data
