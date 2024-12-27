import pandas as pd
from sklearn.preprocessing import StandardScaler


def process_fft_channel(data_path, train_meta_df, test_meta_df):
    df = pd.read_csv(data_path)

    # Count the number of NaN values in each column
    nan_counts = df.isnull().sum()



    # Pivot the DataFrame
    flattened_df = df.pivot(index=["name", "age"], columns='channel_id')

    # Flatten the MultiIndex columns
    flattened_df.columns = ['{}_{}'.format(feature, col) for feature, col in flattened_df.columns]

    # Reset the index to make 'id' a column again
    flattened_df.reset_index(inplace=True)

    # drop na
    flattened_df.dropna(axis=1, inplace=True)

    X = flattened_df.drop(columns=['age', "name"])
    # normalize data except age

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # X back to dataframe
    X = pd.DataFrame(X_scaled, columns=X.columns)

    # get X back with the name and age
    df_scaled = pd.concat([flattened_df[['name', 'age']], X], axis=1)

    # drop na
    df_scaled.dropna(inplace=True)

    # split train test based on train_meta_df
    train_df = df_scaled[df_scaled['name'].isin(train_meta_df['participant_id'])]
    test_df = df_scaled[df_scaled['name'].isin(test_meta_df['participant_id'])]

    X_train = train_df.drop(columns=['age', "name"])
    y_train = train_df['age']

    X_test = test_df.drop(columns=['age', "name"])
    y_test = test_df['age']

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    data_path = "../data/csv/fft_channel_fill.csv"
    indices_train = "../data/train_set.csv"
    indices_test = "../data/test_set.csv"

    train_meta_df = pd.read_csv(indices_train, sep="\t")
    test_meta_df = pd.read_csv(indices_test, sep="\t")

    X_train, y_train, X_test, y_test = process_fft_channel(data_path, train_meta_df, test_meta_df)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)