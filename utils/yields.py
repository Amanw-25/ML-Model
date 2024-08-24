import pandas as pd


def yield_preprocess(data, encoder, scaler):
    # Create a DataFrame from the input data
    df = pd.DataFrame(
        [data],
        columns=[
            "State_Name",
            "Crop_Type",
            "Crop",
            "N",
            "P",
            "K",
            "pH",
            "rainfall",
            "temperature",
            "Area_in_hectares",
        ],
    )

    # Handle categorical variables with OneHotEncoding
    categorical_features = ["State_Name", "Crop_Type", "Crop"]
    X_categorical = encoder.transform(df[categorical_features])

    # Convert the encoded categorical data to a DataFrame
    X_categorical_df = pd.DataFrame(
        X_categorical, columns=encoder.get_feature_names_out(categorical_features)
    )

    # Drop original categorical columns and concatenate encoded columns
    df = df.drop(columns=categorical_features)
    df = pd.concat([df.reset_index(drop=True), X_categorical_df.reset_index(drop=True)], axis=1)

    # Scale numerical features
    numerical_features = ["N", "P", "K", "pH", "rainfall", "temperature", "Area_in_hectares"]
    df[numerical_features] = scaler.transform(df[numerical_features])

    return df
