# Find duplicates and only keep the rows with the maximum amount of values for the studied_features
# Duplicates finding will be based on the features_for_duplicates list
def remove_duplicates(original_dataframe, features_for_duplicates, studied_features):
  if type(features_for_duplicates)!=list:
    features_for_duplicates = [features_for_duplicates]

  dataframe_free_duplicates = original_dataframe.copy()
  # Monitor amount of cleaned rows for each feature
  removed_values_per_feature = {}
  for feature in features_for_duplicates :

    # Create a list to store the indices of the rows to keep
    rows_to_keep = []

    # Iterate over the unique duplicate codes
    for unique_value in dataframe_free_duplicates[dataframe_free_duplicates.duplicated(subset=[feature], keep=False)][feature].dropna().unique():
        # Get the subset of rows with the current duplicate code
        duplicates_subset = dataframe_free_duplicates[dataframe_free_duplicates[feature] == unique_value]

        # Count the number of non-missing values for each row in the subset. 1 or more can be found
        values_counts = duplicates_subset[studied_features].notna().sum(axis=1)

        # Find the index of the row with the maximum number of non-missing values. If several row are reaching the maximum number of non-missing values,
        # then the first index will be arbitrarily chosen
        max_values_index = values_counts.idxmax()

        # Find the index of the row with the maximum number of non-missing values. If several row are reaching the maximum number of non-missing values,
        # then the first index will be arbitrarily chosen
        rows_to_keep.append(max_values_index)

    # Get indices of non-duplicate rows
    non_duplicate_indices = dataframe_free_duplicates[~dataframe_free_duplicates.duplicated(subset=[feature], keep=False)].index

    # Combine indices of non-duplicate rows and rows_to_keep
    all_indices_to_keep = non_duplicate_indices.union(rows_to_keep)

    removed_values_per_feature[feature] = len(dataframe_free_duplicates) - len(all_indices_to_keep)

    # Create the final DataFrame
    dataframe_free_duplicates = dataframe_free_duplicates.loc[all_indices_to_keep]

  return dataframe_free_duplicates, removed_values_per_feature
