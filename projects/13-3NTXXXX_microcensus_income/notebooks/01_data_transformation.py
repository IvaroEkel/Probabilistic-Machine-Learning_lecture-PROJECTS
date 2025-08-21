from basic_functions import *

df_raw = data_load("data/mz2010_cf.csv")
print(df_raw["EF739"].unique())
#renaming all columns
df_selection = select_rename("data/mappings.json",df_raw)
print(df_selection["job"].unique())
df_selection = normalize_missing_values(df_selection)

# applying label mappings for certain columns
df_labels = apply_label_mappings_string(df_selection, "data/mappings.json")
#normalize zero values to NaN and drop them
df_selection = delete_na(df_selection)

#drop 50 (self employed farmer),90 (no income),99 (not specified) from income
df_selection = filter_income(df_selection, ["50","90","99"])


#train-test-split
from sklearn.model_selection import train_test_split

X = df_selection.drop(columns=["income"]).copy()
y = df_selection["income"].copy()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

print("Train:", X_train.shape, "Test:", X_test.shape)



# saving data

df_dict={"X_train.csv": X_train,
            "X_test.csv": X_test,
            "y_train.csv": y_train,
            "y_test.csv": y_test,
            "df_labels.csv": df_labels}

for key in df_dict:
    path = "data/" + key
    save_df(df_dict[key], path)
