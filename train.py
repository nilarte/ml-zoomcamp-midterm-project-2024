# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pickle


# %%
df = pd.read_csv('heart.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.shape


# %% [markdown]
# Stage: Model Training 

# %% [markdown]
# Split data into train validation and test

# %%
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
len(df_train), len(df_val), len(df_test)

# %%
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
y_train = df_train.heartdisease.values
y_val = df_val.heartdisease.values
y_test = df_test.heartdisease.values

del df_train['heartdisease']
del df_val['heartdisease']
del df_test['heartdisease']

# %% [markdown]
# One-hot encoding

# %%
#from imblearn.over_sampling import SMOTE  # If using SMOTE for class imbalance

dv = DictVectorizer(sparse=False)




#train_dict = df_train[categorical_col + numerical_col].to_dict(orient='records')
train_dict = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val.to_dict(orient='records')
X_val = dv.transform(val_dict)

test_dict = df_test.to_dict(orient='records')
X_test = dv.transform(test_dict)
#dv.feature_names_

# # Step 1: Scale the data
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_val_scaled = scaler.transform(X_val)

# # Step 2: Handle class imbalance
# smote = SMOTE()
# X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# %%
output_file = f'rf.bin'
max_depth = 10
min_samples_leaf = 1
df_full_train = df_full_train.reset_index(drop=True)
y_full_train = df_full_train.heartdisease.values
del df_full_train['heartdisease']
dicts_full_train = df_full_train.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)

#rf = RandomForestClassifier(n_estimators=10, random_state=1, n_jobs=-1)
rf = RandomForestClassifier(n_estimators=200,
                            max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf,
                            random_state=1)
rf.fit(X_full_train, y_full_train)
y_pred = rf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print (f'auc={auc}')

# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, rf), f_out)

print(f'the model is saved to {output_file}')




