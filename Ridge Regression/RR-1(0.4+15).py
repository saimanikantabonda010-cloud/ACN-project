import numpy as np
import pandas as pd

# ==========================================
# 1. LOAD DATASETS
# ==========================================

df1 = pd.read_csv("C:/Users/saima/OneDrive/Desktop/ACN project/Datasets/Tuesday-WorkingHours.pcap_ISCX.csv", low_memory=True)
df2 = pd.read_csv("C:/Users/saima/OneDrive/Desktop/ACN project/Datasets/Wednesday-workingHours.pcap_ISCX.csv", low_memory=True)
df3 = pd.read_csv("C:/Users/saima/OneDrive/Desktop/ACN project/Datasets/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", low_memory=True)

# ==========================================
# 2. COMBINE DATASETS
# ==========================================

dataset = pd.concat([df1, df2, df3], ignore_index=True)
dataset.columns = dataset.columns.str.strip()

# ==========================================
# 3. SEPARATE FEATURES AND TARGET
# ==========================================

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# ==========================================
# 4. CONVERT FEATURES TO NUMERIC
# ==========================================

X = X.apply(pd.to_numeric, errors='coerce')

# ==========================================
# 5. HANDLE INFINITE VALUES
# ==========================================

X.replace([np.inf, -np.inf], np.nan, inplace=True)

# ==========================================
# 6. HANDLE MISSING VALUES
# ==========================================

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)

# ==========================================
# 7. ENCODE TARGET VARIABLE
# ==========================================

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# ==========================================
# 8. TRAIN TEST SPLIT
# ==========================================

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0, stratify=y
)

# ==========================================
# 9. PCA
# ==========================================

from sklearn.decomposition import PCA
pca = PCA(n_components=15)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# ==========================================
# 10. RIDGE REGRESSION
# ==========================================

from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)   # alpha is the regularization strength
ridge.fit(X_train, y_train)

# ==========================================
# 11. PREDICTION
# ==========================================

y_pred = ridge.predict(X_test)

# ==========================================
# 12. EVALUATION METRICS
# ==========================================

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# ==========================================
# 13. DISPLAY RESULTS
# ==========================================

print("====================================")
print("RIDGE REGRESSION")
print("TEST SIZE = 0.4")
print("PCA COMPONENTS = 15")
print("====================================")

print("\nR2 Score : {:.4f}".format(r2))
print("MAE      : {:.4f}".format(mae))
print("RMSE     : {:.4f}".format(rmse))
