import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

# 1. Eğitim verisini yükleme
train_data = pd.read_csv("C:/Users/ataka/Desktop/Coderspace Kaggle/data/reytingli_train.csv")
train_data = pd.get_dummies(train_data, columns=["Positions"], drop_first=True)

X_train = train_data.drop(["id", "value_increased"], axis=1)
y_train = train_data["value_increased"]

# 2. Test verisini yükleme
test_data = pd.read_csv("C:/Users/ataka/Desktop/Coderspace Kaggle/data/reytingli_test.csv")
test_data = pd.get_dummies(test_data, columns=["Positions"], drop_first=True)
X_test_final = test_data.drop("id", axis=1)

# 3. Standartlaştırma
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test_final = scaler.transform(X_test_final)

# 4. KNN Modeli ile eğitme
knn = KNeighborsClassifier(n_neighbors= 11)
knn.fit(X_train, y_train)

# 5. CatBoost Modeli ile eğitme
catboost_model = CatBoostClassifier(iterations=100, depth=3, learning_rate=0.1, verbose=False)
catboost_model.fit(X_train, y_train)

# 6. Test verisi üzerinde tahmin yapma
y_pred_knn = knn.predict(X_test_final)
y_pred_cat = catboost_model.predict(X_test_final)

# 7. Submission dosyası oluşturma
submission_knn = pd.DataFrame({'id': test_data['id'], 'value_increased': y_pred_knn})
submission_cat = pd.DataFrame({'id': test_data['id'], 'value_increased': y_pred_cat})

# 8. Tahminleri CSV dosyasına kaydetme
submission_knn.to_csv("submission_knn.csv", index=False)
submission_cat.to_csv("submission_cat.csv", index=False)

print("KNN ve CatBoost modelleri için tahmin dosyaları oluşturuldu.")
