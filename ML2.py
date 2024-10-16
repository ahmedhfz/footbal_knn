import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("./data/reytingli_train.csv")
train_data = pd.get_dummies(train_data, columns=["Positions"], drop_first=True)

X = train_data.drop(["id", "value_increased"], axis=1)
y = train_data["value_increased"]

# X_train , X_test , y_train , y_test = train_test_split(X,y, test_size= 0.2 , random_state= 42)
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, train_data.index, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors= 11,algorithm='kd_tree',leaf_size=100,weights='uniform',p=3,    metric='minkowski',)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(len(X_test))
print(y_pred)
submission_knn = pd.DataFrame({'id':train_data.loc[idx_test, 'id'], 'value_increased': y_pred, 'actual_result':train_data.loc[idx_test, 'value_increased']})
submission_knn['is_correct'] = submission_knn['value_increased'] == submission_knn['actual_result']
accuracy = submission_knn['is_correct'].mean() * 100
print(accuracy)
submission_knn.to_csv("submission_knn1.csv", index=False)

print(f"Doğruluk oranı : {accuracy_score(y_test,y_pred)*100}")