import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

folder_path = 'C:/Users/pgs66/Desktop/GoogleDrive/python/FPS_ML_project/Data_merge/merged_data/'  # 여기에 폴더 경로를 입력하세요.

# 폴더 내의 모든 파일 중 CSV 파일만 선택
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

all_dataframes = []  # 각 CSV 파일의 데이터프레임을 저장할 리스트

# 각 CSV 파일을 데이터프레임으로 읽어서 all_dataframes 리스트에 추가
for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    df = pd.read_csv(file_path)
    all_dataframes.append(df)

# 모든 데이터프레임을 하나로 결합
combined_dataframe = pd.concat(all_dataframes, ignore_index=True)

combined_dataframe.drop('Unnamed: 0', axis=1, inplace=True)

check_0_columns = list(combined_dataframe.columns[0:8])

# 특정 열들이 동시에 0인 행을 제거

filtered_df = combined_dataframe.loc[~(combined_dataframe[check_0_columns] == 0).all(axis=1)]

filtered_df['target'] = filtered_df['target_num'].apply(lambda x: 1 if x.startswith('kill') else 0)

y = filtered_df['target']

filtered_df.drop(['target_num', 'target'], axis=1, inplace=True)

import matplotlib.pyplot as plt

cols = filtered_df.columns

# 히스토그램 그리기
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
for i, colname in enumerate(cols):
    row, col = i // 3, i % 3
    ax[row][col].hist(filtered_df[colname], bins=10)
    ax[row][col].set_xlabel(colname)
    ax[row][col].set_ylabel('Frequency')
plt.show()

#log 변환 해보기

filtered_df_log = np.log1p(filtered_df)

# log 히스토그램 그리기
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
for i, colname in enumerate(cols):
    row, col = i // 3, i % 3
    ax[row][col].hist(filtered_df_log[colname], bins=10)
    ax[row][col].set_xlabel(colname)
    ax[row][col].set_ylabel('Frequency')
plt.show()

# 트레인 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(filtered_df, y,
                                                    test_size=0.2, random_state=100, stratify=y)

import numpy as np
import pandas as pd

#분류 시작
#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

parameters = {'n_estimators': [10, 50, 100, 200],
              'max_depth': [5, 10, 20, 30],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [5, 10, 20]}

rfc = RandomForestClassifier(random_state=100)

grid_rfc = GridSearchCV(rfc, parameters, cv=5)
grid_rfc.fit(X_train, y_train)

print("Best parameters for RandomForestClassifier:", grid_rfc.best_params_)
print("Best score for RandomForestClassifier:", grid_rfc.best_score_)

# 최적의 하이퍼파라미터를 사용하여 모델 생성
best_rfc = RandomForestClassifier(random_state=100, **grid_rfc.best_params_)
best_rfc.fit(X_train, y_train)

# 테스트 데이터에 대한 예측 수행
rfc_pred = best_rfc.predict(X_test)

# 정확도 계산
accuracy = accuracy_score(y_test, rfc_pred)
print("Accuracy:", accuracy)

# ROC curve 계산
y_score = best_rfc.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(24, 8))

# ROC curve 시각화
ax1.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('RFC ROC curve')
ax1.legend(loc="lower right")


#XGBClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

S_Scaler = StandardScaler()

S_Scaler.fit(X_train)

X_train_StandardS = S_Scaler.transform(X_train)
X_test_StandardS = S_Scaler.transform(X_test)

parameters = {'n_estimators': [50, 100, 200],
              'max_depth': [3, 6, 9, 12, 24],
              'learning_rate': [0.01, 0.05, 0.1],
              'gamma': [0.05, 0.1, 0.2, 1, 10]}

xgb = XGBClassifier(random_state=100)

grid_xgb = GridSearchCV(xgb, parameters, cv=5)
grid_xgb.fit(X_train_StandardS, y_train)

print("Best parameters for XGBClassifier:", grid_xgb.best_params_)
print("Best score for XGBClassifier:", grid_xgb.best_score_)

# 최적의 하이퍼파라미터를 사용하여 모델 생성
best_XGB = XGBClassifier(random_state=100, **grid_xgb.best_params_)
best_XGB.fit(X_train_StandardS, y_train)

# 테스트 데이터에 대한 예측 수행
XGB_pred = best_XGB.predict(X_test_StandardS)

# 정확도 계산
accuracy = accuracy_score(y_test, XGB_pred)
print("Accuracy:", accuracy)

# ROC curve 계산
y_score = best_XGB.predict_proba(X_test_StandardS)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# ROC curve 시각화
ax2.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('XGB ROC curve')
ax2.legend(loc="lower right")

#SVC
from sklearn.svm import SVC

parameters = {'C': [0.1, 1, 10, 100],
              'kernel': ['linear', 'rbf', 'sigmoid'],
              'gamma' : [0.01, 0.1, 1, 10]}

svc = SVC(random_state=100)

grid_svc = GridSearchCV(svc, parameters, cv=5)
grid_svc.fit(X_train_StandardS, y_train)

print("Best parameters for SVC:", grid_svc.best_params_)
print("Best score for SVC:", grid_svc.best_score_)

# 최적의 하이퍼파라미터를 사용하여 모델 생성
best_svc = SVC(random_state=100, **grid_svc.best_params_, probability=True)
best_svc.fit(X_train_StandardS, y_train)

# 테스트 데이터에 대한 예측 수행
svc_pred = best_svc.predict(X_test_StandardS)

# 정확도 계산
accuracy = accuracy_score(y_test, svc_pred)
print("Accuracy:", accuracy)

# ROC curve 계산
y_score = best_svc.predict_proba(X_test_StandardS)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# ROC curve 시각화
ax3.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('SVC ROC curve')
ax3.legend(loc="lower right")
plt.show()

#LogisticRegression 시작
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

#스케일러 사용
scaler = MinMaxScaler()

scaler.fit(X_train)

X_train_mmScaled = scaler.transform(X_train)
X_test_mmScaled = scaler.transform(X_test)

lr = LogisticRegression(C = 1, solver = 'liblinear')
lr.fit(X_train_mmScaled, y_train)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# 모델 예측값과 실제값으로 Confusion Matrix 생성
y_pred = lr.predict(X_test_mmScaled)
cm = confusion_matrix(y_test, y_pred)

# Confusion Matrix 시각화
plt.matshow(cm, cmap=plt.cm.Blues)
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks([0, 1], ['Negative', 'Positive'])
plt.yticks([0, 1], ['Negative', 'Positive'])
plt.show()

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 모델 예측값으로 ROC Curve 생성
y_score = lr.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# ROC Curve 시각화
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

#테스트 정확도
y_pred_test = lr.predict(X_test_mmScaled)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Test_Accuracy: {test_accuracy:.3f}")

# confusion matrix 출력
test_confusion = confusion_matrix(y_test, y_pred_test)
print(f"Test_Confusion matrix:\n{test_confusion}")


#트레인 정확도
y_pred_train = lr.predict(X_train_mmScaled)
train_accuracy = accuracy_score(y_train, y_pred_train)
print(f"Train_Accuracy: {train_accuracy:.3f}")

# confusion matrix 출력
train_confusion = confusion_matrix(y_train, y_pred_train)
print(f"Train_Confusion matrix:\n{train_confusion}")

# PolynomialFeatures 클래스를 사용하여 변수 간의 상호작용 항 추가
poly = PolynomialFeatures(interaction_only=True, include_bias=False)

poly.fit(X_train_mmScaled)

X_poly_train = poly.transform(X_train_mmScaled)
X_poly_test = poly.transform(X_test_mmScaled)

lr = LogisticRegression(C = 1, solver = 'liblinear')
lr.fit(X_poly_train, y_train)

# 모델 예측값으로 ROC Curve 생성
y_score = lr.decision_function(X_poly_test)
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# ROC Curve 시각화
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic with poly')
plt.legend(loc="lower right")
plt.show()


# poly 테스트 정확도
y_pred_poly_test = lr.predict(X_poly_test)
poly_test_accuracy = accuracy_score(y_test, y_pred_poly_test)
print(f"Poly_Test_Accuracy: {poly_test_accuracy:.3f}")

# confusion matrix 출력
poly_test_confusion = confusion_matrix(y_test, y_pred_poly_test)
print(f"Poly_Test_Confusion matrix:\n{poly_test_confusion}")

# poly 트레인 정확도
y_pred_poly_train = lr.predict(X_poly_train)
poly_train_accuracy = accuracy_score(y_train, y_pred_poly_train)
print(f"Poly_Train_Accuracy: {poly_train_accuracy:.3f}")

# confusion matrix 출력
poly_train_confusion = confusion_matrix(y_train, y_pred_poly_train)
print(f"Poly_Train_Confusion matrix:\n{poly_train_confusion}")
