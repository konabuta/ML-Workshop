# Machine Learning Workshop

Machine Learning Workshop materials on Azure Machine Learning

## ターゲットユーザ
- Data Scientist
- Citizen Data Scientist
- Business Analyst
- Data Engineer


## 利用するサービス
### Azure Machine Learning service 
Azure Machine Learning service は、機械学習/深層学習のプロセスを全てカバーする分析プラットフォームです。

<img src="https://docs.microsoft.com/en-us/azure/machine-learning/service/media/concept-azure-machine-learning-architecture/workflow.png" width = "600">   

### Power BI
企業のあらゆるデータも可視化・探索することができるビジネスユーザ向けのサービスです。

<img src="https://powerbicdn.azureedge.net/cvt-fb1e2b82bc75f091b9556cde890c10a6ccc1090e0ce83484c26d20dacbcf8e52/pictures/pages/desktop/provide_800_450.gif?636996593618659388" width="600">

### Azure Databricks
マネージドな Spark プラットフォームです。大量データに対する加工や機械学習を高速に実行することができます。

<img src="https://docs.microsoft.com/ja-jp/azure/azure-databricks/media/what-is-azure-databricks/azure-databricks-overview.png" width="400">


<br/>    
  


## 環境準備
#### Azure Machine Learning service Python SDK

```
pip install --upgrade azureml-sdk[notebooks,automl,explain,contrib] azureml-dataprep
```

詳細は構築手順は[こちらのページ](./Setup-AMLservice.md)をご参照ください。

## [品質管理 (Quality Control)](./Quality-Control) ##

### [ビジュアル要因探索](./Quality-Control/KeyInfluencers)
- Power BI (Desktop/service)
    - Key Influencers
 
### [**Decision Tree による品質の要因探索**](./Quality-Control/Statistics-approach)
- Azure Machine Learning service (Python)
    - メトリック記録、モデル管理、データ管理

### [**自動機械学習による品質予測モデル構築**](./Quality-Control/Quality-Prediction)
- Azure Machine Learning service (Python)
    - Automated Machine Learning

### [**モデル解釈手法による品質の要因探索**](./Quality-Control/Root-Cause-Analysis-Explainability)
- Azure Machine Learning service (Python)
    - Automated Machine Learning
    - Interpretability SDK


### 外観検査モデルのアプリケーションへのデプロイ
- Custom Vision Service による画像分類モデル構築
- ONNXモデルのWindows Machine Learning デプロイ

<br/>

## [設備保全 (Predictive Maintenance)](./Predictive-Maintenance) ##
### [**LSTMによる設備保全**](./Predictive-Maintenance/Predict-RUL-lstm-remote)
- LSTMによるRULの時系列予測モデル作成
- GPU Cluster on Machine Learning Compute 

### 時系列データクラスタリング
- Dynamic Time Warping

### 異常検知
- One-Class SVM

<br/>

## 推薦システム (Recommendation) ##


<br/>

## 需要予測 (Demand Forecasting) ##
### 自動機械学習による需要予測モデルj構築
### 状態空間モデルによる時系列モデルの解釈

<br/>  

## 在庫最適化 (Optimization) ##
<br/>

