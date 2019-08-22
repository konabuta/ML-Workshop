# Machine Learning Workshop

Machine Learning Workshop materials on Azure Machine Learning

## ターゲットユーザ
- Data Scientist
- Citizen Data Scientist
- Business Analyst
- Data Engineer


## 利用するサービス
### [Azure Machine Learning service](https://docs.microsoft.com/ja-JP/azure/machine-learning/service/)
Azure Machine Learning service は、機械学習/深層学習のプロセスを効率的に回すオープンな分析プラットフォームです。

<img src="https://docs.microsoft.com/en-us/azure/machine-learning/service/media/concept-azure-machine-learning-architecture/workflow.png" width = "500">   

### [Power BI](https://docs.microsoft.com/ja-jp/power-bi/)
企業のあらゆるデータも可視化・探索することができる Business Analyst 向けのサービスです。

<img src="https://powerbicdn.azureedge.net/cvt-fb1e2b82bc75f091b9556cde890c10a6ccc1090e0ce83484c26d20dacbcf8e52/pictures/pages/desktop/provide_800_450.gif?636996593618659388" width="400">

### [Azure Databricks](https://docs.azuredatabricks.net/)
マネージドな Spark プラットフォームです。大量データに対する加工や機械学習を高速に実行することができます。

<img src="https://docs.microsoft.com/ja-jp/azure/azure-databricks/media/what-is-azure-databricks/azure-databricks-overview.png" width="400">


<br/>    
  


## 環境準備
#### Azure Machine Learning service Python SDK

Azure Machine Learning service が提供している Notebook VM を利用すると、Python SDK が既にインストールされた Jupyter Notebook を利用することができます。

[Notebook VM 利用手順](https://docs.microsoft.com/ja-JP/azure/machine-learning/service/quickstart-run-cloud-notebook)

それ以外の環境で Azure Machine Learning service を利用する際は、Python SDK をインストールします。

```
pip install --upgrade azureml-sdk[notebooks,automl,explain,contrib] azureml-dataprep
```

詳細は構築手順は[こちらのページ](https://docs.microsoft.com/ja-JP/azure/machine-learning/service/how-to-configure-environment#local)をご参照ください。


<br/>  

### [品質管理 (Quality Control)](./Quality-Control)
### [設備保全 (Predictive Maintenance)](./Predictive-Maintenance) ##
### [クラスタリング](./Clustering)
### [Style Transfer](./Style-Transfer)
<!-- 
### 異常検知
- One-Class SVM -->

<!-- ## [推薦システム (Recommendation)](Recommendation) ## -->
<!-- ## 需要予測 (Demand Forecasting) ##
### 自動機械学習による需要予測モデルj構築
### 状態空間モデルによる時系列モデルの解釈 -->
<!-- ## 在庫最適化 (Optimization) ## -->


