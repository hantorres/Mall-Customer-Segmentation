# Mall-Customer-Segmentation
Using a Gaussian Mixture Model to cluster consumer data.
## Mall Customer Segmentation - Python
Python Jupyter Notebook that loads in raw tabular data containing features from mall customers. The notebook leverages advanced probabilistic modeling techniques like Gaussian Model Mixtures to train several models and evaluate model performance and identified clusters. The model undergoes cluster analysis, where domain-specific insights and patterns are extrapolated from model outputs.
### Data Resource - Kaggle
The dataset was sourced from Kaggle. Click [here](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python/data) for access to the dataset.
### Method
The data is first loaded using the pandas library. The data is preprocessed and assessed for building the model. After processing, the data is scaled to properly train the model. The next step is to create a baseline Gaussian mixture model using standard configurations. This is so that we can assess the effectiveness of advanced clustering techniques against baseline models. Using Scikit-Learn, a basic GMM is trained to cluster the data. Next, train an advanced GMM with more in-depth configurations. In this project, I use BIC (Bayesian Inference Criterion) to identify the number of components hyperparameter for my model. This is comparable to using the elbow method in K-Means clustering. The models are evaluated using log-likelihood, BIC, and AIC (Akaike Information Criterion) to measure clustering performance. These metrics are used to compare the performance of the completed model against the baseline model. Finally, the completed model is undergoes cluster analysis, where real-world patterns, insights, and conclusions are made using the model groupings.

## Code Dependencies, Pre-requisites, and Version History
### Dependencies
The program reqires the following libraries:
1) Pandas
2) Matplotlib
3) Seaborn
4) Scikit-Learn

The notebook was tested using Python version 3.13.9.

### Pre-Requisites and Setup
To install missing external libraries on your local machine, open the command prompt and use the following command:

    pip install <library_name>

For the notebook to run properly, ensure the following files and directories exist in the same directory:
1) Mall_Customers.csv
2) Customer_Segmentation_Model.ipynb

### Version History
V1.0 - The Jupyter Notebook is created. All cells and functions have been tested and are functional. Optimal model is found.

## Run Instructions
Once the dependencies are installed and the pre-requisites and setup have been completed, you are all set to run the notebook.
### Instructions
1) Open IDE.
2) Open the directory containing the notebook and the dataset directory.

Using the notebook:
1) Open the notebook in the IDE.
2) Run all cells in the notebook.
3) Validate model training results using the displayed evaluation metrics, including the dataframe containing log-likelihood, BIC, and AIC scores. Furthermore, the biggest observation should be that the final model uniquely groups the dataset into defined groups, which allow reasonable and actionable insights are elucidated in the conclusion cell of the notebook.
4) Best model can now be used in other applications or further tuned.



## Optimization: Mall Customer Segmentation - Python
Python Jupyter Notebook that benchmarks the baseline model against an optimized version of the model. The notebook leverages advanced optimization techniques like efficient numerics representations and learning dynamics modification, evaluating the performance and efficiency across both models in clustering mall customers. The results of model optimization are discussed and evaluated.
### Method
The baseline model code is copied and pasted from the original model notebook. By setting random state seeds in the original model's production, copying and pasting will produce the same model. The scikit-learn library is used to benchmark memory usage, the time library is used to benchmark training time and prediction speed, and scikit-learn is used to benchmark model performance. 
After benchmarking the baseline model, an optimized model is built. Efficient numerics are implemented by changing the data type of the feature data. Model learning dynamics are modified by adding new hyperparameters and modifying existing hyperparameters to improve performance. The benchmarks are then measured the same way. Finally, benchmark metrics between the baseline and optimized model are compared to assess the impacts of optimization. 

## Code Dependencies, Pre-requisites, and Version History
### Dependencies
The program requires the following libraries:
1) Pandas
2) Scikit-Learn

The notebook was tested using Python version 3.13.9.

### Pre-Requisites and Setup
For the notebook to run properly, ensure the following files and directories exist in the same directory:
1) Mall_Customers.csv
2) Customer_Segmentation_Optimization.ipynb

### Version History
V1.0 - The Jupyter Notebook is created. All cells and functions have been tested and are functional. Optimized model is developed.

## Run Instructions
Once the dependencies are installed and the pre-requisites and setup have been completed, you are all set to run the notebook.
### Instructions
1) Open IDE.
2) Open the directory containing the notebook and the dataset directory.

Using the notebook:
1) Open the notebook in the IDE.
2) Run all cells in the notebook.
3) Validate model training results using the displayed evaluation metrics. In production, I observed significantly reduced memory usage and improved performance metrics. 
4) Optimized model can now be used in other applications or further tuned.
