import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import seaborn as sns
import skimage.metrics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from xgboost import DMatrix
import warnings
warnings.simplefilter('ignore')
import xgboost as xgb
from skopt import BayesSearchCV
from sklearn.utils import shuffle

#access the dataset 
df = pd.read_csv('train_dataset.csv')
df = df.dropna()
df_shuffled = shuffle(df, random_state=189)
X = df_shuffled.drop(columns=['sample id','I_q'])
y = df_shuffled['I_q']
#y_norm, min_val, max_val = normalize_target(y)
#load the data into GPU arrays, comment the below lines if you run on CPUs 
#X_gpu = np.asarray(X)
#y_gpu = np.asarray(y)

#define the parameter space to get the optimized values using Bayesian optimization
param_space = {
    'n_estimators': np.arange(50, 1000, 50),
    'max_depth': np.arange(3, 15),
    'learning_rate': np.arange(0.001, 0.1, 0.001),
    'subsample': np.arange(0.5, 1.0, 0.1),
    'colsample_bytree': np.arange(0.5, 1.0, 0.1),
    'gamma': np.arange(0, 1, 0.1),
    'min_child_weight': np.arange(1, 10),
    'reg_lambda': np.arange(0.1, 1, 0.1),
    'reg_alpha': np.arange(0.1, 1, 0.1),
    'colsample_bylevel': np.arange(0.5, 1.0, 0.1)
}

#initialize the XGBoost model, remove device = 'cuda' if you run on CPU's
xgb_reg = xgb.XGBRegressor(tree_method='hist', importance_type='cover', random_state=51)

#We use Skopt library to tume the parameter space
opt = BayesSearchCV(
    xgb_reg,
    param_space,
    n_iter=50,
    cv=5,
    n_jobs=-1,
    random_state=42,
    verbose=0,
    return_train_score=True,
    refit=False,
    optimizer_kwargs={'base_estimator': 'GP'}
)


opt.fit(X, y)


best_params = opt.best_params_

best_score = opt.best_score_
print("Best Parameters:", best_params)
print("Best Score:", best_score)

final_xgb = xgb.XGBRegressor(**best_params, tree_method='hist', importance_type='cover', random_state=51)
final_xgb.fit(X,y)

#get the weights assigned to each feature as cover method type
cover_importance = final_xgb.feature_importances_
print("Feature importance weights:", cover_importance)
#edit the path to save it to desired location
final_xgb.save_model('xgbmodel_vesicles_clustered.json')

df_validate = pd.read_csv('test_dataset.csv')
df_validate = df_validate.dropna()
df_validate.shape

loaded_model = xgb.Booster(model_file='xgbmodel_vesicles_clustered.json')

unique_sample_ids = df_validate['sample id'].unique()
mse_list = []
r2_list = []

fmt     = "%.6f\n"
out_mse = []
out_r2  = []

for i, sample_id in enumerate(unique_sample_ids):
    filtered = df_validate[df_validate['sample id'] == sample_id]
    X_sample_test = filtered.drop(columns=['sample id', 'I_q'])
    y_sample_test = filtered['I_q']
    dmatrix_test = xgb.DMatrix(X_sample_test)
    predicted = loaded_model.predict(dmatrix_test)
    #calculate r2 Score
    r2_score_ = r2_score(y_sample_test, predicted)
    #calculate mse
    mse_ = mean_squared_error(y_sample_test, predicted)
    a_mse = fmt % (mse_)
    out_mse.append(a_mse)
    a_r2 = fmt % (r2_score_)
    out_r2.append(a_r2)

open('Test_Samples_MSE.txt', 'w').writelines(out_mse)
open('Test_Samples_R2.txt', 'w').writelines(out_r2)

df_validate = pd.read_csv('train_dataset.csv')
df_validate = df_validate.dropna()
df_validate.shape

loaded_model = xgb.Booster(model_file='xgbmodel_vesicles_clustered.json')

unique_sample_ids = df_validate['sample id'].unique()
mse_list = []
r2_list = []

fmt     = "%.6f\n"
out_mse = []
out_r2  = []

for i, sample_id in enumerate(unique_sample_ids):
    filtered = df_validate[df_validate['sample id'] == sample_id]
    X_sample_test = filtered.drop(columns=['sample id', 'I_q'])
    y_sample_test = filtered['I_q']
    dmatrix_test = xgb.DMatrix(X_sample_test)
    predicted = loaded_model.predict(dmatrix_test)
    #calculate r2 Score
    r2_score_ = r2_score(y_sample_test, predicted)
    #calculate mse
    mse_ = mean_squared_error(y_sample_test, predicted)
    a_mse = fmt % (mse_)
    out_mse.append(a_mse)
    a_r2 = fmt % (r2_score_)
    out_r2.append(a_r2)

open('Train_Samples_MSE.txt', 'w').writelines(out_mse)
open('Train_Samples_R2.txt', 'w').writelines(out_r2)
