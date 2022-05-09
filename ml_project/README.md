ml_project
==============================

ML in prod HW1 

Project Organization
------------

    ├── README.md                <- The top-level README for developers using this project.
    │
    ├── config
    │   └── train_config.yml     <- Config for this project
    │
    ├── models                   <- Trained and serialized model and transformer
    │   ├── model.pkl
    │   └── transformer.pkl
    │ 
    │
    ├── notebooks                <- Jupyter notebooks
    │    └── EDA.ipynb             
    │                       
    │ 
    ├── src                      <- Source code for use in this project.
    │   ├── __init__.py          <- Makes src a Python module
    │   │
    │   ├── data                 <- Scripts to prepare data
    │   │   ├── __init__.py      <- Makes data a Python module
    │   │   ├── make_dataset.py
    │   │   └── transform_dataset.py  
    │   │
    │   ├── entities            <- dataclasses with parameters for training, prediction, quality measurements 
    │   │   ├── __init__.py.py
    │   │   └── params.py
    │   │
    │   ├── models              <- Scripts to train models and then use trained models to make predictions
    │   │   ├── __init__.py
    │   │   └── train_predict.py
    │
    │   
    ├── tests                   <- Tests for units and full script
    │       ├── conftest.py     
    │       ├── test_make_dataset.py
    │       ├── test_transform_dataset.py
    │       └── test_train_predict.py
    │
    │
    ├── requirements.txt         <- The requirements file for reproducing the analysis environment, e.g.
    │                            generated with `pip freeze > requirements.txt`
    │
    ├── setup.py                 <- makes project pip installable (pip install -e .) so src can be imported
            
             
For training: python train_predict.py --data <PATH_TO_DATA> --model <MODEL_NAME> --mode train --path <MODEL_PATH>
For predict: python train_predict.py --data <PATH_TO_DATA> --model <MODEL_NAME> --mode predict --path <MODEL_PATH>

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
