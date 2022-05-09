import logging
import sys
import pandas as pd
from argparse import ArgumentParser
import pickle
from src.entities import params, TrainPipelineParams
from src.data.make_dataset import read_csv, get_train_test_data
from sklearn.ensemble import RandomForestClassifier
from src.data.transform_dataset import DataPreprocessor


APPLICATION_NAME = 'ML_project'
logger = logging.getLogger(APPLICATION_NAME)


def parse_arguments():
    """Parser to provide path to dataset"""
    parser = ArgumentParser(__doc__)
    parser.add_argument("--data", "-d", help="Path to dataset", default=None)
    parser.add_argument("--model", "-m", help="Choose model type", default="RandomForestClassifier")
    parser.add_argument("--path", "-p", help="Path to model, transformer and predictions")
    parser.add_argument("--mode", help="Choose 'train' or 'predict' mode")
    return parser.parse_args()


def build_model(model_type: str, train_pipe_params: TrainPipelineParams):
    if model_type == "RandomForestClassifier":
        model = RandomForestClassifier(n_estimators=train_pipe_params.train_params.n_estimators,
                                       max_depth=train_pipe_params.train_params.max_depth,
                                       criterion=train_pipe_params.train_params.criterion,
                                       random_state=train_pipe_params.train_params.random_state)
    else:
        raise NotImplemented(f"No {model_type} implemented")
    return model


def _train(train_pipe_params: TrainPipelineParams):
    logger.info("Loading data")
    data = read_csv(args.data)
    features = data.drop(train_pipe_params.feature_params.target_col, axis=1)
    logger.info("Split data into train and tests")
    train_set, _ = get_train_test_data(features)
    y_train, _ = get_train_test_data(data[train_pipe_params.feature_params.target_col].values)
    model = build_model(args.model, train_pipe_params)
    logger.info("Transform data")
    data_preprocessor = DataPreprocessor(train_pipe_params.feature_params.categorical_features,
                                         train_pipe_params.feature_params.numerical_features)
    data_preprocessor.fit(train_set)
    train_set_scaled = data_preprocessor.transform(train_set)
    logger.info("Training model...")
    model.fit(train_set_scaled, y_train)
    logger.info("Saving model and transformer...")
    serialize_model_transformer(model, data_preprocessor, args.path)
    return model, data_preprocessor


def _predict(train_pipe_params: TrainPipelineParams):
    logger.info("Loading data")
    data = read_csv(args.data)
    logger.info("Loading model and transformer")
    model, transformer = deserialize_model_transformer(args.path)
    features = data.drop(train_pipe_params.feature_params.target_col, axis=1)
    test_set_scaled = transformer.transform(features)
    predictions = model.predict(test_set_scaled)
    pd.DataFrame(predictions).to_csv(args.path + "predictions.csv")
    return predictions


def deserialize_model_transformer(path: str):
    with open(path + "model.pkl", "rb") as fin:
        model = pickle.load(fin)
    with open(path + "transformer.pkl", "rb") as fin:
        transformer = pickle.load(fin)
    return model, transformer


def serialize_model_transformer(model: object, transformer: object, path: str) -> None:
    with open(path + "model.pkl", "wb") as fout:
        pickle.dump(model, fout)
    with open(path + "transformer.pkl", "wb") as fout:
        pickle.dump(transformer, fout)


def main(args):
    config = params.read_training_pipeline_params('config/train_config.yml')
    if args.mode == "train":
        _train(config)
    elif args.mode == "predict":
        _predict(config)
    else:
        raise NotImplemented(f"No mode such as {args.mode} implemented")


if __name__ == "__main__":
    args = parse_arguments()
    sys.exit(main(args))
