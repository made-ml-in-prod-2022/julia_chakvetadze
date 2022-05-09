from sklearn.utils.validation import check_is_fitted
from src.models.train_predict import build_model


model_type = 'RandomForestClassifier'


def test_build_model(config):
        model = build_model(model_type, config)
        assert model is not None


# def test_model_can_be_trained():
#         _train(args)
#         assert check_is_fitted()
