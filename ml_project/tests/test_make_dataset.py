from src.data.make_dataset import get_train_test_data


def test_get_train_test_data_works_properly(fake_data, splitting_params):
    train, test = get_train_test_data(fake_data)
    expected_row_num = splitting_params.val_size * len(fake_data)
    expected_col_num = fake_data.shape[1]
    assert (
        test.shape[0] == expected_row_num
    ), f"Validation dataset has {test.shape[0]} rows, expected: {expected_row_num}"
    assert (
        train.shape[1] == test.shape[1] == expected_col_num
    ), f"Data have {train.shape[1]} columns, expected: {expected_col_num}"
