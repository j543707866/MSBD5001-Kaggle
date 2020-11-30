import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error


def extract_date(df, column):
    df[column + "_year"] = df[column].apply(lambda x: x.year)
    df[column + "_month"] = df[column].apply(lambda x: x.month)
    df[column + "_day"] = df[column].apply(lambda x: x.day)
    df[column + "_hour"] = df[column].apply(lambda x: x.hour)
    df[column + "_weekday"] = df[column].apply(lambda x: x.weekday())
    # df[column + "_minute"] = df[column].apply(lambda x: x.minute)
    # df[column + "_second"] = df[column].apply(lambda x: x.second)


if __name__ == "__main__":
    # read data
    train_data = pd.read_csv("train.csv", parse_dates=['date'])
    test_data = pd.read_csv("test.csv", parse_dates=['date'])
    # transform the data to speed, month, day, hour
    # using the month, day, hour as the feature

    extract_date(train_data, 'date')
    extract_date(test_data, 'date')

    test_id = test_data["id"]
    # print(test_id[0])
    # print(train_data)
    train_data = train_data.set_index("id")
    test_data = test_data.set_index("id")

    train_data.drop(["date"], axis=1, inplace=True)
    test_data.drop(["date"], axis=1, inplace=True)

    # print(train_data)
    # print(test_data)

    label = train_data["speed"].astype("float64")
    train_feature = train_data.drop(["speed"], axis=1)

    model = xgb.XGBRegressor(max_depth=7, learning_rate=0.3, n_estimators=160)
    model.fit(train_feature, label)

    loss = mean_squared_error(label, model.predict(train_feature))
    print(loss)
    predict = model.predict(test_data)

    result_len = len(predict)

    # print(predict)
    # maybe can use id?
    prediction_result = []
    for row in range(0, result_len):
        prediction_result.append([int(test_id[row]), predict[row]])
    np_data = np.array(prediction_result)

    pd_data = pd.DataFrame(np_data, columns=['id', 'speed'])
    pd_data.id = pd_data.id.apply(int)
    # print(pd_data)
    pd_data.to_csv('result.csv', index=None)
