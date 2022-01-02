from sklearn.preprocessing import MinMaxScaler, StandardScaler


normalizers = {
    "noop": lambda: lambda x: x,
    "minmax": lambda: MinMaxScaler(feature_range=(-1, 1)).fit_transform,
    "normal": lambda: StandardScaler().fit_transform,
}
