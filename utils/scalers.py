from sklearn.preprocessing import MinMaxScaler, StandardScaler


scalers = {
    "minmax": MinMaxScaler(feature_range=(-1, 1)),
    "standard": StandardScaler(),
}


__all__ = [scalers]