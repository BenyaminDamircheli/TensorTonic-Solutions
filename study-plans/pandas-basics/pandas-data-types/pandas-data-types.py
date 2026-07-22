import pandas as pd

def data_types_overview(data):
    """
    Returns: dict with 'dtypes', 'type_counts', 'num_columns'
    """
    df = pd.DataFrame(data)
    dtypes = {col: str(t) for col, t in df.dtypes.items()}
    type_counts = df.dtypes.astype(str).value_counts().to_dict()
    return {
        "dtypes": dtypes,
        "type_counts": type_counts,
        "num_columns": len(df.columns)
    }