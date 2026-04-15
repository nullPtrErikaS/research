import pandas as pd

df_list = pd.DataFrame({'a': [['hi', 'there'], ['another', 'list']]})
try:
    print("Hashing list...")
    pd.util.hash_pandas_object(df_list)
except Exception as e:
    print(f"List error: {e}")

df_tuple = pd.DataFrame({'a': [('hi', 'there'), ('another', 'list')]})
try:
    print("Hashing tuple...")
    res = pd.util.hash_pandas_object(df_tuple)
    print("Tuple Success!")
except Exception as e:
    print(f"Tuple error: {e}")
