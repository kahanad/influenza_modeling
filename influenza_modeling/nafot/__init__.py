import pandas as pd
from impala.dbapi import connect
# from .borders import *
from .coordinates_conversion import *
# from .samples import *
from .stat_areas import *


# def db_connect(host = 'bdl1.eng.tau.ac.il',  database = 'db_trace', port=21050):
#     conn = connect(host = 'bdl1.eng.tau.ac.il', database = 'db_trace')
#     return conn.cursor()
#
#
# def db_home_connect(port=2324):
#     return db_connect(host='localhost', port=port)
#
#
# def update_imsi_list(df, df_original=None, safe=True):
#     path = './data/imsi_list - full.csv'
#     if df_original is None:
#         df_original = pd.read_csv(path)
#     # assert df.columns == df_original.columns, "Must have same cols"
#     x = df[df.checked]
#
#     x.set_index('imsi')
#     df_original.set_index('imsi')
#
#     df_original.update(x)
#
#     if safe:
#         return df_original
#     else:
#         df_original.to_csv(path)