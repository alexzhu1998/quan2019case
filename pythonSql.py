import pandas as pd
import pandasql
from pandasql import sqldf ,load_births

# df = pd.read_csv("merged.csv")
# df2 = pd.read_csv("room4.csv")
births = load_births()


print(sqldf("SELECT * FROM births where births >250000 limit 5;", locals()))

q = '''
        select
            date(date) as DOB,
            sum(births) as "Total Births"
        from
            births
        group by
            date
            limit 10;
'''
def pysqldf(q):
    return sqldf(q,globals())


print(sqldf(q,locals()))
