from clickhouse_driver import Client
import pandas as pd
import matplotlib.pyplot as plt




client = Client(
                host='127.0.0.1',
                user='savato',
                password='savato',
                database='chpv',
                )

client.execute("use chpv")
sql = 'Select data_format_date,sum(qta_non_offerta) from dump group by data_format_date'
# cols = ['id','cod_cli_for', 'rag_soc', 'cod_prod', 'descr_prod', 'data_doc', 'data_format_date', 'qta_offerta', 'qta_non_offerta', 'val_off', 'val_non_off']
columns = ['data','somma_venduto_non_offerta']
query_result = client.execute(sql, settings = {'max_execution_time' : 3600})
df = pd.DataFrame(query_result, columns = columns)

df.plot(x ='data', y='somma_venduto_non_offerta', kind = 'line')
plt.show()


# columns = ['cod_cli_for', 'rag_soc', 'cod_prod', 'descr_prod', 'data_doc', 'data_format_date', 'qta_offerta', 'qta_non_offerta', 'val_off', 'val_non_off']
