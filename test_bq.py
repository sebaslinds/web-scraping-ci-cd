from google.cloud import bigquery

client = bigquery.Client(project="domainecareycabaneasucre")

df = client.query("SELECT 1 as test").to_dataframe()

print(df)
