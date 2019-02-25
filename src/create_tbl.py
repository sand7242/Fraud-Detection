import psycopg2

try:
    conn = psycopg2.connect(dbname = "fraud_detection", user = "ubuntu", host="/var/run/postgresql")
except:
    print("I am unable to connect to the database")

cur = conn.cursor()
try:
    cur.execute(
        """
        CREATE TABLE test_1 (
            fraud_id integer PRIMARY KEY,
            prediction text,
            percentage float
        )
        """
        );
except:
    print("I can't drop our test database!")

conn.commit() # <--- makes sure the change is shown in the database
conn.close()
cur.close()
