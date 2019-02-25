import psycopg2

class DataSQL():
    def __init__(self):
        self.conn = psycopg2.connect(dbname = "fraud_detection", user = "ubuntu", host="/var/run/postgresql")
        self.cur = self.conn.cursor()

    def inset_data(self, data):
        self.cur.execute("INSERT into fraud_predictions (object_id, prediction, probability_fraud) VALUES (%(fraud_id)s, %(prediction)s, %(probability_fraud)s);", data)
        self.conn.commit()

    def iter_insert(self, tup_ls):
        for tup in tup_ls:
            self.insert_data(tup)

    def get_data(self, ids):
        query = """
                SELECT object_id, prediction, probability_fraud
                FROM fraud_predictions
                ORDER BY idx DESC LIMIT 10;"""
        self.cur.execute(query)
        rows = self.cur.fetchall()
        return rows

    def close(self):
        self.conn.close()
        self.cur.close()

if __name__=='__main__':

    data = 'insert tuple list'
    query = DataSQL()
    query.insert_data(data)
    data = query.get_data(ids)
    query.close()
