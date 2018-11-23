from pymongo import MongoClient
from pymongo import errors as pe
from pymongo import ReadPreference

class ExchangeConn:

    def connect_mongo(self):

        try:
            connection = MongoClient(
                'mongodb://admin:ncc1701d@cluster1-shard-00-00-dvlfk.mongodb.net:27017,cluster1-shard-00-01-dvlfk.mongodb.net:27017,cluster1-shard-00-02-dvlfk.mongodb.net:27017/test?ssl=true&replicaSet=Cluster1-shard-0&authSource=admin&retryWrites=true',
                maxPoolSize=5, connect=True,
                read_preference=ReadPreference.NEAREST,
                readPreference='secondaryPreferred')

            return connection

        except pe.ServerSelectionTimeoutError as e:
            return 'connection fail'
