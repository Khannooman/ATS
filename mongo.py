import pymongo
client = pymongo.MongoClient("mongodb://localhost:27017")

db = client["Resume-Filtering"]
resume =  db["Resumes"]

