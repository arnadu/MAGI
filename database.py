
from pymongo import MongoClient
import os
import re
import json

#TODO: there is a flaw in the logic: a user may inadvertently overwrite an existing assessment by giving it the same name as an existing assessment

def get_db_connection():
    """return a connection to the MongoDB database"""
    CONNECTION_STRING = os.getenv("MONGODB_CONNECTION_STRING")
    MONGO_DB = os.getenv("MONGO_DB", default="MAGI_dev")
    client = MongoClient(CONNECTION_STRING)
    db = client[MONGO_DB]
    return db

def get_list_of_app_templates(username=None):
    #get a list of available application templates from the database
    db = get_db_connection()
    collection = db['ApplicationTemplates']
    

    if username:
        filter =  { "$or": [ { "public": True }, { "owner": username } ] }
    else:
        filter = {"public": True}
    
    ret = collection.distinct("app_name", query=filter)
    
    return ret

def get_revisions_of_app_template(app_name):
    #get a list of available documents from the database
    db = get_db_connection()
    collection = db['ApplicationTemplates']
    files = collection.find({"app_name": app_name}, {"revision": 1, "_id": 0}, sort={'revision': -1})
    ret = [f['revision'] for f in files]
    ret.insert(0, "Latest")
    return ret

def get_list_of_assessments(username=None):
    #get a list of available documents from the database
    db = get_db_connection()
    collection = db['Assessments']

    if username:
        filter =  { "$or": [ { "public": True }, { "owner": username } ] }
    else:
        filter = {"public": True}

    files = collection.find(filter, {"name": 1, "_id": 0})
    ret = [f['name'] for f in files]
    #ret.insert(0, "[new]")
    return ret

def save_assessment_todb(name, assessment):
    #save an assessment to the database
    db = get_db_connection()
    collection = db['Assessments']

    query = {"name": name}
    update = {"$set": assessment}

    #TODO: we should make sure that owner is same if there is an existing assessment of same name
    result = collection.update_one(query, update, upsert=True)
    ret = f"OK: saved as {name}: {result.modified_count} modified, {result.upserted_id} upserted"

    return ret

def load_assessment_fromdb(name):
    #load an assessment from the database   
    db = get_db_connection()
    collection = db['Assessments']
    query = {"name": name}
    assessment = collection.find_one(query)

    #BUG: should check for permission

    #load its application template
    template = get_template_fromdb(assessment["app_name"], None)

    return assessment, template

def get_template_fromdb(name, revision=None):
    """load a template from the database; if version is None, return the latest version; if there is no existing template of this name, return None"""
    db = get_db_connection()
    collection = db['ApplicationTemplates']
    query = {"app_name": name}
    if revision=="Latest":
        revision = None
    if revision:
        query["revision"] = int(revision)
        result = collection.find_one(query)
    else:
        result = collection.find_one(query, sort={'revision': -1})
    if result:
        result.pop("_id", None) #remove the mongodb ObjectId field
    return result

def save_template_todb(name, template):
    """save a template to the database; automatically create a version number = largest version number + 1; returs the template with the new version number"""
    #TODO: is there a way to do this atomically?
    db = get_db_connection()
    current_template = get_template_fromdb(name)
    if current_template:
        template["revision"] = current_template["revision"] + 1
    else:
        template["revision"] = 1
    collection = db['ApplicationTemplates'] 
    collection.insert_one(template)
    return template #return the template with the new revision number


def clean_filename(filename):
    """Replace invalid characters in a filename."""
    filename = re.sub(r'[^a-zA-Z0-9-_.]', '_', filename)
    filename = filename.replace(" ", "_")    
    return filename


def dump_templates(path):
    
    db = get_db_connection()
    collection = db['ApplicationTemplates']
   
    list = collection.distinct("app_name")

    filenames={}

    for app_name in list:
        template = get_template_fromdb(app_name)
        filename = clean_filename(app_name)
        filename = f"{filename}.json"
        filenames[app_name] = filename
        print(f"writing: {filename}")
        with open(os.path.join(path, filename), "w") as outfile:
            json.dump(template, outfile, indent=4)

    filename = "app_templates.json"
    print(f"writing: {filename}")
    with open(os.path.join(path, filename), "w") as outfile:
        json.dump(filenames, outfile)


if __name__ == "__main__":

    #ret = get_list_of_app_templates(username="Arnadu")
    #for r in ret:
    #    print(ret)

    #ret = get_list_of_assessments(username=None)
    #for r in ret:
    #    print(ret)

    dump_templates(".")


"""
DATABASE MANAGEMENT 

https://www.mongodb.com/docs/manual/tutorial/backup-and-restore-tools/

1- download  mongodb tools

2- dumpt the database to a file with your connection string
mongodump --uri="mongodb+srv://username:password@cluster0.example.mongodb.net" --db xyz --collection abc -o dump_dir_path

3- restore the database with a new name from the file
mongorestore --uri="mongodb+srv://username:password@cluster0.example.mongodb.net" --db new_db_name -o dump_dir_path/old_db_name

"""

