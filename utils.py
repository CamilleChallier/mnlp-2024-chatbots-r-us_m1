import json
import os

def get_api_key():
    with open('api_key.json') as f:
        api_keys = json.load(f)
    user = os.environ['USER']
    return api_keys["api_keys"][user]

def get_sciper():
    with open('api_key.json') as f:
        sciper = json.load(f)
    user = os.environ['USERNAME']
    if user == None :
        user = os.environ['USER']
    return sciper["sciper"][user]