import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("-m","--model",help="please give a value for model name",default="topoconv")
parser.add_argument('-d',"--dataset",help="please give a value for dataset name",default="chameleon")
parser.add_argument("-l","--labelrate",help="please give a value for label rate",type=int,default = 20)

args = parser.parse_args()

with open("../configs/{}_{}_{}.json".format("topoconv",args.labelrate,args.dataset)) as f:
    config = json.load(f)

params = config['Data_params']
setting = config['Model_Setup']
# print(params)
