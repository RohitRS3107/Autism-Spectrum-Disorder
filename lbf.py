# URL of the LBF model
import requests 
lbf_model_url = 'https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml'
lbf_model_path = 'lbfmodel.yaml'

# Download the model
response = requests.get(lbf_model_url)
with open(lbf_model_path, 'wb') as file:
    file.write(response.content)
