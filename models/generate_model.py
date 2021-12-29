from models.generate_model_ichar import generate_model_ichar
from models.generate_model_icsr import generate_model_icsr
from models.generate_model_hhar import generate_model_hhar
from models.generate_model_wesad import generate_model_wesad

def generate_model(dataset_name):
    if dataset_name == "ICHAR":
        return generate_model_ichar()
    elif dataset_name == "ICSR":
        return generate_model_icsr()
    elif dataset_name == "HHAR":
        return generate_model_hhar()
    elif dataset_name == "WESAD":
        return generate_model_wesad()
    else:
        return None