import datetime

def get_date_str():
    return datetime.datetime.now().strftime("%y%m%d_%H%M%S")
