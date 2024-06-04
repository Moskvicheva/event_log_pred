from Utils.LogFile import LogFile
from Data.data import Data

BASE_FOLDER = "C://Users/acer/Documents/log_prediction_deeplom/edbn/"

all_data = {"Helpdesk": BASE_FOLDER + "Data/Helpdesk.csv",
            "BPIC12": BASE_FOLDER + "Data/BPIC12.csv"}


def get_data(data_name, sep=",", time="completeTime", case="case", activity="event", resource="role", k=1):
    if data_name in all_data:
        d = Data(data_name, LogFile(all_data[data_name], sep, 0, None, time, case, activity_attr=activity, convert=False, k=k))
        if resource:
            d.logfile.keep_attributes([activity, resource, time])
        else:
            d.logfile.keep_attributes([activity, time])
        return d
    print("ERROR: Datafile not found")
    raise NotImplementedError


