"""
author: Peter Looi
"""

import paramiko
import re

"""
the_master_log_tags
The keys are the framework topology pairs, and each of the
values is another dictionary (a subdictionary) where the keys are data
categories and the values are regular expressesions that 
have "regex groups" that denote the location of the 
data categories. If any of the keys in any of the subdictionaries

"""
the_master_log_tags = {
            "tf:resnet" : 
            {
                "Images/sec" : "total images/sec: ([0-9.]*)",
                "Batch Size" : "batch_size_([0-9.]*)_",
                "Run" : "iteration_([0-9]*)"
            },
            "tf:gnmt":
            {
                "Batch Size" : "batch_size_([0-9.]*)_",
                "Run" : "iteration_([0-9]*)",
                "Time (secs)" : ", time ([0-9.]*)s",
                "ms/record (lat)" : "The latency of the model is ([0-9.]*) ms",
                "Records/sec (thrpt)" : "The throughput of the model is ([0-9.]*) sentences/s"
            },
            "tf:ssd":
            {
                "Batch Size" : "batch_size_([0-9.]*)_",
                "Run" : "iteration_([0-9]*)",
                "Time/Batch" : "Time spent per BATCH: ([0-9.]*) seconds", 
                "Images/sec" : lambda data: 2 * float(data["Batch Size"]) / float(data["Time/Batch"]), 
                "Total Time" : "Time spent : ([0-9.]*) seconds"
            },
            "tf:wnd":
            {
                "Run" : "iteration_([0-9]*)", 
                "Batch Size" : "batch_size_([0-9.]*)_", 
                "Time (sec)" : "End-to-End duration is %s ([0-9.]*)", 
                "ms/record" : "Latency is: %s ([0-9.]*)"
            },
            "mxnet:gnmt":
            {
                "Run" : "iteration_([0-9]*)", 
                "Batch Size" : "batch_size_([0-9.]*)_", 
                "Lines/Batch" : lambda data: float(data["Number of Lines"])/float(data["Number of Batches"]), 
                "Number of Lines_Number of Batches_Total Time (sec)_Latency (sec/sent)_Throughput (sent/sec)_" : "Processed ([0-9.]*) lines in ([0-9.]*) batches\\. Total time: ([0-9.]*), sec/sent: ([0-9.]*), sent/sec: ([0-9.]*)" 
                
            },
            "mxnet:resnet":
            {
                "Run" : "iteration_([0-9]*)", 
                "Batch Size" : "batch_size_([0-9.]*)_", 
                "Images/sec" : "Finished with ([0-9.]*) images per second", 
                "Accuracy" : "INFO:root:\\('accuracy', ([0-9.]*)", 
                "top-k-accuracy-5" : "INFO:root:\\('top_k_accuracy_5', ([0-9.]*)"
            },
            "mxnet:ssd":
            {
                "Run" : "iteration_([0-9]*)", 
                "Batch Size" : "batch_size_([0-9.]*)_", 
                "Images/sec" : "batchsize=[0-9.]* ([0-9.]*) imgs/s",
                "Total time (ms)" : "batchsize=[0-9.]* ([0-9.]*) imgs/s",
                "Total time (sec)" : lambda data: float(data["Total time (ms)"])/1000  
            }
            }
            #Processed 2999 lines in 24 batches. Total time: 235.6916, sec/sent: 0.0786, sent/sec: 12.7243

"""
This class is the internal
"""
class CsvData:
    def __init__(self, framework, topology):
        self.data = {}
        self.known_data_categories = make_known_keys(framework, topology)
        for key in self.known_data_categories:
            self.data[key] = []
    """
    def put(self, data_category, value):
        if not(data_category in self.data.keys()):
            raise Exception("Unrecognized data category " + str(data_category))
        self.data[data_category] = value
    """
    def put(self, data_category, value):
        if not(data_category in self.known_data_categories):
            self.data[data_category] = []
            self.known_data_categories.append(data_category)
        self.data[data_category].append(value)
    def get(self, data_category):
        return self.data[data_category]
    def __str__(self):
        #create the header
        ret = ""
        for key in self.known_data_categories:
            ret += str(key) + ","
        if ret.endswith(","): ret = ret[:len(ret) - 1]
        ret += "\n"
        
        #find the length of the longest data array
        length = 0
        for key in self.known_data_categories:
            dataArray = self.data[key]
            if length < len(dataArray):
                length = len(dataArray)
        
        
        #add all the data entry rows to the csv
        for arrayIndex in range(length):
            for key in self.known_data_categories:
                dataArray = self.data[key]
                if len(dataArray) > arrayIndex:
                    ret += str(dataArray[arrayIndex]) + ","
                else:
                    ret += ","
            if ret.endswith(","): ret = ret[:len(ret) - 1]
            ret += "\n"
        return ret
            
        
        

def make_known_keys(framework, topology):
    
    if framework == "tf" and topology == "gnmt":
        return ["Run", "Batch Size", "Time (secs)", "Records/sec (thrpt)", "ms/record (lat)"]
    elif framework == "tf" and topology == "resnet":
        return ["Run", "Batch Size", "Images/sec"]
    elif framework == "tf" and topology == "ssd":
        return ["Run", "Batch Size", "Time/Batch", "Images/sec", "Total Time"]
    elif framework == "tf" and topology == "wnd":
        return ["Run", "Batch Size", "Time (sec)", "ms/record"]
    elif framework == "mxnet" and topology == "resnet":
        return ["Run", "Batch Size", "Images/sec", "Accuracy", "top-k-accuracy-5"]
    elif framework == "mxnet" and topology == "gnmt":
        return ["Run", "Batch Size", "Lines/Batch", "Total Time (sec)", "Throughput (sent/sec)", "Latency (sec/sent)", "Number of Lines", "Number of Batches"]
    elif framework == "mxnet" and topology == "ssd":
        return ["Run", "Batch Size", "Images/sec", "Total time (sec)", "Total time (ms)"]
    else:
        return []
        #raise Exception("Model " + str(framework) + " with topology " + str(topology) + " is not supported.")

def key_string(framework, topology):
    return framework + ":" + topology
    
    
"""
if log is a string that denotes the path to a log file,
    then this method returns an array where the first 
    and only element is the string that is the contents 
    of the log file
if log is an array of strings that all denote paths to log files
    then this method returns an array of strings where the ith
    element of the array is the text in the log file denoted
    by the ith element of the log array
"""
def load_logs(log, ip_address):
    logs = []
    if type(log) == type(""):
        ret = []
        ret.append(load_single_log(log, ip_address))
        return ret
    elif type(log) == type([]):
        ret = []
        for single_log_file in log:
            ret.append(load_single_log(single_log_file, ip_address))
        return ret
    else:
        raise Exception("Input is of type " + type(log) + " but it should have been of type string or of type array[string...]")

"""
log_file_name is a string which denotes the path to the log file which we need to load
returns a string which is equivalent to the text inside the log file
"""
def load_single_log(log_file_name, ip_address):
    if type(log_file_name) != type(""):
        raise Exception("Input is of type " + type(log_file_name) + " but it should have been of type string.")
    """
    f = open(log_file_name)
    logText = ""
    while True:
        line = f.readline()
        if line == "":
            break
        else:
            logText += line
    f.close()
    """
    return read_file(log_file_name, ip_address)#, local_only = True)
    
"""
This function uses regular expressions, which are 
found in the parameter log_tags, and uses those
regular expressions to find the data that is needed,
and the data is returned in the form of a dictionary
"""
def match(log_contents, framework, topology, logFilePath, master_log_tags = the_master_log_tags):
    log_contents += "\n" + logFilePath
    ret_dict = {}
    master_log_tags_key = key_string(framework, topology)
    this_log_tags = master_log_tags[master_log_tags_key]
    for key in this_log_tags.keys():
        data_categories = key.split("_")
        remove("", data_categories)
        #print("data_categories: " + str(data_categories))
        regex = this_log_tags[key]
        matches = None
        if type(regex) == type(""):
            matches = match_regex(log_contents, regex)
        else:
            continue
        """
        if(len(data_categories) != len(matches)):
            print("Log file " + logFilePath + " for framework " + framework + " and topology " + topology + " cannot be parsed.")
            p("len(data_categories) == len(matches): " + str(data_categories) + " = " + str(matches))
        """
        #p("Data categories: " + str(data_categories))
        #p("Matches: " + str(matches))
        found_match_successful = len(data_categories) == len(matches)
        
        if found_match_successful:
            for i in range(len(data_categories)):
                ret_dict[data_categories[i]] = matches[i]
        else:
            for i in range(len(data_categories)):
                ret_dict[data_categories[i]] = "null"
    for key in this_log_tags.keys():
        if type(this_log_tags[key]) == type(lambda x : x):
            the_function_to_generate_value = this_log_tags[key]
            try:
                ret_dict[key] = the_function_to_generate_value(ret_dict)
            except Exception as e:
                print("EXCEPTION: " + str(e))
                ret_dict[key] = "null"
    return ret_dict
"""
this is where the regex is actually implemented
returns an array that contains all the group matches
"""
def match_regex(log_contents, regex):
    #p("matching regex " + regex + " in log contents " + log_contents)
    group_matches = []
    try:
        m = re.search(regex, log_contents)
        if m == None:
            return []
        num_groups = len(m.groups())
        for i in range(1, num_groups + 1):
            group_matches.append(m.group(i))
        return group_matches
    except Exception as e:
        print("EXCEPTION: " + str(e))
        return []
def p(thing):
    print(thing)
def min(a, b):
    if a < b: return a
    return b
def remove(obj, array):
    for i in range(array.count(obj)):
        array.remove(obj)
def write_file(text, path, ip_address, local_only = False):
    if(local_only):
        f = open(path, "w")
        lines = text.split("\n")
        for line in lines:
            f.write(line+"\n")
        f.close()
        return
        
    text_array = text.split("\n")
    for item in text_array:
        cmd = "echo \"" + item.replace("\"", "\\\"") + "\" >> " + path
        p("executing:" + cmd)
        exec_command(cmd, ip_address)
def read_file(path, ip_address, local_only = False):
    if(local_only):
        f = open(path, "r")
        file_string = ""
        while True:
            line = f.readline()
            if line != "":
                file_string += line
            else:
                return file_string
        f.close()
    cmd = "cat " + path
    p("executing:" + cmd)
    a, file_contents, c = exec_command(cmd, ip_address)
    file_string = file_contents.read()
    return file_string
def exec_command(cmd, ip_address, username_="root", password_="intel123"):
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip_address, username=username_, password=password_)
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(cmd, get_pty=True)
    return ssh_stdin, ssh_stdout, ssh_stderr
def assert_valid_framework_topology(framework, topology):
    the_key_string = key_string(framework, topology)
    if the_key_string in the_master_log_tags.keys():
        return
    raise Exception("Framework " + str(framework) + " with topology " + str(topology) + " does not work :(")
def assert_valid_csv_file_name(csv_file_name_):
    return
    """
    try:
        open(csv_file_name_, "w")
        return
    except Exception as e:
        raise Exception("csv file name " + str(csv_file_name_) + " does not work. " + str(e))
    """
def assert_valid_custom_log_tags(custom_log_tags):
    if custom_log_tags == None: return
    if type(custom_log_tags) == type({}):
        for key in custom_log_tags.keys():
            if type(key) != type(""): raise Exception("The key " + str(key) + " in the custom log tags given should have been a string. :'(")
            if custom_log_tags[key] != None and type(custom_log_tags[key]) != type("") and type(custom_log_tags[key]) != type(lambda x: x): raise Exception("The value of the key " + str(custom_log_tags[key]) + " should have been a string or a function.")
def assert_valid_replace_or_add_log_tags(replace_or_add_log_tags):
    if replace_or_add_log_tags == None: raise Exception("the parameter 'replace_or_add_log_tags' cannot be None")
    if replace_or_add_log_tags != "replace" and replace_or_add_log_tags != "add": raise Exception("the value of the parameter replace_or_add_log_tags was " + replace_or_add_log_tags + " but it should have been 'replace' or 'add'")
def assert_valid_log_file_names(log_file_names):
    if log_file_names == None: raise Exception("parameter log_file_names cannot be None")
    if type(log_file_names) != type("") and type(log_file_names) != type([]): raise Exception("parameter log_file_names should have been of type string or of type list")
    
