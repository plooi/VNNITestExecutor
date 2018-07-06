from support import *
import support



csv_data = None
csv_file_name = None
framework = None
topology = None

translation_dict = {
"ssd_vgg16" : "ssd",
"mn" : "mxnet",
"rn50" : "resnet",


}


def extract_data(framework_, topology_, csv_file_name_, log_file_names, print_on_success = True, custom_log_tags = None, replace_or_add_log_tags = "add"):
    global translation_dict
    if topology_ in translation_dict.keys(): topology_ = translation_dict[topology_]
    if framework_ in translation_dict.keys(): framework_ = translation_dict[framework_]

    setup_logging(framework_, topology_, csv_file_name_, custom_log_tags, replace_or_add_log_tags)
    parse_logs(log_file_names)
    write_logs()
    print("Successfully extracted data from " + str(log_file_names) + " into " + csv_file_name_)

"""
I'm a little confused on what the time and iteration parameters are 

If the_master_log_tags_ is None, then the log tags will just be the default log tags
"""
#def setup_logging(framework_, topology_, time, iteration, csv_file_name_, the_master_log_tags_ = None):
def setup_logging(framework_, topology_, csv_file_name_, custom_log_tags = None, replace_or_add_log_tags = "add"):
    # Check if the arguments are valid
    assert_valid_framework_topology(framework_, topology_)
    assert_valid_csv_file_name(csv_file_name_)
    assert_valid_custom_log_tags(custom_log_tags)
    assert_valid_replace_or_add_log_tags(replace_or_add_log_tags)
    
    global csv_data
    global csv_file_name, framework, topology
    csv_data = CsvData(framework_, topology_)
    #print("csv data is: " + csv_data)
    csv_file_name = csv_file_name_
    framework = framework_
    topology = topology_
    if custom_log_tags != None:
        if replace_or_add_log_tags == "replace":
            support.the_master_log_tags = custom_log_tags
        elif replace_or_add_log_tags == "add":
            for key in custom_log_tags.keys():
                support.the_master_log_tags[key] = custom_log_tags[key]
                
def parse_logs(log_file_name):
    assert_valid_log_file_names(log_file_name)
    global csv_data, framework, topology
    #if type(log_file_name) == type([]):
    #load the log
    #or if it's an array, load all of the logs
    logs = load_logs(log_file_name)#logs is an array of logs. Each log is a string. So it's an array of strings.
    
    log_file_array = None
    if type(log_file_name) == type(''):
        log_file_array = [log_file_name]
    else:
        log_file_array = log_file_name
    i = 0
    for log_contents in logs:
        #p(type(log_contents))
        #p(str((log_contents, framework, topology, log_file_array[i])))
        matches = match(log_contents, framework, topology, log_file_array[i])#AR 
        
        #matches is a dictionary where the keys are the different data categories we want, like throughput, or batch size,
        #and the values of the dictionary are the actual number or string values of the data... like 2 img/sec, or 256
        for data_category in matches.keys():
            csv_data.put(data_category, matches[data_category])
        i += 1
        
    
    
def write_logs():
    global csv_data
    f = open(csv_file_name, "w")
    f.write(str(csv_data))
    f.close()
    
    
    
def main():
    extract_data("mxnet", "ssd", "theOutputCsv.csv", ["iteration_1/mxnet_ssd_batch_size_1__2018_07_03_15_22_50.log"])
    """
    setup_logging("mxnet", "ssd", "theOutputCsv.csv")
    parse_logs(["iteration_1/mxnet_ssd_batch_size_1__2018_07_03_15_22_50.log"])
    write_logs()
    """
    #p(match(load_logs("batch_size_1__2018_07_03_15_22_50.log")[0], "tf", "mxnet", "batch_size_1__2018_07_03_15_22_50.log"))
    #p(match_regex("hello. You have images/sec 500", "images/sec ([0-9.]*).* "))
if __name__ == "__main__": main()
            
#(framework, topology, output_csv_file_name, log_files)

















