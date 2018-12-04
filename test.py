import json
import os
import datetime

data_path = "/home/sha3bola/Datasets/facebook_messages"
messages = {}
message_pairs = []
anomaly_files = []
users = {}
users ["count"] = {}
timeout_seconds = 86400 #
prev_message = ""

os.chdir(data_path)

for folder in os.listdir("."):
    folder_path = os.path.join(data_path, folder)
    print(folder) #Tell us which folder are opening
    os.chdir(folder_path)
    #import pdb; pdb.set_trace()
    for file_name in os.listdir("."):
        print(file_name)
        prev_sender = False
        prev_time = 0
        message_current = None
        question_flag = True
        answer_flag = False

        if file_name == "message.json":
            print("omak") #Found ya3ny but your mother is easier to find
            with open(file_name) as jfile:
                messages = json.load(jfile) ############# File loaded
            #if folder == "saragamal_f115617e43":
            #    import pdb; pdb.set_trace()

            
            if len(messages["messages"]) <= 1: # If no interaction break
                    break
            for message_idx in range(len(messages["messages"])):
                if prev_time == 0:
                    prev_time = messages["messages"][message_idx]["timestamp_ms"]
                    time_diff = 0 #messages["messages"][message_idx]["timestamp_ms"] - prev_time 
                else:
                    time_diff = messages["messages"][message_idx]["timestamp_ms"] - prev_time
                try:
                    message_current = messages["messages"][message_idx]["content"]
                except:
                    print("Empty message ya 5awal")
                    message_current = "Empty Message"
                    pass
                try:
                    _ = messages["messages"][message_idx]["sender_name"]
                except:
                    messages["messages"][message_idx]["sender_name"] = "blocked_user"

                ## If message is still in the same question/statement
                if messages["messages"][message_idx] and time_diff > timeout_seconds:
                    message_current += prev_message
                    prev_message = message_current
                    if message_idx <= len(messages)-1:
                        continue
                    #else:
                        #raise NotImplementedError # Add the 
                else:
                    if question_flag == True:
                        question_flag = False
                        current_question = prev_message
                    else:
                        question_flag = True
                        current_answer = prev_message
                        message_pairs.append( (current_question, current_answer) )            


########################################
##########
#########
########## Problem !!! Disconnect !! Pairs are not put in users !! FIXME Please


            #import pdb; pdb.set_trace()
            users [folder.split("_")[0]] = {}
            users [folder.split("_")[0]] ["messages"] = messages
            users ["count"] [len(messages["messages"])] = folder
        else:
            anomaly_files.append(os.path.join(os.getcwd(),file_name))
    os.chdir("..")

import pdb; pdb.set_trace()