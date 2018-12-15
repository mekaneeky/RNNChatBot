## Inspired by A.Tier's use of transcript objects
import os
import json
import sys

class BaseModel:

    aggregate_user_corpus = ""
    user_objects = {}
    user_files = {}

    def __init__(self):
        self.data_path = "/home/sha3bola/Datasets/facebook_messages"
        self.generate_paths()
        self.generate_users()
        self.generate_aggregates()

        #Not inheritable right away 
        #self.aggregate_questions = []
        #self.aggregate_answers = [] a generator is a better idea !??
        #prev_message = ""

    def generate_paths(self):
        for folder in os.listdir(self.data_path):
            folder_path = os.path.join(self.data_path, folder)
            for file_name in os.listdir(folder_path):
                if file_name == "message.json":
                    self.user_files[folder.split("_")[0]] = os.path.join(folder_path,file_name)
    @classmethod
    def generate_users(cls):
        for user_name, file_path in cls.user_files.items():
            cls.user_objects[user_name] = cls._generate_user_object(file_path)

    @classmethod
    def filter_users(cls):
        for the_user,the_user_object in cls.user_objects.items():
            import pdb;pdb.set_trace()

    @classmethod
    def generate_aggregates(cls):
        #non generator approach
        for user_object in cls.user_objects.values():

            cls.aggregate_user_corpus += user_object.aggregate_user_messages

        
    
    @staticmethod
    def _generate_user_object(user_path):
        return User(user_path)
         


"""     def list_files:
        if self.user_files:
            return self.user_files
        else:
            print("generate files list ")
            return None """

#Why inheritence and not two co-dependant classes
#Add a flip flop that changes state each time __check_sequential_chunk returns False
#

class User(BaseModel):

    def __init__(self, user_path):
        self.blocked_user = False
        self.timeout_seconds = 86400
        self.messages = {}
        self.counts = {}
        self.question_list = []
        self.answer_list = []
        self.detail_message_pairs = {}
        self.user_path = user_path
        self.aggregate_user_messages = ""
        self._harvest_messages()
        self._harvest_meta_data()
        self._test_compliance()
        self._aggregate_messages()
        self._pair_messages()
        #super(BaseModel, self.__init__)


    def _aggregate_messages(self):
        if self.valid:
            for msg_idx in range(len(self.messages["messages"])):
                try:
                    self.aggregate_user_messages += self.messages["messages"][msg_idx]["content"] + "\n"
                except:
                    continue


    def _harvest_meta_data(self):
        try:
            self.interaction_title = self.messages["title"]
        except:
            print("Headless Message")
            self.interaction_title = "Headless Interaction"
        self.participants = [x["name"] for x in self.messages["participants"]]
        self.still_participant = self.messages["is_still_participant"] #unsure whether this means unblocked
                                                        # still on facebook or something else
        self.thread_type = self.messages["thread_type"]

    def __check_sequential_chunk(self, msgidx): # checks if new index is related to a previous chunk

        # Ensure index is not out of range
        if len(self.messages["messages"]) < msgidx:
            print("Fuck index out of range !")
            raise IndexError
            #return False
        
        msgA = self.messages["messages"][msgidx] #Current index value
        try:
            msgB = self.messages["messages"][msgidx-1] #Previous msg index
        except IndexError:
            raise IndexError("First message doesn't have a prior message")# At first message !

        # Check if a prompt response or a new topic
        if msgB["timestamp_ms"] - msgA["timestamp_ms"] < self.timeout_seconds \
            and msgB["sender_name"] == msgA["sender_name"]:
            return "chunk"
        elif msgB["timestamp_ms"] - msgA["timestamp_ms"] < self.timeout_seconds:
            return "response"
        else:
            return False
        # Add orphan = responseless message
        # Check if delayed response !

        #What else to check for ??

    def _harvest_messages(self):

        on_trial_messages = {}
        with open(self.user_path) as jfd:
            on_trial_messages = json.load(jfd)

        #Reverse messages so they're ascending (why? just so ! easier to read !)
        on_trial_messages["messages"] = list(reversed(on_trial_messages["messages"]))
        self.messages = on_trial_messages

    def _test_compliance(self): # I want 2 user message pairs not group chats
                                # Add raising errors 
        if self.thread_type == "Regular":
            pass
        elif self.thread_type == "RegularGroup" or self.thread_type == "Pending":
            self.valid = False
            return
        else:
            print(self.thread_type)
            import pdb; pdb.set_trace()
        
        ## invalid data check
        if len(self.messages["messages"]) <= 1:
            print("Insufficient user data")
            self.valid = False
            return
            #raise UserWarning

        if len(self.messages["participants"]) != 2:
            print("Invalid: Not 2 participants")
            self.valid = False
            return

        #elif list other tests for errors/edge cases
        else:
            self.valid = True

    def _pair_messages(self):

    # Pair together question pairs by linking question_container[0] with answer_container[0]
    # How to deal with one offs and non-answered offsetting the pairs??
    # Should I just assign one participant to questions and the other to answers  !!
        if not self.valid:
            return

        question_container = []
        answer_container = []
        question_flag = True
        previous_question_flag = None
        message_buffer = ""
        for msgidx in range(len(self.messages["messages"])):
            if msgidx == 0:
                question_container.append(self.messages["messages"][msgidx])
                continue

            msg_arxiv = self.messages["messages"]
            
            current_state = self.__check_sequential_chunk(msgidx)
            if current_state == "chunk": #string to previous
                try:
                    message_buffer += (" " + self.messages["messages"][msgidx]["content"])
                except:
                    continue
            elif current_state == "response": #switch previous and start new
                
                if question_flag == True:
                    question_container.append(message_buffer)
                else:
                    answer_container.append(message_buffer)
                question_flag = not question_flag
                message_buffer = ""

            elif not current_state:
                continue

            previous_question_flag = question_flag

            # Dealing with the edge case of having to drop the buffer at the 
            # final iteration since there is no further iteration
            if msgidx == len(self.messages["messages"])-1:
                if question_flag == True:
                    question_container.append(message_buffer)
                else:
                    answer_container.append(message_buffer)

                self.question_list = question_container
                self.answer_list = answer_container

        


#class KerasWordModel(KerasModel):

    
    

###############
#
#
# Add another model for the generation of embeddings/vectors/ a geometrical space
# that sets a relationship between question words/scentences and answer words/scentences
# 

#krass = KerasModel()

##while True:
#    krass.predict_sequence()