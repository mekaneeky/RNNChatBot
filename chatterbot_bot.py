from classcized_dataset import BaseModel
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer, UbuntuCorpusTrainer, ChatterBotCorpusTrainer

class ChatterBotModel(BaseModel):
    
    def __init__(self, name = "shaba7"):
        self.name = name
        self.object = ChatBot(self.name)
    
    def train(self, t_type = "chatter", limit=10000): #accepts "list", "chatter", "twitter"

        if t_type == "chatter":
            self.object.set_trainer(ChatterBotCorpusTrainer)
            self.object.train(
                "chatterbot.corpus.english"
                )

        elif t_type == "list":
            self.object.set_trainer(ListTrainer)
            import pdb;pdb.set_trace()
            self.object.train(self.aggregate_user_corpus[:10000])

        elif t_type == "ubuntu":
            self.object.set_trainer(UbuntuCorpusTrainer)
            self.object.train()
        
    def predict(self):
        while True:
            response = self.object.get_response(input("Input:"))
            print(response)

ass = BaseModel() 
#import pdb;pdb.set_trace()
#namez = input("Who do you want to talk to?\n")
ass = ChatterBotModel()
ass.train()
ass.predict()


    
