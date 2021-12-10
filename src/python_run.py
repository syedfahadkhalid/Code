import os
import sys
import pandas as pd
import io
#os.chdir('Hate_Speech_Detection_MMAI_894_DL/src')

#################################################
# Load Albert pretrain model
################################################
# from main import Albert_pretrain
# Pretrain = Albert_pretrain()
# Pretrain.load_albert()
# print("I hate when I have to call and wake people up")
# Pretrain.predict('I hate when I have to call and wake people up')
# print("The food was meh")
# Pretrain.predict('The food was meh')
# print('He is a best minister india ever had seen')


# data = io.StringIO("""check out our <number>th man .  cowboys nation even with all those faggot ny fans in the stands,
# what happen to them vixen ent bitches they got ran and threw to the side like a foothill bitch,
# <user> im the bitch okay nudes pat,
# the fuck be wrong with these bitches ? nobody knows,
# yall shut up<lolface> make me bitch,
# i hate a i'm pregnant type of bitch .,
# got bitches in the dm  <allcaps> but i don't ever read'em which is y your top <number>,
# baseball season for the win .  yankees this is where the love started,
# little stupid as bitch i don't fuck with yoooooou <elong> . <repeat>,
# overdosing on heavy drugs doesn't sound bad tonight . i do that pussy shit every day .""")
# df = pd.read_csv(data, sep=",",index_col=None)
# Pretrain.predict(df)
#Pretrain.predict(text_dataframe) or List of text

# Pretrain.check_sentiment("I reallly hate this one")
#Pretrain.doc_augmentation("I reallly hate this one")
#Pretrain.corpus_augmentation(text_dataframe)
#Pretrain


#################################################
# Load Albert model and train on your own data
################################################
from main import *
New_data = load_data()
Albert_model = Albert(New_data, 50, 2)    #50 batch size and 2 epoch
Albert_model.fit_albert()

print("Hey, My name is Fahad!")
Albert_model.predict("Hey, My name is Fahad!")
print("You are ugly girl")
Albert_model.predict("You are ugly girl")
#Albert_model.predict(text_dataframe) or List of text

#Albert_model.check_sentiment("I reallly hate this one")
#Albert_model.doc_augmentation("I reallly hate this one")
#Albert_model.corpus_augmentation(text_dataframe)
