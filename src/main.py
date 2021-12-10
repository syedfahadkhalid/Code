from Preprocessing_helper import *
from Data_Model_helper import *
from Data_Augmentation_n_Sentiment_helper import *
from simpletransformers.language_representation import RepresentationModel

from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model



class Albert(object):
    def __init__(self, data, batch_size, num_epochs):
        self.data = data
        self.num_epochs = num_epochs
        self.batch_size = batch_size
      
    def Bert_SentenceEmbedding(self,sentences):  #sentences
      #sentences = ["Machine Learning and Deep Learning are part of AI", "Data Science will excel in future"] #it should always be a list
      model = RepresentationModel(
        model_type="bert",
        model_name="bert-base-uncased",
        use_cuda=False
      )
      sentence_vectors = model.encode_sentences(sentences, combine_strategy='sum')
      return sentence_vectors

    def fit_albert(self):
      clean_data = preprocessing_tweet(self.data)
      train_df, val_df, test_df = split_data(clean_data)
      X_train, y_train = extract_tweet_and_y(train_df)
      X_val, y_val = extract_tweet_and_y(val_df)
      X_test, y_test = extract_tweet_and_y(test_df)

      y_raw, class_weight_raw = prepare_target(clean_data['class'])
      y_train, class_weight_train = prepare_target(y_train)
      y_val, class_weight_val = prepare_target(y_val)
      y_test, class_weight_test = prepare_target(y_test)


      #Albert Model Tokenizer
      X_train_albert, X_val_albert, X_test_albert, vocab_size, word_index = keras_tokenizer(X_train,X_val,X_test, maxnumwords=100)

      #Use GloVe or None Embedding
      # embedding_matrix = GloveTwitterEmbedding(vocab_size,word_index)
      embedding_matrix = None
      # embedding_matrix = self.Bert_SentenceEmbedding(clean_data['clean_tweet'])

      model = SentenceTransformer('bert-base-nli-max-tokens')
      # embedding_matrix1 = np.array(model.encode(clean_data['clean_tweet'], show_progress_bar=True))
      # np.savetxt('test.out', embedding_matrix1, delimiter=',')

      #Albert Model
      Albertmodel = albert_model(param={'Max_length': 100,
                                        'Vocab Size': vocab_size,
                                        'Embedding Matrix': embedding_matrix,
                                        'dropout':0.20,
                                        'first_layer' : 128,
                                        'second_layer' : 64,
                                        })
      Albertmodel = compile_model(Albertmodel)

      self.Albertmodel, history_Albert = train_model(Albertmodel, X_train_albert, y_train, X_val_albert, y_val, batch_size=self.batch_size, num_epochs=self.num_epochs, class_weight=class_weight_train)

      probs = self.Albertmodel.predict(X_test_albert)
      print("#"*100)
      print("----------------------------      Evaluation on Testing Data     -----------------------------------")
      print("#"*100)
      print(" ")
      plot_confusion_matrix(probs, y_test)
      print_classification_report(probs, y_test)

    def predict(self, New_tweet):
      
      if isinstance(New_tweet, list):
        kt = Tokenizer()
        kt.fit_on_texts(New_tweet)
        tweet_vectors = kt.texts_to_sequences(New_tweet) #Converting text to a vector of word indexes
        tweet_padded = pad_sequences(tweet_vectors, maxlen=100, padding='post')
        self.prediction_prods = self.Albertmodel.predict(tweet_padded)
        self.prediction = self.prediction_prods.argmax(1)
        if self.prediction[0]==1 & len(New_tweet) ==1:
          print("This is classified as Offensive")
        elif self.prediction[0] ==0 & len(New_tweet) ==1:
          print("This is classified as Hate")
        elif self.prediction[0] ==2 & len(New_tweet) ==1:
          print("This is classified as Neither Offensive nor Hate")

      elif isinstance(New_tweet, pd.core.series.Series):
        kt = Tokenizer()
        kt.fit_on_texts(New_tweet)
        tweet_vectors = kt.texts_to_sequences(New_tweet) #Converting text to a vector of word indexes
        tweet_padded = pad_sequences(tweet_vectors, maxlen=100, padding='post')
        self.prediction_prods = self.Albertmodel.predict(tweet_padded)
        self.prediction = self.prediction_prods.argmax(1)

      elif isinstance(New_tweet, str):
        kt = Tokenizer()
        kt.fit_on_texts([New_tweet])
        tweet_vectors = kt.texts_to_sequences([New_tweet]) #Converting text to a vector of word indexes
        tweet_padded = pad_sequences(tweet_vectors, maxlen=100, padding='post')
        self.prediction_prods = self.Albertmodel.predict(tweet_padded)
        self.prediction = self.prediction_prods.argmax(1)
        if self.prediction[0]==1:
          print("This is classified as Offensive")
        elif self.prediction[0] ==0:
          print("This is classified as Hate")
        elif self.prediction[0] ==2:
          print("This is classified as Neither Offensive nor Hate")

      elif isinstance(New_tweet, pd.core.frame.DataFrame):
        kt = Tokenizer()
        kt.fit_on_texts(New_tweet.values.tolist())
        tweet_vectors = kt.texts_to_sequences(New_tweet.values.tolist()) #Converting text to a vector of word indexes
        tweet_padded = pad_sequences(tweet_vectors, maxlen=100, padding='post')
        self.prediction_prods = self.Albertmodel.predict(tweet_padded)
        self.prediction = self.prediction_prods.argmax(1)

      else:
        print("Error!\nInput format is not support. Please try other format")
        
    def check_sentiment(self, New_tweet):
      
      if isinstance(New_tweet, list):
        Albert_Sentiment(New_tweet[0])

      elif isinstance(New_tweet, str):
        Albert_Sentiment(New_tweet)

      else:
        print("Error!\nInput format is not support. Please try other format")

    def corpus_augmentation(self, dataframe, class_label, number_of_augmentation):
      self.corpus_augmentation = docs_augment(dataframe, class_label, number_of_augmentation)

    def doc_augmentation(self, New_tweet):
      doc_aug = data_augment_bert_sw(aug_insert_bert, aug_substitute_bert, aug_swap, New_tweet)
      print("Original Text:")
      print(New_tweet)
      print("Augmented Text:")
      print(doc_aug)
      self.doc_augmentation =doc_aug 
        
        
class Albert_pretrain(object):
    # def __init__(self, data):
    #     self.data = data

    def load_albert(self):
      self.Albertmodel  = load_model("C:\\Users\\Fahad Khalid\\Desktop\\Thesis Code\\my_model")
      # Recreate the exact same model, including its weights and the optimizer
      # self.Albertmodel = tf.keras.models.load_model('C:/Users/Fahad Khalid/Desktop/Thesis Code/model/albert_model.h5')

      print("#"*100)
      print("------------------------      Albert Pretrain Model Loaded Successfully     -----------------------")
      print("#"*100)
      print("Below is model summary")
      # Show the model architecture
      self.Albertmodel.summary()

    def predict(self, New_tweet):
      
      if isinstance(New_tweet, list):
        kt = Tokenizer()
        kt.fit_on_texts(New_tweet)
        tweet_vectors = kt.texts_to_sequences(New_tweet) #Converting text to a vector of word indexes
        tweet_padded = pad_sequences(tweet_vectors, maxlen=100, padding='post')
        self.prediction_prods = self.Albertmodel.predict(tweet_padded)
        self.prediction = self.prediction_prods.argmax(1)
        if self.prediction[0]==1 & len(New_tweet) ==1:
          print("This is classified as Offensive")
        elif self.prediction[0] ==0 & len(New_tweet) ==1:
          print("This is classified as Hate")
        elif self.prediction[0] ==2 & len(New_tweet) ==1:
          print("This is classified as Neither Offensive nor Hate")

      elif isinstance(New_tweet, pd.core.series.Series):
        kt = Tokenizer()
        kt.fit_on_texts(New_tweet)
        tweet_vectors = kt.texts_to_sequences(New_tweet) #Converting text to a vector of word indexes
        tweet_padded = pad_sequences(tweet_vectors, maxlen=100, padding='post')
        self.prediction_prods = self.Albertmodel.predict(tweet_padded)
        self.prediction = self.prediction_prods.argmax(1)

      elif isinstance(New_tweet, str):
        kt = Tokenizer()
        kt.fit_on_texts([New_tweet])
        tweet_vectors = kt.texts_to_sequences([New_tweet]) #Converting text to a vector of word indexes
        tweet_padded = pad_sequences(tweet_vectors, maxlen=100, padding='post')
        self.prediction_prods = self.Albertmodel.predict(tweet_padded)
        self.prediction = self.prediction_prods.argmax(1)
        if self.prediction[0]==1:
          print("This is classified as Offensive")
        elif self.prediction[0] ==0:
          print("This is classified as Hate")
        elif self.prediction[0] ==2:
          print("This is classified as Neither Offensive nor Hate")

      elif isinstance(New_tweet, pd.core.frame.DataFrame):
        kt = Tokenizer()
        kt.fit_on_texts(New_tweet.values.tolist())
        tweet_vectors = kt.texts_to_sequences(New_tweet.values.tolist()) #Converting text to a vector of word indexes
        tweet_padded = pad_sequences(tweet_vectors, maxlen=100, padding='post')
        self.prediction_prods = self.Albertmodel.predict(tweet_padded)
        self.prediction = self.prediction_prods.argmax(1)

      else:
        print("Error!\nInput format is not support. Please try other format")
        
    def check_sentiment(self, New_tweet):
      
      if isinstance(New_tweet, list):
        Albert_Sentiment(New_tweet[0])

      elif isinstance(New_tweet, str):
        Albert_Sentiment(New_tweet)

      else:
        print("Error!\nInput format is not support. Please try other format")

    def corpus_augmentation(self, dataframe, class_label, number_of_augmentation):
      self.corpus_augmentation = docs_augment(dataframe, class_label, number_of_augmentation)


    def doc_augmentation(self, New_tweet):
      doc_aug = data_augment_bert_sw(aug_insert_bert, aug_substitute_bert, aug_swap, New_tweet)
      print("Original Text:")
      print(New_tweet)
      print("Augmented Text:")
      print(doc_aug)
      self.doc_augmentation =doc_aug 

if __name__ == "__main__":
    # Pretrain = Albert_pretrain()
    # Pretrain.load_albert()
    # Pretrain.predict("I like it very very much")
    
    Albert = Albert(load_data(), 50, 2)
    Albert.fit_albert()
    Albert.predict("I really hit this one fuck die die die")

