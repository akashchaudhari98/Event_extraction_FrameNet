import pandas as pd
import numpy as np
import jsonlines

class maven:
    def __init__(self,
                    data,
                    data_validation):
        self.training_data = data
        self.testing_data = data_validation
        self.sent_list = []
        self.event_list = []
        self.mention_list = []
        self.extraction()
        self.label_preprocessing()
        self.token_labeling()

    def extraction(self):
        with jsonlines.open(self.training_data) as reader:
            for obj in reader:
            #print(obj['content'])
                dict_ = {}
                for i,sen in enumerate(obj['content']):
                    dict_ = {}
                    dict_['tite'] = [obj["title"]]
                    dict_['doc_id'] = [obj["id"]]
                    dict_['sent_id'] = [i]
                    dict_['sentence'] = [sen['sentence']]
                    dict_['sentence_token'] = [sen['tokens']]
                    #cont_dict[i] = sen['sentence']
                    self.sent_list.append(pd.DataFrame.from_dict(dict_))
                for event in obj['events']:
                    event_dict = {}
                    event_dict['doc_id'] =  [obj["id"]]
                    event_dict['type'] = [event['type']]
                    event_dict['type_id'] = [event['type_id']]
                    self.event_list.append(pd.DataFrame.from_dict(event_dict))
                    
                    for mention in event['mention']:
                        mention_dict = {}
                        mention_dict['doc_id'] = [obj["id"]]
                        mention_dict['type'] = [event['type']]
                        mention_dict['type_id'] = [event['type_id']]
                        mention_dict['mention_id'] = [mention['id']]
                        mention_dict['tigger_word'] = [mention['trigger_word']]
                        mention_dict['sent_id'] = [mention['sent_id']]
                        mention_dict['offset'] = [mention['offset']]
                        self.mention_list.append(pd.DataFrame.from_dict(mention_dict))
        self.df_sent = pd.concat(self.sent_list)
        self.df_event_list = pd.concat(self.event_list)
        self.df_mention_list = pd.concat(self.mention_list)

        self.df_sent.to_csv("sentence_ids.csv")
        self.df_event_list.to_csv("event_id.csv")
        self.df_mention_list.to_csv("mention_ids.csv")

    def label_preprocessing(self):
        labels = self.df_mention_list['type'].unique()
        labels_B = ["B-" + lab for i,lab in enumerate(labels)]
        labels_I = ["I-" + lab for i,lab in enumerate(labels)]
        self.label_final = labels_B + labels_I
        self.label_final = pd.read_csv("label_code.csv")
        self.label_final = self.label_final[['0','1']]
        self.label_final = {label['0']:label['1'] for i,label in self.label_final.iterrows()}

    def token_labeling(self):
            offset_list = []
            for i,sent in self.df_sent.iterrows():
                print(i)
                offset_df = self.df_mention_list[(self.df_mention_list['doc_id'] == sent['doc_id']) & (self.df_mention_list['sent_id'] == sent['sent_id'])]
                for i,df in offset_df.iterrows():
                    dict_ = {}
                    empty_array = np.zeros(len(sent['sentence_token']),dtype=int)
                    dict_['doc_id'] = [df['doc_id']]
                    dict_['type'] = [df['type']]
                    dict_['type_id'] = [df['type_id']]
                    dict_['mention_id'] = [df['mention_id']]
                    dict_['tigger_word'] = [df['tigger_word']]
                    dict_['sent_id'] = [df['sent_id']]
                    dict_['offset'] = [df['offset']]
                    dict_['token'] = [sent['sentence_token']]
                    label1 = "I-" + df['type']
                    label1_no = self.label_final[label1]
                    label2 = "B-"+ df['type']
                    label2_no = self.label_final[label2]
                    empty_array[eval(df['offset'])[0]] =  label2_no
                    empty_array[eval(df['offset'])[1]] =  label1_no
                    dict_['BIO_tags'] =  [empty_array.tolist()]
                    offset_list.append(pd.DataFrame.from_dict(dict_)) 
            offset_final_df = pd.concat(offset_list)
            offset_final_df.to_csv("token_tags.csv")
            
                    