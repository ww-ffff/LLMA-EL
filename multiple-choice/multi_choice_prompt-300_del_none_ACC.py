# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import string
import re
import random
from nltk.tokenize.treebank import TreebankWordDetokenizer
import torch
import matplotlib.pyplot as plt
import time
import json
import math
import ipdb
import ast
import pickle
import os
import openai
import requests

def write_result(file_path,doc_list):
    # 打开 JSON 文件，如果文件不存在则创建一个新的文件
    with open(file_path, 'a') as f:
        # 将每个计算结果转换为 json 格式，并写入文件中
          json.dump(doc_list, f)
          f.write('\n')


def Multi_choise_chatGPT(chatGPT_input):
  prompt = f"""
  You are an awesome knowledge graph accessing agent. There are many entities with similar names that exist in knowledge graphs which cause ambiguity, such as the fruit 'apple' and the company 'Apple'. Given the sentence, and the interested entity mention in it, you are provided with some candidate entities and their information of description followed by the mentioned entities. Now, your task is to consider carefully which of the candidate entities matches the entity mention in sentence.
  ===NOTICE===
  1. Faced with multiple candidates, simply choose the one you think is most likely.
  2. If the candidate entity explicitly matches the target mention, you should definitively select and return it. If there are no explicit matches, then you should return an option that you consider to be the closest to the target mention.
  3. When the entity mentions in the sentence is an abbreviation or a person's name, do not assume that the candidate entities and entity mentions are unrelated simply because the information of the candidate entities cannot cover the entity mentions in the sentence. Use your understanding and imagination how the entities mentioned in the sentence can be related with the candidate entities.
  4. Please do your best to ensure the candidate entity you have choosed is equivalent to the entity mentioned in the sentence. They should belong to the same type. For example, in the sentence "This is a red [apple], very delicious.", you should choose 'apple' instead of 'Apple' because the former is a fruit while the latter is a company.
  5. [IMPORTANT] Must always remember that your task is to select the correct candidate entity rather than answering questions. Never attempt to answer questions in any other form. Must reply "[CAN 1]", "[CAN 2]" ... "[CAN 10]".
  ===INPUT FORMAT===
  You are provided with the [SENTENCE], the [TARGET MENTION] of the target entity mention, [OPTIONS] include no more than 10 candidate entities, and the question [SELECT BEST OPTION].
  ===OUTPUT FORMAT===
  In order to find the correct candidate entity, you should think step by step, and output in json format. First, you should generate your 'thought' understanding and considering the sentence, the target mention, and all the candidate entities. Never answer the question directly in your thought with any other form. Then output your 'choice', which is the entity that best matches the target mention mentioned in sentence like "[CAN 1]", "[CAN 2]" ... "[CAN 10]".
  ===EXAMPLES===
  1.
  Input:
   [SENTENCE]: The song 'Little [Apple]' is very popular in China.
   [TARGET MENTION]: Apple
   [OPTIONS]:
   [CAN 1]: apple(Q89): fruit of the apple tree
   [CAN 2]: Apple(Q312): American multinational technology company  
   [CAN 3]: Apple Music(Q20056642): Internet online music service by Apple
   [CAN 4]: Mac(Q75687): family of personal computers designed, manufactured, and sold by Apple Inc.
   [CAN 5]: Little Apple(Q17324563): 2014 song by Chopstick Brothers.
   [SELECT BEST OPTION]: Please choose the option that best describes the target mention 'Apple' in the given sentence.
  Output:
  {{
     "thought":"The target mention 'Apple' in the context of a song from China. It's likely referring to the song 'Little Apple' song by Chopstick Brothers.
     "choice":"[CAN 5]"
  }}
  2.
  Input:
   [SENTENCE]: SOCCER - LATE GOALS GIVE [JAPAN] WIN OVER SYRIA.
   [TARGET MENTION]: JAPAN
   [OPTIONS]:
   [CAN 1]: Japan(Q17): island country in East Asia
   [CAN 2]: occupation of Japan(Q696251): Allied occupation of Japan following WWII
   [CAN 3]: Japan national football team(Q170566): men's national association football team representing Japan
   [CAN 4]: Sony Music Entertainment Japan(Q732503): Japanese entertainment conglomerate
   [SELECT BEST OPTION]: Please choose the option that best describes the target mention 'JAPAN' in the given sentence.
  Output:
  {{
     "thought": 'The sentence mentions 'JAPAN' in the context of soccer and winning over Syria. It is most likely referring to the 'Japan national football team(Q170566)' in the context of a soccer match victory.',
     "choice": "[CAN 3]"
  }}

  Input: {chatGPT_input}
  """

  #ipdb.set_trace()
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "user", 
       "content": prompt}
    ],
    temperature=0.2
  )
     
  openai_result_content = completion['choices'][0]['message']['content']
  openai_result_dict = json.loads(openai_result_content)
  #print(openai_result_dict)
  return openai_result_dict


def result_mentions(ture_list,openai_result_list):
   count = 0
   for t_item in ture_list:
      for r_item in openai_result_list:
         if r_item['mention'] == t_item['mention']:
            if r_item['wikidata_id'] == t_item['wikidata_id']:
                  #ipdb.set_trace()
                  count += 1                                                                           
   return count
          
def evaluation_chatGPT(data,mention_Can10):
  data_right_num = 0
  data_total_num = 0

  for i, doc_list in enumerate(data):
   if i >= 2:

    doc_right_num = 0
    doc_total_num = 0

    ctxt_limit = 128
    half_ctxt = int(ctxt_limit/2)

    #获取candidates(使用mention直接调用Wikidata，取200)
    doc_can10_from_Wikidata = mention_Can10[i-2]
    doc_can10_from_Wikidata_list = [can_item['mention'] for can_item in doc_can10_from_Wikidata]

    for mention in data[doc_list]['mentions']:
        instances_num = len(data[doc_list]['mentions'][mention]['instances'])
        
        for instance_index,instance in enumerate(data[doc_list]['mentions'][mention]['instances']):
           print('\n----doc',i,'----','mention:',mention,'----',instance['wikidata_id'][0])
           ture_men_id = instance['wikidata_id'][0][0]

           if instances_num == 1:
              mention_name = mention
           else:
              mention_name = mention + str(instance_index)

           ###获取句子Document###
           data[doc_list]['context'][instance['mention_idx']] = '[' + data[doc_list]['context'][instance['mention_idx']] + ']'
           mention_context = data[doc_list]['context']
           
           if len(mention_context)<ctxt_limit:
              document = data[doc_list]['document']
           else:
              ctxt_len = len(mention_context)
              mention_idx = instance['mention_idx']
              if mention_idx<half_ctxt:
                  short_ctxt = mention_context[0:ctxt_limit]
              elif mention_idx>=half_ctxt and mention_idx<ctxt_len-half_ctxt:
                  short_ctxt = mention_context[mention_idx-half_ctxt:mention_idx+half_ctxt]
              else:
                  short_ctxt = mention_context[ctxt_len-ctxt_limit:ctxt_len]
              document = TreebankWordDetokenizer().detokenize(short_ctxt)
              

           #获取该instance对应的candidates
           mention_Candidates_index = doc_can10_from_Wikidata_list.index(mention_name)
           candidates = doc_can10_from_Wikidata[mention_Candidates_index]['candidates']     
           
           
           # 格式化输入
           chatGPT_input = f"""
            [SENTENCE]: {document}
            [TARGET MENTION]: {mention}"""
           # 将候选实体添加到输入中
           for can_i, candidate in enumerate(candidates, start=1):
               chatGPT_input += f"""
            [CAN {can_i}]: {candidate}"""
           
           chatGPT_input += f"""
            [SELECT BEST OPTION]: Please choose the option that best describes the target mention '{mention}' in the given sentence.]"""
 
           #ipdb.set_trace()
           while 1==1:
              try:
                 sub_doc_result = Multi_choise_chatGPT(chatGPT_input) #256一段的结果，[{mention:,descrption:},{mention:,descrption:}]
                 break
              except Exception as e:
                 time.sleep(20)

           #判断选出来的是CAN还是None
           choice_Qid = "Q0"
           if sub_doc_result['choice'] != '[None]':
              choice_option = re.findall(r'\d+', sub_doc_result['choice'])[0]
              if (0 < int(choice_option) < 11) and (candidates != []):
                  choice_Qid = candidates[int(choice_option)-1][0]

           
           print('choice:',sub_doc_result['choice'],'----choice-Qid:',choice_Qid)

           doc_total_num += 1
           if choice_Qid == ture_men_id:
              doc_right_num += 1
           print('--doc_total_num--',doc_total_num,'-----doc_right_num-----',doc_right_num,'\n')
        
    
    write_result('result-ACC/prompt3-bert-prompt6_candidates300_men128_can10-V1(del-none).json',[doc_total_num,doc_right_num])

  data_right_num += doc_right_num
  data_total_num += doc_total_num
  acc = data_right_num/data_total_num
  print('data_right_num',data_right_num,'data_total_num',data_total_num,'acc',acc)
  return  acc

if __name__=="__main__":
    print('Using OpenAI API.')
    openai.api_key = 'your-key'
    openai.organization = 'your-org'

    AIDA_B_data = pickle.load(open('data/AIDA_B_dict.pickle', 'rb'))


    # 从Candidates的JSON文件中加载candidate entity
    with open('prompt3/bert/prompt6_candidates300_men128_can10-V1.json', 'r') as f:
        mention_Can10 = []
        for line in f.readlines():
            doc_item = json.loads(line)
            mention_Can10.append(doc_item)

    acc = evaluation_chatGPT(AIDA_B_data,mention_Can10)
    ipdb.set_trace()
    print('AIDA_B_data的ACC:',acc)