#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_ipython().system('pip install konlpy')
#!pip install rhinoMorph


# In[2]:


from json import JSONDecodeError

from bs4 import BeautifulSoup
import requests
import json
import re
import pandas as pd
import csv
from collections import Counter
import sys
# from konlpy.tag import Mecab
# from prettyPrint import PrettyPrint as pp
import time
from tqdm import tqdm_notebook


# In[3]:


#from google.colab import drive
#drive.mount('/gdrive')


# In[47]:


########## 빅카인즈 데이터 수집 관련 함수 
class DataManager:
    def __init__(self):
        self.json_header = {
            'Referer': 'https://www.bigkinds.or.kr/v2/news/newsDetailView.do?newsId=01400501.20191231192218002',
            'User-agent': "User-Agent' : 'Mozilla/5.0 (iPhone; CPU iPhone OS 9_1 like Mac OS X) AppleWebKit/601.1.46 (KHTML, like Gecko) Version/9.0 Mobile/13B137 Safari/601.1",
            # 'authority': 'www.bigkinds.or.kr',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            # 'content-type': 'application/json; charset=UTF-8',
            'Accept-encoding': 'gzip, deflate, br',
            'Method': 'GET',
            'Host': 'www.bigkinds.or.kr',
            'Sec-fetch-mode': 'cors',
            'Cookie': '_ga=GA1.3.1622290026.1604467134; _gid=GA1.3.1877762275.1608111801; Bigkinds=41885DB8CA4F4EBCA01AA7AE149E2323; qna_692=qna_692; qna_677=qna_677',
            'Sec-fetch-site': 'same-origin',
            'X-requested-with': 'XMLHttpRequest',
            'Connection': 'keep-alive'
        }

    # def url_response(self, news_id): ## 원본
    #     try:
    #         url = "https://www.bigkinds.or.kr/news/newsDetailView.do?newsId=" + news_id
    #         # url = "https://www.bigkinds.or.kr/news/detailView.do?docId=" + news_id
    #         print(url)
    #         req = requests.get(url, self.json_header)
    #         data = json.loads(req.text)
    #         print(data)
    #         # html = response.text
    #         # soup = BeautifulSoup(html, 'lxml')
    #         # print(soup)
    #         # return soup
    #     except:
    #         print('url_respone error (' + news_id + ')')
    #
    #     return False

    def url_response(self, news_id):
        print(news_id)
        url = "https://www.bigkinds.or.kr/news/detailView.do"
        # url = "https://www.bigkinds.or.kr/news/detailView.do?docId=" + news_id
        params = {'docId': news_id, 'returnCnt': 1, 'sectionDiv':1000}
        response = requests.get(url, headers = self.json_header, data=params)
        print(response.status_code)
        # print(req.json())



        # html = response.text
        # soup = BeautifulSoup(html, 'lxml')
        # print(soup)
        # return soup

        return False

    def load_csv(self, file):
        csv_data = pd.read_csv(file, encoding='utf-8', converters={'뉴스 식별자': lambda x: str(x)}) #뉴스 식별자 값이 0으로 시작하기 때문에 문자열로 변환하여 읽어옴
        return csv_data

    def get_news_content(self, soup):
        #contents = soup.select_one("#snsForm > div.snsContent > div > div.snsContentGrp.newsContent > div.doc_title > div")
        contents = soup.select_one("#snsForm > div.snsContent > div > div.snsContentGrp.newsContent > div.doc_title > div.doc_desc")
        content =""
        if contents is not None:
            content = contents.text
        else:
            content = soup.find("div", {"class": "doc_desc"}).text
        return content

    def update_news_contents(self, csv_data):
        index = 0
        total = csv_data.shape[0]
        for news_id in csv_data['뉴스 식별자']:
            print('[{} / {}]  {}'.format(index, total, news_id))
            #print(news_id)
            soup = self.url_response(news_id)
            text = self.get_news_content(soup)
            text = self.text_cleaning(text)
            csv_data.set_value(index, '본문', text)
            index = index + 1

        self.write_csv_file(csv_data)
        
        return csv_data

    def write_csv_file(csv_data, path, file):
        csv_data.to_csv(path+"/"+file, mode='a')
    
    def text_cleanning(text):
        text = re.sub('<.+?>', '', text, 0).strip()
        pattern_email = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9]+\.[a-zA-Z0-9.]+)'
        repl = ''
        text = re.sub(pattern=pattern_email, repl=repl, string=text)
        text = re.sub('[-=+,#/\?:^$.@*\"”※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', repl, text)
        text = text.replace('·', '')
        text = text.replace('“','')
        text = text.replace('”','')
        return text
    
    def getKeywordSentence(dataframe): ##키워드가 포함된 문장만 추출
        index = 0
        total = len(dataframe.index)
        keywordSentences = []
        for lines in dataframe:
            splitList = lines.split(".")
            find_line = []
            for line in splitList:
                if line.find("기업경기") > 0 or line.find("기업 경기") > 0:
                    find_line.append(line)
            keywordSentences = keywordSentences + find_line
            pp.printProgress(index+1, total, "Find Sentence", "Completed")
            time.sleep(0.0025)
            index = index + 1
        return keywordSentences
        
    # def update_keyword(csv_data, path, file):
    #     index = 0
    #     total = csv_data.shape[0]
    #     morpher = Mecab()
    #     keyword_list = []
    #     for text in tqdm_notebook(csv_data['본문']):
    #         nouns = morpher.nouns(text)
    #         new_nouns = [term for term in nouns if len(term) is not 1]
    #         strList = ','.join(new_nouns)
    #         keyword_list.append(strList)
    #         index = index + 1
    #     csv_data['추출명사'] = keyword_list
    #     csv_data.to_csv(path+'/'+file, mode='w')
    #
    #     return csv_data

    # def getSingleDataSet(column_name, csv_data):
    #     iteration = 1
    #     total = csv_data.shape[0]
    #     get_data = []
    #     for keyword in csv_data[column_name]:
    #         get_data.append(keyword)
    #         printProgress(iteration, total, "Get data", "Completed")
    #     return get_data
    
    def csv_load_by_date(file, startDate, endDate):
        csv_data = pd.read_csv(file, encoding='utf-8', converters={'뉴스 식별자': lambda x: str(x)}) #뉴스 식별자 
        csv_data = csv_data[csv_data['일자'] <= endDate]
        csv_data = csv_data[csv_data['일자'] >= startDate]
        csv_data.drop(csv_data.columns[csv_data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
        return csv_data
                
        

