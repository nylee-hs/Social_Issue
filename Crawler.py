from data_manager import DataManager
import pandas as pd
from tqdm import tqdm_notebook
# from konlpy.tag import Mecab

path = 'news_data/'
fileName = 'test_data'


def main(path, fileName):
    dm = DataManager()
    csv_data = pd.read_csv(path+fileName+'.csv', encoding='utf-8-sig', dtype={'뉴스 식별자':object})
    # csv_data['뉴스 식별자'] = csv_data['뉴스 식별자'].astype(str)
    news_ids = csv_data['뉴스 식별자']
    # print(news_ids)
    soup_list = []
    for nids in news_ids:
        print(nids)
        soup = dm.url_response(news_id=nids)
        soup_list.append(soup)
    contents = []
    # for soup in soup_list:
    #     content = dm.get_news_content(soup)
    #     print(content)
    #     #         content = dm.text_cleanning(content)
    #     contents.append(content)
    # index = 0
    # for content in tqdm_notebook(contents):
    #     csv_data.set_value(index, '본문', content)
    #     index = index + 1
    #     print(content)
    #
    # csv_data.to_csv(path + '/' + fileName + '_a.csv', mode='w', encoding='utf-8-sig')
    # #     return csv_data
    #
    # #     csv_data = dm.load_csv(path+'/'+fileName+'_a.csv')
    # #     print(csv_data['본문'][0])
    # #     dm.update_keyword(csv_data, path, fileName+'_final.csv')
    # #     csv_data = dm.load_csv(path+'/'+fileName+'_final.csv')

main(path, fileName)