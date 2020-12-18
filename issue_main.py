from gensim.models import Word2Vec
import gensim
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import networkx as nx
import matplotlib.font_manager as fm
import pickle
import logging
from pprint import pprint as pp
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Configuration:
    def __init__(self):
        self.file_name=self.get_file_name()

    def get_file_name(self):
        file_name = input('Data File Name : ')
        return file_name

class Issue_Main:
    def __init__(self, config):
        self.file_name = config.file_name
        self.clue_terms = ['일으켜', '따라', '인해', '의해서', '의하여', '의해']

    def make_bigram(self, the_list):
      bigram = gensim.models.Phrases(the_list, min_count=10, threshold=1)
      bigram_mod = gensim.models.phrases.Phraser(bigram)
      with open('data/'+self.file_name+'_nng_list.bigram', mode='wb') as f:
          pickle.dump([bigram_mod[doc] for doc in the_list], file=f)
      return [bigram_mod[doc] for doc in the_list]

    def make_cause_result_csv(self, seed_term):
        cause_terms = []
        result_terms = []
        nng_list = pd.read_csv('data/'+self.file_name+'_nng_set.csv', encoding='utf-8-sig')['nng_list'].to_list()
        look_pat_new = []

        with open('data/'+self.file_name+'_pat.csv', 'r', encoding='utf-8-sig') as wf:
            while True:
                cause_temp = []
                result_temp = []
                find_seed_index = 0
                find_clu_index = 0
                clu_term = ''
                data = wf.readline()

                if not data:
                    break

                data = data.replace('[', '').replace(']', '').replace("'", '').replace(' ', '').replace('\n', '')
                print(data)
                new_data = data.split(',')
                print(new_data)
                # news_data.remove('\n')

                for i in range(6):
                    if self.clue_terms[i] in new_data:
                        clu_term = self.clue_terms[i]
                    else:
                        continue
                    print(f'load data = {new_data}')

                    ## look_pat에 단어들이 bigram 단어인지 확인 후 수정
                    new_sentence = []
                    find_clu_index = new_data.index(clu_term)
                    remove_term = []
                    # print(f'No = {i}')
                    #단서 단어 앞에 단어들 확인
                    for i in range(find_clu_index):
                        count = find_clu_index - i

                        if count > 1:
                            if i != 0:
                                temp_1 = new_data[i] + '_' + new_data[i+1]
                                temp_2 = new_data[i-1] + '_' + new_data[i]
                                if temp_1 in nng_list:
                                    # print(f'temp = {temp} : True')
                                    if temp_2 not in nng_list:
                                        new_sentence.append(temp_1)
                                    else:
                                        new_sentence.append(new_data[i])
                            else:
                                temp_1 = new_data[i] + '_' + new_data[i + 1]
                                if temp_1 in nng_list:
                                    new_sentence.append(temp_1)
                                else:
                                    new_sentence.append(new_data[i])
                        else:
                            temp_2 = new_data[i-1] + '_' + new_data[i]
                            if temp_2 not in nng_list:
                                new_sentence.append(new_data[i])
                    new_sentence.append(clu_term)
                    # print(new_sentence)


                    for i in range(find_clu_index+1, len(new_data)):
                        count = len(new_data) - i

                        # print(f'count = {count}')
                        if count > 1:
                            if i != find_clu_index+1:
                                temp_1 = new_data[i] + '_' + new_data[i+1]
                                temp_2 = new_data[i-1] + '_' + new_data[i]
                                if temp_1 in nng_list:
                                    # print(f'remove_term = {remove_term}')
                                    if temp_2 not in nng_list:
                                        new_sentence.append(temp_1)
                                    # print(f'temp = {temp} : False')
                                    else:
                                        new_sentence.append(new_data[i])
                                    # print(new_sentence)
                            else:
                                temp_1 = new_data[i] + '_' + new_data[i+1]
                                if temp_1 in nng_list:
                                    new_sentence.append(temp_1)
                                else:
                                    new_sentence.append(new_data[i])
                        else:
                            temp_2 = new_data[i-1] + '_' + new_data[i]
                            # print(temp_2, news_data[i])
                            if temp_2 not in nng_list:
                                new_sentence.append(new_data[i])

                        # removed_sentence = []
                        # print(f'new_sentence = {new_sentence}')
                        # for i in range(len(new_sentence)) :
                        #     # print(f'new_sentence[{i}] = {new_sentence[i]}', end = " ")
                        #     if i not in remove_term:
                        #         # print('True')
                        #         removed_sentence.append(new_sentence[i])
                        #         # print(removed_sentence)
                    new_data = new_sentence
                    look_pat_new.append(new_sentence)

                    print(f'new_data = {new_data}')
                    if seed_term in new_data:
                        find_seed_index = new_data.index(seed_term)
                        # print(f'find_seed_index = {find_seed_index}')
                        find_clu_index = new_data.index(clu_term)
                        if find_clu_index < find_seed_index:
                            for i in range(find_clu_index):
                                cause_temp.append(new_data[i])
                            print(cause_temp)
                            cause_terms.extend(cause_temp)
                        if find_clu_index > find_seed_index:
                            for i in range(find_clu_index+1, len(new_data)):
                                result_temp.append(new_data[i])
                            print(result_temp)
                            result_terms.extend(result_temp)

                    print()
        cause_terms.append(seed_term)
        c_terms_set = set(cause_terms)
        c_terms_set_list = list(c_terms_set)

        result_terms.append(seed_term)
        r_terms_set = set(result_terms)
        r_terms_set_list = list(r_terms_set)

        df_c = pd.DataFrame({'seed_term': seed_term, 'cause_terms':c_terms_set_list})
        df_r = pd.DataFrame({'seed_term': seed_term, 'result_terms':r_terms_set_list})
        df_c.to_csv('data/result/'+ self.file_name+'_C.csv', encoding='utf-8-sig', mode='w')
        df_r.to_csv('data/result/'+ self.file_name + '_R.csv', encoding='utf-8-sig', mode='w')


        df_look_pat = pd.DataFrame({'pattern': look_pat_new})
        df_look_pat.to_csv('data/'+self.file_name+'_pat_new.csv', mode='w', encoding='utf-8-sig')


    def get_relations(self, seed_term, model: gensim.models.Word2Vec):
        df_c = pd.read_csv('data/result/'+self.file_name+'_C.csv', encoding='utf-8-sig')
        df_r = pd.read_csv('data/result/' + self.file_name+ '_R.csv', encoding='utf-8-sig')

        cause_terms = df_c['cause_terms']
        result_c = [model.similarity(seed_term, term) for term in cause_terms]
        print(result_c[:10])


    def get_data(self, seed_term):
        sentence = []
        with open('data/'+self.file_name+'.csv', 'r', encoding='utf-8') as wf:
            while True:
                data = wf.readline()
                if not data:
                    break
                data = data.replace('[', '')
                data = data.replace(']', '')
                data = data.replace("'", '')
                data = data.replace(',', '')
                data = data.split()
                # print(data)
                # data = ','.join(data)
                sentence.append(data)

        df = pd.DataFrame({'seed_term': seed_term, 'nng_list':sentence})
        return df

    def build_w2v_model(self, seed_term):
        sentence_df = pd.read_csv('data/'+self.file_name+'_nng_list_bigram.csv', encoding='utf-8-sig')
        sentences_df_list = sentence_df['sentence'].tolist()

        sentences = [sentence.replace('[', '').replace(']', '').replace(',', '').replace("'", '') for sentence in sentences_df_list]
        sentences = [sentence.split() for sentence in sentences]

        nng_set_df = pd.read_csv('data/'+self.file_name + '_nng_set.csv', encoding='utf-8-sig')
        nng_set = nng_set_df['nng_list'].tolist()

        model = Word2Vec(sentences, workers=4, size=100, window=5, min_count=1, sg=1, iter=100)

        sim_values = []
        for term in nng_set:
            value = model.similarity(seed_term, term)
            print(f'{term} : {value}')
            sim_values.append(value)

        df = pd.DataFrame({'terms': nng_set, 'sim_values': sim_values})
        df.to_csv('data/result/'+self.file_name+'_w2v.csv', encoding='utf-8-sig')

        # model_name = 'w2v.model'
        #
        model.save('data/model/'+self.file_name+'.model')
        # return model
        # model.init_sims(replace=True)
        # sol = model.most_similar('대기오염', topn=100)
        # vector_list.append(sol)
        # print(*sol, sep='\n')


    def rebuild_look_pat(self):
        # loot_pat = pd.read_csv('data/look_pat.csv', encoding='utf-8')
        nng_set = pd.read_csv('data/nng_set.csv', encoding = 'utf-8-sig')

        # print(loot_pat.head())
        print(nng_set.head())

    def make_graph2(self):
        w2v_data = pd.read_csv('data/result/'+self.file_name+'_final_w2v_result.csv', encoding='utf-8-sig')
        cause_df = w2v_data[(w2v_data['type']=='C') & (w2v_data['value'] > 0.0)]
        result_df = w2v_data[(w2v_data['type'] == 'R') & (w2v_data['value'] > 0.0)]

        cause_list_seed = cause_df['seed_term'].to_list()
        result_list_seed = result_df['seed_term'].to_list()

        cause_list_weight = cause_df['value'].to_list()
        nsize_cause = np.array([v for v in cause_list_weight])
        nsize_cause = 2000 * (nsize_cause - min(nsize_cause)) / (max(nsize_cause) - min(nsize_cause))
        cause_list_weight = nsize_cause.tolist()
        print(cause_list_weight)
        cause_list_terms = cause_df['terms'].to_list()

        result_list_terms = result_df['terms'].to_list()
        result_list_weight = result_df['value'].to_list()

        cause_set = []
        result_set = []

        df_cause = pd.DataFrame({'from' : cause_list_seed, 'to' : cause_list_terms, 'weight':cause_list_weight})
        df_result = pd.DataFrame({'from' : result_list_seed, 'to' : result_list_terms, 'weight':result_list_weight})
        print(df_cause)
        # i = 0
        # for s, c, w in zip(cause_list_seed, cause_list_terms, cause_list_weight):
        #     # print(f'[{i}] seed : {s} | term : {c} | weight : {w}')
        #     cause_set.append((s, c, {'weight': w}))
        #     i += 1
        # df_cause = pd.DataFrame({'items':cause_set})
        #
        # j = 0
        # for s, c, w in zip(result_list_seed, result_list_terms, result_list_weight):
        #     # print(f'[{j}] seed : {s} | term : {c} | weight : {w}')
        #     result_set.append((s, c, {'weight': w}))
        #     j += 1
        # df_result = pd.DataFrame({'items':result_set})

        # cause_list = [(s,c, {'weight':w}) for s, c, w in zip(cause_list_seed, cause_list_terms, cause_list_weight)]

        fm.get_fontconfig_fonts()
        # font_location = '/usr/share/fonts/truetype/nanum/NanumGothicOTF.ttf'
        font_location = 'C:/Windows/Fonts/NanumGothic.ttf'  # For Windows
        font_name = fm.FontProperties(fname=font_location).get_name()
        plt.rc('font', family=font_name)

        G_cause = nx.Graph()
        G_cause = nx.from_pandas_edgelist(df_cause, 'from', 'to', create_using=nx.DiGraph())
        # ar_cause = (df_cause['items'])
        # G_cause.add_edges_from(ar_cause)
        # print(ar_cause)
        G_result = nx.Graph()
        G_result = nx.from_pandas_edgelist(df_result, 'from', 'to', create_using=nx.DiGraph())
        # ar_result = (df_result['items'])
        # G_result.add_edges_from(ar_result)
        # nsize = np.array([v for v in cause_list_weight])
        # cause_list_weight.insert(0, 1.000)
        # result_list_weight.insert(0, 1.000)


        # nsize_cause = np.array([v for v in cause_list_weight])
        # nsize_cause = 2000 * (nsize_cause-min(nsize_cause)) / (max(nsize_cause)- min(nsize_cause))
        # print(nsize_cause)
        # nsize_result = np.array([v for v in result_list_weight])
        # nsize_result = 2000 * (nsize_result - min(nsize_result)) / (max(nsize_result) - min(nsize_result))
        # print(nsize_cause)
        # nsize_cause = np.insert(nsize_cause, 0, 1000)
        # print(nsize_cause)
        pos_cause = nx.spring_layout(G_cause)
        pos_result = nx.spring_layout(G_result)
        plt.figure(figsize=(16,12))
        plt.title('원인')
        cmap = cm.get_cmap('Dark2')
        print(G_cause.nodes)
        # nx.draw_networkx(G_cause, font_size=14, font_family=font_name,pos=pos_cause, node_color=list(cause_list_weight), node_size=nsize_cause, alpha=0.7, edge_color='.5', cmap=cmap)
        # nx.draw_networkx(G_cause, font_size=14, font_family=font_name, pos=pos_cause, node_color=list(cause_list_weight), node_size=nsize_cause, alpha=0.7, edge_color='.5', cmap=cmap)

        nx.draw_networkx(G_cause, pos = pos_cause, node_size = 1000, node_color = 'dark', alpha = .1, font_family = font_name, with_labels=True)
        plt.savefig('data/result/'+self.file_name+'_cause.png', bbox_inches='tight')

        nx.draw(G_result, font_family = font_name, with_labels=True)
        # nx.draw_networkx(G_result, font_size=14, font_family=font_name, pos=pos_result, node_color=list(result_list_weight), node_size=nsize_result, alpha=0.7, edge_color='.5', cmap=cmap)
        plt.savefig('data/result/' + self.file_name + '_result.png', bbox_inches='tight')

    def make_graph(self):
        w2v_data = pd.read_csv('data/result/'+self.file_name+'_final_w2v_result.csv', encoding='utf-8-sig')
        cause_df = w2v_data[(w2v_data['type']=='C') & (w2v_data['value'] > 0.0)]
        result_df = w2v_data[(w2v_data['type'] == 'R') & (w2v_data['value'] > 0.0)]
        print(cause_df)
        cause_list_seed = cause_df['seed_term'].to_list()
        result_list_seed = result_df['seed_term'].to_list()

        cause_list_weight = cause_df['value'].to_list()
        nsize_cause = np.array([v for v in cause_list_weight])
        nsize_cause = 2000 * (nsize_cause - min(nsize_cause)) / (max(nsize_cause) - min(nsize_cause))
        cause_list_weight = nsize_cause.tolist()
        print(cause_list_weight)
        cause_list_terms = cause_df['terms'].to_list()

        result_list_terms = result_df['terms'].to_list()
        result_list_weight = result_df['value'].to_list()

        cause_set = []
        result_set = []

        df_cause = pd.DataFrame({'from' : cause_list_seed, 'to' : cause_list_terms, 'weight':cause_list_weight})
        df_result = pd.DataFrame({'from' : result_list_seed, 'to' : result_list_terms, 'weight':result_list_weight})
        print(df_cause)
        # i = 0
        # for s, c, w in zip(cause_list_seed, cause_list_terms, cause_list_weight):
        #     # print(f'[{i}] seed : {s} | term : {c} | weight : {w}')
        #     cause_set.append((s, c, {'weight': w}))
        #     i += 1
        # df_cause = pd.DataFrame({'items':cause_set})
        #
        # j = 0
        # for s, c, w in zip(result_list_seed, result_list_terms, result_list_weight):
        #     # print(f'[{j}] seed : {s} | term : {c} | weight : {w}')
        #     result_set.append((s, c, {'weight': w}))
        #     j += 1
        # df_result = pd.DataFrame({'items':result_set})

        # cause_list = [(s,c, {'weight':w}) for s, c, w in zip(cause_list_seed, cause_list_terms, cause_list_weight)]

        fm.get_fontconfig_fonts()
        # font_location = '/usr/share/fonts/truetype/nanum/NanumGothicOTF.ttf'
        font_location = 'C:/Windows/Fonts/NanumGothic.ttf'  # For Windows
        font_name = fm.FontProperties(fname=font_location).get_name()
        plt.rc('font', family=font_name)

        G_cause = nx.Graph()
        G_cause = nx.from_pandas_edgelist(df_cause, 'from', 'to', create_using=nx.DiGraph())
        # ar_cause = (df_cause['items'])
        # G_cause.add_edges_from(ar_cause)
        # print(ar_cause)
        G_result = nx.Graph()
        G_result = nx.from_pandas_edgelist(df_result, 'from', 'to', create_using=nx.DiGraph())
        # ar_result = (df_result['items'])
        # G_result.add_edges_from(ar_result)
        # nsize = np.array([v for v in cause_list_weight])
        # cause_list_weight.insert(0, 1.000)
        # result_list_weight.insert(0, 1.000)


        # nsize_cause = np.array([v for v in cause_list_weight])
        # nsize_cause = 2000 * (nsize_cause-min(nsize_cause)) / (max(nsize_cause)- min(nsize_cause))
        # print(nsize_cause)
        # nsize_result = np.array([v for v in result_list_weight])
        # nsize_result = 2000 * (nsize_result - min(nsize_result)) / (max(nsize_result) - min(nsize_result))
        # print(nsize_cause)
        # nsize_cause = np.insert(nsize_cause, 0, 1000)
        # print(nsize_cause)
        pos_cause = nx.spring_layout(G_cause)
        pos_result = nx.spring_layout(G_result)
        plt.figure(figsize=(16,12))
        plt.title('원인')
        cmap = cm.get_cmap('Dark2')
        print(G_cause.nodes)
        # nx.draw_networkx(G_cause, font_size=14, font_family=font_name,pos=pos_cause, node_color=list(cause_list_weight), node_size=nsize_cause, alpha=0.7, edge_color='.5', cmap=cmap)
        # nx.draw_networkx(G_cause, font_size=14, font_family=font_name, pos=pos_cause, node_color=list(cause_list_weight), node_size=nsize_cause, alpha=0.7, edge_color='.5', cmap=cmap)

        nx.draw_networkx(G_cause, pos = pos_cause, node_size = 1000, node_color = 'dark', alpha = .1, font_family = font_name, with_labels=True)
        plt.savefig('data/result/'+self.file_name+'_cause.png', bbox_inches='tight')

        nx.draw(G_result, font_family = font_name, with_labels=True)
        # nx.draw_networkx(G_result, font_size=14, font_family=font_name, pos=pos_result, node_color=list(result_list_weight), node_size=nsize_result, alpha=0.7, edge_color='.5', cmap=cmap)
        plt.savefig('data/result/' + self.file_name + '_result.png', bbox_inches='tight')




        # G = nx.Graph()
        # ar =


def main():
    config = Configuration()
    issue_main = Issue_Main(config)
    while(True):
        print('=================================')
        print('== 1. make bigram sentences   ===')
        print('== 2. select clue term        ===')
        print('== 3. build w2v model         ===')
        print('== 4. get similarity matrix   ===')
        print('== 5. get network graph       ===')
        print('== 6. re-try to select file   ===')
        print('== 7. Quit                    ===')
        print('=================================')
        choice = input('Select your choice : ')


        if choice == '1' :
            df = issue_main.get_data('대기_오염')
            word_list = df['nng_list']
            bi_gram = issue_main.make_bigram(word_list)

            # final_list = [','.join(sentence).replace('_', '').split(',') for sentence in bi_gram]
            # print(final_list[:10])
            # print(len(','.join([','.join(sentence) for sentence in bi_gram]).split(',')))
            nng_set = list(set(','.join([','.join(sentence) for sentence in bi_gram]).split(',')))
            # print(nng_set[:10])
            nng_set.remove('')
            nng_df = pd.DataFrame({'nng_list': nng_set})
            nng_df.to_csv('data/'+issue_main.file_name+'_nng_set.csv', encoding='utf-8-sig', mode='w')
            new_df = pd.DataFrame({'sentence':bi_gram})
            new_df.to_csv('data/'+issue_main.file_name+'_nng_list_bigram.csv', encoding='utf-8-sig', mode='w')
            # model = build_w2v_model(final_list)
            # print(model)

        if choice == '2':
            issue_main.make_cause_result_csv('대기_오염')

        if choice == '3':
            issue_main.build_w2v_model('대기_오염')

        if choice == '4':
            model = Word2Vec.load('data/model/'+issue_main.file_name+'.model')
            nng_set = pd.read_csv('data/'+issue_main.file_name+'_nng_set.csv', encoding='utf-8-sig')['nng_list'].tolist()

            df_c = pd.read_csv('data/result/'+issue_main.file_name+'_C.csv')
            seed_c = df_c['seed_term'].tolist()
            term_c = df_c['cause_terms'].tolist()
            df_r = pd.read_csv('data/result/'+issue_main.file_name+'_R.csv')
            seed_r = df_r['seed_term'].tolist()
            term_r = df_r['result_terms'].tolist()
            result_r = [(seed, term, model.similarity(seed, term)) for seed, term in zip(seed_r, term_r) if term in nng_set]
            result_c = [(seed, term, model.similarity(seed, term)) for seed, term in zip(seed_c, term_c) if term in nng_set]

            final_cv = [result[2] for result in result_c]
            final_rv = [result[2] for result in result_r]

            final_df_c = pd.DataFrame({'seed_term': [result[0] for result in result_c], 'terms': [result[1] for result in result_c], 'value': [result[2] for result in result_c], 'type':'C'})
            final_df_r = pd.DataFrame({'seed_term': [result[0] for result in result_r], 'terms': [result[1] for result in result_r], 'value': [result[2] for result in result_r], 'type':'R'})
            final_df = final_df_c.append(final_df_r, ignore_index=True)
            final_df.to_csv('data/result/'+issue_main.file_name+'_final_w2v_result.csv', encoding='utf-8-sig', mode='w')


        if choice == '5':
            issue_main.make_graph()

        if choice == '6':
            config = Configuration()

        if choice == '7':
            break



    # make_bigram()

if __name__ == '__main__':
    main()
    # make_cause_result_csv('대기_오염')
# build_w2v_model()
# run_w2v('대기오염')
# print(cause_terms[:5])
# print(result_terms[:5])


# w2v_dust_cause()
#w2v_go()
#w2v_go2()
#w2v_result()