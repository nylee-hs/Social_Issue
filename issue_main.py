from gensim.models import Word2Vec
import gensim
import pandas as pd
import os
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
clue_terms = ['일으켜', '따라', '인해', '의해서', '의하여', '의해']

def make_bigram(the_list):
  bigram = gensim.models.Phrases(the_list, min_count=20, threshold=1)
  bigram_mod = gensim.models.phrases.Phraser(bigram)
  return [bigram_mod[doc] for doc in the_list]

def make_cause_result_csv(seed_term):
    cause_terms = []
    result_terms = []
    select = input('  >> 1) all   2) 따라   3) 의해   4) 의해서   5) 일으켜   6) 의하여   6) 인해 : ' )
    with open('data\look_pat.csv', 'r', encoding='utf-8') as wf:
        while True:
            cause_temp = []
            result_temp = []
            find_seed_index = 0
            find_clu_index = 0
            clu_term = ''
            data = wf.readline()
            if not data:
                break

            new_data = data.split(',')
            new_data.remove('\n')

            if select == '1':
                for i in range(6):
                    if clue_terms[i] in new_data:
                        clu_term = clue_terms[i]
                    else:
                        continue

            if seed_term in new_data:
                find_seed_index = new_data.index('대기오염')
                find_clu_index = new_data.index(clu_term)
                if find_clu_index < find_seed_index:
                    for i in range(find_clu_index):
                        cause_temp.append(new_data[i])
                    cause_terms.extend(cause_temp)
                elif find_clu_index > find_seed_index:
                    for i in range(find_clu_index+1, len(new_data)):
                        result_temp.append(new_data[i])
                    result_terms.extend(result_temp)

                # else:

                # print(f'sentence : {new_data}  ||  clue_term : {clu_term} ')
                # print(f'cause_terms : {cause_temp} || result_terms : {result_temp}')
                # print()

    c_terms_set = set(cause_terms)
    c_terms_set_list = list(c_terms_set)
    r_terms_set = set(result_terms)
    r_terms_set_list = list(r_terms_set)

    df_c = pd.DataFrame({'seed_term': seed_term, 'cause_terms':c_terms_set_list})
    df_r = pd.DataFrame({'seed_term': seed_term, 'result_terms':r_terms_set_list})
    df_c.to_csv('data/result/'+ seed_term+'C.csv', encoding='utf-8', mode='w')
    df_r.to_csv('data/result/'+ seed_term + 'R.csv', encoding='utf-8', mode='w')


def get_similiarity(seed_term, model: gensim.models.Word2Vec):
    df_c = pd.read_csv('data/result/'+seed_term+'C.csv', encoding='cp949')
    df_r = pd.read_csv('data/result/' + seed_term + 'R.csv', encoding='cp949')

    cause_terms = df_c['cause_terms']
    result_c = [model.similarity(seed_term, term) for term in cause_terms]
    print(result_c[:10])


def get_data(seed_term):
    sentence = []
    with open('data/nng_list_new.csv', 'r') as wf:
        while True:
            data = wf.readline()
            if not data:
                break
            data = data.replace(',', '')
            data = data.split()
            # data = ','.join(data)
            sentence.append(data)
    df = pd.DataFrame({'seed_term': seed_term, 'nng_list':sentence})
    return df

def build_w2v_model(sentence):
    global model
    model = Word2Vec(sentence, workers=2, size=100, window=5, min_count=1, sg=0, iter=100)
    model_name = 'w2v.model'
    model.save('data/model/'+model_name)
    return model
    # model.init_sims(replace=True)
    # sol = model.most_similar('대기오염', topn=100)
    # vector_list.append(sol)
    # print(*sol, sep='\n')




def main():

    print('=================================')
    print('== 1. build W2V model         ===')
    print('== 2. get similarity matrix   ===')
    print('== 3. select clue term   ===')
    print('== 4. Quit                    ===')
    print('=================================')

    while(True):
        choice = input('Select your choice : ')
        if choice == '1' :
            df = get_data('대기오염')
            word_list = df['nng_list']
            bi_gram = make_bigram(word_list)
            final_list = [','.join(sentence).replace('_', '').split(',') for sentence in bi_gram]
            nng_set = list(set(','.join([','.join(sentence) for sentence in final_list]).split(',')))
            nng_df = pd.DataFrame({'nng_list': nng_set})
            nng_df.to_csv('data/nng_set.csv', encoding='utf-8', mode='w')
            # new_df = pd.DataFrame({'sentence':final_list})
            # new_df.to_csv('data/nng_list_bigram.csv', encoding='utf-8', mode='w')
            # model = build_w2v_model(final_list)
            # print(model)

        if choice == '2':
            model = Word2Vec.load('data/model/w2v.model')
            nng_set = pd.read_csv('data/nng_set.csv', encoding='utf-8')['nng_list'].tolist()

            df_c = pd.read_csv('data/result/대기오염C.csv')
            seed_c = df_c['seed_term'].tolist()
            term_c = df_c['cause_terms'].tolist()
            df_r = pd.read_csv('data/result/대기오염R.csv')
            seed_r = df_r['seed_term'].tolist()
            term_r = df_r['result_terms'].tolist()
            result_r = [(seed, term, model.similarity(seed, term)) for seed, term in zip(seed_r, term_r) if term in nng_set]
            result_c = [(seed, term, model.similarity(seed, term)) for seed, term in zip(seed_c, term_c) if term in nng_set]

            final_cv = [result[2] for result in result_c]
            final_rv = [result[2] for result in result_r]

            final_df_c = pd.DataFrame({'seed_term': [result[0] for result in result_c], 'terms': [result[1] for result in result_c], 'value': [result[2] for result in result_c], 'type':'C'})
            final_df_r = pd.DataFrame({'seed_term': [result[0] for result in result_r], 'terms': [result[1] for result in result_r], 'value': [result[2] for result in result_r], 'type':'R'})
            final_df = final_df_c.append(final_df_r, ignore_index=True)
            final_df.to_csv('data/result/final_w2v_result.csv', encoding='cp949', mode='w')




        if choice == '3':
            make_cause_result_csv('대기오염')

        if choice == '4':
            break


    # make_bigram()

if __name__ == '__main__':
    main()

# make_cause_result_csv('대기오염')
# build_w2v_model()
# run_w2v('대기오염')
# print(cause_terms[:5])
# print(result_terms[:5])


# w2v_dust_cause()
#w2v_go()
#w2v_go2()
#w2v_result()