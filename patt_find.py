import re


clue_list = []
clue_fin = []
data_list = []
tu_list = []
pat_list = []
dict_list = {}
fin_list = []
file_name = '/Users/blossom/Downloads/물부족_mecab.csv'

def first_func():
    print('first')
    with open(file_name, 'r', encoding='utf-8') as sf:
        for fk in range(500):
            data = sf.readline()
            if not data: break
            else:
                if len(data) > 4:  # 비어있는 혹은 너무 짧아 의미없는 기사를 날리기위한 구간입니다.
                    data_list.append(eval(data))

    for aj in data_list:
        for ka in aj:
            if ka[1] == 'VV+EC':
                dict_list[ka[0]] = ka[1]
        #print(*dict_list.items(), sep='\n')


def patt_fi():
    li = ['인해', '따라', '의해', '일으켜', '의해서', '의하여']
    print('second')
    #patt_1 = re.compile("\(.*, 'NNG'\), .*, \('인해', 'VV\+EC'\), .*, \(.*, 'JKO'\)+")
    #patt_1 = re.compile("\(.*, 'NNG'\), .*, \('따라', 'VV\+EC'\), .*, \(.*, 'JKO'\)+")
    #patt_1 = re.compile("\(.*, 'NNG'\), .*, \('의해', 'VV\+EC'\), .*, \(.*, 'JKO'\)+")
    #patt_1 = re.compile("\(.*, 'NNG'\), .*, \('일으켜', 'VV\+EC'\), .*, \(.*, 'JKO'\)+")
    #patt_1 = re.compile("\(.*, 'NNG'\), .*, \('의해서', 'VV\+EC'\), .*, \(.*, 'JKO'\)+")
    #patt_1 = re.compile("\(.*, 'NNG'\), .*, \('의하여', 'VV\+EC'\), .*, \(.*, 'JKO'\)+")

    for ps in li:
        pat_str = "\(.*, 'NNG'\), .*, \('%s', 'VV\+EC'\), .*, \(.*, 'JKO'\)+"%ps
        print(pat_str)
        print((ps, 'VV+EC'))
        patt_1 = re.compile(pat_str)
        for f_data in data_list:
            find_patt = patt_1.findall(str(f_data))
            #print(find_patt)
            if len(find_patt) > 0:  # 패턴에 해당하지 않는 비어있는 리스트를 제외하기 위한 조건문 입니다.
                pat_list.append(find_patt)
        #print(*pat_list, sep='\n')
        for k in pat_list:
            for tu_data in k:
                tu_list.append(list(eval(tu_data)))

        print('third')
        try:
            for find_a in tu_list:
                # print(find_a.index(('인해', 'VV+EC')))
                clue = find_a.index((ps, 'VV+EC'))
                print(clue)
            while True:
                if find_a[clue][1] == 'JKO':
                    clue_list.append(find_a[:clue + 1])
                    break
                clue += 1
        except: continue

        try:
            for find_b in clue_list:
                clue2 = find_b.index((ps, 'VV+EC'))
                find_break = clue2 - 1
                while True:
                    if find_b[find_break][1] == 'NNG' and find_b[find_break - 1][1] != 'NNG':
                        # if find_b[find_break][1] == 'NNP' and find_b[find_break - 1][1] != 'NNP':
                        print(find_b[find_break - 1:])
                        # print(find_b.index(find_b[find_break]))
                        clue_fin.append(find_b[find_break:])
                        break
                    find_break -= 1
        except: continue

        for fin_data in clue_fin:
            te_list = []
            for clear_n in fin_data:
                if clear_n[1] == 'NNG' or clear_n[1] == 'VV+EC':
                    # if clear_n[1] == 'NNP' or clear_n[1] == 'VV+EC':
                    te_list.append(clear_n)
            fin_list.append(te_list)
        print(*fin_list, sep='\n')
        # print(*clue_fin, sep='\n')
        '''with open('D:/PycharmProjects/codefile/csv_file/가뭄_pattern.csv', 'a', encoding='utf-8') as pf:
            for k in fin_list:
                pf.write(str(k) + '\n')'''

first_func()
patt_fi()