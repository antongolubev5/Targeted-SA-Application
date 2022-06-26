# на вход системы новостной текст + координаты сущности, определенные через NER.
# на выход оценочная разметка для соотв. сущностей

# потестировать bert-base модели на сформированном несбалансированном датасете
# заявленная точность = f1 60%

from sklearn import metrics
import pandas as pd


def calc_metrics():
    true = []
    pred = []
    key_vocab = {'нейтрально': 2, 'положительно': 1, 'отрицательно': 0}

    with open('test_ep_3.txt') as f:
        for line in f:
            pred.append(int(line.strip().split()[0]))

    df = pd.read_csv('data/df_test_mod.csv', sep='\t')
    true = df['label'].apply(lambda x: key_vocab[x]).values

    print('accuracy = ' + str(round(metrics.accuracy_score(true, pred), 2)))
    print(metrics.classification_report(true, pred, target_names=['отрицательный класс', 'положительный класс', 'нейтральный класс']))


if __name__ == '__main__':
    calc_metrics()
