from collections import OrderedDict

import spacy

from os import listdir
from os.path import isfile, join
from tqdm import tqdm


def get_text_path(directory_path, file):
    """ Путь к текстовому файлу документа
    """
    return join(directory_path, file + '.txt')


def is_correct(directory_path, file):
    """ Функция проверки на то, что документ подходит для выполнения анализа
    """
    ann_path = join(directory_path, file + '.ann')
    return isfile(ann_path) and \
           isfile(get_text_path(directory_path=directory_path, file=file))


def calculate_sentences_stat_per_doc(directory_path, limit=None):
    """ Подсчитываем число предложений для каждого документа
    """
    files = sorted(list(set([file[:-4] for file in listdir(directory_path)])))

    if limit is not None:
        files = files[:limit]

    sentences_per_doc = OrderedDict()
    nlp = spacy.load('ru_core_news_sm')

    for file in tqdm(files):

        if not is_correct(directory_path=directory_path, file=file):
            # Пропускаем документ
            continue

        # Анализируем файл с текстом документа
        with open(get_text_path(directory_path=directory_path, file=file)) as f:

            f_total_text = f.read()
            doc = nlp(f_total_text)

            # Фиксируем число предложений
            sentences_per_doc[file] = len(list(doc.sents))

    return sentences_per_doc
