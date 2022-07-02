import collections

from sklearn.model_selection import train_test_split
import spacy
import pandas as pd
from tqdm import tqdm
import os

pd.options.mode.chained_assignment = None  # default='warn'


def check_data():
    texts = []

    with open('/home/anton/Documents/brat/sentiment_dataset/013.ann') as f:
        for line in f:
            if len(line.strip()) > 0:
                texts.append(line.strip())

    print(texts)


def prepare_tsa_dataset(file_name):
    """
    приведение собранного датасета к формату модели BERT-QA
    """
    sentence_1 = []
    sentence_2 = []
    label = []
    tonal_word = []
    text_id = []

    label_vocab = {0: 'нейтрально', 1: 'положительно', -1: 'отрицательно'}

    df = pd.read_csv(file_name, sep='\t')

    for i in range(len(df)):
        sentence_1.append(df['text'][i].replace(df['entity'][i], 'location - 1'))
        sentence_2.append('что вы думаете вообще о location - 1 ?')
        label.append(label_vocab[df['label'][i]])
        tonal_word.append(df['entity'][i])
        text_id.append(0)

    df_modified = pd.DataFrame(
        {
            'sentence_1': sentence_1,
            'sentence_2': sentence_2,
            'label': label,
            'tonal_word': tonal_word,
            'text_id': text_id
        })

    df_modified.to_csv(file_name[:-4] + '_modified.csv', sep='\t')


def check_type_of_entity(entity_pos_start, entity_pos_end, entity, directory_path, file_name):
    """
    функция, определяющая тип сущности по файлу разметки из датасета brat
    :param directory_path: директория с файлами .ann
    :param file_name: название файла .ann, по которому осуществлять поиск
    :param entity_pos_start: абсолютная стартовая позиция сущности в файле .txt
    :param entity_pos_end: абсолютная конечная позиция сущности в файле .txt
    :param entity: имя сущности
    :return: сущность [str], если не нашли то 'none'
    """
    output = 'none'

    with open(os.path.join(directory_path, file_name + '.ann')) as f:
        for sent in f.readlines():
            # проверяем строки файла с разметкой пока не найдем упоминание входящего параметра
            if sent.strip().split('\t')[1].split()[1].isdigit() and sent.strip().split('\t')[1].split()[2].isdigit():
                if int(sent.strip().split('\t')[1].split()[1]) == entity_pos_start \
                        and int(sent.strip().split('\t')[1].split()[2]) == entity_pos_end \
                        and sent.strip().split('\t')[-1] == entity:
                    output = sent.strip().split('\t')[1].split()[0]
                    break

    return output


def search_entity(directory_path, file_name, t_label):
    """
    :param t_label: метка сущности в формате T__
    :param directory_path: директория с файлами .ann
    :param file_name: название файла .ann, по которому осуществлять поиск
    функция, определяющая параметры сущности по указателю T_ в заданном файле
    :return: строку из файла .ann с параметрами, однозначно определяющими сущность
    """
    output = 'none'

    with open(os.path.join(directory_path, file_name + '.ann')) as f:
        for sent in f.readlines():
            # проверяем строки файла с разметкой пока не найдем упоминание входящего параметра
            if sent.strip().split()[0] == t_label:
                output = sent
                break

    return output


def create_tsa_dataset():
    """
    Тональность может быть выведена из трех составных частей:
    1.- сама сущность отмечена author_pos или author_neg — это отношение автора

    2.1. positive_to, negative_to — нужно брать второй атрибут и ставить соотв. тональность.  Это значит кто-то
    относится позитивно к сущности 2.2  opinion_relates_to  — иногда носитель мнения не упомянут, но мнение есть —
    тогда нам важно это отношение

    Наша сущность — это второй аргумент отношения opinion_relates_to . Тональность второго аргумента определяется от
    тональности первого аргумента, который может быть размечен так: негативная тональность — opinion_word_neg или
    argument_neg, позитивная тональность — opinion_word_pos или argument_pos.

    4. Наконец, имеет смысл смотреть тональность не только к человеку и организации, но и к странам  — COUNTRY

    для каждого предложения проставлять соотв метку (с помощью чего было отобрано предложение) в поле source
    в качестве нейтральной части нужно отобрать предложения, содержащие сущности без какой-либо разметки отношений
    """
    directory_path = 'brat/sentiment_dataset'
    files = list(sorted([file[:-4] for file in os.listdir(directory_path)]))

    nlp = spacy.load('ru_core_news_sm')  # sentencizer

    entities_types = ['PERSON', 'ORGANIZATION', 'COUNTRY', 'PROFESSION', 'NATIONALITY']  # список отбираемых типов

    # итерация (1 файл)
    entity = []
    entity_pos_start = []
    entity_pos_end = []
    label = []
    sentence_pos_start = []
    sentence_pos_end = []
    source = []
    entity_tag = []

    # итоговый датасет
    out_sentence = []
    out_entity = []
    out_label = []
    out_source = []
    out_entity_tag = []

    for file in tqdm(files):
        if os.path.isfile(os.path.join(directory_path, file + '.ann')) and os.path.isfile(
                os.path.join(directory_path, file + '.txt')):

            # анализ разметки .ann для каждого файла
            with open(os.path.join(directory_path, file + '.ann')) as f:
                for sent in f.readlines():

                    # 1 - наличие оценки от автора [author_pos, author_neg]
                    if 'AUTHOR_NEG' in sent or 'AUTHOR_POS' in sent:
                        tag = check_type_of_entity(int(sent.strip().split('\t')[1].split()[1]),
                                                   int(sent.strip().split('\t')[1].split()[2]),
                                                   sent.strip().split('\t')[-1],
                                                   directory_path,
                                                   file).upper()
                        # проверка принадлежности к списку тэгов
                        if tag in entities_types:
                            entity.append(sent.strip().split('\t')[-1])
                            entity_tag.append(tag)
                            entity_pos_start.append(int(sent.strip().split('\t')[1].split()[1]))
                            entity_pos_end.append(int(sent.strip().split('\t')[1].split()[2]))
                            label.append(1 if sent.strip().split('\t')[1].split()[0] == 'AUTHOR_POS' else -1)
                            source.append('AUTHOR_POS/AUTHOR_NEG')

                    # 2.1 - наличие отношения к сущности [positive_to, negative_to], берем 2-ой атрибут
                    if 'POSITIVE_TO' in sent or 'POSITIVE_NEG' in sent:
                        entity_info = search_entity(directory_path, file, sent.split()[3].split(':')[1])
                        # проверка принадлежности к списку тэгов
                        if entity_info.strip().split()[1] in entities_types:
                            entity.append(entity_info.strip().split('\t')[-1])
                            entity_tag.append(entity_info.strip().split()[1])
                            entity_pos_start.append(int(entity_info.strip().split('\t')[1].split()[1]))
                            entity_pos_end.append(int(entity_info.strip().split('\t')[1].split()[2]))
                            label.append(1 if sent.strip().split('\t')[1].split()[0] == 'POSITIVE_TO' else -1)
                            source.append('POSITIVE_TO/NEGATIVE_TO')

                    # 2.2 - наличие opinion_relates_to, мнение есть, но автора нет

            # анализ содержимого .txt для каждого файла (если нашли что-то ранее)
            with open(os.path.join(directory_path, file + '.txt')) as f:
                # sentencizing
                f_total_text = f.read()
                doc = nlp(f_total_text)
                sentencized_file = [str(sent).strip() for sent in doc.sents if len(str(sent).strip()) > 0]

                # поиск границ каждого предложения
                for sent in sentencized_file:
                    sentence_pos_start.append(f_total_text.find(sent))
                    sentence_pos_end.append(f_total_text.find(sent) + len(sent))

                # на основании разметки из .ann и sentencized-файла .txt добавляем сэмплы к генерируемому датасету
                for i in range(len(entity)):
                    entity_pos = (entity_pos_end[i] + entity_pos_start[i]) // 2
                    for j in range(len(sentencized_file)):
                        if sentence_pos_start[j] < entity_pos < sentence_pos_end[j] and entity[i].upper() in \
                                sentencized_file[j].upper():
                            out_sentence.append(sentencized_file[j])
                            out_entity.append(entity[i])
                            out_label.append(label[i])
                            out_entity_tag.append(entity_tag[i])
                            out_source.append(source[i])

            # обнуление списков после обработки каждого файла
            entity = []
            entity_tag = []
            entity_pos_start = []
            entity_pos_end = []
            label = []
            source = []

    out_df = pd.DataFrame(
        {
            'sentence': out_sentence,
            'entity': out_entity,
            'entity_tag': out_entity_tag,
            'label': out_label,
            'source': out_source
        })

    out_df = out_df.drop_duplicates()
    out_df.to_csv('data/tsa_dataset.csv', sep='\t', index=False)
    print(collections.Counter(out_df['source']))


if __name__ == '__main__':
    create_tsa_dataset()
    # prepare_tsa_dataset()
