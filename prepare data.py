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


def create_tsa_dataset():
    """
    Тональность может быть выведена из трех составных частей:
    1.- сама сущность отмечена author_pos или author_neg — это отношение автора

    2.1. positive_to, negative_to — нужно брать второй атрибут и ставить соотв. тональность.  Это значит кто-то относится позитивно к сущности
    2.2  opinion_relates_to  — иногда носитель мнения не упомянут, но мнение есть — тогда нам важно это отношение

    Наша сущность — это второй аргумент отношения opinion_relates_to .
    Тональность второго аргумента определяется от тональности первого аргумента, который может быть размечен так:
    негативная тональность — opinion_word_neg или  argument_neg, позитивная тональность — opinion_word_pos или argument_pos.

    4. Наконец, имеет смысл смотреть тональность не только к человеку и организации, но и к странам  — COUNTRY

    для каждого предложения проставлять соотв метку (с помощью чего было отобрано предложение) в поле source
    в качестве нейтральной части нужно отобрать предложения, содержащие сущности без какой-либо разметки отношений
    :return:
    """
    directory_path = 'brat/sentiment_dataset'
    files = list(sorted([file[:-4] for file in os.listdir(directory_path)]))

    nlp = spacy.load('ru_core_news_sm')  # sentencizer

    # итерация
    entity = []
    entity_pos_start = []
    entity_pos_end = []
    label = []
    sentence_pos_start = []
    sentence_pos_end = []
    entity_tag = []

    # итоговый датасет
    out_sentence = []
    out_entity = []
    out_label = []
    out_source = []
    out_entity_tag = []

    # 1 - наличие оценки от автора [author_pos, author_neg]
    for file in tqdm(files):
        if os.path.isfile(os.path.join(directory_path, file + '.ann')) and os.path.isfile(
                os.path.join(directory_path, file + '.txt')):
            # анализ разметки для каждого файла
            with open(os.path.join(directory_path, file + '.ann')) as f:
                for sent in f.readlines():
                    if 'AUTHOR_NEG' in sent or 'AUTHOR_POS' in sent:
                        # отбираем сущности с тэгами из [PERSON, ORGANIZATION, COUNTRY, PROFESSION, NATIONALITY]
                        if check_type_of_entity(int(sent.strip().split('\t')[1].split()[1]),
                                                int(sent.strip().split('\t')[1].split()[2]),
                                                sent.strip().split('\t')[-1],
                                                directory_path,
                                                file).upper() in ['PERSON', 'ORGANIZATION', 'COUNTRY', 'PROFESSION',
                                                                  'NATIONALITY']:
                            entity.append(sent.strip().split('\t')[-1])
                            entity_pos_start.append(int(sent.strip().split('\t')[1].split()[1]))
                            entity_pos_end.append(int(sent.strip().split('\t')[1].split()[2]))
                            label.append(1 if sent.strip().split('\t')[1].split()[0] == 'AUTHOR_POS' else -1)
            # анализ каждого файла
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
                # проверять, сущность в entity или нет
                for i in range(len(entity)):
                    entity_pos = (entity_pos_end[i] + entity_pos_start[i]) // 2
                    for j in range(len(sentencized_file)):
                        if sentence_pos_start[j] < entity_pos < sentence_pos_end[j] and entity[i].upper() in \
                                sentencized_file[j].upper():
                            out_sentence.append(sentencized_file[j])
                            out_entity.append(entity[i])
                            out_label.append(label[i])
                            out_source.append('AUTHOR_POS/AUTHOR_NEG')

            # обнуление списков после каждой итерации
            entity = []
            entity_pos_start = []
            entity_pos_end = []
            label = []

    out_df = pd.DataFrame(
        {
            'sentence': out_sentence,
            'entity': out_entity,
            'label': out_label,
            'source': out_source
        })

    out_df = out_df.drop_duplicates()
    out_df.to_csv('data/tsa_dataset.csv', sep='\t')


if __name__ == '__main__':
    create_tsa_dataset()
    # prepare_tsa_dataset()
