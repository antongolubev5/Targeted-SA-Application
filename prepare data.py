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


def prepare_dataset():
    """
    разметка датасета по собранной коллекции из brat
    :return:
    """
    # вытаскиваем из .ann всю инфу по PERSON + ORG с абсолютными координатами в тексте
    # если нет тональности, оставляем 0, иначе проставляем соотв тэги
    # text, entity, entity_type, label, entity_pos_start, entity_pos_end, source

    entity = []
    entity_type = []
    entity_pos_abs_start = []  # начало сущности во всем файле
    entity_pos_abs_end = []  # конец сущности во всем файле
    entity_pos_rel_start = []  # начало сущности в предложении
    entity_pos_rel_end = []  # конец сущности в предложении
    label = []
    text = []
    texts = []
    source = []
    texts_pos_abs_start = []
    texts_pos_abs_end = []

    directory_path = '/home/anton/Documents/brat/sentiment_dataset'
    files = list(sorted([file[:-4] for file in os.listdir(directory_path)]))

    file_total = pd.DataFrame(
        {
            'entity_type': entity_type,
            'entity': entity,
            'entity_pos_abs_start': entity_pos_abs_start,
            'entity_pos_abs_end': entity_pos_abs_end,
            'entity_pos_rel_start': entity_pos_rel_start,
            'entity_pos_rel_end': entity_pos_rel_end,
            'label': label,
            'text': text,
            'source': source
        })

    # в цикле для каждого файла создавать df на основании инфы о PERSON/ORG,
    # дообогащать его информацией о тональности
    # потом конкатенировать его с общим и очищать для новой итерации (файла)
    for file in tqdm(files):
        with open(os.path.join(directory_path, file + '.ann')) as f:
            for line in f:
                if ('PERSON' in line or 'ORGANIZATION' in line) and line.strip().split()[2].isdigit() and \
                        line.strip().split()[3].isdigit():
                    entity.append(' '.join(line.split()[4:]))
                    entity_type.append(line.strip().split()[1])
                    entity_pos_abs_start.append(int(line.strip().split()[2]))
                    entity_pos_abs_end.append(int(line.strip().split()[3]))
                    label.append(0)
                    texts.append('')
                    source.append(file + '.txt')
                    entity_pos_rel_start.append('')
                    entity_pos_rel_end.append('')

        file_loop = pd.DataFrame(
            {
                'entity_type': entity_type,
                'entity': entity,
                'entity_pos_abs_start': entity_pos_abs_start,
                'entity_pos_abs_end': entity_pos_abs_end,
                'entity_pos_rel_start': entity_pos_rel_start,
                'entity_pos_rel_end': entity_pos_rel_end,
                'label': label,
                'text': texts,
                'source': source
            })

        file_loop = file_loop.sort_values(by='entity_pos_abs_start', ascending=True)

        # собираем из файла .ann информацию о тональной разметке
        entity = []
        entity_type = []
        entity_pos_abs_start = []
        entity_pos_abs_end = []

        with open(os.path.join(directory_path, file + '.ann')) as f:
            for line in f:
                if ('EFFECT_NEG' in line or 'EFFECT_POS' in line) and line.strip().split()[2].isdigit() and \
                        line.strip().split()[3].isdigit():
                    entity.append(' '.join(line.split()[4:]))
                    entity_type.append(line.strip().split()[1])
                    entity_pos_abs_start.append(int(line.strip().split()[2]))
                    entity_pos_abs_end.append(int(line.strip().split()[3]))

        # идем по созданному df и проставляем метки тональности, извлеченные из файла
        for i in range(len(file_loop)):
            for j in range(len(entity)):
                if entity_pos_abs_start[j] == file_loop['entity_pos_abs_start'][i] and entity_pos_abs_end[j] == \
                        file_loop['entity_pos_abs_end'][i] and entity[j] == file_loop['entity'][i]:
                    if entity_type[j] == 'EFFECT_POS':
                        file_loop['label'][i] = 1
                    elif entity_type[j] == 'EFFECT_NEG':
                        file_loop['label'][i] = -1

        # обнуляем df
        entity = []
        entity_type = []
        entity_pos_abs_start = []
        entity_pos_abs_end = []

        # к df с размеченными по тональности сущностями добавить тексты, в которых они учитываются
        # тональный словарь готов. теперь идем по по текстам .txt и собираем корпус
        # по координатам abs_start и abs_end определить какой текст вставить в поле text

        with open(os.path.join(directory_path, file + '.txt')) as f:
            file_content = f.read()

        # собрали предложения и их абсолютные координаты в списки
        with open(os.path.join(directory_path, file + '.txt')) as f:
            for line in f:
                for sent in line.strip().split('. '):
                    if len(sent.strip()) > 0:
                        text.append(sent.strip())
                        texts_pos_abs_start.append(file_content.find(sent.strip()))
                        texts_pos_abs_end.append(file_content.find(sent.strip()) + len(sent.strip()))

        #  идем циклом по file_loop и дообогащаем его данными
        for i in range(len(file_loop)):
            for j in range(len(text)):
                if texts_pos_abs_start[j] < (
                        file_loop['entity_pos_abs_end'][i] + file_loop['entity_pos_abs_start'][i]) // 2 < \
                        texts_pos_abs_end[j]:
                    file_loop['text'][i] = text[j]
                    file_loop['entity_pos_rel_start'][i] = text[j].find(file_loop['entity'][i])
                    file_loop['entity_pos_rel_end'][i] = text[j].find(file_loop['entity'][i]) + len(
                        file_loop['entity'][i])

        # конкатенируем df с итерации с основным df
        file_total = pd.concat([file_total, file_loop], axis=0)

        entity_type = []
        entity = []
        entity_pos_abs_start = []
        entity_pos_abs_end = []
        entity_pos_rel_start = []
        entity_pos_rel_end = []
        label = []
        texts = []
        source = []

    file_total = file_total[
        ['text', 'entity', 'label', 'entity_type', 'entity_pos_rel_start', 'entity_pos_rel_end', 'source']]
    file_total = file_total.loc[file_total['text'].apply(lambda x: len(x.split()) > 4)]
    file_total = file_total.drop_duplicates()
    file_total = file_total[file_total['entity_pos_rel_start'] != -1]

    file_total.to_csv('dataset.csv', sep='\t', index=False)


def prepare_data(file_name):
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

    df_modified.to_csv(file_name[:-4] + '_mod.csv', sep='\t')


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
    directory_path = '/home/anton/Documents/brat/sentiment_dataset'
    files = list(set([file[:-4] for file in os.listdir(directory_path)]))

    nlp = spacy.load('ru_core_news_sm')  # sentencizer

    # итерация
    entity = []
    entity_pos_start = []
    entity_pos_end = []
    label = []
    sentence_pos_start = []
    sentence_pos_end = []

    # итоговый датасет
    out_sentence = []
    out_entity = []
    out_label = []
    out_source = []

    # 1 - наличие оценки от автора [author_pos, author_neg]
    for file in tqdm(files):
        if os.path.isfile(os.path.join(directory_path, file + '.ann')) and os.path.isfile(
                os.path.join(directory_path, file + '.txt')):
            # анализ разметки для каждого файла
            with open(os.path.join(directory_path, file + '.ann')) as f:
                for sent in f.readlines():
                    if 'AUTHOR_NEG' in sent or 'AUTHOR_POS' in sent:
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
                    entity_pos = (entity_pos_end[i] + entity_pos_start[i])//2
                    for j in range(len(sentencized_file)):
                        if sentence_pos_start[j] < entity_pos < sentence_pos_end[j]:
                            if entity[i].upper() in sentencized_file[j].upper():
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
    out_df.to_csv('new_by_nv.csv', sep='\t')


if __name__ == '__main__':
    create_tsa_dataset()
    # prepare_dataset()
    # prepare_data('df_train.csv')
    # prepare_data('df_test.csv')
