import numpy as np

from split_data import calculate_sentences_stat_per_doc


def calculate_fold_stat(sentences_per_doc, folds):
    """ подсчет общего числа предложений в каждом разбиении
    """
    return [sum(map(lambda f: sentences_per_doc[f], folds[i])) for i in folds.keys()]


if __name__ == '__main__':
    """ Построение статистики кросс-валидационного разбиения
    """

    k_fold = 5
    sentences_per_doc = calculate_sentences_stat_per_doc('data/sentiment_dataset')

    # Заполняем пустой список
    folds = {}
    for fold_index in range(k_fold):
        folds[fold_index] = []

    fold_index = 0
    for file, sentences_count in reversed(sorted(sentences_per_doc.items(), key=lambda item: item[1])):
        fold_stats = calculate_fold_stat(sentences_per_doc=sentences_per_doc, folds=folds)
        fold_index = np.argmin(fold_stats)
        folds[fold_index].append(file)

    print("Статистика разбиений:", calculate_fold_stat(sentences_per_doc=sentences_per_doc, folds=folds))

    # Сохраняем результат в отдельный файл.
    with open("data/split_{}_fold.txt".format(k_fold), "w") as out:
        for fold_index in range(k_fold):
            out.write(",".join(folds[fold_index]))
            out.write("\n")
