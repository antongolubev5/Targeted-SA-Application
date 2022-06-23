import numpy as np
from sklearn.model_selection import train_test_split
from split_data import calculate_sentences_stat_per_doc


def calc_average_sentences(sentences_per_doc, files):
    return round(np.average([sentences_per_doc[file] for file in files]), 4)


if __name__ == '__main__':
    sentences_per_doc = calculate_sentences_stat_per_doc("data/sentiment_dataset")
    files = list(sentences_per_doc.keys())
    train, test = train_test_split(files, test_size=0.3, train_size=0.7, random_state=0)

    print("Среднее число предложений в TRAIN", calc_average_sentences(
        sentences_per_doc=sentences_per_doc, files=train))

    print("Среднее число предложений в TEST", calc_average_sentences(
        sentences_per_doc=sentences_per_doc, files=test))

    # Сохраняем результат в отдельный файл.
    with open("data/split_fixed.txt", "w") as out:
        out.write(",".join(train))
        out.write("\n")
        out.write(",".join(test))
