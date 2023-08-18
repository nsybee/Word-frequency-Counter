import string
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from spellchecker import SpellChecker
from textblob import TextBlob
import nltk

nltk.download('stopwords')
nltk.download('punkt')


def tokenize(text):
    # Split text into words and remove punctuation
    translator = str.maketrans("", "", string.punctuation)
    words = text.translate(translator).lower().split()
    return words


def count_word_frequencies(words):
    # Count word frequencies using Counter
    word_counts = Counter(words)
    return word_counts


def remove_stop_words(words):
    stop_words = set(stopwords.words('english'))
    return [word for word in words if word not in stop_words]


def plot_word_frequencies(word_counts, top_n=10):
    # Plot the top N most common words
    common_words = word_counts.most_common(top_n)
    words, counts = zip(*common_words)

    plt.bar(words, counts)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title(f'Top {top_n} Most Common Words')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def generate_word_cloud(words):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


def sentiment_analysis(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score


def named_entity_recognition(text):
    blob = TextBlob(text)
    named_entities = [word for word, tag in blob.tags if tag == "NNP"]
    return named_entities


def spell_check(words):
    spell = SpellChecker()
    misspelled = spell.unknown([word.lower() for word in words])
    return misspelled


def main():
    input_text = input("Enter text: ")
    words = tokenize(input_text)
    words = remove_stop_words(words)
    word_counts = count_word_frequencies(words)

    print("\nTotal Number of Words:", len(words))

    print("\nWord Frequencies:")
    for word, count in word_counts.items():
        print(f"{word}: {count}")

    # plot word frequency and generate word cloud
    plot_word_frequencies(word_counts)
    generate_word_cloud(words)

    # perform sentiment analysis
    sentiment_score = sentiment_analysis(input_text)
    if sentiment_score > 0:
        sentiment = "positive"
    elif sentiment_score < 0:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    print(f"Sentiment: {sentiment} (Score: {sentiment_score:.2f})")

    # name entity recognition
    entities = named_entity_recognition(input_text)
    if entities:
        print("\nNamed Entities:")
        for entity in entities:
            print(entity)

    # misspelled words analysis/no. of occurrences
    misspelled_words = spell_check(words)
    if misspelled_words:
        misspelled_word_counts = Counter([word.lower() for word in words])
        print("\nMisspelled Words:")
        for word in misspelled_words:
            corrected_word = spell.correction(word)
            print(f"{word} -> {corrected_word}")


if __name__ == '__main__':
    main()
