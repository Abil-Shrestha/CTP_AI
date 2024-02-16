from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")
sentiment_classifier = pipeline(
    task="sentiment-analysis", 
    model="siebert/sentiment-roberta-large-english"
)

sentences = [
    "This is my first CTP AI class homework",
    "It was a lot to process.",
    "I didn't know how to feel about the hugginface tutorial",
    "I didn't really understand much but it was fun.",
    "They need to make a new tutorial."
]

results = sentiment_classifier(sentences)

for result in results:
    print(f"label: {result['label']}, with score: {result['score']:.4f}")


qa_pipeline = pipeline(
    task="question-answering",
    model="bert-large-uncased-whole-word-masking-finetuned-squad"
)

context = """
The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York Harbor in New York City, in the United States. 
The copper statue, a gift from the people of France to the people of the United States, was designed by French sculptor Frédéric Auguste Bartholdi 
and its metal framework was built by Gustave Eiffel. The statue was dedicated on October 28, 1886.
"""

question = "Who designed the Statue of Liberty?"

answer = qa_pipeline(question=question, context=context)

print(f"Question: {question}")
print(f"Answer: {answer['answer']}")

