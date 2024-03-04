from textractor import Textractor
from textractor.data.constants import TextractFeatures
from PIL import Image
import time

queries = [
    # specifying the label works but, otherwise it seems it takes first name is taken as the first name it encounters rather than the label.
    "What is the label first name value",
    "What is the label last name value",
    "Which clinic site was the 1st dose COVID-19 administrated?",
    "Who is the manufacturer for 1st dose of COVID-19?",
    "What is the date for the 2nd dose covid-19?",
    "What is the patient number",
    "Who is the manufacturer for 2nd dose of COVID-19?",
    "Which clinic site was the 2nd dose covid-19 administrated?",
    "What is the lot number for 2nd dose covid-19?",
    "What is the date for the 1st dose covid-19?",
    "What is the lot number for 1st dose covid-19?",
    "What is the MI?",
    "MI",
]


if __name__ == "__main__":
    extractor = Textractor(profile_name="default")
    t0 = time.time()
    document = extractor.analyze_document(
        file_source=Image.open("./aws/vaccination.jpg"),
        features=[TextractFeatures.QUERIES],
        queries=queries,
    )
    t1 = time.time()
    print(f"Time taken: {t1 - t0:.2f} seconds")
    print(document)
    for query in document.queries:
        if query.result:
            print(
                f"{query.query}\n\tAnswer: {query.result.answer}\n\tConfidence: {query.result.confidence}\n"
            )
        else:
            print(f"{query.query}\n\tNo Answer\n")
