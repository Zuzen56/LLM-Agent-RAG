from pathlib import Path
from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import fetch_archive_from_http
from haystack.pipelines import Pipeline
from haystack.nodes import FileTypeClassifier, TextConverter, PreProcessor, BM25Retriever, FARMReader
from gtts import gTTS
import os

# Initialize the DocumentStore
document_store = InMemoryDocumentStore(use_bm25=True)

# Get the documents
documents_path = "data/tutorial17"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt17.zip"
fetch_archive_from_http(url=s3_url, output_dir=documents_path)

# List all the paths
file_paths = [p for p in Path(documents_path).glob("**/*")]

# NOTE: In this example we're going to use only one text file from the wiki
file_paths = [p for p in file_paths if "Stormborn" in p.name]

# Prepare some basic metadata for the files
files_metadata = [{"name": path.name} for path in file_paths]

# Makes sure the file is a TXT file (FileTypeClassifier node)
classifier = FileTypeClassifier()

# Converts a file into text and performs basic cleaning (TextConverter node)
text_converter = TextConverter(remove_numeric_tables=True)

# Pre-processes the text by performing splits and adding metadata to the text (Preprocessor node)
preprocessor = PreProcessor(clean_header_footer=True, split_length=200, split_overlap=20)

# Here we create a basic indexing pipeline
indexing_pipeline = Pipeline()
indexing_pipeline.add_node(classifier, name="classifier", inputs=["File"])
indexing_pipeline.add_node(text_converter, name="text_converter", inputs=["classifier.output_1"])
indexing_pipeline.add_node(preprocessor, name="preprocessor", inputs=["text_converter"])
indexing_pipeline.add_node(document_store, name="document_store", inputs=["preprocessor"])

# Run the indexing pipeline with the documents and their metadata as input
indexing_pipeline.run(file_paths=file_paths, meta=files_metadata)

# Initialize the retriever and the reader
retriever = BM25Retriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

# Define a function to convert text to speech using gtts
def answer_to_speech(answer_text, file_name="answer.mp3"):
    tts = gTTS(text=answer_text, lang='zh')  # 语言可以根据需要设置
    audio_file_path = Path("./audio_answers") / file_name
    tts.save(str(audio_file_path))
    return audio_file_path

# Create the audio processing pipeline
audio_pipeline = Pipeline()
audio_pipeline.add_node(retriever, name="Retriever", inputs=["Query"])
audio_pipeline.add_node(reader, name="Reader", inputs=["Retriever"])

# Run the query through the pipeline
prediction = audio_pipeline.run(
    query="Who is the father of Arya Stark?", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
)

# Print the prediction
from pprint import pprint
pprint(prediction)

# The document from which the first answer was extracted
original_document = [doc for doc in prediction["documents"] if doc.id == prediction["answers"][0].document_ids[0]][0]
pprint(original_document)

# Display the answer and play the audio
answer_text = prediction["answers"][0].meta["answer_text"]
print("Answer: ", answer_text)
audio_file = answer_to_speech(answer_text, "answer.mp3")
print(f"Audio saved to {audio_file}")
os.system(f"start {audio_file}")  # Windows

# Display the context and play the audio
context_text = prediction["answers"][0].meta["context_text"]
print("Context: ", context_text)
audio_file_context = answer_to_speech(context_text, "context.mp3")
print(f"Audio saved to {audio_file_context}")
os.system(f"start {audio_file_context}")  # Windows
