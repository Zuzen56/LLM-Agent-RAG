from haystack.nodes import FARMReader
from haystack.utils import fetch_archive_from_http
from haystack.schema import Document
from pprint import pprint

reader = FARMReader(model_name_or_path="distilbert-base-uncased-distilled-squad", use_gpu=True)

data_dir = "data/fine-tuning"

fetch_archive_from_http(
    url="https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/squad20.tar.gz", output_dir=data_dir
)

#reader.train(data_dir=data_dir, train_filename="squad20/dev-v2.0.json", use_gpu=True, n_epochs=1, save_dir="my_model")

new_reader = FARMReader(model_name_or_path="my_model")

prediction = new_reader.predict(
    query="What is the capital of Germany?", documents=[Document(content="The capital of Germany is Berlin")]
)

pprint(prediction)