from haystack.utils import fetch_archive_from_http


# This fetches some sample files to work with
doc_dir = "data/tutorial8"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/preprocessing_tutorial8.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)\


from haystack.nodes import TextConverter, PDFToTextConverter, DocxToTextConverter, PreProcessor


converter = TextConverter(remove_numeric_tables=True, valid_languages=["en"])
doc_txt = converter.convert(file_path="data/tutorial8/classics.txt", meta=None)[0]

converter = PDFToTextConverter(remove_numeric_tables=True, valid_languages=["en"])
doc_pdf = converter.convert(file_path="data/tutorial8/bert.pdf", meta=None)[0]

converter = DocxToTextConverter(remove_numeric_tables=False, valid_languages=["en"])
doc_docx = converter.convert(file_path="data/tutorial8/heavy_metal.docx", meta=None)[0]

from haystack.utils import convert_files_to_docs


all_docs = convert_files_to_docs(dir_path=doc_dir)


from haystack.nodes import PreProcessor


# This is a default usage of the PreProcessor.
# Here, it performs cleaning of consecutive whitespaces
# and splits a single large document into smaller documents.
# Each document is up to 1000 words long and document breaks cannot fall in the middle of sentences
# Note how the single document passed into the document gets split into 5 smaller documents

preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=False,
    split_by="word",
    split_length=100,
    split_respect_sentence_boundary=True,
)
docs_default = preprocessor.process([doc_txt])
print(f"n_docs_input: 1\nn_docs_output: {len(docs_default)}")

# Not respecting sentence boundary vs respecting sentence boundary

preprocessor_nrsb = PreProcessor(split_respect_sentence_boundary=False)
docs_nrsb = preprocessor_nrsb.process([doc_txt])

print("RESPECTING SENTENCE BOUNDARY")
end_text = docs_default[0].content[-50:]
print('End of document: "...' + end_text + '"')
print()
print("NOT RESPECTING SENTENCE BOUNDARY")
end_text_nrsb = docs_nrsb[0].content[-50:]
print('End of document: "...' + end_text_nrsb + '"')

# Sliding window approach

preprocessor_sliding_window = PreProcessor(split_overlap=3, split_length=10, split_respect_sentence_boundary=False)
docs_sliding_window = preprocessor_sliding_window.process([doc_txt])

doc1 = docs_sliding_window[0].content[:200]
doc2 = docs_sliding_window[1].content[:100]
doc3 = docs_sliding_window[2].content[:100]

print('Document 1: "' + doc1 + '..."')
print('Document 2: "' + doc2 + '..."')
print('Document 3: "' + doc3 + '..."')


all_docs = convert_files_to_docs(dir_path=doc_dir)
preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=False,
    split_by="word",
    split_length=100,
    split_respect_sentence_boundary=True,
)
docs = preprocessor.process(all_docs)

print(f"n_files_input: {len(all_docs)}\nn_docs_output: {len(docs)}")