import setup_nltk

resources = [
    "punkt_tab",
    "averaged_perceptron_tagger_eng",
    "universal_tagset",
]

for r in resources:
    setup_nltk.download(r, download_dir="/home/radhofan/miniconda3/envs/ollama_pdf_rag/nltk_data")

setup_nltk.data.path.append("/home/radhofan/miniconda3/envs/ollama_pdf_rag/nltk_data")
