from langchain_community.document_loaders import WebLoader

def load_data(url):
    loader = WebLoader(url)
    documents = loader.load()
    return documents
