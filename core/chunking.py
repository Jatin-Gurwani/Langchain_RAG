from langchain_text_splitters import RecursiveCharacterTextSplitter,Language
from langchain_core.documents import Document
from typing import List
from datetime import datetime
from tqdm import tqdm

def recursive_chunking(Documents:List[Document]) -> List[Document]:
    extension_dir ={
            '.py': RecursiveCharacterTextSplitter.from_language(Language.PYTHON,chunk_size=1000, chunk_overlap=100),
            '.js': RecursiveCharacterTextSplitter.from_language(Language.JS,chunk_size=1000, chunk_overlap=100),
            '.html':RecursiveCharacterTextSplitter.from_language(Language.HTML,chunk_size=700, chunk_overlap=100),
            '.md': RecursiveCharacterTextSplitter.from_language(Language.MARKDOWN,chunk_size=700, chunk_overlap=100),
            '.java':RecursiveCharacterTextSplitter.from_language(Language.JAVA,chunk_size=1000, chunk_overlap=100),
            '.cs' : RecursiveCharacterTextSplitter.from_language(Language.CSHARP,chunk_size=500, chunk_overlap=100),
            '.ts': RecursiveCharacterTextSplitter.from_language(Language.TS,chunk_size=500, chunk_overlap=100),     
        }
    chunks = []
    for doc in tqdm(Documents,desc='splitting code into chunks'):
        spiltter = extension_dir.get(doc.metadata.get('file_type')) or RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)  
        temp_chunks= spiltter.create_documents([str(doc.page_content)],[doc.metadata])
        for tc in temp_chunks:
            chunks.append(tc)
                    
    print(f"{datetime.now()} Total Chunks Created -> {len(chunks)}")
    return chunks