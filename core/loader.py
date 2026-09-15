
from langchain_community.document_loaders import (
    GitLoader,
    PyMuPDFLoader,
    PyPDFium2Loader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    NotebookLoader,
    TextLoader,
)
from langchain_core.documents import Document
from dotenv import load_dotenv
from datetime import datetime
from os import getenv, path,listdir
from typing import List
from config import get_settings
from tempfile import TemporaryDirectory

settings = get_settings()
Skip_Extensions = {'.zip','.tar','.7z','.png', '.jpg', '.gif', '.lock', '.ico', '.woff', '.ttf', '.pyc','.venv','.mp4','.wav','.mkv','.python-version'}
local_repo_path = settings.local_repo_path

class SmartLoaderManager:

    docs_loaded:List[Document] 

    _PDF_EXTS = {".pdf"}
    _DOCX_EXTS = {".docx"}
    _MARKDOWN_EXTS = {".md", ".mdx"}
    _TEXT_EXTS = {".txt", ".rst", ".log"}
    _CSV_EXTS = {".csv"}
    _PPT_EXTS = {".ppt", ".pptx"}
    _IPYNB_EXTS = {".ipynb"}
    _CODE_EXTS = {
        ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".cc", ".cpp",
          ".cs", ".html", ".htm", ".css", ".json", ".xml", ".yaml",
        ".yml", ".sql", ".sh", ".bash", ".zsh", ".env", ".toml", ".ini",
        ".cfg", ".dockerfile"
    }

    def __init__(self):
        self.docs_loaded = []

    def get_from_folder(self,fpath:str):
        if not path.isdir(fpath) :
            raise ValueError()
        files_list  = [path.join(fpath, f) for f in listdir(fpath) if path.isfile(path.join(fpath, f))]
        self.get_from_file(files_list)
               
    def get_from_file(self,files_list:List[str]):
        for f in files_list:
                file_name=path.basename(f)
                file_ext= path.splitext(f)[1]
                if file_ext in self._PDF_EXTS:
                    cur_docs =self.fetch_pdf_attachment(f)
                elif file_ext in self._PPT_EXTS:
                    cur_docs = self.fetch_ppt_attachment(f)
                elif file_ext in self._DOCX_EXTS:
                    cur_docs = self.fetch_docx_attachment(f)
                elif file_ext in self._MARKDOWN_EXTS:
                    cur_docs = self.fetch_md_attachment(f)
                elif file_ext in self._CSV_EXTS:
                    cur_docs = self.fetch_csv_attachment(f)
                elif file_ext in self._IPYNB_EXTS:
                    cur_docs= self.fetch_ipynb_attachment(f)
                elif file_ext in self._CODE_EXTS:
                    cur_docs = self.fetch_default_type_attachment(f)
                elif file_ext in Skip_Extensions:
                    continue
                else :
                    cur_docs= self.fetch_default_type_attachment(f)

                for doc in cur_docs :
                    doc.metadata['file_type']=file_ext
                    doc.metadata['file_name']= file_name
                    #doc.metadata['file_path']= f

                self.docs_loaded.extend(cur_docs)
                
    def fetch_git_repo(self,repo_link:str,repo_name:str,branch='main'):
        with TemporaryDirectory() as tempdir :
            git_loader = GitLoader(repo_path=path.join(tempdir,repo_name),clone_url=repo_link,branch=branch)
            docs = git_loader.load()
        filter_docs = [doc for doc in  docs if doc.metadata.get('file_type') not in Skip_Extensions]
        print(f"{datetime.now()} Total  Documents -> {len(docs)}")
        print(f"{datetime.now()} Documents after filtered -> {len(filter_docs)}")
        self.docs_loaded.extend(filter_docs)
        return filter_docs

    def fetch_pdf_attachment(self, file_path: str):
        """Load PDF attachments with proper error handling."""
        if not path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        try:
            return PyPDFium2Loader(file_path=file_path).load()
        except Exception as e:
            raise RuntimeError(f"Failed to load PDF {file_path}: {str(e)}")

    def fetch_csv_attachment(self, file_path: str):
        """Load CSV attachments with proper error handling."""
        if not path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        try:
            return CSVLoader(file_path=file_path, encoding='utf-8').load()
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV {file_path}: {str(e)}")

    def fetch_docx_attachment(self, file_path: str):
        """Load DOCX attachments with proper error handling."""
        if not path.exists(file_path):
            raise FileNotFoundError(f"DOCX file not found: {file_path}")
        try:
            return Docx2txtLoader(file_path).load()
        except Exception as e:
            raise RuntimeError(f"Failed to load DOCX {file_path}: {str(e)}")

    def fetch_md_attachment(self, file_path: str):
        """Load Markdown attachments with proper error handling."""
        if not path.exists(file_path):
            raise FileNotFoundError(f"Markdown file not found: {file_path}")
        try:
            return UnstructuredMarkdownLoader(file_path, encoding='utf-8').load()
        except Exception as e:
            raise RuntimeError(f"Failed to load Markdown {file_path}: {str(e)}")

    def fetch_ppt_attachment(self, file_path: str):
        """Load PowerPoint attachments with proper error handling."""
        if not path.exists(file_path):
            raise FileNotFoundError(f"PowerPoint file not found: {file_path}")
        try:
            return UnstructuredPowerPointLoader(file_path=file_path).load()
        except Exception as e:
            raise RuntimeError(f"Failed to load PowerPoint {file_path}: {str(e)}")

    def fetch_ipynb_attachment(self, file_path: str):
        if path.exists(file_path):
            return NotebookLoader(file_path,remove_newline=True).load()
        raise FileNotFoundError()

    def fetch_default_type_attachment(self, file_path: str):
        if path.exists(file_path):
            return TextLoader(file_path=file_path, encoding='utf-8').load()
        raise FileNotFoundError()


if __name__ == "__main__":
    sm = SmartLoaderManager()
    #sm.get_from_folder("D:/git_projects/local-docker-mcp")
    sm.get_from_folder("D:\\Study_Docs\\GFG\\Live_lecture_notes\\genai_course")
    mata = [ d.metadata for d in sm.docs_loaded]
    print(mata)