from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredExcelLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
import os
import aiofiles.os as aos
import logging


class FileLoaderAndsplitter:
    """FileLoaderAndsplitter class to load and split files"""

    async def _txt_loader_and_splitting(
        self, file_path: str, text_splitter, encoding: str = "utf-8"
    ) -> list:
        """Return the splitted documents from the txt file path."""

        loader = TextLoader(file_path, encoding=encoding)
        documents = await loader.aload()

        return await text_splitter.split_documents(documents)

    async def _pdf_loader_and_splitting(self, file_path: str, text_splitter) -> list:
        """Return the splitted documents from the pdf file path."""

        loader = PyPDFLoader(file_path)
        documents = await loader.aload()

        return text_splitter.split_documents(documents)

    async def _docx_loader_and_splitting(self, file_path: str, text_splitter) -> list:
        """Return the splitted documents from the docx file path."""

        loader = Docx2txtLoader(file_path)
        documents = await loader.aload()

        return await text_splitter.split_documents(documents)

    async def _xlsx_loader_and_splitting(
        self, file_path: str, text_splitter, sheet_name: str | None = None
    ) -> list:
        """Return the splitted documents from the excel file path."""

        if sheet_name != None:
            loader = UnstructuredExcelLoader(file_path, mode="elements")
            documents = await loader.aload()

            return await text_splitter.split_documents(documents)
        else:
            raise Exception("Sheet name is not provided!")

    def _get_text_splitter(
        self, text_splitter_name: str, chunk_size: int, chunk_overlap: int
    ) -> CharacterTextSplitter | RecursiveCharacterTextSplitter:
        """Return the text splitting method based on the name."""

        match text_splitter_name:
            case "recursive":
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n\n", "\n\n", " ", ""],
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    strip_whitespace=True,
                )
            case "_":
                text_splitter = CharacterTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator="\n\n"
                )

        return text_splitter

    async def load_and_split_file(
        self,
        file_path: str,
        text_splitter_name: str,
        chunk_size: int,
        chunk_overlap: int,
        encoding: str = "utf-8",
        sheet_name: str | None = None,
    ) -> list:
        """Return the splitted documents from the file path.
        Args:
            file_path (str): Path of the file
            text_splitter_name (str): Name of the text splitting method [recursive, _]
                default: CharacterTextSplitter
            chunk_size (int): Chunk size for the text splitting
            chunk_overlap (int): Chunk overlap for the text splitting
            encoding (str): Encoding of the file
            sheet_name (str): Sheet name for the excel file

        Returns:
            list: List of splitted documents"""

        logging.getLogger("logger").debug(f"Loading and splitting file: {file_path}")

        text_splitter = self._get_text_splitter(
            text_splitter_name, chunk_size, chunk_overlap
        )

        if await aos.path.exists(file_path):

            extension = os.path.splitext(file_path)[1].strip()

            if extension == ".pdf":
                return await self._pdf_loader_and_splitting(file_path, text_splitter)

            elif extension == ".docx" or extension == ".doc":
                return await self._docx_loader_and_splitting(file_path, text_splitter)

            elif extension == ".xlsx":
                return await self._xlsx_loader_and_splitting(
                    file_path, text_splitter, sheet_name
                )

            elif extension == ".txt":
                return await self._txt_loader_and_splitting(
                    file_path, text_splitter, encoding=encoding
                )

            else:
                raise Exception("File type is not supported!")
        else:
            raise FileNotFoundError("File path is not valid!")
