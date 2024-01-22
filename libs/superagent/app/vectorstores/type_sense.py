import typesense
from decouple import config
from langchain.docstore.document import Document

DEFAULT_TEXT_KEY = "text"
DEFAULT_METADATA_KEY = "metadata"
DEFAULT_COLLECTION_NAME = "superagent"


class TypesenseVectorStore:
    def __init__(
        self,
        index_name: str = config("TYPESENSE_COLLECTION", DEFAULT_COLLECTION_NAME),
        typesense_host: str = config("TYPESENSE_HOST", ""),
        typesense_api_key: str = config("TYPESENSE_API_KEY", ""),
    ) -> None:
        if not typesense_api_key:
            raise ValueError(
                "Please provide a Typesense API key via the "
                "`TYPESENSE_API_KEY` environment variable."
            )

        if not typesense_host:
            raise ValueError(
                "Please provide a Typesense host via the "
                "`TYPESENSE_HOST` environment variable."
            )

        self._collection_name = index_name
        self._client = typesense.Client(
            {
                "nodes": [
                    {
                        "host": typesense_host,
                        "port": "443",
                        "protocol": "https",
                    }
                ],
                "api_key": typesense_api_key,
            }
        )

    def delete(self, datasource_id: str):
        self._client.collections[self._collection_name].documents.delete(
            {"filter_by": f"datasource_id={datasource_id}"}
        )

    def _stringify_dict(self, input_dict):
        for key, value in input_dict.items():
            if isinstance(value, (dict, list)):
                # If the value is a nested dictionary or list, recurse into it
                input_dict[key] = self._stringify_dict(value)
            elif not isinstance(value, (str, int)):
                # If the value is not a string or integer, convert it to a string
                input_dict[key] = str(value)
        return input_dict

    def embed_documents(self, documents: list[Document], batch_size: int = 20):
        to_upsert = [
            {
                "id": f"{doc.metadata['datasource_id']}_{i}",
                DEFAULT_TEXT_KEY: doc.page_content,
                DEFAULT_METADATA_KEY: {
                    **self._stringify_dict(doc.metadata),
                    "chunk": i,
                },
            }
            for i, doc in enumerate(documents)
        ]
        return self._client.collections[self._collection_name].documents.import_(
            to_upsert, {"action": "upsert"}, batch_size=batch_size
        )
