import json
import logging
from datetime import datetime
from typing import Any, AsyncIterable, Callable, List, Optional, cast

import typesense
from decouple import config
from fastapi import APIRouter, Depends
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index import (
    Document,
    OpenAIEmbedding,
    PromptTemplate,
    ServiceContext,
    VectorStoreIndex,
)
from llama_index.bridge.pydantic import Field
from llama_index.callbacks import CallbackManager
from llama_index.chat_engine.types import ChatMode, StreamingAgentChatResponse
from llama_index.embeddings.openai import OpenAIEmbeddingMode
from llama_index.llms import OpenAI
from llama_index.llms.types import ChatMessage, MessageRole
from llama_index.memory import ChatMemoryBuffer
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.schema import NodeWithScore, QueryBundle
from llama_index.utils import get_tokenizer
from llama_index.vector_stores import PineconeVectorStore
from llama_index.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.vector_stores.typesense import TypesenseVectorStore
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from app.models.response import (
    AgentInvoke as AgentInvokeResponse,
)
from app.utils.api import get_current_api_user
from app.utils.langfuse import LangfuseHandler
from app.utils.llm import LLM_MAPPING
from app.utils.prisma import prisma
from app.vectorstores.pinecone import PineconeVectorStore as pinecone_client
from prisma.enums import DatasourceStatus
from prisma.models import Agent

router = APIRouter()
logging.basicConfig(level=logging.INFO)


class AgentRAGInvoke(BaseModel):
    input: str
    chatHistory: Optional[List[Any]]
    enableStreaming: Optional[bool] = False
    tokenLimitMemory: Optional[int] = 12288
    tokenLimitContext: Optional[int] = 8192
    llmModel: Optional[str]


class AgentRAGSearch(BaseModel):
    input: str
    llmModel: Optional[str]


@router.post(
    "/agents_rag/{agent_id}/search",
    name="search",
    description="Search Agent Memory",
    response_model=AgentInvokeResponse,
)
async def search(
    agent_id: str, body: AgentRAGSearch, api_user=Depends(get_current_api_user)
):
    agent_config = await prisma.agent.find_first(
        where={"id": agent_id, "apiUserId": api_user.id},
        include={
            "datasources": {"include": {"datasource": True}},
        },
    )

    datasource_ids = [ds.datasourceId for ds in agent_config.datasources]
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="filter_by",
                operator=FilterOperator.EQ,
                value=f"metadata.datasource_id:=[{','.join(datasource_ids)}]"
                if datasource_ids
                else "metadata.datasource_id:=[NO_DATASOURCES_SENTINEL]",
            )
        ],
    )

    service_context = create_service_context(
        api_user_id=api_user.id,
        agent_id=agent_id,
        llm_model=body.llmModel or agent_config.llmModel,
    )

    vector_store = create_vector_store()
    index = VectorStoreIndex.from_vector_store(vector_store)
    query_engine = index.as_query_engine(
        service_context=service_context,
        filters=filters,
        similarity_top_k=8,
        verbose=True,
    )

    recency = create_recency(agent_config)

    return {
        "success": True,
        "data": {
            "input": body.input,
            "output": (await query_engine.aquery(body.input)).response,
            "recency": recency,
        },
    }


@router.post(
    "/agents_rag/{agent_id}/invoke",
    name="search",
    description="Invoke a RAG agent",
    response_model=AgentInvokeResponse,
)
async def invoke(
    agent_id: str, body: AgentRAGInvoke, api_user=Depends(get_current_api_user)
):
    agent_config = await prisma.agent.find_first(
        where={"id": agent_id, "apiUserId": api_user.id},
        include={
            "datasources": {"include": {"datasource": True}},
        },
    )

    datasource_ids = [ds.datasourceId for ds in agent_config.datasources]
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="filter_by",
                operator=FilterOperator.EQ,
                value=f"metadata.datasource_id:=[{','.join(datasource_ids)}]"
                if datasource_ids
                else "metadata.datasource_id:=[NO_DATASOURCES_SENTINEL]",
            )
        ],
    )

    chat_history = create_chat_history(chat_history=body.chatHistory)

    service_context = create_service_context(
        api_user_id=api_user.id,
        agent_id=agent_id,
        llm_model=body.llmModel or agent_config.llmModel,
    )

    vector_store = create_vector_store()
    index = VectorStoreIndex.from_vector_store(vector_store)

    chat_engine = index.as_chat_engine(
        service_context=service_context,
        chat_mode=ChatMode.CONTEXT,
        chat_history=chat_history,
        memory=ChatMemoryBuffer.from_defaults(
            chat_history=chat_history,
            token_limit=body.tokenLimitMemory,
        ),
        filters=filters,
        context_template=create_prompt_template(
            additional_instructions=agent_config.prompt
        ),
        similarity_top_k=64,
        node_postprocessors=[
            TokenLimitingPostprocessor(token_limit=body.tokenLimitContext)
        ],
    )

    recency = create_recency(agent_config)

    if not body.enableStreaming:
        return {
            "success": True,
            "data": {
                "input": body.input,
                "output": (
                    await chat_engine.achat(
                        message=body.input, chat_history=chat_history
                    )
                ).response,
                "recency": recency,
            },
        }

    async def stream_message_generator(
        stream: StreamingAgentChatResponse,
    ) -> AsyncIterable[str]:
        async for delta in stream.async_response_gen():
            data = f"data: {json.dumps({'delta': delta})}\n\n"
            yield data

    chat = await chat_engine.astream_chat(message=body.input, chat_history=chat_history)

    return StreamingResponse(
        content=stream_message_generator(stream=chat),
        media_type="text/event-stream",
        headers={
            "x-superagent-recency-progress": repr(recency.get("progress", 1)),
            "x-superagent-recency-staleness": recency.get(
                "staleness", datetime.now()
            ).isoformat(),
        },
    )


def create_recency(agent_config: Agent | None):
    return {
        "progress": (
            len(
                [
                    ds.datasource
                    for ds in agent_config.datasources
                    if ds.datasource.status == DatasourceStatus.DONE
                ]
            )
            / len(agent_config.datasources)
        )
        if agent_config.datasources
        else 0.0,
        "staleness": min(
            [ds.datasource.updatedAt for ds in agent_config.datasources]
            if agent_config.datasources
            else [datetime.now()],
        ),
    }


def create_vector_store():
    vectorstore = config("VECTORSTORE", "pinecone")
    if vectorstore == "typesense":
        tsvs = TypesenseVectorStore(
            client=typesense.Client(
                {
                    "nodes": [
                        {
                            "host": config("TYPESENSE_HOST", ""),
                            "port": int(config("TYPESENSE_PORT", "443")),
                            "protocol": config("TYPESENSE_PROTOCOL", "https"),
                        }
                    ],
                    "api_key": config("TYPESENSE_API_KEY", ""),
                }
            ),
            collection_name=config("TYPESENSE_COLLECTION", "superagent"),
        )
        tsvs.is_embedding_query = True
        return tsvs
    else:
        return PineconeVectorStore(pinecone_index=pinecone_client().index)


def create_chat_history(chat_history: Optional[List[Any]]):
    return (
        [
            ChatMessage(
                role=MessageRole.ASSISTANT
                if message["type"] == "ai"
                else MessageRole.USER,
                content=message["content"],
            )
            for message in chat_history
        ]
        if chat_history
        else []
    )


def create_service_context(api_user_id: str, agent_id: str, llm_model: str):
    langfuse_handler = LangfuseHandler(
        debug=False, agent_id=agent_id, api_user_id=api_user_id
    )
    callback_manager = CallbackManager([langfuse_handler])
    return ServiceContext.from_defaults(
        callback_manager=callback_manager,
        llm=OpenAI(temperature=0, model=LLM_MAPPING[llm_model]),
        embed_model=OpenAIEmbedding(mode=OpenAIEmbeddingMode.SIMILARITY_MODE),
    )


def create_prompt_template(additional_instructions: Optional[str]):
    qa_template = []

    query_template = (
        "You are to answer the query using only the given context information, "
        "additional instructions (if any) and chat history. "
        "DO NOT USE PRIOR KNOWLEDGE.\n\n"
    )
    qa_template.append(query_template)

    context_template = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
    )
    qa_template.append(context_template)

    if additional_instructions:
        instructions_template = (
            "Additional instructions are below.\n"
            "---------------------\n"
            f"{additional_instructions}\n"
            "---------------------\n"
        )
        qa_template.append(instructions_template)

    qa_tmpl_str = "".join(qa_template)
    return PromptTemplate(qa_tmpl_str)


class TokenLimitingPostprocessor(BaseNodePostprocessor):
    token_limit: int = Field(description="Token limit for context.")
    tokenizer_fn: Callable[[str], List] = Field(
        # NOTE: mypy does not handle the typing here well, hence the cast
        default_factory=cast(Callable[[], Any], get_tokenizer()),
        exclude=True,
    )

    def __init__(
        self,
        token_limit: int = 8192,
        tokenizer_fn=get_tokenizer(),
    ) -> None:
        super().__init__(token_limit=token_limit, tokenizer_fn=tokenizer_fn)

    @classmethod
    def class_name(_) -> str:
        return "TokenLimitingPostProcessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,  # noqa: F841
    ) -> List[NodeWithScore]:
        token_count = 0

        for i, n in enumerate(nodes):
            text = n.get_content()
            tokens = len(self.tokenizer_fn(text))

            if token_count + tokens > self.token_limit:
                if i == 0:  # special case if the top hit is already too long
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.token_limit,
                        chunk_overlap=0,
                        length_function=lambda x: len(self.tokenizer_fn(x)),
                    )
                    texts = text_splitter.create_documents([text])

                    for j, t in enumerate(texts):
                        toks = len(self.tokenizer_fn(t.page_content))

                        if token_count + toks > self.token_limit:
                            logging.info(
                                (
                                    "RAGging with truncated hit: "
                                    f"{token_count}/{tokens} tokens "
                                    f"(limited to {self.token_limit})"
                                )
                            )
                            return [
                                NodeWithScore(node=Document(text=tex.page_content))
                                for tex in texts[:j]
                            ]

                        token_count += toks

                    logging.info(
                        (
                            "RAGging with truncated hit: "
                            f"{token_count}/{tokens} tokens "
                            f"(limited to {self.token_limit})"
                        )
                    )
                    return [
                        NodeWithScore(node=Document(text=tex.page_content))
                        for tex in texts
                    ]

                if (
                    tokens > token_count
                ):  # special case if the next hit is very long versus the higher hits
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.token_limit - token_count,
                        chunk_overlap=0,
                        length_function=lambda x: len(self.tokenizer_fn(x)),
                    )
                    texts = text_splitter.create_documents([text])

                    token_accum = 0
                    for j, t in enumerate(texts):
                        toks = len(self.tokenizer_fn(t.page_content))

                        if token_accum + toks > self.token_limit - token_count:
                            logging.info(
                                (
                                    "RAGging with {i+1} hits: "
                                    f"{token_count + token_accum} tokens "
                                    f"(limited to {self.token_limit})"
                                )
                            )
                            return nodes[:i] + [
                                NodeWithScore(node=Document(text=tex.page_content))
                                for tex in texts[:j]
                            ]

                        token_accum += toks

                    logging.info(
                        (
                            f"RAGging with {i+1} hits: "
                            f"{token_count + token_accum} tokens "
                            f"(limited to {self.token_limit})"
                        )
                    )
                    return nodes[:i] + [
                        NodeWithScore(node=Document(text=tex.page_content))
                        for tex in texts
                    ]

                logging.info(
                    (
                        f"RAGging with {i} hits: "
                        f"{token_count} tokens "
                        f"(limited to {self.token_limit})"
                    )
                )
                return nodes[:i]

            token_count += tokens

        logging.info(
            (
                f"RAGging with {len(nodes)} hits: "
                f"{token_count} tokens "
                f"(below {self.token_limit})"
            )
        )
        return nodes
