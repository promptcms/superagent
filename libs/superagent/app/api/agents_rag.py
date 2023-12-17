import logging
from datetime import datetime
from typing import Any, List, Optional, cast, AsyncIterable
from fastapi import APIRouter, Depends
from prisma.enums import DatasourceStatus
from prisma.models import Agent
from starlette.responses import StreamingResponse

from app.models.response import (
    AgentInvoke as AgentInvokeResponse,
)
from app.utils.api import get_current_api_user
from app.utils.llm import LLM_MAPPING
from app.utils.prisma import prisma
from app.vectorstores.pinecone import PineconeVectorStore as pinecone_client

from langchain.text_splitter import RecursiveCharacterTextSplitter

from llama_index import (
    Document,
    PromptTemplate,
    ServiceContext,
    VectorStoreIndex,
)
from llama_index.bridge.pydantic import Field
from llama_index.chat_engine.types import ChatMode, StreamingAgentChatResponse
from llama_index.llms import OpenAI
from llama_index.llms.types import MessageRole, ChatMessage
from llama_index.memory import ChatMemoryBuffer
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.schema import NodeWithScore, QueryBundle
from llama_index.service_context import ServiceContext
from llama_index.utils import GlobalsHelper
from llama_index.vector_stores import PineconeVectorStore

from pydantic import BaseModel

router = APIRouter()
logging.basicConfig(level=logging.INFO)


class AgentDatasourceRecency(BaseModel):
    success: bool
    data: Any

@router.get(
    "/agents_rag/{agent_id}/datasource_recency",
    name="datasource_recency",
    description="Get RAG agent datasource recency",
    response_model=AgentDatasourceRecency,
)
async def get_datasource_recency(
        agent_id: str, api_user=Depends(get_current_api_user)
):
    agent_config = await prisma.agent.find_first(
        where={"id": agent_id, "apiUserId": api_user.id},
        include={
            "datasources": {"include": {"datasource": True}},
        },
    )
    return {
        "success": True,
        "data": {
            "recency": create_recency(agent_config),
        },
    }


class AgentRAGInvoke(BaseModel):
    input: str
    chatHistory: Optional[List[Any]]
    enableStreaming: Optional[bool] = False


@router.post(
    "/agents_rag/{agent_id}/invoke",
    name="invoke",
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

    metadata_filters = {"datasource_id": {"$in": datasource_ids}}
    vector_store = create_vector_store(metadata_filters)
    index = VectorStoreIndex.from_vector_store(vector_store)

    chat_history = create_chat_history(body)

    chat_engine = index.as_chat_engine(
        service_context=create_service_context(agent_config.llmModel),
        chat_mode=ChatMode.CONTEXT,
        chat_history=chat_history,
        memory=ChatMemoryBuffer.from_defaults(
            chat_history=chat_history, token_limit=12288 # GPT 3.5 Turbo limit of 16,384 - 4,096 output token limit
        ),
        context_template=create_prompt_template(agent_config.prompt),
        similarity_top_k=64,
        node_postprocessors=[TokenLimitingPostprocessor(8192)] # Memory token limit of 12,288 - 4,096 system message & user message token limit
    )
    async def stream_message_generator(stream: StreamingAgentChatResponse) -> AsyncIterable[str]:
        async for response in stream.async_response_gen():
            yield f"data: {response}\n\n"

    if body.enableStreaming:
        chat = await chat_engine.astream_chat(message=body.input, chat_history=chat_history)
        return StreamingResponse(stream_message_generator(chat), media_type="text/event-stream")

    chat = await chat_engine.achat(message=body.input, chat_history=chat_history)

    return {
        "success": True,
        "data": {
            "input": body.input,
            "output": chat.response,
            "recency": create_recency(agent_config),
        },
    }


def create_vector_store(metadata_filters: any):
    return PineconeVectorStore(
        pinecone_index=pinecone_client().index, metadata_filters=metadata_filters
    )


def create_chat_history(body: AgentRAGInvoke):
    return (
        [
            ChatMessage(
                role=MessageRole.ASSISTANT
                if message["type"] == "ai"
                else MessageRole.USER,
                content=message["content"],
            )
            for message in body.chatHistory
        ]
        if body.chatHistory
        else []
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
        if agent_config.datasources is not None
        else 1.0,
        "staleness": min(
            [ds.datasource.updatedAt for ds in agent_config.datasources]
            if agent_config.datasources is not None
            else [datetime.DateTme.now()],
            default=datetime.now(),
        ),
    }


def create_service_context(llmModel: str):
    return ServiceContext.from_defaults(
        llm=OpenAI(temperature=0, model=LLM_MAPPING[llmModel])
    )


def create_prompt_template(prompt: Optional[str]):
    qa_template = []

    query_template = "You are to answer the query using only the given context information, additional instructions (if any) and chat history. DO NOT USE PRIOR KNOWLEDGE.\n\n"
    qa_template.append(query_template)

    context_template = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
    )
    qa_template.append(context_template)

    if prompt:
        instructions_template = (
            "Additional instructions are below.\n"
            "---------------------\n"
            f"{prompt}\n"
            "---------------------\n"
        )
        qa_template.append(instructions_template)

    qa_tmpl_str = "".join(qa_template)
    return PromptTemplate(qa_tmpl_str)


"""LLM reranker."""
from typing import Callable, List, Optional


class TokenLimitingPostprocessor(BaseNodePostprocessor):
    token_limit: int = Field(
        description="Token limit upon which to truncate context."
    )
    tokenizer_fn: Callable[[str], List] = Field(
        # NOTE: mypy does not handle the typing here well, hence the cast
        default_factory=cast(Callable[[], Any], GlobalsHelper().tokenizer),
        exclude=True,
    )

    def __init__(
        self,
        token_limit: int = 4096,
        tokenizer_fn = GlobalsHelper().tokenizer,
    ) -> None:
        super().__init__(token_limit=token_limit, tokenizer_fn=tokenizer_fn)

    @classmethod
    def class_name(cls) -> str:
        return "TokenLimitingPostProcessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        token_count = 0

        for i, n in enumerate(nodes):
            text = n.get_content()
            tokens = len(self.tokenizer_fn(text))

            if token_count + tokens > self.token_limit:
                if i == 0: # special case if the top hit is already too long
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size = self.token_limit,
                        chunk_overlap = 0,
                        length_function = lambda x: len(self.tokenizer_fn(x)),
                    )
                    texts = text_splitter.create_documents([text])

                    for j, t in enumerate(texts):
                        toks = len(self.tokenizer_fn(t.page_content))

                        if token_count + toks > self.token_limit:
                            logging.info(f"RAGging with truncated top hit comprising {token_count} tokens capped at {self.token_limit} limit with full hit comprising {tokens} tokens")
                            return [NodeWithScore(node=Document(text=tex.page_content)) for tex in texts[:j]]
                    
                        token_count += toks

                    logging.info(f"RAGging with truncated top hit comprising {token_count} tokens capped at {self.token_limit} limit with full hit comprising {tokens} tokens")
                    return [NodeWithScore(node=Document(text=tex.page_content)) for tex in texts]
                
                if tokens > token_count: # special case if the next hit is very long versus the higher hits
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size = self.token_limit - token_count,
                        chunk_overlap = 0,
                        length_function = lambda x: len(self.tokenizer_fn(x)),
                    )
                    texts = text_splitter.create_documents([text])

                    token_accum = 0
                    for j, t in enumerate(texts):
                        toks = len(self.tokenizer_fn(t.page_content))

                        if token_accum + toks > self.token_limit - token_count:
                            logging.info(f"RAGging with top-{i+1} hits comprising {token_count + token_accum} tokens capped at {self.token_limit} limit with final hit comprising {token_accum} out of {tokens} tokens")
                            return nodes[:i] + [NodeWithScore(node=Document(text=tex.page_content)) for tex in texts[:j]]

                        token_accum += toks

                    logging.info(f"RAGging with top-{i+1} hits comprising {token_count + token_accum} tokens capped at {self.token_limit} limit with final hit comprising {token_accum} out of {tokens} tokens")
                    return nodes[:i] + [NodeWithScore(node=Document(text=tex.page_content)) for tex in texts]

                logging.info(f"RAGging with top-{i} hits comprising {token_count} tokens capped at {self.token_limit} limit with next hit comprising {tokens} tokens")
                return nodes[:i]
            
            token_count += tokens
        
        logging.info(f"RAGging with top-{len(nodes)} hits comprising {token_count} tokens not hitting {self.token_limit} limit")
        return nodes
