import logging
from decouple import config
from typing import Any, List, Optional
from fastapi import APIRouter, Depends

from app.models.response import (
    AgentInvoke as AgentInvokeResponse,
)
from app.utils.api import get_current_api_user
from app.utils.llm import LLM_MAPPING
from app.utils.prisma import prisma
from app.vectorstores.pinecone import PineconeVectorStore as pinecone_client

from llama_index import (
    VectorStoreIndex,
    PromptTemplate,
    ServiceContext,
)
from llama_index.chat_engine.types import ChatMode
from llama_index.llms import OpenAI
from llama_index.llms.types import MessageRole, ChatMessage
from llama_index.memory import ChatMemoryBuffer
from llama_index.vector_stores import PineconeVectorStore

from pydantic import BaseModel

router = APIRouter()
logging.basicConfig(level=logging.INFO)


class AgentRAGInvoke(BaseModel):
    input: str
    chatHistory: Optional[List[Any]]


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

    chat_history = createChatHistory(body)

    chat_engine = index.as_chat_engine(
        service_context=create_service_context(agent_config.llmModel),
        chat_mode=ChatMode.CONTEXT,
        chat_history=chat_history,
        memory=ChatMemoryBuffer.from_defaults(
            chat_history=chat_history, token_limit=28672
        ),
        context_template=create_prompt_template(agent_config.prompt),
    )
    chat = await chat_engine.achat(message=body.input, chat_history=chat_history)

    return {
        "success": True,
        "data": {
            "input": body.input,
            "output": chat.response,
        },
    }


def create_vector_store(metadata_filters: any):
    return PineconeVectorStore(
        pinecone_index=pinecone_client().index, metadata_filters=metadata_filters
    )


def createChatHistory(body: AgentRAGInvoke):
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
