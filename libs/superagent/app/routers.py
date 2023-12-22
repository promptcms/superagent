from fastapi import APIRouter

from app.api import agents_rag, agents, api_user, datasources, llms, tools, workflows

router = APIRouter()
api_prefix = "/api/v1"

router.include_router(agents_rag.router, tags=["AgentRAG"], prefix=api_prefix)
router.include_router(agents.router, tags=["Agent"], prefix=api_prefix)
router.include_router(llms.router, tags=["LLM"], prefix=api_prefix)
router.include_router(api_user.router, tags=["Api user"], prefix=api_prefix)
router.include_router(datasources.router, tags=["Datasource"], prefix=api_prefix)
router.include_router(tools.router, tags=["Tool"], prefix=api_prefix)
router.include_router(workflows.router, tags=["Workflow"], prefix=api_prefix)
