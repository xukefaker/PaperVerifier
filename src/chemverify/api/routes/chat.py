from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..schemas import PaperChatRequest, PaperChatResponse

router = APIRouter()


@router.post("/chat/paper", response_model=PaperChatResponse)
def chat_about_paper(request: Request, payload: PaperChatRequest) -> PaperChatResponse:
    with request.app.state.service_manager.acquire_services() as services:
        try:
            response = services.deep_chat.answer(
                paper_id=payload.paper_id,
                query=payload.query,
                history=[message.model_dump() for message in payload.history],
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Paper not found") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        return PaperChatResponse.model_validate(response)
