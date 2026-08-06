from fastapi import APIRouter
from utils.logger import logger

router = APIRouter(prefix="/history", tags=["Conversation History"])

@router.get("")
async def get_history():
    """Retrieves conversation history log of clinical assistant interactions."""
    logger.info("History endpoint requested.")
    # In future module (Module 10 Integration / SQLite), this queries SQLite database.
    return [
        {
            "id": 1,
            "timestamp": "2026-08-04T10:15:30Z",
            "language": "ta",
            "user_query_local": "தலைவலி மருந்து என்ன?",
            "response_local": "பாராசிட்டமால் 500 மி.கி மாத்திரை வழங்கவும்.",
            "response_en": "Provide Paracetamol 500mg table.",
            "has_image": False
        },
        {
            "id": 2,
            "timestamp": "2026-08-04T12:05:00Z",
            "language": "hi",
            "user_query_local": "बुखार के लिए क्या करें?",
            "response_local": "रोगी के तापमान की जांच करें और पैरासिटामोल दें।",
            "response_en": "Assess patient temperature and give Paracetamol.",
            "has_image": True
        }
    ]
