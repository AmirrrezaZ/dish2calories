"""
CalorieNet FastAPI Server - Simple, Standard, Advanced
"""
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from dataset_loaders import DatasetInfo
from models import CalorieNet
from predict import CalorieNetPredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
class Config:
    MODEL_TYPE = "efficientnet_b0"
    DEVICE = "cuda"
    DATASET_CONFIG = "configs/dataset_info.json"
    WEIGHTS_DIR = "weights"
    CLS_THRESHOLD = 0.5
    TEMPERATURE = 0.5
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Response models
class NutritionResponse(BaseModel):
    total_protein: float = Field(..., description="Protein in grams")
    total_fat: float = Field(..., description="Fat in grams") 
    total_carbs: float = Field(..., description="Carbohydrates in grams")
    total_calories: float = Field(..., description="Total calories")
    total_mass: float = Field(..., description="Mass in grams")
    labels_conf: List[float] = Field(..., description="Confidence scores")
    mean_conf: float = Field(..., description="Mean confidence")
    labels_txt: List[str] = Field(..., description="Detected foods")

class ErrorResponse(BaseModel):
    error: str
    detail: str = None

# Global predictor
predictor: CalorieNetPredictor = None

# Utility functions
def validate_file(file: UploadFile) -> None:
    """Validate uploaded file"""
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No filename provided")
    
    ext = Path(file.filename).suffix.lower()
    if ext not in Config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, 
            f"Invalid file type. Allowed: {list(Config.ALLOWED_EXTENSIONS)}"
        )

def format_results(results: Dict) -> Dict:
    """Format prediction results for JSON response"""
    try:
        return {
            "total_protein": float(results["total_protein"][0]),
            "total_fat": float(results["total_fat"][0]),
            "total_carbs": float(results["total_carbs"][0]),
            "total_calories": float(results["total_calories"][0]),
            "total_mass": float(results["total_mass"][0]),
            "labels_conf": [float(c) for c in results["labels_conf"]],
            "mean_conf": float(results["mean_conf"]),
            "labels_txt": results["labels_txt"],
        }
    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"Error formatting results: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Error processing results")

# Startup/shutdown
async def startup():
    """Load model on startup"""
    global predictor
    
    try:
        logger.info("Loading CalorieNet model...")
        
        # Validate paths
        config_path = Path(Config.DATASET_CONFIG)
        weights_path = Path(Config.WEIGHTS_DIR) / f"{Config.MODEL_TYPE}.pth"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        
        # Load components
        info = DatasetInfo.from_json(str(config_path))
        model, _ = CalorieNet.from_checkpoint(str(weights_path))
        
        # Initialize predictor
        predictor = CalorieNetPredictor(
            model=model,
            norm_ingr=info.norm_ingr,
            scaler=info.scalers["total_mass"],
            idx2cls=info.idx2cls,
            img_size=info.img_size,
            cls_threshold=Config.CLS_THRESHOLD,
            temperature=Config.TEMPERATURE,
            mode="probs",
            device=Config.DEVICE,
        )
        
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

async def shutdown():
    """Cleanup on shutdown"""
    global predictor
    predictor = None
    logger.info("Shutdown complete")

@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup()
    yield
    await shutdown()

# FastAPI app
app = FastAPI(
    title="CalorieNet API",
    description="Food nutrition analysis from images",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Routes
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "CalorieNet API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Upload image for nutrition analysis",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy" if predictor else "loading",
        "model_loaded": predictor is not None
    }

@app.post("/predict", response_model=NutritionResponse)
async def predict(file: UploadFile = File(...)):
    """Predict nutrition from food image"""
    
    # Check if model is loaded
    if not predictor:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Model not loaded. Please wait."
        )
    
    # Validate file
    validate_file(file)
    
    # Process file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        try:
            # Save uploaded file
            content = await file.read()
            tmp.write(content)
            tmp.flush()
            
            # Run prediction
            logger.info(f"Processing: {file.filename}")
            results = predictor.predict_img(Path(tmp.name))
            
            # Format response
            formatted = format_results(results)
            logger.info(f"Prediction complete. Foods: {formatted['labels_txt']}")
            
            return formatted
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                f"Prediction failed: {str(e)}"
            )
        finally:
            # Cleanup temp file
            Path(tmp.name).unlink(missing_ok=True)

# Run server
if __name__ == "__main__":
    uvicorn.run(
        "inference:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
