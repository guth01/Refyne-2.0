from fastapi import FastAPI, status, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from datetime import datetime
import asyncio
from typing import Optional
import io
from pathlib import Path
import gc
import logging
from tqdm import tqdm

# Your existing imports
import models
from database import engine, SessionLocal
import auth
from auth import get_current_user

# Import the DataQualityEnhancer class from the separate file
from data_quality_enhancer import DataQualityEnhancer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI App Setup
app = FastAPI(title="CSV Data Enhancement & Todo API", version="1.0.0")

# Include your existing auth router
app.include_router(auth.router)

# CORS Configuration for CSV processing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create database tables
models.Base.metadata.create_all(bind=engine)

# Your existing database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]
user_dependency = Annotated[dict, Depends(get_current_user)]

# Store active temp directories for cleanup tracking
active_temp_dirs = set()

# CSV Processing Functions
def process_large_dataset(file_path: str, chunk_size: int = 10000) -> pd.DataFrame:
    """
    Process a large dataset in chunks
    """
    try:
        # Read data in chunks
        chunks = pd.read_csv(file_path, chunksize=chunk_size)
        enhancer = DataQualityEnhancer(memory_efficient=True, chunk_size=chunk_size)
        
        results = []
        for chunk in chunks:
            enhanced_chunk = enhancer.enhance_data(chunk)
            results.append(enhanced_chunk)
            
        return pd.concat(results, axis=0)
    except Exception as e:
        logger.error(f"Error processing large dataset: {e}")
        raise

def handle_file(file_path: str) -> pd.DataFrame:
    """
    Main file processing function
    """
    LARGE_FILE_SIZE = 30 * 1024 * 1024  # 30 MB in bytes
    
    try:
        file_size = os.path.getsize(file_path)
        logger.info(f"File size: {file_size} bytes")
        
        if file_size > LARGE_FILE_SIZE:
            logger.info("Processing large dataset...")
            return process_large_dataset(file_path)
        else:
            logger.info("Processing small dataset...")
            df = pd.read_csv(file_path)
            enhancer = DataQualityEnhancer(memory_efficient=False)
            enhanced_df = enhancer.enhance_data(df)
            return enhanced_df
            
    except Exception as e:
        logger.error(f"Error in handle_file: {e}")
        raise

def safe_iterfile(file_path: str, temp_dir: str):
    """
    Safely iterate over file content and cleanup afterwards
    """
    try:
        # Verify file exists before opening
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
            
        logger.info(f"Starting to stream file: {file_path}")
        
        with open(file_path, mode="rb") as file_like:
            while True:
                chunk = file_like.read(8192)  # Read in 8KB chunks
                if not chunk:
                    break
                yield chunk
                
    except Exception as e:
        logger.error(f"Error streaming file: {e}")
        raise
    finally:
        # Cleanup temp directory after streaming is complete
        try:
            if temp_dir in active_temp_dirs:
                active_temp_dirs.remove(temp_dir)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as cleanup_error:
            logger.warning(f"Could not cleanup temporary directory {temp_dir}: {cleanup_error}")

# Your existing root endpoint (modified to show both functionalities)
@app.get("/", status_code=status.HTTP_200_OK)
async def user(user: user_dependency, db: db_dependency):
    if user is None:
        raise HTTPException(status_code=401, detail='Authentication Failed')
        
    return {
        "User": user,
        "message": "CSV Data Enhancement & Todo API",
        "version": "1.0.0",
        "endpoints": {
            "todos": "Your existing todo endpoints",
            "process_csv": "/process-csv (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)"
        }
    }

# Alternative approach: Return processed data directly in memory
@app.post("/process-csv-memory")
async def process_csv_file_memory(
    current_user: user_dependency,
    file: UploadFile = File(...)
):
    """
    Process uploaded CSV file and return enhanced version (in-memory approach)
    """
    # Check authentication
    if current_user is None:
        raise HTTPException(status_code=401, detail='Authentication Failed')
    
    # Validate file type
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported"
        )
    
    try:
        # Read file content directly into memory
        content = await file.read()
        
        # Convert to pandas dataframe
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        logger.info(f"Processing file: {file.filename}")
        logger.info(f"File shape: {df.shape}")
        
        # Process the dataframe
        enhancer = DataQualityEnhancer(memory_efficient=True)
        enhanced_df = enhancer.enhance_data(df)
        
        # Convert processed dataframe back to CSV string
        output = io.StringIO()
        enhanced_df.to_csv(output, index=False)
        output.seek(0)
        
        output_filename = f"enhanced_{file.filename}"
        
        logger.info("Processing completed successfully")
        
        # Return as streaming response
        return StreamingResponse(
            io.StringIO(output.getvalue()),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={output_filename}"
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing CSV: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing CSV file: {str(e)}"
        )

# Original approach with fixed file handling
@app.post("/process-csv")
async def process_csv_file(
    current_user: user_dependency,
    file: UploadFile = File(...)
):
    """
    Process uploaded CSV file and return enhanced version (file-based approach)
    """
    # Check authentication
    if current_user is None:
        raise HTTPException(status_code=401, detail='Authentication Failed')
    
    # Validate file type
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported"
        )
    
    # Create temporary directory for processing
    temp_dir = tempfile.mkdtemp(prefix="csv_processing_")
    active_temp_dirs.add(temp_dir)
    
    input_file_path = None
    output_file_path = None
    
    try:
        # Save uploaded file temporarily
        input_file_path = os.path.join(temp_dir, f"input_{file.filename}")
        
        with open(input_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Processing file: {file.filename}")
        logger.info(f"File size: {os.path.getsize(input_file_path)} bytes")
        
        # Process the file
        enhanced_df = handle_file(input_file_path)
        
        # Save processed file
        output_filename = f"enhanced_{file.filename}"
        output_file_path = os.path.join(temp_dir, output_filename)
        enhanced_df.to_csv(output_file_path, index=False)
        
        # Verify output file was created successfully
        if not os.path.exists(output_file_path):
            raise FileNotFoundError(f"Failed to create output file: {output_file_path}")
            
        file_size = os.path.getsize(output_file_path)
        logger.info(f"Processing completed. Output saved to: {output_file_path}")
        logger.info(f"Output file size: {file_size} bytes")
        
        # Return processed file as download
        # Note: temp_dir cleanup happens inside safe_iterfile after streaming
        return StreamingResponse(
            safe_iterfile(output_file_path, temp_dir),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={output_filename}",
                "Content-Length": str(file_size)
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing CSV: {str(e)}")
        
        # Cleanup on error
        try:
            if temp_dir in active_temp_dirs:
                active_temp_dirs.remove(temp_dir)
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as cleanup_error:
            logger.warning(f"Could not cleanup temporary files: {cleanup_error}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing CSV file: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "message": "CSV Data Enhancement & Todo API is running",
        "active_temp_dirs": len(active_temp_dirs)
    }

@app.get("/file-info")
async def get_file_info(current_user: user_dependency):
    """Get information about supported file formats and limits"""
    if current_user is None:
        raise HTTPException(status_code=401, detail='Authentication Failed')
        
    return {
        "supported_formats": ["CSV"],
        "max_file_size": "100MB",
        "endpoints": {
            "/process-csv": "File-based processing (for large files)",
            "/process-csv-memory": "Memory-based processing (for smaller files, more reliable)"
        },
        "features": [
            "Data type optimization",
            "Missing value handling",
            "Feature engineering",
            "Correlation removal",
            "Data standardization",
            "Categorical encoding"
        ]
    }

# Cleanup function for server shutdown
@app.on_event("shutdown")
async def cleanup_temp_dirs():
    """Cleanup any remaining temp directories on shutdown"""
    for temp_dir in active_temp_dirs.copy():
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temp directory on shutdown: {temp_dir}")
        except Exception as e:
            logger.warning(f"Could not cleanup temp directory {temp_dir}: {e}")
    active_temp_dirs.clear()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)