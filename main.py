import io
import logging
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from data_pipeline import create_cleaning_report, process_data, validate_dataframe
from src import (
    HealthResponse,
    MovieCache,
    MovieDataResponse,
    MovieProcessingReport,
    MovieStorage,
    MovieUploadInfo,
    ProcessConfig,
    log_requests,
    setup_logging,
)
from src.storage import MOVIE_COLUMNS

setup_logging(verbose=True)
logger = logging.getLogger(__name__)

app = FastAPI(title="Movie API", version="1.0.0")
app.middleware("http")(log_requests)
storage = MovieStorage()
cache = MovieCache()


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Return the application status and version."""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/upload", response_model=MovieUploadInfo)
async def upload_movies(file: UploadFile = File(...)):
    """Read a CSV upload and replace the stored movie collection."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix != ".csv":
        logger.warning(f"Unsupported file format: {suffix}")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{suffix}'. Only .csv files are accepted.",
        )

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as e:
        logger.warning(f"Could not read CSV '{file.filename}': {e}")
        raise HTTPException(status_code=400, detail=f"Could not read CSV file: {e}")

    if len(df.columns) != len(MOVIE_COLUMNS) or set(df.columns) != set(MOVIE_COLUMNS):
        logger.warning(f"Unexpected columns: {list(df.columns)}")
        raise HTTPException(
            status_code=400,
            detail=f"CSV must have exactly these columns: {MOVIE_COLUMNS}",
        )
    if df.empty:
        logger.warning("Uploaded CSV has no data rows")
        raise HTTPException(status_code=400, detail="CSV has no data rows")

    df = df[MOVIE_COLUMNS]
    storage.save(df)
    cache.invalidate()
    logger.info(f"Uploaded movies: filename={file.filename}, rows={len(df)}")
    return {
        "filename": file.filename,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
    }


@app.get("/movies", response_model=MovieDataResponse)
def get_movies(limit: int = Query(default=100, ge=1)):
    """Return up to limit stored movies and the total count."""
    cached = cache.get(limit)
    if cached is not None:
        logger.info(f"Cache hit for movies: limit={limit}")
        return cached

    data = storage.select_columns(MOVIE_COLUMNS, limit=limit)
    total = storage.count_movies()
    response = {"data": data, "total_rows": total}
    cache.set(limit, response)
    logger.info(f"Retrieved movies: returned_rows={len(data)}, total_rows={total}")
    return response


@app.get("/movies/search", response_model=list[dict])
def query_movies(
    columns: list[str] = Query(default=["title"]),
    min_rating: float = Query(default=0, ge=0, le=10),
    limit: int = Query(default=100, ge=1),
):
    """Return selected fields for movies at or above min_rating."""
    try:
        result = storage.query_movies(columns, min_rating, limit)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    logger.info(
        f"Queried movies: columns={columns}, min_rating={min_rating}, "
        f"returned_rows={len(result)}"
    )
    return result


@app.post("/movies/process", response_model=MovieProcessingReport)
def process_movies(config: ProcessConfig):
    """Validate and process the stored movies using MP1 functions."""
    df = storage.get()
    if df.empty:
        logger.warning("No movies stored to process")
        raise HTTPException(status_code=400, detail="No movies stored to process")

    config_dict = config.model_dump()
    required_columns = config_dict["validation"]["required_columns"]
    numeric_columns = config_dict["validation"]["numeric_columns"]

    missing_numeric = [c for c in numeric_columns if c not in df.columns]
    if missing_numeric:
        logger.warning(f"Numeric columns not found: {missing_numeric}")
        raise HTTPException(
            status_code=400,
            detail=f"Numeric columns not found: {missing_numeric}",
        )

    try:
        validated = validate_dataframe(df, required_columns, numeric_columns)
        validation_removed = len(df) - len(validated)
        validated_copy = validated.copy()
        processed = process_data(validated, config_dict)
        cleaning_report = create_cleaning_report(validated_copy, processed)
    except ValueError as error:
        logger.warning(f"Processing failed: {error}")
        raise HTTPException(status_code=400, detail=str(error))

    storage.save(processed)
    cache.invalidate()
    logger.info(
        f"Processed movies: validation_removed={validation_removed}, "
        f"rows_before={cleaning_report['rows_before']}, "
        f"rows_after={cleaning_report['rows_after']}"
    )
    return {
        "validation_rows_removed": validation_removed,
        "cleaning_report": cleaning_report,
    }