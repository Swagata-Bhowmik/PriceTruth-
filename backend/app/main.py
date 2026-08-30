"""
Price Truth - FastAPI Main Application Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Price Truth API",
    description="ML-powered e-commerce discount verification and shrinkflation tracking",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint - API status"""
    return {
        "message": "Price Truth API",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "price-truth-api"
    }


# Import routers (will be added in later phases)
# from app.api.v1 import discount_checker, shrinkflation, price_comparison, buy_timing
# app.include_router(discount_checker.router, prefix="/api/v1", tags=["Discount Checker"])
# app.include_router(shrinkflation.router, prefix="/api/v1", tags=["Shrinkflation"])
# app.include_router(price_comparison.router, prefix="/api/v1", tags=["Price Comparison"])
# app.include_router(buy_timing.router, prefix="/api/v1", tags=["Buy Timing"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
