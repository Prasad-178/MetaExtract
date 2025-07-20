#!/usr/bin/env python3
"""
Start the MetaExtract API server.
"""
import uvicorn

def main():
    """Start the API server."""
    print("Starting MetaExtract API...")
    print("Docs: http://localhost:8000/docs")
    print("Health: http://localhost:8000/api/v1/health")
    print("Press Ctrl+C to stop")
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

if __name__ == "__main__":
    main() 