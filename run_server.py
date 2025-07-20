#!/usr/bin/env python3
"""
Simple script to start the MetaExtract API server.
"""
import os
import sys
import uvicorn
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Start the FastAPI server."""
    print("🚀 Starting MetaExtract API Server...")
    print("=" * 50)
    print("📚 API Documentation will be available at:")
    print("   🔗 Swagger UI: http://localhost:8000/docs")
    print("   🔗 ReDoc: http://localhost:8000/redoc")
    print("   🔗 OpenAPI Schema: http://localhost:8000/openapi.json")
    print()
    print("🔍 Health Check: http://localhost:8000/api/v1/health")
    print("🏠 Root Endpoint: http://localhost:8000/")
    print()
    print("💡 To test the API, run the demo script:")
    print("   python demo_api.py")
    print()
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 