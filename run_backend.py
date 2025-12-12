import uvicorn
import os
import sys

if __name__ == "__main__":
    # Add the current directory to sys.path so that 'api.main' can be found
    # assuming this script is run from the 'psai' directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    
    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        # Use simple forward slashes for the glob pattern
        reload_excludes=["**/custom/*"],
        # Ensure imports work relative to the psai module
        app_dir=current_dir
    )
