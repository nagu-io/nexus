import os

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "nexus.api:app",
        host="127.0.0.1",
        port=int(os.environ.get("NEXUS_BACKEND_PORT", "8000")),
    )