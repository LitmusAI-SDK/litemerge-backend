from api.routes.health import router as health
from api.routes.projects import router as projects
from api.routes.reports import router as reports
from api.routes.runs import router as runs

__all__ = ["health", "projects", "reports", "runs"]
