import os, sys
os.environ["PORTFOLIO_SKIP_INIT"] = "1"        # unit tests inject their own state; never set in production
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
