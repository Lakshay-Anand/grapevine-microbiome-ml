"""Shared constants and class definitions for the manuscript analyses."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "FeatureDataWoOut.pkl"
RESULTS_DIR = PROJECT_ROOT / "results"
N_METADATA_COLUMNS = 6
RANDOM_SEED = 5

APPROACH_A_CLASSES = {
    "Country": ["USA", "Spain", "Croatia", "France", "Hungary", "Italy", "Argentina", "South Africa", "Australia", "Denmark", "Portugal", "Germany"],
    "Continent": ["NAm", "Eur", "SAm", "Afr", "Oce"],
    "Grape_variety": ["Merlot", "Tempranillo", "Callet", "Cabernet Sauvignon", "Verdejo", "Sauvignon Blanc", "Cabernet Franc", "Chardonnay", "Pinot Noir", "Grenache", "Airen", "Shiraz", "Barbera", "Malbec", "Rondo", "Solaris", "Verdelho", "Riesling", "Sangiovese"],
}

APPROACH_B_CLASSES = {
    "Country": ["Australia", "Denmark", "Italy", "Spain", "USA"],
    "Continent": ["Eur", "NAm", "Oce"],
    "Grape_variety": ["Cabernet Sauvignon", "Chardonnay", "Merlot", "Sangiovese", "Shiraz", "Tempranillo"],
    "Rootstock": ["110R", "3309C", "420A", "SO4"],
    "comb": ["Chardonnay-110R", "Chardonnay-3309C", "Chardonnay-420A", "Chardonnay-SO4", "Merlot-3309C", "Merlot-420A", "Sangiovese-110R", "Sangiovese-420A"],
}
