"""
Portfolio model - holdings, SIP, bonds across India, USA, UK
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class Holding:
    ticker: str
    market: str
    asset_type: str  # stock, bond, etf, gold
    quantity: float
    avg_cost: float
    current_price: float = 0

    @property
    def value(self) -> float:
        return self.quantity * (self.current_price or self.avg_cost)

    @property
    def pnl_pct(self) -> float:
        if self.avg_cost <= 0:
            return 0
        return ((self.current_price or self.avg_cost) / self.avg_cost - 1) * 100


@dataclass
class SIPEntry:
    ticker: str
    market: str
    amount: float
    frequency: str  # Monthly, Weekly, Quarterly
    start_date: str
    currency: str = "INR"


class Portfolio:
    """Portfolio across markets."""

    def __init__(self):
        self.holdings: list[Holding] = []
        self.sips: list[SIPEntry] = []
        self.cash: dict[str, float] = {"india": 0, "usa": 0, "uk": 0}
        self.currency = {"india": "₹", "usa": "$", "uk": "£"}

    def add_holding(self, h: Holding):
        self.holdings.append(h)

    def add_sip(self, s: SIPEntry):
        self.sips.append(s)

    def total_value(self, market: Optional[str] = None) -> float:
        total = sum(h.value for h in self.holdings if market is None or h.market == market)
        if market:
            total += self.cash.get(market, 0)
        else:
            total += sum(self.cash.values())
        return total

    def by_asset_type(self, market: Optional[str] = None) -> dict:
        d = {}
        for h in self.holdings:
            if market and h.market != market:
                continue
            d[h.asset_type] = d.get(h.asset_type, 0) + h.value
        return d

    def by_market(self) -> dict:
        d = {}
        for h in self.holdings:
            d[h.market] = d.get(h.market, 0) + h.value
        for m, c in self.cash.items():
            d[m] = d.get(m, 0) + c
        return d
