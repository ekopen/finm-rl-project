"""Quick yfinance connectivity smoke test."""

from __future__ import annotations

import sys

import yfinance as yf


def main() -> None:
	"""Download a tiny window of data and print a short summary."""
	ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
	print(f"Fetching latest data for {ticker}…")
	df = yf.download(ticker, period="1mo", interval="1d")

	if df.empty:
		raise RuntimeError(
			"yfinance returned no rows (likely a network or Yahoo Finance issue)."
		)

	print(df.tail())
	print(
		f"\nGot {len(df)} rows from {df.index.min().date()} "
		f"to {df.index.max().date()}."
	)


if __name__ == "__main__":
	main()

