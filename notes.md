# Notes

## Current objective

Build a read-only market scanner that gets public market data from Hyperliquid.

## What the first program should do

1. Connect to Hyperliquid's public Info API
2. Get all mid prices
3. Print BTC, ETH, and SOL prices
4. Handle errors cleanly
5. Avoid anything involving wallets, private keys, or real orders

## Dangerous words for current stage

- private_key
- secret
- wallet signing
- exchange endpoint
- order
- buy
- sell
- leverage

## Safe words for current stage

- info endpoint
- allMids
- prices
- market data
- requests
- read-only