# Optional: anchor ledger hashes on a local Ethereum network

Status: **contract provided, not part of the core app and not tested in the CI of this project.**

1. Start a dev chain: `npx hardhat node` (or Ganache).
2. Deploy `TrustChainAnchor.sol` (Remix IDE "Dev - Hardhat Provider" is the easiest way).
3. After saving a checkpoint in the dashboard, copy the latest block hash (Blockchain Verification page -> "Full hashes")
   and call `anchor(0x<hash>, <block_count>)`.
4. To audit later, call `matches(index, 0x<hash>)` with the hash recomputed from your database.

Never put private keys in this repository; use environment variables.
