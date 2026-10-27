// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title TrustChainAnchor
/// @notice OPTIONAL extension: anchors the Python ledger's latest SHA-256 hash on an Ethereum
///         development network so a rewrite of the local database becomes detectable by third parties.
///         It proves WHEN a hash existed, not that the underlying supply-chain data was true.
contract TrustChainAnchor {
    address public immutable owner;
    struct Anchor { bytes32 blockHash; uint256 blockCount; uint256 timestamp; }
    Anchor[] public anchors;

    event Anchored(uint256 indexed index, bytes32 blockHash, uint256 blockCount, uint256 timestamp);

    constructor() { owner = msg.sender; }

    function anchor(bytes32 blockHash, uint256 blockCount) external {
        require(msg.sender == owner, "only owner");
        anchors.push(Anchor(blockHash, blockCount, block.timestamp));
        emit Anchored(anchors.length - 1, blockHash, blockCount, block.timestamp);
    }

    function count() external view returns (uint256) { return anchors.length; }

    function matches(uint256 index, bytes32 blockHash) external view returns (bool) {
        return anchors[index].blockHash == blockHash;
    }
}
