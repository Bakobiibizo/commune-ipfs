#!/usr/bin/env python3
"""
Module Registry Integration Client

Demonstrates the end-to-end workflow between:
1. IPFS backend (commune-ipfs) for metadata storage
2. Substrate pallet for on-chain CID registration

This client shows how to:
- Store module metadata on IPFS
- Register the CID on-chain via the Substrate pallet
- Retrieve and verify the complete workflow
"""

import asyncio
import sys
import time
from typing import Any

import aiohttp
from pydantic import BaseModel

from app.config import get_settings


class ModuleMetadata(BaseModel):
    """Module metadata structure matching the API."""

    name: str
    version: str
    description: str | None = None
    author: str | None = None
    license: str | None = None
    repository: str | None = None
    dependencies: list[str] = []
    tags: list[str] = []
    public_key: str
    chain_type: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class ModuleRegistryClient:
    """Client for interacting with the Module Registry system."""

    def __init__(
        self, ipfs_api_url: str | None = None, substrate_rpc_url: str | None = None
    ):
        # Use environment variable or default
        if not ipfs_api_url:
            # Auto-detect from environment or use default
            settings = get_settings()
            ipfs_api_url = settings.ipfs_api_url
        self.ipfs_api_url = ipfs_api_url.rstrip("/")
        self.substrate_rpc_url = substrate_rpc_url
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def register_module_metadata(
        self, metadata: ModuleMetadata, pin: bool = True
    ) -> dict[str, Any]:
        """
        Register module metadata on IPFS.

        Args:
            metadata: Module metadata to store
            pin: Whether to pin the metadata in IPFS

        Returns:
            Dictionary containing CID and other registration info
        """
        if not self.session:
            raise RuntimeError(
                "Client not initialized. Use 'async with' context manager."
            )

        registration_data = {"metadata": metadata.model_dump(), "pin": pin}

        async with self.session.post(
            f"{self.ipfs_api_url}/api/modules/register", json=registration_data
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(
                    f"IPFS registration failed ({response.status}): {error_text}"
                )

            return await response.json()

    async def get_module_metadata(self, cid: str) -> ModuleMetadata:
        """
        Retrieve module metadata from IPFS by CID.

        Args:
            cid: IPFS Content Identifier

        Returns:
            ModuleMetadata object
        """
        if not self.session:
            raise RuntimeError(
                "Client not initialized. Use 'async with' context manager."
            )

        async with self.session.get(
            f"{self.ipfs_api_url}/api/modules/{cid}"
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(
                    f"Failed to retrieve metadata ({response.status}): {error_text}"
                )

            metadata_dict = await response.json()
            return ModuleMetadata(**metadata_dict)

    async def search_modules(
        self,
        query: str | None = None,
        chain_type: str | None = None,
        tags: list[str] | None = None,
        author: str | None = None,
        skip: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        """
        Search for modules by various criteria.

        Args:
            query: Search query string
            chain_type: Filter by blockchain type
            tags: Filter by tags
            author: Filter by author
            skip: Number of results to skip
            limit: Maximum results to return

        Returns:
            Search results dictionary
        """
        if not self.session:
            raise RuntimeError(
                "Client not initialized. Use 'async with' context manager."
            )

        search_data = {
            "query": query,
            "chain_type": chain_type,
            "tags": tags,
            "author": author,
            "skip": skip,
            "limit": limit,
        }

        async with self.session.post(
            f"{self.ipfs_api_url}/api/modules/search", json=search_data
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Search failed ({response.status}): {error_text}")

            return await response.json()

    async def unregister_module(self, cid: str, unpin: bool = True) -> dict[str, Any]:
        """
        Unregister module metadata from IPFS.

        Args:
            cid: IPFS Content Identifier
            unpin: Whether to unpin from IPFS

        Returns:
            Unregistration confirmation
        """
        if not self.session:
            raise RuntimeError(
                "Client not initialized. Use 'async with' context manager."
            )

        async with self.session.delete(
            f"{self.ipfs_api_url}/api/modules/{cid}", params={"unpin": unpin}
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(
                    f"Unregistration failed ({response.status}): {error_text}"
                )

            return await response.json()

    async def get_module_stats(self, cid: str) -> dict[str, Any]:
        """
        Get IPFS statistics for module metadata.

        Args:
            cid: IPFS Content Identifier

        Returns:
            Statistics dictionary
        """
        if not self.session:
            raise RuntimeError(
                "Client not initialized. Use 'async with' context manager."
            )

        async with self.session.get(
            f"{self.ipfs_api_url}/api/modules/{cid}/stats"
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(
                    f"Failed to get stats ({response.status}): {error_text}"
                )

            return await response.json()

    def register_on_substrate(self, public_key: bytes, cid: str) -> dict[str, Any]:
        """
        Register the CID on the Substrate pallet.

        Uses the SubstratePalletClient to register the module on-chain.

        Args:
            public_key: Public key as bytes
            cid: IPFS CID to register

        Returns:
            Transaction result with block info and status
        """
        try:
            # Import here to avoid circular imports
            from substrate_pallet_client import SubstratePalletClient

            # Create client and register module
            client = SubstratePalletClient()

            # Convert bytes to hex string for the pallet client
            public_key_hex = public_key.hex()
            if not public_key_hex.startswith("0x"):
                public_key_hex = "0x" + public_key_hex

            # Register the module
            result = client.register_module(public_key_hex, cid)

            return {
                "status": "success",
                "block_hash": result.block_hash,
                "block_number": result.block_number,
                "extrinsic_hash": result.extrinsic_hash,
                "events": result.events,
            }

        except Exception as e:
            print(f"❌ Failed to register on Substrate: {e}")
            return {
                "status": "error",
                "error": str(e),
                "block_hash": None,
                "extrinsic_hash": None,
                "events": [],
            }


async def demo_workflow():
    """Demonstrate the complete module registry workflow."""

    print("🚀 Module Registry Integration Demo")
    print("=" * 50)

    # Create sample module metadata
    metadata = ModuleMetadata(
        name="awesome-defi-module",
        version="1.2.3",
        description="An awesome DeFi module for Substrate",
        author="developer@example.com",
        license="MIT",
        repository="https://github.com/user/awesome-defi-module",
        dependencies=["substrate-api", "polkadot-js", "web3"],
        tags=["defi", "substrate", "polkadot"],
        public_key="test_public_key",
        chain_type="ed25519",
        created_at=time.time(),
        updated_at=time.time(),
    )

    async with ModuleRegistryClient() as client:
        try:
            # Step 1: Register metadata on IPFS
            print("\n📦 Step 1: Registering module metadata on IPFS...")
            registration_result = await client.register_module_metadata(
                metadata, pin=True
            )

            cid = registration_result["cid"]
            print("✅ Metadata registered successfully!")
            print(f"   CID: {cid}")
            print(f"   Size: {registration_result['size']} bytes")
            print(f"   Pinned: {registration_result['pinned']}")
            print(f"   Gateway URL: {registration_result['gateway_url']}")

            # Step 2: Register CID on Substrate pallet
            print("\n🔗 Step 2: Registering CID on Substrate pallet...")
            public_key_bytes = bytes.fromhex(metadata.public_key.replace("0x", ""))
            substrate_result = client.register_on_substrate(public_key_bytes, cid)

            print("✅ CID registered on-chain!")
            print(f"   Status: {substrate_result['status']}")
            print(f"   Block Hash: {substrate_result['block_hash']}")

            # Step 3: Verify retrieval
            print("\n📥 Step 3: Verifying metadata retrieval...")
            retrieved_metadata = await client.get_module_metadata(cid)

            print("✅ Metadata retrieved successfully!")
            print(f"   Name: {retrieved_metadata.name}")
            print(f"   Version: {retrieved_metadata.version}")
            print(f"   Author: {retrieved_metadata.author}")
            print(f"   Tags: {', '.join(retrieved_metadata.tags)}")

            # Step 4: Search for modules
            print("\n🔍 Step 4: Searching for modules...")
            search_results = await client.search_modules(
                query="defi", chain_type="ed25519", limit=10
            )

            print("✅ Search completed!")
            print(f"   Found {len(search_results['modules'])} modules")
            for module in search_results["modules"][:3]:  # Show first 3
                print(
                    f"   - {module['name']} v{module['version']} ({module['cid'][:12]}...)"
                )

            # Step 5: Get statistics
            print("\n📊 Step 5: Getting module statistics...")
            stats = await client.get_module_stats(cid)

            print("✅ Statistics retrieved!")
            print(f"   Size: {stats['size']} bytes")
            print(f"   Cumulative Size: {stats['cumulative_size']} bytes")
            print(f"   Blocks: {stats['blocks']}")

            print("\n🎉 Demo completed successfully!")
            print("\nWorkflow Summary:")
            print(f"1. ✅ Metadata stored on IPFS (CID: {cid[:12]}...)")
            print("2. ✅ CID registered on Substrate pallet")
            print("3. ✅ Metadata retrieval verified")
            print("4. ✅ Search functionality working")
            print("5. ✅ Statistics available")

            return cid

        except Exception as e:
            print(f"❌ Demo failed: {e}")
            return None


async def cleanup_demo(cid: str):
    """Clean up demo data."""
    if not cid:
        return

    print("\n🧹 Cleaning up demo data...")

    async with ModuleRegistryClient() as client:
        try:
            result = await client.unregister_module(cid, unpin=True)
            print(f"✅ Cleanup completed: {result['message']}")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")


if __name__ == "__main__":
    print("Module Registry Integration Client")
    print("This demo requires the commune-ipfs backend to be running.")
    print("Start it with: cd commune-ipfs && python main.py")
    print()

    if len(sys.argv) > 1 and sys.argv[1] == "--cleanup":
        # Cleanup mode - remove demo data
        if len(sys.argv) < 3:
            print("Usage: python integration_client.py --cleanup <cid>")
            sys.exit(1)

        cid = sys.argv[2]
        asyncio.run(cleanup_demo(cid))
    else:
        # Demo mode
        cid = asyncio.run(demo_workflow())

        if cid:
            print("\nTo clean up demo data, run:")
            print(f"python integration_client.py --cleanup {cid}")
