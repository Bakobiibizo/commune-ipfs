"""
Module Registry API endpoints for IPFS Storage System.

Provides specialized endpoints for module metadata storage and retrieval,
designed to work with the Substrate Module Registry Pallet.
"""

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.config import get_settings
from app.database import DatabaseService, FileRecord
from app.logging_config import get_logger, log_file_operation, log_ipfs_operation
from app.services.ipfs import IPFSService

router = APIRouter()
settings = get_settings()
logger = get_logger(__name__)


# Pydantic models for module registry
class ModuleMetadata(BaseModel):
    """Module metadata structure for IPFS storage."""

    name: str = Field(..., description="Module name")
    version: str = Field(..., description="Module version")
    description: str | None = Field(None, description="Module description")
    author: str | None = Field(None, description="Module author")
    license: str | None = Field(None, description="Module license")
    repository: str | None = Field(None, description="Repository URL")
    dependencies: list[str] | None = Field(
        default_factory=list, description="Module dependencies"
    )
    tags: list[str] | None = Field(default_factory=list, description="Module tags")
    public_key: str = Field(..., description="Public key identifier")
    chain_type: str | None = Field(
        None, description="Blockchain type (ed25519, ethereum, solana)"
    )
    created_at: str | None = Field(None, description="Creation timestamp")
    updated_at: str | None = Field(None, description="Last update timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "my-awesome-module",
                "version": "1.0.0",
                "description": "An awesome module for doing great things",
                "author": "developer@example.com",
                "license": "MIT",
                "repository": "https://github.com/user/my-awesome-module",
                "dependencies": ["substrate-api", "polkadot-js"],
                "tags": ["utility", "defi", "substrate"],
                "public_key": "0x1234567890abcdef...",
                "chain_type": "ed25519",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
            }
        }


class ModuleRegistrationRequest(BaseModel):
    """Request model for module registration."""

    metadata: ModuleMetadata = Field(..., description="Module metadata")
    pin: bool = Field(True, description="Whether to pin the metadata in IPFS")


class ModuleRegistrationResponse(BaseModel):
    """Response model for module registration."""

    cid: str = Field(..., description="IPFS CID of the stored metadata")
    metadata: ModuleMetadata = Field(..., description="Stored module metadata")
    gateway_url: str = Field(..., description="IPFS gateway URL for the metadata")
    size: int = Field(..., description="Size of the stored metadata in bytes")
    pinned: bool = Field(..., description="Whether the metadata is pinned in IPFS")


class ModuleSearchRequest(BaseModel):
    """Request model for module search."""

    query: str | None = Field(
        None, description="Search query (name, description, tags)"
    )
    chain_type: str | None = Field(None, description="Filter by blockchain type")
    tags: list[str] | None = Field(None, description="Filter by tags")
    author: str | None = Field(None, description="Filter by author")
    skip: int = Field(0, ge=0, description="Number of results to skip")
    limit: int = Field(50, ge=1, le=100, description="Maximum results to return")


class ModuleSearchResponse(BaseModel):
    """Response model for module search."""

    modules: list[dict[str, Any]] = Field(..., description="List of matching modules")
    total: int = Field(..., description="Total number of matching modules")
    skip: int = Field(..., description="Number of results skipped")
    limit: int = Field(..., description="Maximum results returned")


def get_ipfs_service():
    """Dependency to get IPFS service instance."""
    return IPFSService()


def get_database_service():
    """Dependency to get database service instance."""
    return DatabaseService()


@router.post("/modules/register", response_model=ModuleRegistrationResponse)
async def register_module(
    request: Request,
    registration_request: ModuleRegistrationRequest,
    ipfs_service: IPFSService = Depends(get_ipfs_service),
    db_service: DatabaseService = Depends(get_database_service),
):
    """
    Register a module by storing its metadata on IPFS.

    This endpoint:
    1. Validates the module metadata
    2. Stores the metadata as JSON on IPFS
    3. Returns the CID for use with the Substrate pallet
    4. Optionally pins the metadata to prevent garbage collection

    - **metadata**: Complete module metadata
    - **pin**: Whether to pin the metadata in IPFS (default: True)
    """
    try:
        logger.info(
            f"📦 Registering module: {registration_request.metadata.name} v{registration_request.metadata.version}"
        )

        # Convert metadata to JSON
        metadata_dict = registration_request.metadata.model_dump()
        metadata_json = json.dumps(metadata_dict, indent=2, sort_keys=True)

        # Upload JSON directly to IPFS (bypass file type restrictions)
        import time

        start_time = time.time()
        ipfs_result = await ipfs_service.add_json_content(
            content=metadata_json,
            filename=f"{registration_request.metadata.name}-{registration_request.metadata.version}-metadata.json",
        )
        upload_duration = time.time() - start_time

        logger.info(
            f"✅ Module metadata uploaded to IPFS: {ipfs_result['cid']} ({upload_duration:.3f}s)"
        )
        log_ipfs_operation(
            "MODULE_REGISTER", ipfs_result["cid"], True, duration=upload_duration
        )

        # Get client IP
        client_ip = request.client.host if request.client else None

        # Store metadata in database with module-specific tags
        module_tags = ["module", "registry"]
        if registration_request.metadata.chain_type:
            module_tags.append(registration_request.metadata.chain_type)
        if registration_request.metadata.tags:
            module_tags.extend(registration_request.metadata.tags)

        file_record = db_service.create_file_record(
            cid=ipfs_result["cid"],
            filename=ipfs_result["filename"],
            original_filename=ipfs_result["filename"],
            content_type="application/json",
            size=ipfs_result["size"],
            description=f"Module metadata for {registration_request.metadata.name} v{registration_request.metadata.version}",
            tags=json.dumps(module_tags),
            uploader_ip=client_ip,
        )

        # Pin if requested
        pinned = False
        if registration_request.pin:
            pinned = await ipfs_service.pin_file(ipfs_result["cid"])
            if pinned:
                # Update database record
                db = db_service.get_session()
                try:
                    file_record.is_pinned = 1
                    db.commit()
                except Exception as e:
                    db.rollback()
                    logger.warning(f"Failed to update pin status: {e}")
                finally:
                    db.close()

        log_file_operation(
            "MODULE_REGISTER",
            file_record.cid,
            file_record.filename,
            file_record.size,
            client_ip,
        )

        logger.info(
            f"📁 Module registered successfully: {registration_request.metadata.name} -> {file_record.cid}"
        )

        return ModuleRegistrationResponse(
            cid=file_record.cid,
            metadata=registration_request.metadata,
            gateway_url=ipfs_service.get_gateway_url(file_record.cid),
            size=ipfs_result["size"],
            pinned=pinned,
        )

    except HTTPException:
        raise
    except Exception as e:
        msg = f"Module registration failed: {str(e)}"
        logger.error(msg)
        raise HTTPException(status_code=500, detail=msg) from e


@router.get("/modules/{cid}", response_model=ModuleMetadata)
async def get_module_metadata(
    cid: str,
    ipfs_service: IPFSService = Depends(get_ipfs_service),
    db_service: DatabaseService = Depends(get_database_service),
):
    """
    Retrieve module metadata by IPFS CID.

    This endpoint:
    1. Fetches the metadata JSON from IPFS
    2. Validates and parses the metadata
    3. Returns the structured module metadata

    - **cid**: IPFS Content Identifier for the module metadata
    """
    try:
        logger.info(f"📥 Retrieving module metadata: {cid}")

        # Get metadata from IPFS
        metadata_bytes = await ipfs_service.get_file(cid)
        metadata_json = metadata_bytes.decode("utf-8")
        metadata_dict = json.loads(metadata_json)

        # Validate and parse metadata
        metadata = ModuleMetadata(**metadata_dict)

        logger.info(
            f"✅ Module metadata retrieved: {metadata.name} v{metadata.version}"
        )

        return metadata

    except json.JSONDecodeError as e:
        msg = f"Invalid JSON in module metadata: {str(e)}"
        logger.error(msg)
        raise HTTPException(status_code=422, detail=msg) from e
    except Exception as e:
        msg = f"Failed to retrieve module metadata: {str(e)}"
        logger.error(msg)
        raise HTTPException(status_code=404, detail=msg) from e


@router.post("/modules/search", response_model=ModuleSearchResponse)
async def search_modules(
    search_request: ModuleSearchRequest,
    ipfs_service: IPFSService = Depends(get_ipfs_service),
    db_service: DatabaseService = Depends(get_database_service),
):
    """
    Search for modules by various criteria.

    This endpoint:
    1. Searches the database for matching module metadata files
    2. Retrieves and parses the metadata from IPFS
    3. Filters results based on search criteria
    4. Returns paginated results

    - **query**: Search query (searches name, description, tags)
    - **chain_type**: Filter by blockchain type
    - **tags**: Filter by specific tags
    - **author**: Filter by author
    - **skip**: Number of results to skip (pagination)
    - **limit**: Maximum results to return
    """
    try:
        logger.info(
            f"🔍 Searching modules: query='{search_request.query}', chain_type='{search_request.chain_type}'"
        )

        # Search database for module files (tagged with "module" and "registry")
        db = db_service.get_session()
        try:
            query = db.query(FileRecord).filter(
                FileRecord.content_type == "application/json"
            )

            # Filter by tags containing "module" and "registry"
            if search_request.query:
                query = query.filter(
                    (FileRecord.filename.contains(search_request.query))
                    | (FileRecord.description.contains(search_request.query))
                    | (FileRecord.tags.contains(search_request.query))
                )

            # Apply pagination
            file_records = (
                query.offset(search_request.skip).limit(search_request.limit).all()
            )

        finally:
            db.close()

        # Retrieve and parse metadata for each file
        modules = []
        for file_record in file_records:
            try:
                # Get metadata from IPFS
                metadata_bytes = await ipfs_service.get_file(file_record.cid)
                metadata_json = metadata_bytes.decode("utf-8")
                metadata_dict = json.loads(metadata_json)

                # Apply additional filters
                if (
                    search_request.chain_type
                    and metadata_dict.get("chain_type") != search_request.chain_type
                ):
                    continue

                if (
                    search_request.author
                    and metadata_dict.get("author") != search_request.author
                ):
                    continue

                if search_request.tags:
                    module_tags = metadata_dict.get("tags", [])
                    if not any(tag in module_tags for tag in search_request.tags):
                        continue

                # Add file metadata
                module_info = {
                    **metadata_dict,
                    "cid": file_record.cid,
                    "gateway_url": ipfs_service.get_gateway_url(file_record.cid),
                    "upload_date": (
                        file_record.upload_date.isoformat()
                        if file_record.upload_date
                        else None
                    ),
                    "is_pinned": bool(file_record.is_pinned),
                    "size": file_record.size,
                }

                modules.append(module_info)

            except Exception as e:
                logger.warning(
                    f"Failed to parse module metadata for CID {file_record.cid}: {e}"
                )
                continue

        logger.info(f"✅ Module search completed: {len(modules)} results found")

        return ModuleSearchResponse(
            modules=modules,
            total=len(modules),  # Note: This is filtered total, not database total
            skip=search_request.skip,
            limit=search_request.limit,
        )

    except Exception as e:
        msg = f"Module search failed: {str(e)}"
        logger.error(msg)
        raise HTTPException(status_code=500, detail=msg) from e


@router.delete("/modules/{cid}")
async def unregister_module(
    cid: str,
    unpin: bool = Query(True, description="Whether to unpin from IPFS"),
    ipfs_service: IPFSService = Depends(get_ipfs_service),
    db_service: DatabaseService = Depends(get_database_service),
):
    """
    Unregister a module by removing its metadata.

    This endpoint:
    1. Removes the metadata from the database
    2. Optionally unpins the metadata from IPFS
    3. Returns confirmation of removal

    - **cid**: IPFS Content Identifier for the module metadata
    - **unpin**: Whether to unpin from IPFS (default: True)
    """
    try:
        logger.info(f"🗑️ Unregistering module: {cid}")

        # Check if file exists
        file_record = db_service.get_file_by_cid(cid)
        if not file_record:
            raise HTTPException(status_code=404, detail="Module not found")

        # Get module name for logging
        module_name = "unknown"
        try:
            metadata_bytes = await ipfs_service.get_file(cid)
            metadata_json = metadata_bytes.decode("utf-8")
            metadata_dict = json.loads(metadata_json)
            module_name = metadata_dict.get("name", "unknown")
        except Exception:
            pass  # Continue with deletion even if we can't get the name

        # Delete from database
        success = db_service.delete_file_record(cid)
        if not success:
            raise HTTPException(
                status_code=500, detail="Failed to delete module record"
            )

        # Optionally unpin from IPFS
        unpinned = False
        if unpin:
            unpinned = await ipfs_service.unpin_file(cid)

        logger.info(f"✅ Module unregistered: {module_name} ({cid})")

        return {
            "message": "Module unregistered successfully",
            "cid": cid,
            "module_name": module_name,
            "unpinned": unpinned,
        }

    except HTTPException:
        raise
    except Exception as e:
        msg = f"Module unregistration failed: {str(e)}"
        logger.error(msg)
        raise HTTPException(status_code=500, detail=msg) from e


@router.get("/modules/{cid}/stats")
async def get_module_stats(
    cid: str,
    ipfs_service: IPFSService = Depends(get_ipfs_service),
):
    """
    Get IPFS statistics for module metadata.

    - **cid**: IPFS Content Identifier for the module metadata
    """
    try:
        stats = await ipfs_service.get_file_stats(cid)

        return {
            "cid": cid,
            "size": stats.get("DataSize", 0),
            "cumulative_size": stats.get("CumulativeSize", 0),
            "blocks": stats.get("NumLinks", 0),
            "type": stats.get("Type", "unknown"),
        }

    except Exception as e:
        msg = f"Failed to get module stats: {str(e)}"
        raise HTTPException(status_code=500, detail=msg) from e
