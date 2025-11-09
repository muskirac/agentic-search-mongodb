"""MongoDB access, tools, and sample data for agentic search demo."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable, Iterable, Literal, Mapping

import certifi
from pymongo import ASCENDING, DESCENDING, AsyncMongoClient

from agent import MongoTool

# Global MongoDB clients
mongo_client: AsyncMongoClient | None = None
demo_db: Any | None = None
products_collection: Any | None = None


demo_products: list[dict[str, Any]] = [
    # Static catalogue used throughout the talk. Embedding scores are handcrafted
    # to illustrate vector search while remaining easy to reason about on stage.
    {
        "_id": "prod-espresso-lab-001",
        "sku": "ESP-LAB-001",
        "name": "Atlas Espresso Lab Pro",
        "description": "Energy-efficient modular espresso station designed for professional teams.",
        "category": "coffee-equipment",
        "tags": ["espresso", "barista", "energy-efficient", "modular"],
        "price": 7890,
        "currency": "EUR",
        "rating": 4.9,
        "popularity": 0.94,
        "sustainability": {
            "score": 0.82,
            "certifications": ["EnergyStar"],
            "notes": "Heat-recovery boiler",
        },
        "inventory": {"warehouse": "Berlin", "stock": 6},
        "embedding": {
            "espresso-barista": 0.97,
            "travel-kit": 0.12,
            "eco-office": 0.48,
        },
    },
    {
        "_id": "prod-filter-flight-002",
        "sku": "FLT-TRIO-002",
        "name": "Nordic Filter Flight Set",
        "description": "Includes three filter brewing stations and reusable stainless filters.",
        "category": "coffee-equipment",
        "tags": ["filter", "sustainable", "slow-coffee"],
        "price": 1290,
        "currency": "EUR",
        "rating": 4.6,
        "popularity": 0.78,
        "sustainability": {
            "score": 0.74,
            "certifications": ["CradleToCradle"],
            "notes": "Filters designed for infinite reuse",
        },
        "inventory": {"warehouse": "Amsterdam", "stock": 18},
        "embedding": {
            "espresso-barista": 0.55,
            "travel-kit": 0.33,
            "eco-office": 0.71,
        },
    },
    {
        "_id": "prod-travel-kit-003",
        "sku": "TRV-KIT-003",
        "name": "Nomad Brew Travel Kit",
        "description": "Foldable, solar-assisted travel brewing kit for field teams.",
        "category": "travel",
        "tags": ["mobile", "camping", "solar", "portable"],
        "price": 480,
        "currency": "EUR",
        "rating": 4.4,
        "popularity": 0.69,
        "sustainability": {
            "score": 0.77,
            "certifications": ["SolarImpact"],
            "notes": "Includes integrated solar panel",
        },
        "inventory": {"warehouse": "Lisbon", "stock": 25},
        "embedding": {
            "espresso-barista": 0.36,
            "travel-kit": 0.93,
            "eco-office": 0.58,
        },
    },
    {
        "_id": "prod-office-bar-004",
        "sku": "OFF-BAR-004",
        "name": "Eco Office Coffee Bar",
        "description": "Intelligent energy-managed coffee bar and dispenser for large offices.",
        "category": "office",
        "tags": ["office", "smart", "energy", "IoT"],
        "price": 9450,
        "currency": "EUR",
        "rating": 4.7,
        "popularity": 0.88,
        "sustainability": {
            "score": 0.9,
            "certifications": ["LEED"],
            "notes": "Recycled aluminum chassis",
        },
        "inventory": {"warehouse": "Berlin", "stock": 4},
        "embedding": {
            "espresso-barista": 0.79,
            "travel-kit": 0.41,
            "eco-office": 0.95,
        },
    },
    {
        "_id": "prod-bulk-roast-005",
        "sku": "BLK-RST-005",
        "name": "Circular Roast Bulk Program",
        "description": "Carbon-neutral roasted bean subscription for office consumption.",
        "category": "beans",
        "tags": ["subscription", "carbon-neutral", "organic"],
        "price": 620,
        "currency": "EUR",
        "rating": 4.8,
        "popularity": 0.83,
        "sustainability": {
            "score": 0.92,
            "certifications": ["FairTrade", "B-Corp"],
            "notes": "Compostable packaging",
        },
        "inventory": {"warehouse": "Hamburg", "stock": 55},
        "embedding": {
            "espresso-barista": 0.51,
            "travel-kit": 0.47,
            "eco-office": 0.89,
        },
    },
    {
        "_id": "prod-ceramic-cups-006",
        "sku": "CRM-CUP-006",
        "name": "Thermo Ceramic Cup Duo",
        "description": "Dual-wall ceramic cup set that retains heat, finished with sustainable glaze.",
        "category": "serveware",
        "tags": ["cup", "ceramic", "reusable"],
        "price": 64,
        "currency": "EUR",
        "rating": 4.3,
        "popularity": 0.58,
        "sustainability": {
            "score": 0.68,
            "certifications": ["ZeroWaste"],
            "notes": "Fired using recovered waste heat",
        },
        "inventory": {"warehouse": "Warsaw", "stock": 200},
        "embedding": {
            "espresso-barista": 0.42,
            "travel-kit": 0.28,
            "eco-office": 0.81,
        },
    },
    {
        "_id": "prod-smart-scale-007",
        "sku": "SMT-SCL-007",
        "name": "Cloud Sync Brew Scale",
        "description": "IoT-enabled scale for barista teams with data analytics and recipe sharing.",
        "category": "accessories",
        "tags": ["IoT", "analytics", "barista"],
        "price": 350,
        "currency": "EUR",
        "rating": 4.5,
        "popularity": 0.73,
        "sustainability": {
            "score": 0.65,
            "certifications": [],
            "notes": "Powered by renewable energy supply",
        },
        "inventory": {"warehouse": "Prague", "stock": 32},
        "embedding": {
            "espresso-barista": 0.83,
            "travel-kit": 0.37,
            "eco-office": 0.67,
        },
    },
    {
        "_id": "prod-upcycled-snacks-008",
        "sku": "UPC-SNK-008",
        "name": "Upcycled Coffee Snack Box",
        "description": "Snack and energy bar set produced from upcycled coffee grounds.",
        "category": "snacks",
        "tags": ["upcycle", "vegan", "office"],
        "price": 120,
        "currency": "EUR",
        "rating": 4.1,
        "popularity": 0.61,
        "sustainability": {
            "score": 0.88,
            "certifications": ["ZeroWaste"],
            "notes": "Locally produced",
        },
        "inventory": {"warehouse": "Cologne", "stock": 140},
        "embedding": {
            "espresso-barista": 0.39,
            "travel-kit": 0.54,
            "eco-office": 0.9,
        },
    },
    {
        "_id": "prod-carbon-tracker-009",
        "sku": "CBN-TRK-009",
        "name": "Carbon Tracker Dashboard",
        "description": "SaaS platform and API that tracks coffee program emissions.",
        "category": "software",
        "tags": ["SaaS", "analytics", "sustainability"],
        "price": 2100,
        "currency": "EUR",
        "rating": 4.9,
        "popularity": 0.81,
        "sustainability": {
            "score": 0.96,
            "certifications": ["ScienceBasedTargets"],
            "notes": "Data centers run on renewable energy",
        },
        "inventory": {"warehouse": "Remote", "stock": 999},
        "embedding": {
            "espresso-barista": 0.45,
            "travel-kit": 0.36,
            "eco-office": 0.98,
        },
    },
    {
        "_id": "prod-modular-cart-010",
        "sku": "MOD-CART-010",
        "name": "Modular Cold Brew Cart",
        "description": "Mobile service cart that prepares cold brew and monitors water consumption.",
        "category": "coffee-equipment",
        "tags": ["cold brew", "mobile", "water-saving"],
        "price": 2680,
        "currency": "EUR",
        "rating": 4.4,
        "popularity": 0.66,
        "sustainability": {
            "score": 0.8,
            "certifications": ["WaterSense"],
            "notes": "Closed-loop water recovery system",
        },
        "inventory": {"warehouse": "Munich", "stock": 11},
        "embedding": {
            "espresso-barista": 0.6,
            "travel-kit": 0.59,
            "eco-office": 0.77,
        },
    },
]


async def init_mongodb_client() -> None:
    """Initialize MongoDB client and collections."""
    global mongo_client, demo_db, products_collection

    if mongo_client is not None and products_collection is not None:
        return

    mongodb_url = os.environ.get("MONGODB_URL")
    if not mongodb_url:
        import sys
        print("\n❌ Error: MONGODB_URL environment variable is not set.", file=sys.stderr)
        print("\nExport the variable:", file=sys.stderr)
        print("  export MONGODB_URL='your-mongodb-connection-string'", file=sys.stderr)
        print("\nSee README.md for detailed setup instructions.\n", file=sys.stderr)
        sys.exit(1)
    mongo_client = AsyncMongoClient(mongodb_url, tlsCAFile=certifi.where())
    await mongo_client.admin.command("ping")
    demo_db = mongo_client.get_database("agentic_search_demo")
    products_collection = demo_db.get_collection("products")
    print("Connected to MongoDB using PyMongo async driver.")


async def shutdown_mongodb_client() -> None:
    """Close MongoDB client."""
    global mongo_client, demo_db, products_collection

    if mongo_client is not None:
        await mongo_client.close()
        mongo_client = None

    demo_db = None
    products_collection = None


async def seed_demo_products() -> int:
    """Reset the MongoDB collection with the curated demo product catalog."""
    if demo_db is None or products_collection is None:
        raise RuntimeError(
            "MongoDB resources are not initialised. Call init_mongodb_client() first."
        )

    timestamp = datetime.now(UTC)
    enriched_payload: list[dict[str, Any]] = []
    for product in demo_products:
        enriched_payload.append(
            product
            | {
                "created_at": timestamp,
                "updated_at": timestamp,
                "sustainability_index": round(product["sustainability"]["score"] * 100),
                "price_bucket": "premium" if product["price"] > 1500 else "core",
            }
        )

    await demo_db.drop_collection("products")
    await products_collection.insert_many(enriched_payload)
    # Index layout mirrors the operations performed by the agent's tools.
    await products_collection.create_index(
        [("name", "text"), ("description", "text"), ("tags", "text")],
        name="product_text",
    )
    await products_collection.create_index(
        [("category", ASCENDING)], name="category_idx"
    )
    await products_collection.create_index(
        [("sustainability_index", DESCENDING)], name="sustainability_idx"
    )
    await products_collection.create_index(
        [("price_bucket", ASCENDING), ("category", ASCENDING)],
        name="price_category_idx",
    )
    return len(enriched_payload)


def _ensure_collection() -> Any:
    """Guard helper to verify MongoDB initialisation before a query runs."""
    if products_collection is None:
        raise RuntimeError(
            "Products collection is not initialised. Call init_mongodb_client() first."
        )
    return products_collection


def _clean_doc(
    raw: Mapping[str, Any], extra_keys: Iterable[str] | None = None
) -> dict[str, Any]:
    """Prepare a MongoDB document for display by selecting stable fields."""
    keep_keys = {
        "_id",
        "name",
        "category",
        "price",
        "price_bucket",
        "rating",
        "popularity",
        "sustainability_index",
        "tags",
    }
    if extra_keys:
        keep_keys.update(extra_keys)

    clean: dict[str, Any] = {}
    for key in keep_keys:
        if key in raw:
            value = raw[key]
            if isinstance(value, datetime):
                clean[key] = value.isoformat()
            else:
                clean[key] = value
    return clean


def _format_inputs(inputs: dict[str, Any]) -> str:
    """Return deterministic JSON for logging tool payloads."""
    return json.dumps(inputs, indent=2, ensure_ascii=False)


async def mongo_keyword_search(
    term: str | None = None,
    *,
    keywords: list[str] | None = None,
    limit: int = 6,
    fields: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    sort_by: tuple[str, Literal[1, -1]] | None = None,
) -> dict[str, Any]:
    """Execute a MongoDB text search."""
    collection = _ensure_collection()

    search_term = term
    if keywords:
        search_term = " ".join(str(keyword) for keyword in keywords if keyword)
    if not search_term:
        raise ValueError(
            "mongo.find.keyword payload must include 'term' or 'keywords'."
        )

    print(f"[mongo_keyword_search] term='{search_term}' limit={limit}")

    projection: dict[str, Any] = {"score": {"$meta": "textScore"}}
    base_projection = [
        "name",
        "category",
        "price",
        "price_bucket",
        "rating",
        "popularity",
        "sustainability_index",
        "tags",
    ]
    for proj_field in fields or base_projection:
        projection[proj_field] = 1

    query: dict[str, Any] = {"$text": {"$search": search_term}}
    if filters:
        query.update(filters)

    cursor = collection.find(query, projection)
    cursor = cursor.sort([("score", {"$meta": "textScore"})])
    if sort_by:
        cursor = cursor.sort([sort_by])
    cursor = cursor.limit(limit)
    docs = [doc async for doc in cursor]
    result = {
        "documents": [
            _clean_doc(doc, extra_keys=["score"]) | {"score": doc.get("score")}
            for doc in docs
        ],
        "meta": {"count": len(docs)},
        "summary": "",
    }

    result["summary"] = (
        f"Keyword search for '{search_term}' returned {result['meta']['count']} docs (limit {limit})."
    )
    return result


async def mongo_faceted_search(
    match: dict[str, Any] | None = None,
    facet_fields: list[str] | None = None,
    limit: int = 12,
    sort: dict[str, int] | None = None,
    facet_limit: int = 6,
) -> dict[str, Any]:
    """Run a faceted aggregation that surfaces both exemplars and buckets."""
    collection = _ensure_collection()
    match = match or {}
    facet_fields = facet_fields or ["category", "price_bucket"]

    pipeline: list[dict[str, Any]] = []
    if match:
        pipeline.append({"$match": match})

    top_documents_pipeline: list[dict[str, Any]] = []
    if sort:
        top_documents_pipeline.append({"$sort": sort})
    else:
        top_documents_pipeline.append(
            {"$sort": {"sustainability_index": -1, "popularity": -1}}
        )
    top_documents_pipeline.append({"$limit": limit})
    top_documents_pipeline.append(
        {
            "$project": {
                "name": 1,
                "category": 1,
                "price": 1,
                "price_bucket": 1,
                "sustainability_index": 1,
                "popularity": 1,
                "tags": 1,
            }
        }
    )

    facets: dict[str, list[dict[str, Any]]] = {"top_documents": top_documents_pipeline}
    for facet_field in facet_fields:
        facets[f"{facet_field}_facet"] = [
            {"$group": {"_id": f"${facet_field}", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": facet_limit},
        ]

    pipeline.append({"$facet": facets})
    agg_cursor = await collection.aggregate(pipeline, allowDiskUse=True)
    agg_result = [doc async for doc in agg_cursor]
    raw_result = agg_result[0] if agg_result else {key: [] for key in facets}

    docs = [_clean_doc(doc) for doc in raw_result.get("top_documents", [])]
    facets = {
        key: raw_result.get(key, []) for key in raw_result if key != "top_documents"
    }
    summary = f"Faceted search matched {len(docs)} exemplar docs; facets: {', '.join(facets.keys()) or 'none'}."
    return {"documents": docs, "facets": facets, "summary": summary}


async def mongo_pipeline_search(pipeline: list[dict[str, Any]]) -> dict[str, Any]:
    """Execute an arbitrary aggregation pipeline with minimal validation."""
    collection = _ensure_collection()

    # Check if pipeline has a $limit stage
    has_limit = any("$limit" in stage for stage in pipeline)

    # Validate existing limit or auto-inject one
    MAX_LIMIT = 20  # Adjust this value if you need more results
    if has_limit:
        # Check that limit value is <= MAX_LIMIT
        for stage in pipeline:
            if "$limit" in stage:
                limit_value = stage["$limit"]
                if limit_value > MAX_LIMIT:
                    raise ValueError(f"mongo.aggregate.pipeline requires a $limit stage <= {MAX_LIMIT} (got {limit_value})")
    else:
        # Auto-inject a default $limit at the end
        pipeline.append({"$limit": MAX_LIMIT})

    agg_cursor = await collection.aggregate(pipeline, allowDiskUse=True)
    docs = [
        _clean_doc(doc, extra_keys=["score", "vector_score", "match_reason"])
        async for doc in agg_cursor
    ]
    summary = f"Pipeline executed with $limit producing {len(docs)} docs."
    return {"documents": docs, "summary": summary, "pipeline": pipeline}


async def mongo_vector_search(
    query_embedding_label: str, top_k: int = 6
) -> dict[str, Any]:
    """Simulate a vector query by sorting on precomputed embedding scores."""
    collection = _ensure_collection()

    if query_embedding_label not in {"espresso-barista", "travel-kit", "eco-office"}:
        raise ValueError("Invalid embedding label; must match system prompt contract.")
    if top_k > 10:
        raise ValueError("mongo.aggregate.vector top_k must be <= 10")

    embed_field = f"embedding.{query_embedding_label}"

    pipeline = [
        {"$addFields": {"vector_score": {"$ifNull": [f"${embed_field}", 0]}}},
        {"$sort": {"vector_score": -1}},
        {"$limit": top_k},
        {
            "$project": {
                "name": 1,
                "category": 1,
                "price": 1,
                "price_bucket": 1,
                "vector_score": 1,
                "tags": 1,
                "sustainability_index": 1,
            }
        },
    ]
    agg_cursor = await collection.aggregate(pipeline, allowDiskUse=True)
    cleaned = [_clean_doc(doc, extra_keys=["vector_score"]) async for doc in agg_cursor]
    summary = f"Vector-style search on '{query_embedding_label}' produced {len(cleaned)} candidates."
    return {"documents": cleaned, "summary": summary}


def format_tool_transcript(
    tool: MongoTool, inputs: dict[str, Any], outcome: dict[str, Any]
) -> str:
    """Produce a human-friendly log entry summarising a tool invocation."""
    parts = [
        f"tool={tool.value}",
        f"inputs={_format_inputs(inputs)}",
        f"summary={outcome.get('summary', 'n/a')}",
    ]
    documents = outcome.get("documents") or []
    if documents:
        preview_lines = []
        for doc in documents[:3]:
            preview_lines.append(
                f"- {doc.get('_id')} | {doc.get('name')} | category={doc.get('category')} | sustainability_index={doc.get('sustainability_index')}"
            )
        parts.append("top_documents=\n" + "\n".join(preview_lines))
    facets = outcome.get("facets")
    if facets:
        facet_lines = []
        for facet_name, buckets in facets.items():
            sample = ", ".join(
                f"{bucket['_id']}:{bucket['count']}" for bucket in buckets[:3]
            )
            facet_lines.append(f"{facet_name} -> {sample}")
        parts.append("facets=" + " | ".join(facet_lines))
    return "\n".join(parts)


tool_dispatch: dict[
    MongoTool, Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
] = {
    MongoTool.KEYWORD: lambda payload: mongo_keyword_search(
        term=payload.get("term"),
        keywords=payload.get("keywords"),
        limit=payload.get("limit", 6),
        fields=payload.get("fields"),
        filters=payload.get("filters"),
        sort_by=tuple(payload["sort_by"]) if payload.get("sort_by") else None,
    ),
    MongoTool.FACETED: lambda payload: mongo_faceted_search(
        match=payload.get("match"),
        facet_fields=payload.get("facet_fields"),
        limit=payload.get("limit", 12),
        sort=payload.get("sort"),
        facet_limit=payload.get("facet_limit", 6),
    ),
    MongoTool.PIPELINE: lambda payload: mongo_pipeline_search(
        pipeline=payload["pipeline"]
    ),
    MongoTool.VECTOR: lambda payload: mongo_vector_search(
        query_embedding_label=payload["query_embedding_label"],
        top_k=payload.get("top_k", 6),
    ),
}
