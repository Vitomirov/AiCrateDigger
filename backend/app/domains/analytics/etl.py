import asyncio
import json
import logging
from sqlalchemy import select
from app.core.db.database import session_factory, init_db, SearchResponseCacheORM
from app.core.config import get_settings
from app.domains.analytics.models import SearchAnalyticsORM

logger = logging.getLogger(__name__)

async def run_etl_pipeline():
    settings = get_settings()
    url = settings.resolved_database_url
    
    if not url:
        logger.error("ETL aborted: Database is not configured.")
        return

    # Inicijalizujemo bazu sa URL-om koji config.py već razrešava
    await init_db(database_url=url, debug=settings.debug)

    async_session = session_factory()

    async with async_session() as session:
        try:
            # --- 1. EXTRACT (Izvlačenje) ---
            result = await session.execute(select(SearchResponseCacheORM))
            cache_rows = result.scalars().all()
            
            logger.info(f"Extracted {len(cache_rows)} rows from cache for ETL processing.")

            # --- 2. TRANSFORM (Transformacija i čišćenje) ---
            aggregated_queries = {}

            for row in cache_rows:
                try:
                    payload = json.loads(row.payload_json)
                    raw_query = payload.get("query")
                    
                    if not raw_query:
                        continue
                    
                    clean_query = raw_query.strip().lower()
                    
                    if clean_query in aggregated_queries:
                        aggregated_queries[clean_query] += 1
                    else:
                        aggregated_queries[clean_query] = 1
                        
                except Exception as e:
                    logger.warning(f"Failed to parse payload_json: {e}")
                    continue

            # --- 3. LOAD (Učitavanje u analitičku tabelu) ---
            for query_text, count in aggregated_queries.items():
                stmt = select(SearchAnalyticsORM).where(SearchAnalyticsORM.clean_query == query_text)
                existing = (await session.execute(stmt)).scalar_one_or_none()

                if existing:
                    existing.search_count = count
                else:
                    new_entry = SearchAnalyticsORM(
                        clean_query=query_text,
                        search_count=count
                    )
                    session.add(new_entry)

            await session.commit()
            logger.info("ETL pipeline successfully completed and loaded into database!")

        except Exception as e:
            await session.rollback()
            logger.error(f"ETL pipeline failed: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_etl_pipeline())