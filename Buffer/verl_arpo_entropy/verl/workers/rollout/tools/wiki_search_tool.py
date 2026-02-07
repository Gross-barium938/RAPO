import os
import json
import time
import queue
import logging
import threading
import requests
from typing import Optional, Tuple, Dict, List, Any
import traceback
import uuid

from verl.workers.rollout.tools.base_tool import BaseTool

DEFAULT_TIMEOUT = 30  # Default search request timeout
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 1
API_TIMEOUT = 10
logger = logging.getLogger(__name__)

def call_search_api(retrieval_service_url: str, query_list: List[str], topk: int = 3, return_scores: bool = True, timeout: int = DEFAULT_TIMEOUT) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    request_id = str(uuid.uuid4())
    log_prefix = f"[Search Request ID: {request_id}] "

    payload = {"queries": query_list, "topk": topk, "return_scores": return_scores}

    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: Calling search API at {retrieval_service_url}")
            session = requests.Session()
            session.trust_env = False  # 忽略环境变量的代理
            response = session.post(
                retrieval_service_url,
                headers=headers,
                json=payload,
                timeout=timeout,
                proxies={"http": None, "https": None}
            )
            # Check for Gateway Timeout (504) and other server errors for retrying
            if response.status_code in [500, 502, 503, 504]:
                last_error = f"{log_prefix}API Request Error: Server Error ({response.status_code}) on attempt {attempt + 1}/{MAX_RETRIES}"
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                    time.sleep(delay)
                continue

            # Check for other HTTP errors (e.g., 4xx)
            response.raise_for_status()

            # If successful (status code 2xx)
            logger.info(f"{log_prefix}Search API call successful on attempt {attempt + 1}")
            return response.json(), None

        except requests.exceptions.ConnectionError as e:
            last_error = f"{log_prefix}Connection Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.Timeout as e:
            last_error = f"{log_prefix}Timeout Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"
            break  # Exit retry loop on other request errors
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}, Response: {raw_response_text[:200]}"
            break  # Exit retry loop on JSON decode errors
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"
            break  # Exit retry loop on other unexpected errors

    # If loop finishes without returning success, return the last recorded error
    logger.error(f"{log_prefix}Search API call failed. Last error: {last_error}")
    return None, last_error.replace(log_prefix, "API Call Failed: ") if last_error else "API Call Failed after retries"


def perform_single_search_batch(retrieval_service_url: str, query_list: List[str], topk: int = 3, concurrent_semaphore: Optional[threading.Semaphore] = None, timeout: int = DEFAULT_TIMEOUT) -> Tuple[str, Dict[str, Any]]:
    api_response = None
    error_msg = None

    try:
        if concurrent_semaphore:
            with concurrent_semaphore:
                api_response, error_msg = call_search_api(retrieval_service_url=retrieval_service_url, query_list=query_list, topk=topk, return_scores=True, timeout=timeout)
        else:
            api_response, error_msg = call_search_api(retrieval_service_url=retrieval_service_url, query_list=query_list, topk=topk, return_scores=True, timeout=timeout)
    except Exception as e:
        error_msg = f"API Request Exception during batch search: {e}"
        traceback.print_exc()

    result_text = "Search request failed or timed out after retries."

    if error_msg:
        result_text = f"Search error: {error_msg}"
        logger.error(f"Batch search: API error occurred: {error_msg}")
    elif api_response:
        logger.debug(f"Batch search: API Response: {api_response}")

        try:
            raw_results = api_response.get("result", [])
            if raw_results:
                pretty_results = []
                total_results = 0

                for retrieval in raw_results:
                    formatted = _passages2string(retrieval)
                    pretty_results.append(formatted)
                    total_results += len(retrieval) if isinstance(retrieval, list) else 1

                final_result = "\n---\n".join(pretty_results)
                result_text = final_result
                logger.info(f"Batch search: Successful, got {total_results} total results")
            else:
                result_text = "No search results found."
                logger.info("Batch search: No results found")
        except Exception as e:
            error_msg = f"Error processing search results: {e}"
            result_text = error_msg
            logger.error(f"Batch search: {error_msg}")
    else:
        result_text = "Unknown API state (no response and no error message)."
        logger.error("Batch search: Unknown API state.")

    return result_text


def _passages2string(retrieval_result):
    """Convert retrieval results to formatted string."""
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx + 1} (Title: {title})\n{text}\n\n"
    return format_reference.strip()


class WikiSearchTool(BaseTool):
    def __init__(
        self,
        retrieval_service_url: str,
        topk: int,
        timeout: int,
        cache_file: Optional[str] = None
    ):
        self.timeout = timeout
        self.retrieval_service_url = retrieval_service_url
        self.topk = topk

    @property
    def name(self) -> str:
        """Tool name identifier."""
        return "wikiSearch"

    @property
    def trigger_tag(self) -> str:
        """Tag used to trigger this tool."""
        return "search"


    def execute(self, query) -> str:
        timeout = self.timeout
        query_list = [query] # temporarily

        if not query_list or not isinstance(query_list, list):
            return ""

        result_text = perform_single_search_batch(
            retrieval_service_url=self.retrieval_service_url,
            query_list=query_list,
            topk=self.topk,
            concurrent_semaphore=None,  # Ray handles concurrency control
            timeout=timeout,
        )
        return result_text