# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed  the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # Adapted from https://github.com/PeterGriffinJin/Search-R1/blounderb/main/search_r1/search/retrieval_server.py

import argparse
import os
import json
import time
import warnings
import threading
import asyncio
import logging
from typing import List, Optional
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime

import datasets
import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# 全局请求统计
class RequestMonitor:
    def __init__(self):
        self.active_requests = 0
        self.total_requests = 0
        self.request_times = []
        self.processing_times = []  # 存储请求处理时间
        self.lock = threading.Lock()
        self.start_time = time.time()
        
    def start_request(self):
        with self.lock:
            self.active_requests += 1
            self.total_requests += 1
            self.request_times.append(time.time())
            
    def end_request(self, processing_time=None):
        with self.lock:
            self.active_requests = max(0, self.active_requests - 1)
            if processing_time is not None:
                self.processing_times.append(processing_time)
            
    def get_stats(self):
        with self.lock:
            current_time = time.time()
            # 清理旧的请求时间记录
            self.request_times = [t for t in self.request_times if current_time - t < 300]  # 保留5分钟内的记录
            
            # 计算平均处理时间
            if self.processing_times:
                avg_processing_time = sum(self.processing_times) / len(self.processing_times)
                # 只保留最近1000个请求的处理时间，避免内存无限增长
                if len(self.processing_times) > 1000:
                    self.processing_times = self.processing_times[-1000:]
            else:
                avg_processing_time = 0
            
            # 计算平均每分钟吞吐量
            uptime = current_time - self.start_time
            uptime_minutes = uptime / 60.0
            avg_throughput_per_min = self.total_requests / uptime_minutes if uptime_minutes > 0 else 0
            
            return {
                'active_requests': self.active_requests,
                'total_requests': self.total_requests,
                'avg_throughput_per_min': round(avg_throughput_per_min, 2),
                'avg_processing_time': round(avg_processing_time, 3),
                'uptime_seconds': round(uptime, 1)
            }

# 全局监控实例
request_monitor = RequestMonitor() 

def start_monitoring_thread():
    def monitor():
        while True:
            stats = request_monitor.get_stats()
            print(
                f"[retrieve] active_requests={stats['active_requests']} "
                f"total_requests={stats['total_requests']} "
                f"avg_throughput_per_min={stats['avg_throughput_per_min']} "
                f"avg_processing_time={stats['avg_processing_time']}s "
                f"uptime_seconds={stats['uptime_seconds']}s"
            )
            time.sleep(5)

    t = threading.Thread(target=monitor, daemon=True)
    t.start()


def load_corpus(corpus_path: str):
    corpus = []
    with open(corpus_path, 'r') as f:
        for line in tqdm(f):
            obj = json.loads(line)
            corpus.append(obj)
    return corpus # list of dict


def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results


class Encoder:
    def __init__(self, model_name, max_length):
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', use_fast=True, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32, device_map='cuda:0')
        self.encoder.eval()
    @torch.no_grad()
    def encode(self, query_list: list[str]) -> np.ndarray:
        if isinstance(query_list, str):
            query_list = [query_list]
        marker = "\nuser\n"
        processed_query_list = []
        for query in query_list:
            pos = query.find(marker)
            processed_text = query[pos + len(marker):]
            processed_text = processed_text[-1300:]
            processed_query_list.append(processed_text)
        encoder_input = self.tokenizer(
            processed_query_list,
            padding=True, 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        encoder_input_ids = encoder_input['input_ids'].to(self.encoder.device)
        encoder_attention_mask = encoder_input['attention_mask'].to(self.encoder.device)

        model_output = self.encoder(
            input_ids=encoder_input_ids, 
            attention_mask=encoder_attention_mask
        )
        output = model_output.last_hidden_state[:, -1, :]
        query_emb = torch.nn.functional.normalize(output, dim=-1)
        query_emb = query_emb.detach().cpu().numpy()

        del encoder_input, encoder_input_ids, encoder_attention_mask, model_output, output
        torch.cuda.empty_cache()
        return query_emb


class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk

        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)

    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)


class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher(self.index_path)
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8

    def _check_contain_doc(self):
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            if return_score:
                return [], []
            else:
                return []
        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn("Not enough documents retrieved!", stacklevel=2)
        else:
            hits = hits[:num]

        if self.contain_doc:
            all_contents = [json.loads(self.searcher.doc(hit.docid).raw())["contents"] for hit in hits]
            results = [{"title": content.split("\n")[0].strip('"'), "text": "\n".join(content.split("\n")[1:]), "contents": content} for content in all_contents]
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        else:
            return results


class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.index = faiss.read_index(self.index_path)

        self.corpus = load_corpus(self.corpus_path)
        self.encoder = Encoder(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            max_length=config.retrieval_query_max_length
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size
        self.embedding_dim = self.index.d
        dummy_query = np.random.random((1, self.embedding_dim)).astype('float32')
        self.index.search(dummy_query, k=10)


    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]
        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores.tolist()
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        try:
            if isinstance(query_list, str):
                query_list = [query_list]
            if num is None:
                num = self.topk

            results = []
            scores = []
            for start_idx in range(0, len(query_list), self.batch_size):
                query_batch = query_list[start_idx : start_idx + self.batch_size]
                batch_emb = self.encoder.encode(query_batch)
                batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
                batch_scores = batch_scores.tolist()
                batch_idxs = batch_idxs.tolist()
                # load_docs is not vectorized, but is a python list approach
                flat_idxs = sum(batch_idxs, [])
                batch_results = load_docs(self.corpus, flat_idxs)
                # chunk them back
                batch_results = [batch_results[i * num : (i + 1) * num] for i in range(len(batch_idxs))]

                results.extend(batch_results)
                scores.extend(batch_scores)

                del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
                torch.cuda.empty_cache()

            if return_score:
                return results, scores
            else:
                return results
        except Exception as e:
            print(e)
            if return_score:
                return [], []
            else:
                return []


def get_retriever(config):
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    else:
        return DenseRetriever(config)


#####################################
# FastAPI server below
#####################################


class Config:
    """
    Minimal config class (simulating your argparse)
    Replace this with your real arguments or load them dynamically.
    """

    def __init__(
        self,
        retrieval_method: str = "dense",
        retrieval_topk: int = 10,
        index_path: str = None,
        corpus_path: str = None,
        dataset_path: str = "./rollout",
        data_split: str = "train",
        retrieval_query_max_length: int = 512,
        retrieval_batch_size: int = 512,
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_batch_size = retrieval_batch_size


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()
BATCH_MAX_WAIT_MS = int(os.getenv("RETRIEVER_BATCH_MAX_WAIT_MS", "100"))
BATCH_MAX_REQS = int(os.getenv("RETRIEVER_BATCH_MAX_REQS", "256"))


@dataclass
class _PendingReq:
    queries: List[str]
    topk: int
    return_scores: bool
    future: asyncio.Future


class _SimpleBatcher:
    def __init__(self, retriever_obj, cfg_obj):
        self.queue: asyncio.Queue[_PendingReq] = asyncio.Queue()
        self.retriever = retriever_obj
        self.config = cfg_obj
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="retrieval-simple-batcher")

    async def submit(self, queries: List[str], topk: int, return_scores: bool):
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self.queue.put(_PendingReq(queries=queries, topk=topk, return_scores=return_scores, future=fut))
        return await fut

    async def _loop(self):
        while True:
            req: _PendingReq = await self.queue.get()
            batch: List[_PendingReq] = [req]
            deadline = asyncio.get_running_loop().time() + (BATCH_MAX_WAIT_MS / 1000.0)
            while len(batch) < BATCH_MAX_REQS:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    nxt: _PendingReq = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                    batch.append(nxt)
                except asyncio.TimeoutError:
                    break

            try:
                flat_queries: List[str] = []
                segments: List[tuple[int, int, _PendingReq]] = []
                for r in batch:
                    start_idx = len(flat_queries)
                    q = r.queries if isinstance(r.queries, list) else [r.queries]
                    flat_queries.extend(q)
                    segments.append((start_idx, len(q), r))

                if not flat_queries:
                    for _, _, r in segments:
                        if not r.future.done():
                            r.future.set_result(([], [], 0))
                    continue

                batch_topk = max(r.topk for r in batch)
                results, scores = self.retriever.batch_search(
                    query_list=flat_queries, num=batch_topk, return_score=True
                )


                for start, count, r in segments:
                    sub_results = results[start : start + count]
                    sub_scores = scores[start : start + count] if scores is not None else None
                    if batch_topk != r.topk:
                        sub_results = [lst[: r.topk] for lst in sub_results]
                        if sub_scores is not None:
                            sub_scores = [lst[: r.topk] for lst in sub_scores]

                    if not r.future.done():
                        r.future.set_result((sub_results, sub_scores, len(flat_queries)))
            except Exception as e:
                for r in batch:
                    if not r.future.done():
                        r.future.set_exception(e)


batcher: Optional[_SimpleBatcher] = None


@app.post("/retrieve")
async def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.

    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }

    Output format (when return_scores=True，similarity scores are returned):
    {
        "result": [
            [   # Results for each query
                    {"document": doc, "reward": reward, "score": score}
                # ... more documents
            ],
            # ... results for other queries
        ]
    }
    """
    # 开始请求监控
    request_monitor.start_request()
    start_time = time.time()
    
    try:
        topk = request.topk or config.retrieval_topk
        # 提交到批处理器，等待本次请求的结果
        results, scores, batch_flat_query_count = await batcher.submit(queries=request.queries, topk=topk, return_scores=request.return_scores)

        resp = []
        for i, single_result in enumerate(results):
            if request.return_scores and scores is not None:
                combined = []
                for doc, score in zip(single_result, scores[i]):
                    combined.append({
                        "document": doc["value"],
                        "reward": doc["reward"],
                        "score": score
                    })
                resp.append(combined)
            else:
                resp.append(single_result)


        processing_time = time.time() - start_time

        return {"result": resp}
        
    except Exception as e:
        processing_time = time.time() - start_time
        raise e
    finally:
        # 结束请求监控，传递处理时间
        processing_time = time.time() - start_time
        request_monitor.end_request(processing_time)


@app.on_event("startup")
async def _on_startup():
    # 启动异步批处理器
    global batcher
    if 'retriever' in globals() and 'config' in globals():
        batcher = _SimpleBatcher(retriever, config)
        await batcher.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    parser.add_argument(
        "--index_path", type=str, default="./rollout/Step_Trace_Buffer.index", help="Corpus indexing file."
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="./rollout/Step_Trace_Buffer.jsonl",
        help="Local corpus file.",
    )
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")


    args = parser.parse_args()

    config = Config(
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        retrieval_query_max_length=512,
        retrieval_batch_size=512,
    )

    retriever = get_retriever(config)

    start_monitoring_thread()

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    print("Start...")
    uvicorn.run(app, host="127.0.0.1", port=8005, access_log=False)
